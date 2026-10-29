"""Measured, stateful routing with explicit calibration costs and fail-closed gates.

Three routes: batched within, legacy exact-tuple-compressed within, fereg-core.
No solver is silently given looser numerical acceptance. All calibration uses
full rows and the actual RHS block, never a row-subsampled graph. The measured
planner is intended for repeated projections; its first call is deliberately
more expensive than a single preselected solver.
"""
from __future__ import annotations
import copy
import math
from time import perf_counter
import numpy as np
from .structure import compact_codes, positive_weights, tuple_sketch

ROUTES = ("within", "compressed_within", "fereg")

class PlanningError(RuntimeError):
    pass

class ProjectionDisagreement(PlanningError):
    pass

class _Route:
    def __init__(self, name, codes, weights, threads, scratch_bytes):
        self.name = name
        if name == "compressed_within":
            from _econhdfe_within import NativeSolver
            self.core = NativeSolver(codes, weights=weights, strategy="compressed",
                                     threads=threads, workspace_bytes=scratch_bytes)
        else:
            from econhdfe_dual import DualAbsorber
            self.core = DualAbsorber(codes, weights=weights, backend=name,
                                    threads=threads, scratch_bytes=scratch_bytes,
                                    allow_fallback=False)

    def project(self, data, tol, max_iter):
        if self.name != "compressed_within":
            return self.core.residualize(data, tol=tol, max_iter=max_iter, return_info=True)
        out, flags, iterations, errors = self.core.solve(data, tol=tol, max_iter=max_iter)
        accepted = (all(flags) and np.isfinite(out).all() and
                    all(np.isfinite(e) and e <= max(tol, 128*np.finfo(float).eps) for e in errors))
        info = dict(accepted=bool(accepted), backend="compressed_within",
                    route=self.core.route, iterations=iterations, certificate=errors,
                    n_cells=self.core.n_cells)
        if not accepted:
            raise PlanningError(f"compressed within certificate rejected: {info}")
        return out, info

    def update_weights(self, w):
        self.core.update_weights(w)


def _weighted_gap(a, b, w, chunk=65536):
    """Columnwise weighted relative L2 difference; scale before squaring."""
    scale = np.maximum(np.max(np.abs(a), axis=0), np.max(np.abs(b), axis=0))
    scale = np.maximum(scale, np.finfo(float).tiny)
    numerator = np.zeros(a.shape[1])
    denominator = np.zeros(a.shape[1])
    for start in range(0, len(a), chunk):
        aa = a[start:start+chunk] / scale
        bb = b[start:start+chunk] / scale
        ww = w[start:start+chunk, None]
        numerator += np.sum(ww * (aa - bb)**2, axis=0)
        denominator += np.sum(ww * (aa*aa + bb*bb), axis=0)
    return float(np.sqrt(numerator / np.maximum(denominator, np.finfo(float).tiny)).max(initial=0))


class PlannedAbsorber:
    """An explicit experimental planner; does not change EconHDFE defaults.

    backend='auto' measures admitted candidates on the first real RHS block.
    probe_repeats controls additional same-input solves per candidate. Choose
    workload='reweight' to include an identical-weight numeric refresh in each
    warm probe. New weights may change relative performance; call retune() for
    an explicit new calibration. The plan is otherwise stable across solves.

    probe_budget_seconds is checked BETWEEN candidates: it cannot interrupt a
    Rust call and is NOT a hard timeout. A budget-exhausted candidate is reported
    as untested, not slower. scratch_bytes is not a process-memory/fill bound.
    """
    def __init__(self, codes, *, weights=None, backend="auto", threads=2,
                 scratch_bytes=512 << 20, probe_repeats=1,
                 probe_budget_seconds=None, workload="fixed", allow_fallback=True,
                 compression_threshold=0.7, agreement_tolerance=2e-8):
        if backend not in (*ROUTES, "auto"):
            raise ValueError("invalid backend")
        for key, val, lower in (("threads", threads, 1), ("scratch_bytes", scratch_bytes, 1024),
                                ("probe_repeats", probe_repeats, 1)):
            if isinstance(val, bool) or not isinstance(val, (int, np.integer)) or val < lower:
                raise ValueError(f"{key} must be an integer >= {lower}")
        if workload not in ("fixed", "reweight"):
            raise ValueError("workload must be fixed or reweight")
        if (probe_budget_seconds is not None and
            (not np.isfinite(probe_budget_seconds) or probe_budget_seconds <= 0)):
            raise ValueError("probe_budget_seconds must be positive or None")
        if not 0 < compression_threshold < 1 or not np.isfinite(agreement_tolerance) or agreement_tolerance <= 0:
            raise ValueError("invalid compression/agreement tolerance")
        t = perf_counter()
        self._codes, self._levels = compact_codes(codes)
        self._n = len(self._codes)
        self._w = positive_weights(weights, self._n)
        self._requested, self._threads, self._scratch = backend, int(threads), int(scratch_bytes)
        self._probe_repeats, self._budget = int(probe_repeats), probe_budget_seconds
        self._workload, self._fallback = workload, bool(allow_fallback)
        self._agreement = float(agreement_tolerance)
        self._threshold = float(compression_threshold)
        self._sketch = None
        self._admitted = None
        self._route = None
        self._selected = None
        self._generation = 0
        self._plan_id = 0
        self._pending_update = 0.0
        self._last = {}
        self._plan = {}
        self._constructor_seconds = perf_counter() - t
        self._last_input_shape = None

    def _candidates(self):
        if self._admitted is not None:
            return list(self._admitted)
        t = perf_counter()
        self._sketch = tuple_sketch(self._codes)
        compressed = self._sketch.ratio_upper <= self._threshold
        small = sum(self._levels) - max(self._levels) <= 4096
        if compressed:
            names = ["compressed_within", "fereg", "within"]
        elif small:
            names = ["fereg", "within"]
        else:
            names = ["within", "fereg"]
        self._admitted = tuple(names)
        self._sketch_seconds = perf_counter() - t
        return names

    def _new_route(self, name):
        return _Route(name, self._codes, self._w, self._threads, self._scratch)

    def _calibrate(self, data, tol, max_iter):
        begin = perf_counter()
        names = self._candidates()
        trials = []
        best = best_out = best_info = None
        best_score = math.inf
        for name in names:
            if best is not None and self._budget is not None and perf_counter() - begin >= self._budget:
                trials.append(dict(backend=name, status="untested_budget"))
                continue
            candidate = None
            try:
                t = perf_counter()
                candidate = self._new_route(name)
                construction = perf_counter() - t
                t = perf_counter()
                out, info = candidate.project(data, tol, max_iter)
                first = perf_counter() - t
                if not info.get("accepted"):
                    raise PlanningError("candidate did not return an accepted certificate")
                if best_out is not None:
                    gap = _weighted_gap(out, best_out, self._w)
                    # Near-zero projections use the original RHS as denominator,
                    # because relative residual error is undefined at exact zero.
                    if gap > self._agreement:
                        # Compute a stable input-scaled gap directly instead.
                        scale = np.maximum(np.max(np.abs(data),axis=0), np.finfo(float).tiny)
                        delta = np.max(np.abs(out/scale-best_out/scale), axis=0)
                        if float(np.max(delta)) > self._agreement:
                            raise ProjectionDisagreement(f"certified candidates disagree: {name}, gap={gap:.3e}")
                else:
                    gap = 0.0
                warm = []
                for _ in range(self._probe_repeats):
                    t = perf_counter()
                    if self._workload == "reweight":
                        candidate.update_weights(self._w)
                    update = perf_counter() - t
                    t = perf_counter()
                    again, again_info = candidate.project(data, tol, max_iter)
                    elapsed = perf_counter() - t
                    if not again_info.get("accepted"):
                        raise PlanningError("warm probe certificate rejected")
                    if not np.allclose(again, out, rtol=self._agreement, atol=self._agreement * max(1.0,float(np.max(np.abs(data))))):
                        raise ProjectionDisagreement("candidate changed projection during unchanged-weight warm probe")
                    warm.append(dict(update_seconds=update, solve_seconds=elapsed, total_seconds=update+elapsed))
                    out, info = again, again_info
                score = float(np.median([x["total_seconds"] for x in warm]))
                trial = dict(backend=name, status="accepted", construction_seconds=construction,
                             first_solve_seconds=first, cold_seconds=construction+first,
                             warm=warm, score_seconds=score, cross_candidate_gap=gap,
                             actual_route=info.get("route", info.get("backend")))
                trials.append(trial)
                if score < best_score:
                    best, best_out, best_info, best_score = candidate, out, info, score
                # Only the incumbent and one challenger may coexist.
                del candidate, out, again
            except ProjectionDisagreement:
                self._last = dict(accepted=False, phase="calibration", trials=trials)
                raise
            except (RuntimeError, ValueError, MemoryError) as exc:
                trials.append(dict(backend=name, status="failed", error=f"{type(exc).__name__}: {exc}"))
                del candidate
        if best is None:
            self._last = dict(accepted=False, phase="calibration", trials=trials)
            raise PlanningError(f"no certified candidate: {trials}")
        self._route = best
        self._selected = best.name
        self._plan_id += 1
        self._last_input_shape = data.shape
        # No alternative engines are retained after calibration.
        self._plan = dict(plan_id=self._plan_id, selected=self._selected,
                          reason="lowest_measured_warm_cost_on_full_data",
                          objective="future_call_cost; calibration is sunk and separately reported",
                          calibration_seconds=perf_counter()-begin,
                          geometry_seconds=getattr(self, "_sketch_seconds", 0.0),
                          tuple_sketch=self._sketch.to_dict(), trials=trials,
                          calibrated_rhs=data.shape[1], calibrated_tolerance=tol,
                          calibrated_weight_generation=self._generation,
                          workload=self._workload, probe_repeats=self._probe_repeats,
                          hardware_specific=True)
        return best_out, best_info

    def residualize(self, data, *, tol=1e-10, max_iter=6000, return_info=False):
        y = np.asarray(data, dtype=np.float64)
        vector = y.ndim == 1
        if vector:
            y = y[:, None]
        if y.ndim != 2 or len(y) != self._n or not np.isfinite(y).all():
            raise ValueError("RHS must be finite with shape (n,) or (n,rhs)")
        if not np.isfinite(tol) or not 0 < tol <= 0.1:
            raise ValueError("tol must be in (0,0.1]")
        if isinstance(max_iter,bool) or not isinstance(max_iter,(int,np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        t_all = perf_counter()
        if y.shape[1] == 0:
            out = y.copy()
            self._last = dict(accepted=True, backend="none", rhs=0, plan_id=self._plan_id,
                              weight_generation=self._generation, call_seconds=perf_counter()-t_all)
            return (out, self.diagnostics) if return_info else out
        calibrated = self._requested == "auto" and self._route is None and self._selected is None
        if calibrated:
            out, native = self._calibrate(y, tol, int(max_iter))
        else:
            if self._route is None:
                self._selected = self._selected or self._requested
                self._route = self._new_route(self._selected)
            try:
                out, native = self._route.project(y, tol, int(max_iter))
            except (RuntimeError, ValueError, MemoryError) as first:
                if not self._fallback:
                    raise
                failed_name = self._selected
                self._route = None
                errors = [dict(backend=failed_name,error=str(first))]
                for name in self._candidates():
                    if name == failed_name:
                        continue
                    try:
                        alternate = self._new_route(name)
                        out, native = alternate.project(y, tol, int(max_iter))
                        self._route, self._selected = alternate, name
                        self._plan = dict(self._plan, selected=name, reason="certification_failover",
                                          fallback_errors=errors)
                        break
                    except (RuntimeError, ValueError, MemoryError) as exc:
                        errors.append(dict(backend=name,error=str(exc)))
                else:
                    self._selected = failed_name
                    raise PlanningError(f"all fallback candidates failed: {errors}") from first
        self._last = dict(accepted=True, backend=self._selected, calibrated_this_call=calibrated,
                          plan_id=self._plan_id, weight_generation=self._generation,
                          call_seconds=perf_counter()-t_all, update_seconds=self._pending_update,
                          constructor_seconds=self._constructor_seconds,
                          plan_rhs_changed=(self._last_input_shape is not None and self._last_input_shape[1]!=y.shape[1]),
                          plan=self._plan, projection=native)
        self._pending_update = 0.0
        if vector:
            out = out[:, 0]
        return (out, self.diagnostics) if return_info else out

    def update_weights(self, weights):
        """Validate before mutation; internal failure invalidates numerical cache.

        The old public weights and generation remain authoritative on error.
        The next solve reconstructs the selected kernel for the OLD weights.
        """
        t = perf_counter()
        new_w = positive_weights(weights, self._n)
        try:
            if self._route is not None:
                self._route.update_weights(new_w)
        except (RuntimeError, ValueError, MemoryError):
            self._route = None
            raise
        self._w = new_w
        self._generation += 1
        self._pending_update += perf_counter()-t

    def retune(self):
        """Explicitly remeasure on the next nonempty RHS; no background work."""
        if self._requested != "auto":
            raise ValueError("retune is only available with backend='auto'")
        self._route = None
        self._selected = None

    @property
    def backend(self):
        return self._selected or "uninitialized"

    @property
    def weight_generation(self):
        return self._generation

    @property
    def diagnostics(self):
        return copy.deepcopy(self._last)
