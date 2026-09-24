"""Shared-setup CPU comparator for finite augmented wild-gradient QR.

Credit and scope
----------------
This is a strong *known-method baseline*, not a new QR algorithm. Residual-sign
preprocessing, leverage normalization, tail aggregation, sign repair and doubling
come from Portnoy--Koenker (1997), Chernozhukov--Fernandez-Val--Melly (2020), and
quantreg's boot.rq.pxy/pwxy implementations:
https://arxiv.org/abs/1909.05782
https://github.com/cran/quantreg/blob/master/R/boot.R

CFM's tail globs are implemented here by fixing tail dual coordinates at their
box bounds and moving their sufficient statistics to the equality RHS. When
the final tail signs are correct this is algebraically the same compression;
we do not introduce artificial big-M tail observations. The actual quantreg
finite pseudo-observation (W/tau, n*max(abs(y))) is free by default. Optional
positive_pseudo first tries its upper dual face, using an outward strict sign
guard and the original finite checks. Rejected trials restart the original
free strategy; the finite target is never replaced by an unrestricted tilt.

Differences from literal CFM bootstrap code: these are gradient-perturbed fits,
not weighted-pairs fits; the default selector additionally uses a kernel
Jacobian one-step location and the supplied W second moment to guess the active
set. Both are only heuristics for selection. A full-sample residual/dual check
and exact reduced LP are still required; no one-step estimate is returned.
selector='cfm' disables that predictor and uses the shared CFM quantile window.
"""

from pathlib import Path
import argparse
import json
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .positive_pseudo import pseudo_residual_lower

try:
    from .lp import _lp, full_fit
except ImportError:
    from .lp import _lp, full_fit


class CFMPreprocessor:
    """One setup per (X, y, tau, beta0), then fit_batch(W[p, B]).

    Dense design only. Memory is O(np + p^2 + n + Bp), excluding caller-owned
    inputs and temporary bounded row blocks. No n-by-B array is allocated.
    Inputs must remain unchanged while the object is used. Setup timings are
    reported separately and must be charged in any end-to-end comparison.
    """

    def __init__(self, X, y, tau, beta0, *, selector="covariance", factor=None,
                 max_rounds=10, block_rows=8192, solver_limit=30.0,
                 sign_tolerance=1e-9, equality_tolerance=1e-6,
                 gap_tolerance=1e-6, positive_pseudo=False):
        start = time.perf_counter()
        self.x = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64).reshape(-1)
        self.beta0 = np.asarray(beta0, dtype=np.float64).reshape(-1).copy()
        if self.x.ndim != 2:
            raise ValueError("X must be a dense n-by-p matrix")
        self.n, self.p = self.x.shape
        if self.n < self.p or self.p == 0 or self.y.size != self.n:
            raise ValueError("Require nonempty n >= p and matching X/y")
        if self.beta0.size != self.p or not 0 < tau < 1:
            raise ValueError("beta0 must have p entries and 0 < tau < 1")
        if not all(np.isfinite(a).all() for a in (self.x, self.y, self.beta0)):
            raise ValueError("Inputs must be finite and missing values removed")
        if selector not in ("covariance", "cfm"):
            raise ValueError("selector must be 'covariance' or 'cfm'")
        if factor is None:
            # quantreg's shared-coefficient pxy implementation uses 3*sqrt(np).
            # The directed predictor uses a smaller, explicitly separate rule.
            factor = 3.0 if selector == "cfm" else 1.0
        if factor <= 0 or max_rounds < 1 or block_rows < 1:
            raise ValueError("Invalid size or iteration controls")
        self.tau = float(tau)
        self.selector = selector
        self.factor = float(factor)
        self.max_rounds = int(max_rounds)
        self.block_rows = int(block_rows)
        self.solver_limit = float(solver_limit)
        if not isinstance(positive_pseudo, (bool, np.bool_)):
            raise ValueError('positive_pseudo must be boolean')
        self.positive_pseudo = bool(positive_pseudo)
        self.sign_tolerance = float(sign_tolerance)
        self.equality_tolerance = float(equality_tolerance)
        self.gap_tolerance = float(gap_tolerance) * (1 + np.mean(np.abs(self.y)))
        self.r0 = self.y - self.x @ self.beta0
        self.pseudo_y = self.n * np.max(np.abs(self.y))
        self.base_a = np.where(self.r0 < 0, self.tau - 1, self.tau)
        self.base_total = self.x.T @ self.base_a
        self.gram = self.x.T @ self.x
        self.gram_inverse, self.gram_pseudoinverse = self._inverse(self.gram)
        self.leverage = np.sqrt(np.maximum(self._diag_quad(self.gram_inverse), 0))
        self.leverage = np.maximum(self.leverage, np.finfo(float).eps)
        self.cfm_order = np.argsort(self.r0 / self.leverage)
        spread = min(float(np.std(self.r0)),
                     float(np.subtract(*np.quantile(self.r0, [.75, .25]))) / 1.349)
        self.residual_scale = max(spread, 1e-10 * (1 + np.mean(np.abs(self.y))))
        self.bandwidth = .9 * self.residual_scale * self.n ** (-.2)
        self.jacobian_inverse = None
        self.jacobian_pseudoinverse = False
        if selector == "covariance":
            # Smooth curvature is used ONLY for screening, never for estimates
            # or standard errors. A bad density estimate just causes repairs.
            z = self.r0 / self.bandwidth
            density = np.exp(-.5 * z * z) / (np.sqrt(2 * np.pi) * self.bandwidth)
            jacobian = np.zeros((self.p, self.p))
            for sl in self._blocks():
                xx = self.x[sl]
                jacobian += xx.T @ (xx * density[sl, None])
            self.jacobian_inverse, self.jacobian_pseudoinverse = self._inverse(jacobian)
        self.setup_seconds = time.perf_counter() - start

    def _blocks(self):
        for lo in range(0, self.n, self.block_rows):
            yield slice(lo, min(self.n, lo + self.block_rows))

    @staticmethod
    def _inverse(matrix):
        try:
            factor = cho_factor(matrix, lower=True, check_finite=False)
            return cho_solve(factor, np.eye(matrix.shape[0]), check_finite=False), False
        except np.linalg.LinAlgError:
            # Only the screening scale uses this inverse. Numerical rank issues
            # remain visible, and LP/certificate failures are not suppressed.
            return np.linalg.pinv(matrix, hermitian=True), True

    def _diag_quad(self, matrix):
        answer = np.empty(self.n)
        for sl in self._blocks():
            xx = self.x[sl]
            answer[sl] = np.einsum("ij,ij->i", xx @ matrix, xx)
        return answer

    def _cfm_indices(self, size):
        center = int(round(self.n * self.tau))
        lo = max(0, min(self.n - size, center - size // 2))
        return self.cfm_order[lo:lo + size]

    def _diagnostics(self, w, beta, a, a0, residual):
        # Same local Fenchel calculation as cpu_candidate.check, reusing the
        # already-computed full residual. Feasibility is checked independently
        # with X.T@a, rather than relying on cached tail-sum cancellation.
        d = w / self.tau
        pseudo_r = self.pseudo_y - d @ beta
        loss = np.maximum(self.tau * residual, (self.tau - 1) * residual)
        local = loss - a * residual
        local0 = max(self.tau * pseudo_r, (self.tau - 1) * pseudo_r) - a0 * pseudo_r
        eq = self.x.T @ a + d * a0
        gap = float(np.maximum(local, 0).sum() + max(local0, 0))
        equality = float(np.max(np.abs(eq)))
        box = float(max(np.max(a - self.tau), np.max(self.tau - 1 - a),
                        a0 - self.tau, self.tau - 1 - a0, 0))
        return dict(gap=gap, equality_residual=equality, box_violation=box,
                    pseudo_residual=float(pseudo_r), pseudo_dual=float(a0),
                    numerically_checked=bool(gap <= self.gap_tolerance and
                                             equality <= self.equality_tolerance and
                                             box <= self.equality_tolerance))

    def fit_batch(self, W, *, W_cov=None):
        """Return (beta[B,p], diagnostics).

        W is the gradient used by quantreg's finite augmented problem, NOT its
        negation. Optional W_cov[p,p] may be the exact cluster-score covariance;
        otherwise W@W.T/B estimates its uncentered second moment (E[W]=0).
        This covariance controls screening only. B=1 is supported. Supplied
        columns are deterministic; there is no randomization or discarded draw.
        """
        start = time.perf_counter()
        w = np.asarray(W, dtype=np.float64)
        if w.ndim == 1:
            w = w[:, None]
        if w.ndim != 2 or w.shape[0] != self.p or not np.isfinite(w).all():
            raise ValueError("W must be a finite p-by-B matrix")
        count = w.shape[1]
        predicted_delta = np.zeros_like(w)
        scale = self.leverage
        if self.selector == "covariance" and count:
            inv = self.jacobian_inverse
            predicted_delta = inv @ w
            cov = w @ w.T / count if W_cov is None else np.asarray(W_cov, dtype=float)
            if cov.shape != (self.p, self.p) or not np.isfinite(cov).all():
                raise ValueError("W_cov must be finite p-by-p")
            cov = (cov + cov.T) / 2
            # The leverage component prevents low-rank small-B empirical
            # covariance from claiming zero uncertainty in other directions.
            beta_cov = inv @ cov @ inv.T
            variance = np.maximum(self._diag_quad(beta_cov), 0)
            scale = np.sqrt(variance + (self.residual_scale * self.leverage) ** 2)
            scale = np.maximum(scale, np.finfo(float).eps)
        preparation_seconds = time.perf_counter() - start
        result = np.empty((count, self.p))
        records = []
        for j in range(count):
            result[j], record = self._fit_one(w[:, j], predicted_delta[:, j], scale)
            record["replicate"] = j
            records.append(record)
        return result, dict(selector=self.selector, setup_seconds=self.setup_seconds,
                            preparation_seconds=preparation_seconds,
                            batch_seconds=time.perf_counter() - start,
                            fallback_count=sum(r["fallback"] for r in records),
                            gram_pseudoinverse=self.gram_pseudoinverse,
                            jacobian_pseudoinverse=self.jacobian_pseudoinverse,
                            all_checked=all(r["numerically_checked"] for r in records),
                            fits=records)

    def _fit_one(self, w, delta, scale):
        start = time.perf_counter()
        active = np.zeros(self.n, dtype=bool)
        size = min(self.n, max(4 * self.p, int(np.ceil(self.factor * np.sqrt(self.n * self.p)))))
        if self.selector == "cfm":
            guess_r = self.r0
            score = None
        else:
            guess_r = self.r0 - self.x @ delta
            score = np.abs(guess_r) / scale
        anchor = np.where(guess_r < 0, self.tau - 1, self.tau)
        changed = anchor != self.base_a
        # Unlike X[fixed].T@a[fixed] this only scans selected/flipped design rows
        # on each reduced solve; the shared O(np) base sum was computed once.
        anchor_total = self.base_total.copy()
        if changed.any():
            anchor_total += self.x[changed].T @ (anchor[changed] - self.base_a[changed])

        def expand(target, bounded=False):
            if target >= self.n:
                active[:] = True
            else:
                selected = (self._cfm_indices(target) if self.selector == 'cfm' else
                            np.argpartition(score, target - 1)[:target])
                if bounded:
                    slots = max(0, int(target)-int(active.sum()))
                    selected = selected[~active[selected]][:slots]
                active[selected] = True

        expand(size)
        sizes, repairs, failures = [], [], []
        d = w / self.tau
        # Translate the primal origin by beta0 to improve reduced-LP scaling.
        # For fixed RHS this changes the dual objective by a constant only.
        pseudo_r0 = self.pseudo_y - d @ self.beta0
        positive_info = dict(enabled=self.positive_pseudo, attempts=0, lp_failures=0,
                             accepted=False, fallback_reason=None, residual_lower_bound=None,
                             active_sizes=[], free_restart=False)
        initial_size = int(active.sum())
        for positive_face in ((True, False) if self.positive_pseudo else (False,)):
            if self.positive_pseudo and not positive_face:
                # A rejected face must not contaminate the original free solve.
                active[:] = False
                expand(initial_size)
                positive_info['free_restart'] = True
                if positive_info['fallback_reason'] is None:
                    positive_info['fallback_reason'] = 'positive_round_limit'
            for iteration in range(self.max_rounds):
                indices = np.flatnonzero(active)
                sizes.append(int(indices.size))
                if positive_face:
                    positive_info['active_sizes'].append(int(indices.size))
                xx = self.x[indices]
                rhs = -anchor_total + xx.T @ anchor[indices]
                if positive_face and indices.size == self.n:
                    rhs = np.zeros(self.p)
                try:
                    if positive_face:
                        positive_info['attempts'] += 1
                        # The represented finite row d defines this candidate's
                        # face equality; replacing d*tau by W changes the model.
                        step, ordinary_a = _lp(xx, self.r0[indices], self.tau,
                                               rhs-d*self.tau, limit=self.solver_limit)
                        lower = pseudo_residual_lower(d, self.pseudo_y, self.beta0+step)
                        positive_info['residual_lower_bound'] = lower
                        if lower <= 0:
                            positive_info['fallback_reason'] = 'pseudo_residual_not_strictly_positive'
                            break
                        reduced_a = np.r_[ordinary_a, self.tau]
                    else:
                        step, reduced_a = _lp(np.vstack((xx, d)),
                                              np.r_[self.r0[indices], pseudo_r0], self.tau,
                                              rhs, limit=self.solver_limit)
                except RuntimeError as error:
                    failures.append(str(error))
                    if positive_face:
                        positive_info['lp_failures'] += 1
                    if active.all():
                        if positive_face:
                            positive_info['fallback_reason'] = 'positive_lp_failed_at_full_design'
                        break
                    size = min(self.n, 2 * len(indices))
                    expand(size, bounded=positive_face)
                    continue
                beta = self.beta0 + step
                residual = (self.y - self.x @ beta if positive_face else
                            self.r0 - self.x @ step)
                wrong = (~active) & (((anchor == self.tau) & (residual < -self.sign_tolerance)) |
                                     ((anchor == self.tau - 1) & (residual > self.sign_tolerance)))
                bad = int(wrong.sum())
                repairs.append(bad)
                if bad:
                    if positive_face:
                        # All wrong signs remain in the next full-data check;
                        # only a bounded number of new score rows enter the LP.
                        expand(min(self.n, 2 * len(indices)), bounded=True)
                        continue
                    active[wrong] = True
                    if bad > .1 * len(indices):
                        size = min(self.n, max(2 * len(indices), int(active.sum())))
                        expand(size)
                    continue
                a = anchor.copy()
                a[indices] = reduced_a[:-1]
                certificate = self._diagnostics(w, beta, a, reduced_a[-1], residual)
                if certificate["numerically_checked"]:
                    positive_info['accepted'] = bool(positive_face)
                    return beta, dict(**certificate, active_sizes=sizes, sign_repairs=repairs,
                                      lp_failures=failures, fallback=False,
                                      positive_pseudo=positive_info,
                                      seconds=time.perf_counter() - start)
                failures.append("Full-sample numerical diagnostics did not pass")
                if active.all():
                    if positive_face:
                        positive_info['fallback_reason'] = 'positive_full_data_check_failed'
                    break
                size = min(self.n, 2 * len(indices))
                expand(size, bounded=positive_face)
        # The original finite full LP remains the final free-branch fallback.
        full = full_fit(self.x, self.y, self.tau, w)
        residual = self.y - self.x @ full.beta
        certificate = self._diagnostics(w, full.beta, full.dual[:-1], full.dual[-1], residual)
        return full.beta, dict(**certificate, active_sizes=sizes + [self.n],
                               sign_repairs=repairs, lp_failures=failures, fallback=True,
                               positive_pseudo=positive_info,
                               seconds=time.perf_counter() - start)


def fit_batch(X, y, tau, W, beta0, *, W_cov=None, **options):
    """Convenience interface; construct CFMPreprocessor explicitly to reuse setup."""
    return CFMPreprocessor(X, y, tau, beta0, **options).fit_batch(W, W_cov=W_cov)
