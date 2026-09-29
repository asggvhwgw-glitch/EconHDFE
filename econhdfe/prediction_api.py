"""Chunked prediction from frozen linear-model state, independent of fit internals."""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
import numpy as np
import pandas as pd

from .errors import InputError, ShapeError, SpecificationError, InferenceError


def _spec(message, code):
    return SpecificationError(message, code=f"prediction.{code}", stage="postestimation")


def _row_count(data):
    if isinstance(data, pd.DataFrame):
        return len(data)
    if not isinstance(data, Mapping):
        raise _spec("data must be a raw-label DataFrame or column mapping", "data_type")
    sizes = []
    for value in data.values():
        if np.ndim(value) != 1:
            raise ShapeError("prediction mapping columns must be vectors", stage="postestimation")
        sizes.append(len(value))
    if sizes and min(sizes) != max(sizes):
        raise ShapeError("prediction columns must have equal lengths", stage="postestimation")
    return sizes[0] if sizes else 0


def _column(data, name, lo, hi):
    if name is None:
        raise _spec("opaque fitted regressors require an explicit transformed X matrix", "opaque_design")
    if name not in data:
        raise _spec(f"prediction data is missing column {name!r}", "missing_column")
    # Slice pandas before materializing nullable columns to bound workspace.
    value = data[name]
    value = value.iloc[lo:hi] if hasattr(value, "iloc") else value[lo:hi]
    out = np.asarray(value)
    if out.ndim != 1 or len(out) != hi - lo:
        raise ShapeError("prediction source must resolve to one aligned column", stage="postestimation")
    return out


def _index(arrays):
    return (pd.Index(arrays[0], tupleize_cols=False) if len(arrays) == 1
            else pd.MultiIndex.from_arrays(arrays))


def _lookup(levels, values):
    try:
        return levels.get_indexer(_index(values))
    except (TypeError, pd.errors.InvalidIndexError) as exc:
        raise InputError("prediction categories must be unique, scalar fitted labels",
                         code="prediction.categories", stage="postestimation") from exc


def _invalid(rows, unknown, code):
    if unknown == "raise" and np.any(rows):
        raise InputError("prediction rows contain unknown, nonfinite or unidentified values",
                         code=f"prediction.{code}", stage="postestimation",
                         details={"invalid_rows_in_chunk": int(np.count_nonzero(rows))})


def _compiled_design(state):
    """Compile lookups once per prediction, using indices instead of guessed names."""
    compiled = []
    offset = 0
    for role in state.coefficient_roles:
        design = state.design(role)
        flattened = tuple(name for term in design.terms for name in term.active_names)
        if flattened != design.materialized_names:
            raise _spec("fitted design provenance is incomplete or misaligned", "design_state")
        position = {i: offset + j for j, i in enumerate(design.active_indices)}
        cursor = 0
        for term in design.terms:
            columns = []
            for cell, keep in enumerate(term.active_mask):
                if keep:
                    if cursor in position:
                        columns.append((cell, position[cursor]))
                    cursor += 1
            if not columns:
                continue
            if not term.reconstructable:
                raise _spec("opaque fitted regressors require an explicit transformed X matrix", "opaque_design")
            levels = tuple(_index((enc.levels,)) for enc in term.categorical)
            cells = None if not levels else _index(tuple(term.cell_codes.T))
            compiled.append((term, columns, levels, cells))
        offset += len(design.active_names)
    if offset != len(state.coefficient_names):
        raise _spec("fitted coefficient roles do not align", "design_state")
    return compiled


def _design_chunk(compiled, data, lo, hi, k, unknown):
    X = np.zeros((hi - lo, k), dtype=np.float64)
    bad = np.zeros(hi - lo, dtype=bool)
    for term, columns, levels, cells in compiled:
        monomial = np.ones(hi - lo)
        for source in term.continuous:
            raw = _column(data, source.source, lo, hi)
            try:
                numeric = np.asarray(pd.to_numeric(pd.Series(raw), errors="raise"), dtype=float)
            except (ValueError, TypeError) as exc:
                raise InputError("prediction regressors must be numeric",
                                 code="prediction.non_numeric", stage="postestimation") from exc
            with np.errstate(over="ignore", invalid="ignore"):
                monomial *= numeric
        numeric_bad = ~np.isfinite(monomial)
        _invalid(numeric_bad, unknown, "nonfinite")
        bad |= numeric_bad
        if levels:
            codes = tuple(_lookup(idx, (_column(data, enc.source, lo, hi),))
                          for idx, enc in zip(levels, term.categorical))
            missing = np.any(np.column_stack(codes) < 0, axis=1)
            _invalid(missing, unknown, "unknown_level")
            cell = _lookup(cells, codes)
            absent = cell < 0
            _invalid(absent & ~missing, unknown, "unobserved_cell")
            bad |= missing | absent
            for index, output in columns:
                hit = cell == index
                X[hit, output] = monomial[hit]
        else:
            for _, output in columns:
                X[:, output] = monomial
    X[bad] = 0.0
    return X, bad


def _fe_chunk(payload, indexes, data, lo, hi, unknown):
    codes = tuple(_lookup(index, tuple(_column(data, s, lo, hi) for s in term.sources))
                  for term, index in zip(payload.terms, indexes))
    bad = np.any(np.column_stack(codes) < 0, axis=1)
    _invalid(bad, unknown, "unknown_fe_level")
    safe = tuple(np.maximum(code, 0) for code in codes)
    component = [comp[safe[i]] for i, comp in zip(payload.effective_indices, payload.components)]
    incompatible = np.zeros(hi - lo, dtype=bool)
    for other in component[1:]:
        incompatible |= other != component[0]
    if component:
        incompatible |= ~np.isin(component[0], payload.identified_components)
    for nesting in payload.nesting:
        incompatible |= nesting.coarse_by_fine[safe[nesting.fine]] != safe[nesting.coarse]
    _invalid(incompatible & ~bad, unknown, "unidentified_combination")
    bad |= incompatible
    out = np.zeros(hi - lo)
    for i, coef in zip(payload.effective_indices, payload.coefficients):
        out += coef[safe[i]]
    return out, bad


def _has_automatic_regressor_omissions(result):
    info = result.collinearity_info
    if not isinstance(info, dict):
        return False
    blocks = (info,) if "omitted" in info else tuple(
        info.get(role, {}) for role in ("exogenous", "endogenous")
    )
    return any(not item.get("selected_by_user", False)
               for block in blocks for item in (block.get("omitted") or ()))


def predict_linear(result, *, data=None, X=None, kind="response", unknown="raise",
                   restore_sample=False, chunk_size=65536):
    """Linear predictions only; stdp excludes FE uncertainty and outcome noise."""
    if not isinstance(kind, str) or kind not in {"response", "xb", "fe", "stdp"}:
        raise _spec("kind must be response, xb, fe or stdp", "kind")
    if not isinstance(unknown, str) or unknown not in {"raise", "nan"}:
        raise _spec("unknown must be raise or nan", "unknown_policy")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size <= 0:
        raise _spec("chunk_size must be a positive integer", "chunk_size")
    if not isinstance(restore_sample, (bool, np.bool_)):
        raise _spec("restore_sample must be boolean", "sample_alignment")
    if data is not None and X is not None:
        raise _spec("provide either data or transformed X, not both", "ambiguous_input")
    if restore_sample and (data is not None or X is not None):
        raise _spec("restore_sample only applies to stored in-sample predictions", "sample_alignment")
    state = result.prediction_state
    if data is None and X is None:
        if kind == "response":
            out = np.array(result.fitted, dtype=float, copy=True)
        elif kind == "stdp":
            raise _spec("stdp requires data or transformed X; raw design is not retained", "data_required")
        elif not result.fe_names:
            out = np.array(result.fitted, dtype=float, copy=True) if kind == "xb" else np.zeros_like(result.fitted)
        elif result.fixed_effects is not None and result.fixed_effects.converged:
            out = result.fixed_effects.fitted
            if kind == "xb":
                out = result.fitted - out
        else:
            raise _spec("in-sample xb/fe require saved effects or explicit prediction data", "fe_unavailable")
        if restore_sample:
            if state is None:
                raise _spec("this fitted path has no raw-row sample mapping", "sample_unavailable")
            if len(out) != state.sample.nobs_final:
                raise _spec("stored fitted values do not align with the fitted sample", "sample_alignment")
            aligned = np.full(state.sample.nobs_raw, np.nan)
            aligned[state.sample.mask()] = out
            return aligned
        return out

    if state is None or state.link != "identity" or state.response != "identity":
        raise _spec("this result does not carry supported linear prediction state", "state_unavailable")
    beta = np.asarray(result.params, dtype=float)
    names = tuple(name for role in state.coefficient_roles for name in state.design(role).active_names)
    if tuple(result.names) != state.coefficient_names or names != tuple(result.names) or beta.shape != (len(names),):
        raise _spec("result coefficients and fitted design state are not aligned", "coefficient_alignment")
    if not np.isfinite(beta).all():
        raise _spec("result coefficients must be finite", "coefficient_alignment")
    has_fe = state.fixed_effects.has_fixed_effects
    include_fe = kind in {"response", "fe"} and has_fe
    if kind == "response" and _has_automatic_regressor_omissions(result):
        raise _spec("new-row response prediction with automatically omitted regressors is not yet certified; "
                    "xb explicitly targets the retained coefficient parameterization", "omitted_estimability")
    payload = state.fixed_effects.categorical
    indexes = ()
    if include_fe:
        if X is not None or not state.can_include_fixed_effects:
            raise _spec("new-row FE predictions require named categorical effects saved with save_fe=True", "fe_unavailable")
        if not np.array_equal(beta, payload.beta):
            raise _spec("coefficients changed after the fixed effects were saved", "coefficient_alignment")
        indexes = tuple(_index(term.labels) for term in payload.terms)
    if X is not None:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != len(names):
            raise ShapeError("transformed X must have one column per reported coefficient, in names order",
                             code="prediction.matrix_shape", stage="postestimation")
        n = len(X)
        compiled = ()
    else:
        n = _row_count(data)
        compiled = _compiled_design(state) if kind != "fe" else ()
        sources = {source.source for term, _, _, _ in compiled
                   for source in (*term.continuous, *term.categorical)}
        if include_fe:
            sources.update(source for term in payload.terms for source in term.sources)
        for source in sources:
            _column(data, source, 0, 0)  # missing columns fail even on empty batches
    V = None
    if kind == "stdp":
        V = np.asarray(result.vcov, dtype=float)
        if V.shape != (len(names), len(names)) or not np.isfinite(V).all():
            raise InferenceError("stdp requires a finite aligned covariance", stage="postestimation")
        scale = np.max(np.abs(V), initial=0.0)
        if np.max(np.abs(V - V.T), initial=0.0) > 1e-10 * scale:
            raise InferenceError("stdp requires a symmetric covariance", stage="postestimation")
        V = (V + V.T) * 0.5
    out = np.empty(n)
    for lo in range(0, n, int(chunk_size)):
        hi = min(n, lo + int(chunk_size))
        if kind == "fe":
            values, bad = np.zeros(hi - lo), np.zeros(hi - lo, dtype=bool)
        else:
            if X is None:
                block, bad = _design_chunk(compiled, data, lo, hi, len(names), unknown)
            else:
                try:
                    block = np.array(X[lo:hi], dtype=float, copy=True)
                except (ValueError, TypeError) as exc:
                    raise InputError("transformed X must be numeric", stage="postestimation") from exc
                bad = ~np.isfinite(block).all(axis=1)
                _invalid(bad, unknown, "nonfinite")
                block[bad] = 0.0
            if kind == "stdp":
                values = np.einsum("ij,ij->i", block @ V, block)
                magnitude = np.einsum("ij,ij->i", np.abs(block) @ np.abs(V), np.abs(block))
                if np.any(values < -64 * np.finfo(float).eps * max(len(names), 1) * magnitude):
                    raise InferenceError("negative prediction variance", code="prediction.invalid_variance", stage="postestimation")
                values = np.sqrt(np.maximum(values, 0.0))
            else:
                values = block @ beta
        if include_fe:
            contribution, bad_fe = _fe_chunk(payload, indexes, data, lo, hi, unknown)
            values += contribution
            bad |= bad_fe
        nonfinite = ~np.isfinite(values)
        _invalid(nonfinite & ~bad, unknown, "nonfinite")
        values[bad | nonfinite] = np.nan
        out[lo:hi] = values
    return out
