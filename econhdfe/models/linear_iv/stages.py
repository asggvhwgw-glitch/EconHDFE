"""Small synchronous IV result snapshots; never retain observation-level arrays."""
from copy import deepcopy
import numpy as np
from ...errors import SpecificationError


def validate_callback(callback):
    if callback is not None and not callable(callback):
        raise SpecificationError("on_stage must be callable or None")


def bind_callback(callback, names, outcome, *, df_absorbed):
    if callback is None:
        return None
    names = tuple(names)
    def emit(event):
        if event["stage"] == "estimation":
            # The core may retain one intercept in its small-sample correction
            # when every FE is nested in clusters. Report frontend absorbed DoF,
            # just as RegressionResult does; the core's residual DoF is correct.
            event["df_absorbed"] = df_absorbed
        callback(dict(event, names=names, outcome=outcome if isinstance(outcome, str) else None))
    return emit


def estimation_ready(callback, beta, vcov, scale, *, nobs_raw, effective_n,
                     rank, df_absorbed, vce, clusters):
    if callback is None:
        return
    df = max(effective_n - rank - df_absorbed, 0.)
    if str(vce).lower().replace("-", "_") == "cluster" and clusters:
        df = min(df, float(min(len(np.unique(c)) for c in clusters) - 1))
    # The cores have not restored units. Independent small arrays also keep
    # callbacks from mutating the ongoing fit.
    params = beta / scale
    V = vcov / scale[:, None] / scale[None, :]
    callback(dict(stage="estimation", status="complete", params=params, vcov=V,
                  stderr=np.sqrt(np.clip(np.diag(V), 0, None)), nobs=int(round(effective_n)),
                  nobs_raw=nobs_raw, rank=rank, df_resid=df, df_absorbed=df_absorbed,
                  vce=vce, diagnostics_status="pending"))


def first_stage_ready(callback, first, advanced):
    if callback is not None:
        callback(dict(stage="first_stage", status="complete",
                      diagnostics=deepcopy(first["diagnostics"]), advanced_statistics=advanced))


def diagnostics_ready(callback, diagnostics, advanced):
    if not advanced:
        diagnostics["advanced_statistics"] = {
            "enabled": False, "status": "not_requested",
            "omitted": ("shea_partial_r2", "sanderson_windmeijer",
                        "cragg_donald_f", "stock_yogo"),
        }
    if callback is not None:
        callback(dict(stage="diagnostics", status="complete", diagnostics=deepcopy(diagnostics),
                      advanced_statistics=advanced))


def completed(callback, result, *, outcome, cached=False):
    if callback is not None:
        # Publication output omits fitted_endog/residuals and includes final
        # units, confidence intervals and frontend metadata.
        callback(dict(stage="complete", status="complete", names=tuple(result.names),
                      outcome=outcome if isinstance(outcome, str) else None, cached=cached,
                      result=deepcopy(result.publication_output(include_diagnostics=True))))
