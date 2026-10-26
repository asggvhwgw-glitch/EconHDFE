"""Compare econhdfe outputs with Stata golden results on identical fixtures.

The comparator is intentionally strict about every statistic that Stata records:
coefficients, standard errors, covariance, N, absorbed DoF, and available weak-ID
statistics.  v0.8-only solver/canonicalization/omission semantics are additionally
asserted in Python because Stata has no corresponding metadata surface.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from econhdfe import (
    FixedEffect,
    OmittedVariableWarning,
    factor,
    interaction,
    ivreghdfe,
    omit_column,
    omit_level,
    reg_interaction,
    reghdfe,
)


def _row(model, r, check_name: str | None = None):
    names = list(r.names)

    def get(name):
        j = names.index(name)
        return float(r.params[j]), float(r.stderr[j])

    ix = names.index("x")
    iw = names.index("w")
    bx, sex = get("x")
    bw, sew = get("w")
    kp = (r.diagnostics or {}).get("kleibergen_paap", {})
    if check_name is None:
        bcheck = secheck = np.nan
    else:
        bcheck, secheck = get(check_name)
    return dict(
        model=model,
        bx=bx,
        sex=sex,
        bw=bw,
        sew=sew,
        covxw=float(r.vcov[ix, iw]),
        bcheck=bcheck,
        secheck=secheck,
        N=float(r.nobs),
        dfa=float(r.df_absorbed),
        widstat=float(kp.get("rk_wald_f", np.nan)),
        idstat=float(kp.get("rk_lm", np.nan)),
    )


def _assert_v080_semantics(df: pd.DataFrame, rows: list[dict]) -> None:
    # Canonicalization must actually remove the hierarchy, not merely produce
    # a numerically similar answer through the raw redundant system.
    r = reghdfe(
        df,
        y="y",
        x=["x", "w"],
        absorb=["firm", "year", interaction("province", "year"), interaction("city", "year")],
        vce="robust",
    )
    canon = r.absorb_info["canonicalization"]
    if tuple(canon["effective"]) != ("firm", "city#year"):
        raise AssertionError(f"unexpected canonical FE system: {canon['effective']}")
    if r.absorb_info["solver_selection"]["resolved"] != "twoway":
        raise AssertionError("canonical two-FE system did not route to specialized two-way solver")
    rows.append(_row("ols_canonical_hierarchy_robust", r))

    # Plain firm/year auto selection must also resolve to the specialized solver.
    r = reghdfe(df, y="y", x=["x", "w"], absorb=["firm", "year"], vce="robust", method="auto")
    if r.absorb_info["solver_selection"]["resolved"] != "twoway":
        raise AssertionError("method='auto' did not route firm/year to two-way solver")
    rows.append(_row("ols_twoway_auto_robust", r))

    # User-selected omission must be exposed separately from automatic rank loss.
    r = reghdfe(
        df,
        y="y",
        x=["x", "w", "x3"],
        absorb=["firm", "year"],
        omit=[omit_column("x3")],
        collinearity="warn",
        vce="robust",
    )
    user = tuple(o["name"] for o in r.user_omitted_variables)
    if user != ("x3",):
        raise AssertionError(f"explicit omission metadata mismatch: {user}")
    if any(o["name"] == "x3" for o in r.automatic_omitted_variables):
        raise AssertionError("user-selected x3 was incorrectly reported as automatic omission")
    rows.append(_row("ols_user_omit_x3_robust", r))

    # Structural factor block should be pruned before materialization because
    # year is in the span of city#year.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OmittedVariableWarning)
        r = reghdfe(
            df,
            y="y",
            x=["x", "w", factor("year")],
            absorb=["firm", interaction("city", "year")],
            vce="robust",
        )
    structural = r.collinearity_info.get("structural") or {}
    if structural.get("requested_columns", 0) <= structural.get("materialized_columns", 0):
        raise AssertionError("structural year-factor block was not pruned before materialization")
    if not all(o.get("structural") for o in r.automatic_omitted_variables):
        raise AssertionError("structural omissions were not identified as structural")
    rows.append(_row("ols_structural_factor_robust", r))

    # Explicit event-study reference must agree with an algebraically equivalent
    # factor-base parameterization, while the never-treated zero interaction is
    # still recorded as a distinct automatic omission.
    full = reg_interaction(
        factor("event_code", drop_base=False, name="event_code"),
        "ever",
        name="event_study",
    )
    based = reg_interaction(
        factor("event_code", base=2, drop_base=True, name="event_code"),
        "ever",
        name="event_study",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OmittedVariableWarning)
        explicit = reghdfe(
            df,
            y="y",
            x=["x", "w", full],
            absorb=["firm", "year"],
            omit=[omit_level("event_code", 2, term="event_study")],
            collinearity="warn",
            vce="robust",
        )
        base = reghdfe(
            df,
            y="y",
            x=["x", "w", based],
            absorb=["firm", "year"],
            collinearity="warn",
            vce="robust",
        )
    if tuple(explicit.names) != tuple(base.names):
        raise AssertionError("event-study explicit-reference and factor-base columns differ")
    np.testing.assert_allclose(explicit.params, base.params, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(explicit.stderr, base.stderr, rtol=2e-9, atol=2e-9)
    if tuple(o["name"] for o in explicit.user_omitted_variables) != ("event_code[2]#ever",):
        raise AssertionError("event-study reference is not exposed as the sole user omission")
    if "event_code[7]#ever" not in {o["name"] for o in explicit.automatic_omitted_variables}:
        raise AssertionError("never-treated zero interaction was not separately auto-omitted")
    rows.append(_row("ols_event_ref_robust", explicit, check_name="event_code[3]#ever"))

    # The automatic composite-collinearity path must remain loud and distinct
    # from the explicit user-reference path.
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", OmittedVariableWarning)
        auto = reghdfe(
            df,
            y="y",
            x=["x", "w", "x3"],
            absorb=["firm", "year"],
            collinearity="warn",
        )
    if not auto.automatic_omitted_variables or auto.user_omitted_variables:
        raise AssertionError("automatic collinearity omission metadata is incomplete")
    if not any(issubclass(w.category, OmittedVariableWarning) for w in seen):
        raise AssertionError("automatic omission did not emit OmittedVariableWarning")


def run_python(df: pd.DataFrame, group_df: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    ols = [
        ("ols_iid", dict()),
        ("ols_robust", dict(vce="robust")),
        ("ols_cluster3", dict(vce="cluster", cluster=["c1", "c2", "c3"])),
        ("ols_dk", dict(vce="dkraay", time="year", bandwidth=4)),
    ]
    for model, kw in ols:
        r = reghdfe(df, y="y", x=["x", "w"], absorb=["firm", "year"], **kw)
        rows.append(_row(model, r))

    slope = reghdfe(
        df,
        y="y",
        x=["x", "w"],
        absorb=[FixedEffect("firm", slopes=("trend",), intercept=True), "year"],
        vce="robust",
    )
    rows.append(_row("ols_slope_robust", slope))

    weighted_ols = [
        ("ols_fweight_robust", "fw", "fweight", dict(vce="robust")),
        ("ols_aweight_robust", "aw", "aweight", dict(vce="robust")),
        ("ols_pweight", "pw", "pweight", dict()),
    ]
    for model, weight, weight_type, kw in weighted_ols:
        r = reghdfe(
            df,
            y="y",
            x=["x", "w"],
            absorb=["firm", "year"],
            weights=weight,
            weight_type=weight_type,
            **kw,
        )
        rows.append(_row(model, r))

    ivs = [
        ("iv_2sls_robust", dict(estimator="2sls", vce="robust")),
        ("iv_2sls_cluster2", dict(estimator="2sls", vce="cluster", cluster=["c1", "c2"])),
        ("iv_liml_robust", dict(estimator="liml", vce="robust")),
        ("iv_gmm2s_robust", dict(estimator="gmm2s", vce="robust")),
    ]
    for model, kw in ivs:
        r = ivreghdfe(
            df,
            y="y",
            exog=["w"],
            endog=["x"],
            instruments=["z1", "z2"],
            absorb=["firm", "year"],
            **kw,
        )
        rows.append(_row(model, r))

    weighted_ivs = [
        ("iv_fweight_robust", "fw", "fweight", dict(vce="robust")),
        ("iv_aweight_robust", "aw", "aweight", dict(vce="robust")),
        ("iv_pweight", "pw", "pweight", dict()),
    ]
    for model, weight, weight_type, kw in weighted_ivs:
        r = ivreghdfe(
            df,
            y="y",
            exog=["w"],
            endog=["x"],
            instruments=["z1", "z2"],
            absorb=["firm", "year"],
            weights=weight,
            weight_type=weight_type,
            **kw,
        )
        rows.append(_row(model, r))

    _assert_v080_semantics(df, rows)

    if group_df is not None:
        for aggregation in ("mean", "sum"):
            r = reghdfe(
                group_df,
                y="y",
                x=["x", "w"],
                absorb=["year", "inventor"],
                group="patent",
                individual="inventor",
                aggregation=aggregation,
                method="lsmr",
                vce="robust",
            )
            rows.append(_row(f"ols_group_ind_{aggregation}", r))
        r = ivreghdfe(
            group_df,
            y="y",
            exog=["w"],
            endog=["x"],
            instruments=["z1", "z2"],
            absorb=["year", "inventor"],
            group="patent",
            individual="inventor",
            aggregation="mean",
            method="lsmr",
            vce="robust",
        )
        rows.append(_row("iv_group_ind_mean", r))
    return pd.DataFrame(rows)


def _close(got: float, exp: float, *, atol: float, rtol: float = 0.0) -> bool:
    return bool(np.isclose(got, exp, atol=atol, rtol=rtol, equal_nan=False))


def _check_metric(failures, model, metric, got, exp, *, atol, rtol=0.0, required_if_expected=True):
    exp_finite = np.isfinite(exp)
    got_finite = np.isfinite(got)
    if not exp_finite:
        return
    if required_if_expected and not got_finite:
        failures.append(f"{model}:{metric}=missing")
        return
    if got_finite and not _close(got, exp, atol=atol, rtol=rtol):
        failures.append(f"{model}:{metric}")


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument("--data", type=Path, default=root / "fixtures" / "golden_data.csv")
    ap.add_argument("--expected", type=Path, default=root / "fixtures" / "golden_results_stata.csv")
    ap.add_argument("--group-data", type=Path, default=root / "fixtures" / "golden_group_individual.csv")
    ap.add_argument("--params-atol", type=float, default=1e-7)
    ap.add_argument("--se-atol", type=float, default=5e-5)
    ap.add_argument("--cov-atol", type=float, default=5e-6)
    ap.add_argument("--n-atol", type=float, default=0.0)
    ap.add_argument("--dfa-atol", type=float, default=0.0)
    ap.add_argument("--diag-atol", type=float, default=1e-5)
    ap.add_argument("--diag-rtol", type=float, default=5e-4)
    args = ap.parse_args()

    if not args.expected.exists():
        raise SystemExit(f"missing {args.expected}; run Stata validation/stata/generate_golden.do first")
    group_df = pd.read_csv(args.group_data) if args.group_data.exists() else None
    got = run_python(pd.read_csv(args.data), group_df).set_index("model")
    exp = pd.read_csv(args.expected).set_index("model")

    missing = got.index.difference(exp.index)
    if len(missing):
        raise SystemExit("Stata golden file missing expected models: " + ", ".join(missing))
    unexpected = exp.index.difference(got.index)
    if len(unexpected):
        print("note: Stata golden contains extra rows not used by this comparator: " + ", ".join(unexpected))

    failures: list[str] = []
    header = "model                                    max|d coef|  max|d se|    dN      ddfa"
    print(header)
    print("-" * len(header))
    for m in got.index:
        g = got.loc[m]
        e = exp.loc[m]
        coef_diffs = [abs(g.bx - e.bx), abs(g.bw - e.bw)]
        se_diffs = [abs(g.sex - e.sex), abs(g.sew - e.sew)]
        if np.isfinite(e.get("bcheck", np.nan)):
            coef_diffs.append(abs(g.bcheck - e.bcheck))
        if np.isfinite(e.get("secheck", np.nan)):
            se_diffs.append(abs(g.secheck - e.secheck))
        print(f"{m:40s} {max(coef_diffs):11.3e} {max(se_diffs):11.3e} {g.N-e.N:+7.1f} {g.dfa-e.dfa:+9.1f}")

        _check_metric(failures, m, "bx", g.bx, e.bx, atol=args.params_atol)
        _check_metric(failures, m, "bw", g.bw, e.bw, atol=args.params_atol)
        _check_metric(failures, m, "sex", g.sex, e.sex, atol=args.se_atol)
        _check_metric(failures, m, "sew", g.sew, e.sew, atol=args.se_atol)
        _check_metric(failures, m, "covxw", g.covxw, e.get("covxw", np.nan), atol=args.cov_atol)
        _check_metric(failures, m, "bcheck", g.bcheck, e.get("bcheck", np.nan), atol=args.params_atol)
        _check_metric(failures, m, "secheck", g.secheck, e.get("secheck", np.nan), atol=args.se_atol)
        _check_metric(failures, m, "N", g.N, e.N, atol=args.n_atol)
        _check_metric(failures, m, "dfa", g.dfa, e.dfa, atol=args.dfa_atol)
        _check_metric(
            failures, m, "widstat", g.widstat, e.get("widstat", np.nan),
            atol=args.diag_atol, rtol=args.diag_rtol,
        )
        _check_metric(
            failures, m, "idstat", g.idstat, e.get("idstat", np.nan),
            atol=args.diag_atol, rtol=args.diag_rtol,
        )

    if failures:
        raise SystemExit("golden mismatch: " + ", ".join(failures))
    print(
        "golden parity checks passed: coefficients, SEs, covariance, N, absorbed DoF, "
        "available IV diagnostics, and v0.8 semantic assertions"
    )


if __name__ == "__main__":
    main()
