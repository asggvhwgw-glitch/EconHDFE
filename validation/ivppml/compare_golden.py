from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from econhdfe import ivppmlhdfe, IVPPMLConfig


def cfg(**kw):
    base = dict(engine="replica", tolerance=1e-8, target_inner_tol=1e-9)
    base.update(kw)
    return IVPPMLConfig(**base)


def _fit(df, spec):
    absorb = [df[c].to_numpy() for c in spec.get("absorb", ())]
    clusters = [df[c].to_numpy() for c in spec.get("clusters", ())] or None
    kwargs = dict(
        y=df[spec.get("y", "y")].to_numpy(),
        exog=df[list(spec.get("exog", ()))].to_numpy() if spec.get("exog") else None,
        endog=df[list(spec["endog"])].to_numpy(),
        instruments=df[list(spec["inst"])].to_numpy(),
        absorb=absorb,
        vce=spec.get("vce", "robust"),
        cluster=clusters,
        exog_names=spec.get("exog", ()), endog_names=spec["endog"], instrument_names=spec["inst"],
        config=cfg(standardize=spec.get("standardize", False), separation=spec.get("separation", ("fe","simplex","relu"))),
    )
    if "weight" in spec:
        kwargs["weights"] = df[spec["weight"]].to_numpy()
        kwargs["weight_type"] = spec["weight_type"]
    if "exposure" in spec:
        kwargs["exposure"] = df[spec["exposure"]].to_numpy()
    if "offset" in spec:
        kwargs["offset"] = df[spec["offset"]].to_numpy()
    return ivppmlhdfe(**kwargs)


SPECS = {
    "m01_base": dict(exog=("c",), endog=("e",), inst=("z",), absorb=("g1","g2")),
    "m02_cluster2": dict(exog=("c",), endog=("e",), inst=("z",), absorb=("g1","g2"), vce="cluster", clusters=("cl1","cl2")),
    "m03_standardize": dict(exog=("c",), endog=("e",), inst=("z",), absorb=("g1","g2"), standardize=True),
    "m04_fweight": dict(exog=("c",), endog=("e",), inst=("z",), absorb=("g1","g2"), weight="fw", weight_type="fweight"),
    "m05_pweight": dict(exog=("c",), endog=("e",), inst=("z",), absorb=("g1","g2"), weight="pw", weight_type="pweight"),
    "m06_exposure": dict(y="yexp", exog=("c",), endog=("e",), inst=("z",), absorb=("g1","g2"), exposure="expo"),
    "m07_multiendog": dict(exog=("c",), endog=("e","e2"), inst=("z","z2","z3"), absorb=("g1","g2")),
    "m08_mu": dict(exog=("c",), endog=("e",), inst=("z",), absorb=("g1","g2"), separation=("fe","simplex","relu","mu")),
}


def python_results(df):
    coef_rows, v_rows = [], []
    for model, spec in SPECS.items():
        r = _fit(df, spec)
        for j, term in enumerate(r.names):
            coef_rows.append(dict(
                model=model, term=term, b=r.params[j], se=r.stderr[j], N=r.nobs,
                df_a=r.df_absorbed, num_sep=r.n_separated,
                num_sep_mu=int(r.diagnostics.get("num_sep_mu", 0)), iters=r.iterations,
            ))
            for k, term2 in enumerate(r.names):
                v_rows.append(dict(model=model, row=term, col=term2, v=r.vcov[j,k]))
    return pd.DataFrame(coef_rows), pd.DataFrame(v_rows)


def compare(root: Path, atol_b=8e-7, atol_v=2e-7, rtol_v=3e-5):
    df = pd.read_csv(root/"fixtures"/"ivppml_feature_fixture.csv")
    py, pyv = python_results(df)
    gcoef, gvcov = root/"stata_golden.csv", root/"stata_golden_vcov.csv"
    if not gcoef.exists() or not gvcov.exists():
        py.to_csv(root/"python_expected.csv", index=False)
        pyv.to_csv(root/"python_expected_vcov.csv", index=False)
        print("Stata golden files not found; wrote Python expectations. Run stata/generate_golden.do on a licensed Stata host.")
        return 2
    st, stv = pd.read_csv(gcoef), pd.read_csv(gvcov)
    z = st.merge(py, on=["model","term"], suffixes=("_stata","_python"), validate="one_to_one")
    z["db"] = np.abs(z.b_stata-z.b_python)
    z["dse"] = np.abs(z.se_stata-z.se_python)
    fail = (z.db > atol_b) | (z.dse > np.maximum(atol_v, rtol_v*np.abs(z.se_stata)))
    fail |= (z.N_stata != z.N_python) | (z.num_sep_stata != z.num_sep_python) | (z.num_sep_mu_stata != z.num_sep_mu_python)
    # Feature fixture has only <=2 FE, so raw absorbed DoF is an exact gate here.
    fail |= z.df_a_stata != z.df_a_python
    vz = stv.merge(pyv, on=["model","row","col"], suffixes=("_stata","_python"), validate="one_to_one")
    vz["dv"] = np.abs(vz.v_stata-vz.v_python)
    vfail = vz.dv > np.maximum(atol_v, rtol_v*np.abs(vz.v_stata))
    print(z.to_string(index=False))
    print(f"VCE max |diff| = {vz.dv.max():.3e}")
    if fail.any() or vfail.any():
        if fail.any(): print("FAILED coefficient rows:\n", z.loc[fail].to_string(index=False))
        if vfail.any(): print("FAILED VCE rows:\n", vz.loc[vfail].head(30).to_string(index=False))
        return 1
    print("IV-PPML feature golden comparison passed")
    return 0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args=ap.parse_args()
    raise SystemExit(compare(args.root))

if __name__ == "__main__": main()
