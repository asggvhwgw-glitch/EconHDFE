from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from econhdfe import ivppmlhdfe, IVPPMLConfig


def _interaction(*cols):
    mi = pd.MultiIndex.from_arrays([np.asarray(c) for c in cols])
    return pd.factorize(mi, sort=False)[0].astype(np.int64)


def _cfg(**kw):
    d = dict(engine="replica", tolerance=1e-8, target_inner_tol=1e-9)
    d.update(kw)
    return IVPPMLConfig(**d)


def fit_class(label, df):
    common = dict(
        y=df.y.to_numpy(), exog=df[["x2"]].to_numpy(), endog=df[["x1"]].to_numpy(),
        instruments=df[["z"]].to_numpy(), exog_names=("x2",), endog_names=("x1",),
        instrument_names=("z",),
    )
    if label == "ClassA":
        return ivppmlhdfe(**common, absorb=[df.id.to_numpy(), df.year.to_numpy()], vce="robust", config=_cfg())
    ey = _interaction(df.exp, df.year); iy = _interaction(df.imp, df.year)
    if label == "ClassB":
        return ivppmlhdfe(**common, absorb=[ey, iy], vce="cluster", cluster=[df.pair.to_numpy()], config=_cfg())
    if label == "ClassC":
        pair = _interaction(df.exp, df.imp)
        return ivppmlhdfe(
            **common, absorb=[pair, ey, iy], vce="cluster", cluster=[df.pair.to_numpy()],
            config=_cfg(standardize=True, separation=("fe","simplex","relu","mu")),
        )
    raise ValueError(label)


def collect(data_dir: Path):
    rows, vrows = [], []
    for label in ("ClassA","ClassB","ClassC"):
        path = data_dir / f"ivppmlhdfe_{label}.dta"
        if not path.exists():
            raise FileNotFoundError(path)
        r = fit_class(label, pd.read_stata(path))
        for j,t in enumerate(r.names):
            rows.append(dict(model=label,term=t,b=r.params[j],se=r.stderr[j],N=r.nobs,df_a=r.df_absorbed,
                             num_sep=r.n_separated,num_sep_mu=r.diagnostics.get("num_sep_mu",0),iters=r.iterations))
            for k,t2 in enumerate(r.names):
                vrows.append(dict(model=label,row=t,col=t2,v=r.vcov[j,k]))
    return pd.DataFrame(rows), pd.DataFrame(vrows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="Directory containing upstream Class A/B/C .dta files")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--atol-b", type=float, default=8e-7)
    ap.add_argument("--rtol-v", type=float, default=5e-5)
    args=ap.parse_args()
    py, pyv = collect(args.data_dir)
    g, gv = args.root/"official_stata_golden.csv", args.root/"official_stata_golden_vcov.csv"
    if not g.exists() or not gv.exists():
        py.to_csv(args.root/"official_python_expected.csv", index=False)
        pyv.to_csv(args.root/"official_python_expected_vcov.csv", index=False)
        print("Official Stata golden missing; wrote Python results. Run stata/generate_official_corpus_golden.do on licensed Stata.")
        raise SystemExit(2)
    st, stv = pd.read_csv(g), pd.read_csv(gv)
    z=st.merge(py,on=["model","term"],suffixes=("_stata","_python"),validate="one_to_one")
    z["db"]=np.abs(z.b_stata-z.b_python); z["dse"]=np.abs(z.se_stata-z.se_python)
    fail=(z.db>args.atol_b)|(z.dse>np.maximum(2e-7,args.rtol_v*np.abs(z.se_stata)))
    fail|=(z.N_stata!=z.N_python)|(z.num_sep_stata!=z.num_sep_python)|(z.num_sep_mu_stata!=z.num_sep_mu_python)
    # A/B have <=2 effective FE sets; C's raw df_a is diagnostic only because
    # econhdfe may use a structurally tighter 3+FE convention than reghdfe.
    fail |= ~z.model.eq("ClassC") & (z.df_a_stata != z.df_a_python)
    vz=stv.merge(pyv,on=["model","row","col"],suffixes=("_stata","_python"),validate="one_to_one")
    vz["dv"]=np.abs(vz.v_stata-vz.v_python)
    vfail=vz.dv>np.maximum(2e-7,args.rtol_v*np.abs(vz.v_stata))
    print(z.to_string(index=False)); print(f"VCE max |diff|={vz.dv.max():.3e}")
    if fail.any() or vfail.any():
        if fail.any(): print("FAILED coefficient rows:\n",z.loc[fail].to_string(index=False))
        if vfail.any(): print("FAILED VCE rows:\n",vz.loc[vfail].head(30).to_string(index=False))
        raise SystemExit(1)
    print("Official Class A/B/C parity passed")

if __name__=="__main__": main()
