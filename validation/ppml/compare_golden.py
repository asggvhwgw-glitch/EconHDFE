from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from econhdfe import PPMLConfig, ppmlhdfe
from econhdfe.models.ppml.separation_relu import relu_separation
from econhdfe.hdfe.plan import FEPlan


def cfg(engine="replica"):
    return PPMLConfig(engine=engine, tolerance=1e-8, target_inner_tol=1e-9)


def python_results(df):
    X = df[["x1", "x2"]].to_numpy()
    specs = {
        "m01_no_fe": dict(y="y", absorb=(), vce="robust"),
        "m02_one_fe": dict(y="y", absorb=("g1",), vce="robust"),
        "m03_two_fe_cluster": dict(y="y", absorb=("g1", "g2"), vce="cluster", clusters=("cl1",)),
        "m04_three_fe_2cluster": dict(y="y", absorb=("g1", "g2", "g3"), vce="cluster", clusters=("cl1", "cl2")),
        "m05_exposure": dict(y="yexp", absorb=("g1", "g2"), vce="robust", exposure="expo"),
    }
    rows=[]
    for model,s in specs.items():
        absorb=[df[c].to_numpy() for c in s["absorb"]]
        clusters=[df[c].to_numpy() for c in s.get("clusters",())] or None
        r=ppmlhdfe(
            y=df[s["y"]].to_numpy(), x=X, absorb=absorb,
            exposure=None if "exposure" not in s else df[s["exposure"]].to_numpy(),
            vce=s["vce"], cluster=clusters, names=("x1","x2"), config=cfg(),
        )
        for term in ("x1","x2") + (("_cons",) if model=="m01_no_fe" else ()):
            j=r.names.index(term)
            rows.append(dict(model=model,term=term,b=r.params[j],se=r.stderr[j],N=r.nobs,df_a=r.df_absorbed,num_sep=r.n_separated,iters=r.iterations))
    return pd.DataFrame(rows)


def compare(root: Path, atol_b=5e-7, rtol_se=2e-5):
    fixture=pd.read_csv(root/"fixtures"/"ppml_fixture.csv")
    py=python_results(fixture)
    stata_path=root/"stata_golden.csv"
    if not stata_path.exists():
        out=root/"python_expected.csv"; py.to_csv(out,index=False)
        print(f"Stata golden not found; wrote {out}")
        return 2
    st=pd.read_csv(stata_path)
    z=st.merge(py,on=["model","term"],suffixes=("_stata","_python"),validate="one_to_one")
    z["db"]=np.abs(z.b_stata-z.b_python)
    z["rel_se"]=np.abs(z.se_stata-z.se_python)/np.maximum(np.abs(z.se_stata),1e-15)
    fail=(z.db>atol_b)|(z.rel_se>rtol_se)|(z.N_stata!=z.N_python)|(z.num_sep_stata!=z.num_sep_python)
    # For <=2 FE the DoF convention should also agree exactly. 3+ FE may be
    # a documented pairwise/exact-rank convention difference and is reported.
    check_dfa=~z.model.eq("m04_three_fe_2cluster")
    fail |= check_dfa & (z.df_a_stata!=z.df_a_python)
    print(z.to_string(index=False))
    if fail.any():
        print("FAILED rows:")
        print(z.loc[fail].to_string(index=False))
        return 1
    print("ppmlhdfe golden comparison passed")
    return 0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    args=ap.parse_args()
    raise SystemExit(compare(args.root))


if __name__=="__main__": main()
