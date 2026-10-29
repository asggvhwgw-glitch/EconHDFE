"""End-to-end PPML on identical synthetic FE topologies: fereg.fepois vs EconHDFE within.

Unlike kernel-pair, these timings include estimator, IRLS, FE absorption and variance.
Same y, regressors, observations, FE, positive outcomes, no singleton/separation, 2 threads.
"""
from __future__ import annotations
import argparse, json, os, pathlib, resource, sys, time, traceback
import numpy as np
import pandas as pd

ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fereg_within_topologies import generate


def corpus(case,n,seed):
    codes,x,_,_=generate(n,case,seed)
    rng=np.random.default_rng(seed+113)
    eta=-0.15+0.18*x
    for j in range(codes.shape[1]):
        eta+=0.11/(1+j)*np.sin(codes[:,j].astype(np.float64)*(0.007+0.002*j))
    # Positive continuous outcomes: separation not a confounder; E[y|x,FE] approximately exp(eta).
    y=np.exp(eta+0.18*rng.normal(size=n)-0.5*0.18**2)
    cols={f"f{j}":codes[:,j].astype(np.int64,copy=True) for j in range(codes.shape[1])}
    cols.update(y=y,x=x)
    df=pd.DataFrame(cols)
    return codes,x,y,df


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--case",required=True,choices=["one_big","two_big","three_big","weak_links","hub","repeated_cells","reweight"])
    p.add_argument("--rows",type=int,default=120000)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--repeat",type=int,default=2)
    p.add_argument("--seed",type=int,default=20061029)
    a=p.parse_args()
    for k in ("RAYON_NUM_THREADS","OMP_NUM_THREADS","NUMBA_NUM_THREADS"):
        os.environ[k]=str(a.threads)
    os.environ["OPENBLAS_NUM_THREADS"]="1"
    import fereg
    from econhdfe.config import ExecutionConfig
    from econhdfe.models.ppml.api import ppmlhdfe
    from econhdfe.models.ppml.config import PPMLConfig
    codes,x,y,df=corpus(a.case,a.rows,a.seed)
    cfg=PPMLConfig(engine="within",separation=(),tolerance=1e-9,
                   target_inner_tol=1e-10,start_inner_tol=1e-5,max_iter=100,min_ok=2)
    ex=ExecutionConfig(threads=a.threads,memory_budget_mb=4096)
    formula="y ~ x | "+" + ".join(f"f{j}" for j in range(codes.shape[1]))
    records=[]
    all_valid=True
    for rep in range(a.repeat):
        order=["fereg","econhdfe"] if rep%2==0 else ["econhdfe","fereg"]
        estimates={}
        for name in order:
            try:
                t0=time.perf_counter()
                if name=="fereg":
                    fit=fereg.fepois(formula,df,vcov="iid",iwls_tol=1e-9,
                                    iwls_maxiter=100,fixef_tol=1e-10,
                                    fixef_maxiter=6000,fixef_rm="none",separation_check=[],
                                    nthreads=a.threads)
                    beta=float(fit.coef()["x"])
                    nfit=int(fit._N)
                    # fereg's default residual convention may differ across APIs; compare beta.
                    diagnostics={"n_obs":nfit}
                else:
                    fit=ppmlhdfe(y=y,x=x.reshape(-1,1),absorb=tuple(codes[:,j] for j in range(codes.shape[1])),
                                 names=("x",),config=cfg,vce="model",execution_config=ex)
                    beta=float(fit.params[0])
                    nfit=int(fit.nobs)
                    diagnostics={"n_obs":nfit, "iterations":int(fit.iterations),
                                 "route":fit.diagnostics["projection_resources"].get("route")}
                elapsed=time.perf_counter()-t0
                valid=np.isfinite(beta) and nfit==len(y)
                estimates[name]=beta
                records.append({"case":a.case,"rows":len(y),"rep":rep,"name":name,"seconds":elapsed,
                                "beta":beta,"valid":bool(valid),"diagnostics":diagnostics})
                all_valid &=bool(valid)
            except Exception as e:
                records.append({"case":a.case,"rows":len(y),"rep":rep,"name":name,"valid":False,
                                "error":f"{type(e).__name__}: {e}",
                                "traceback":traceback.format_exc()[-1200:]})
                all_valid=False
        if len(estimates)==2:
            diff=abs(estimates["fereg"]-estimates["econhdfe"])
            records.append({"case":a.case,"rows":len(y),"rep":rep,"comparison":"slope",
                            "beta_abs_gap":diff,"valid":bool(diff<1e-5)})
            all_valid &=diff<1e-5
    out=dict(case=a.case,rows=len(y),threads=a.threads,records=records,
             valid=bool(all_valid),peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             scope="Full fereg PPML versus full EconHDFE PPML/within; different estimator implementation, same FE topology and data",
             limitation="No fereg-core direct IRLS solver reuse diagnosis; no IV-PPML competitor in fereg's Python API")
    print("BENCH_PPML "+json.dumps(out,allow_nan=False),flush=True)
    if not all_valid:raise SystemExit(2)


if __name__=="__main__":
    main()
