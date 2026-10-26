from __future__ import annotations
import argparse, json, time
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from econhdfe import olshdfe, ivhdfe, ppmlhdfe, ivppmlhdfe, PPMLConfig, IVPPMLConfig


def arr(x):
    a=np.asarray(x,dtype=float)
    return a.tolist()

def linear_rec(r):
    return dict(names=list(r.names),params=arr(r.params),stderr=arr(r.stderr),vcov=np.asarray(r.vcov,float).tolist(),
                nobs=int(r.nobs),rank=int(r.rank),df_absorbed=int(r.df_absorbed),df_resid=float(r.df_resid),
                iterations=int(r.iterations),dropped_singletons=int(r.dropped_singletons),cluster_counts=list(r.cluster_counts),
                solver=(r.absorb_info or {}).get('method'))

def pp_rec(r):
    return dict(names=list(r.names),coef=arr(r.params),stderr=arr(r.stderr),vcov=np.asarray(r.vcov,float).tolist(),
                nobs=int(r.nobs),df_absorbed=int(r.df_absorbed),iterations=int(r.iterations),
                n_separated=int(r.n_separated),cluster_counts=list(r.cluster_counts),converged=bool(r.converged))

def medtime(fn,reps=3):
    fn()  # warm
    vals=[]; out=None
    for _ in range(reps):
        t=time.perf_counter(); out=fn(); vals.append(time.perf_counter()-t)
    return float(np.median(vals)),out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=18000); ap.add_argument('--reps',type=int,default=3); ap.add_argument('--out',required=True)
    a=ap.parse_args(); n=a.n
    rng=np.random.default_rng(20260913)
    g1=rng.integers(0,max(80,n//60),n,dtype=np.int32)
    g2=rng.integers(0,80,n,dtype=np.int16)
    g3=rng.integers(0,24,n,dtype=np.int8)
    x1=rng.normal(size=n); x2=rng.normal(size=n); c=rng.normal(size=n); z1=rng.normal(size=n); z2=rng.normal(size=n)
    e=.75*z1+.22*z2+.18*c+rng.normal(scale=.65,size=n)
    a1=rng.normal(scale=.4,size=int(g1.max())+1); a2=rng.normal(scale=.25,size=int(g2.max())+1); a3=rng.normal(scale=.12,size=int(g3.max())+1)
    y=.32*x1-.17*x2+.10*c+a1[g1]+a2[g2]+a3[g3]+rng.normal(scale=.8,size=n)
    yiv=.28*e+.12*c+a1[g1]+a2[g2]+a3[g3]+rng.normal(scale=.8,size=n)
    mu=np.exp(np.clip(.08*x1-.05*x2+.03*c+.12*a1[g1]+.08*a2[g2]+.04*a3[g3],-2,2)); ypp=rng.poisson(mu)
    muiv=np.exp(np.clip(.10*e+.04*c+.10*a1[g1]+.06*a2[g2],-2,2)); yivpp=rng.poisson(muiv)
    df=pd.DataFrame(dict(y=y,yiv=yiv,ypp=ypp,yivpp=yivpp,x1=x1,x2=x2,c=c,e=e,z1=z1,z2=z2,g1=g1,g2=g2,g3=g3))
    out={'n':n,'models':{},'timings':{}}
    cases=[]
    cases.append(('ols_2fe_robust',lambda:olshdfe(df,y='y',x=['x1','x2','c'],absorb=['g1','g2'],vce='robust'),linear_rec))
    cases.append(('ols_3fe_cluster2',lambda:olshdfe(df,y='y',x=['x1','x2','c'],absorb=['g1','g2','g3'],vce='cluster',cluster=['g1','g2']),linear_rec))
    cases.append(('iv_2fe_robust',lambda:ivhdfe(df,y='yiv',exog=['c'],endog=['e'],instruments=['z1','z2'],absorb=['g1','g2'],vce='robust'),linear_rec))
    cases.append(('iv_3fe_cluster2',lambda:ivhdfe(df,y='yiv',exog=['c'],endog=['e'],instruments=['z1','z2'],absorb=['g1','g2','g3'],vce='cluster',cluster=['g1','g2']),linear_rec))
    pcfg=PPMLConfig(separation=('fe',),standardize=False,tolerance=1e-8,target_inner_tol=1e-9,max_iter=200,engine='optimized')
    cases.append(('ppml_2fe_robust',lambda:ppmlhdfe(y=df.ypp.to_numpy(),x=df[['x1','x2','c']].to_numpy(),absorb=[g1,g2],vce='robust',names=('x1','x2','c'),config=pcfg),pp_rec))
    cases.append(('ppml_3fe_cluster2',lambda:ppmlhdfe(y=df.ypp.to_numpy(),x=df[['x1','x2','c']].to_numpy(),absorb=[g1,g2,g3],vce='cluster',cluster=[g1,g2],names=('x1','x2','c'),config=pcfg),pp_rec))
    icfg=IVPPMLConfig(separation=('fe',),standardize=False,tolerance=1e-8,target_inner_tol=1e-9,max_iter=250,engine='optimized')
    cases.append(('ivppml_2fe_robust',lambda:ivppmlhdfe(y=df.yivpp.to_numpy(),exog=df[['c']].to_numpy(),endog=df[['e']].to_numpy(),instruments=df[['z1','z2']].to_numpy(),absorb=[g1,g2],vce='robust',exog_names=('c',),endog_names=('e',),instrument_names=('z1','z2'),config=icfg),pp_rec))
    for name,fn,rec in cases:
        t,r=medtime(fn,a.reps); out['models'][name]=rec(r); out['timings'][name]=t
    Path(a.out).write_text(json.dumps(out,indent=2))
    print(a.out)
if __name__=='__main__': main()
