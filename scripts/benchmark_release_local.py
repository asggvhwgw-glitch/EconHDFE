"""Bounded local probe, not a universal performance or memory guarantee.

Run each version/case in a fresh process with the same thread settings. Includes
one warmup; compilation and profiling are reported separately from repetitions.
Peak RSS, where available, includes interpreter/imports/data/profiling.
"""
from __future__ import annotations
import argparse
import cProfile
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import sys
from time import perf_counter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--case',choices=['ols_kernel','iv_kernel','ols_3fe','ppml_2fe','ivppml_2fe'],required=True)
    p.add_argument('--n',type=int,default=50000)
    p.add_argument('--repeat',type=int,default=5)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.n<100 or a.repeat<1: p.error('n>=100 and repeat>=1 are required')
    sys.path.insert(0,str(a.source.resolve()))
    import numpy as np
    import econhdfe as e
    from econhdfe.models.ols import _fit_ols
    from types import SimpleNamespace
    from econhdfe.iv.solve import weighted_2sls
    rng=np.random.default_rng(62026)
    X=rng.normal(size=(a.n,12));z=rng.normal(size=(a.n,3))
    X[:,1]=.6*z[:,0]+.3*X[:,0]+.3*X[:,1]
    g=rng.integers(0,200,a.n);h=rng.integers(0,30,a.n);j=rng.integers(0,15,a.n)
    y=X@np.linspace(.02,.2,12)+rng.normal(scale=.2,size=200)[g]+rng.normal(size=a.n)
    yp=rng.poisson(np.exp(.3+.12*X[:,0]+.24*X[:,1]+.03*(g%5)-.02*(h%3))).astype(float)
    if a.case=='ols_kernel':
        def call():
            b,V,*_= _fit_ols(y,X,vce='robust')
            return SimpleNamespace(beta=b,vcov=V)
    elif a.case=='iv_kernel':
        Z=np.column_stack([X[:,:1],z]);T=X[:,:2].copy()
        call=lambda: weighted_2sls(y,T,Z)
    elif a.case=='ols_3fe':
        call=lambda:e.olshdfe(y=y,x=X,absorb=[g,h,j],vce='robust',tol=1e-10,absorb_threads=1)
    elif a.case=='ppml_2fe':
        cfg=e.PPMLConfig(separation=(),tolerance=1e-9,target_inner_tol=1e-10)
        call=lambda:e.ppmlhdfe(y=yp,x=X[:,:2],absorb=[g,h],vce='robust',config=cfg)
    else:
        cfg=e.IVPPMLConfig(separation=(),tolerance=1e-9,target_inner_tol=1e-10)
        call=lambda:e.ivppmlhdfe(y=yp,exog=X[:,:1],endog=X[:,1:2],instruments=z,absorb=[g,h],config=cfg)
    t=perf_counter();r=call();warm=perf_counter()-t
    times=[]
    for _ in range(a.repeat):
        t=perf_counter();r=call();times.append(perf_counter()-t)
    profiler=cProfile.Profile();profiler.enable();call();profiler.disable()
    ps=pstats.Stats(profiler)
    entries=[]
    for (filename,line,name),(cc,nc,tt,ct,_) in ps.stats.items():
        if 'econhdfe/' in filename:
            entries.append(dict(file=filename.split('econhdfe/',1)[1],line=line,function=name,
                                calls=nc,self_seconds=tt,cumulative_seconds=ct))
    entries.sort(key=lambda x:x['cumulative_seconds'],reverse=True)
    try:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    except ImportError:
        peak=None
    beta=getattr(r,'beta',None)
    if beta is None:beta=getattr(r,'coef',None)
    if beta is None:beta=r.params
    if isinstance(beta,dict): beta=list(beta.values())
    V=getattr(r,'vcov',getattr(r,'bread',None))
    result=dict(case=a.case,n=a.n,repeat=a.repeat,source=str(a.source.resolve()),version=e.__version__,
                python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,
                thread_settings={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS']},
                warmup_seconds=warm,times_seconds=times,median_seconds=statistics.median(times),
                process_peak_rss_mib=peak,peak_scope='includes imports, data, warmup, repetitions, and separate profile',
                beta=np.asarray(beta).tolist(),vcov_or_bread=None if V is None else np.asarray(V).tolist(),
                converged=getattr(r,'converged',None),iterations=getattr(r,'iterations',None),
                profile_top=entries[:25])
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['case','version','n','median_seconds','process_peak_rss_mib','converged']}))


if __name__=='__main__':main()
