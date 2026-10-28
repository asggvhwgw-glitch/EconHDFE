"""Same-process-policy IV-PPML outer benchmark; run modes in fresh processes.

Example: OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=4 python
benchmarks/ivppml_outer.py --n 1000000 --case crossed3 --mode moment --output result.json
JIT warm-up and data generation are excluded; full estimation/inference is included.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, resource, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from econhdfe import ivppmlhdfe, IVPPMLConfig, ExecutionConfig


def data(n,case,seed=20261028):
    rng=np.random.default_rng(seed)
    q=3 if case in ('crossed3','many_instruments','weak') else 2
    # Large-scale low-dimensional case limits data/solver peak memory.
    kc=1;ke=1;ki=8 if case=='many_instruments' else (3 if case=='overidentified' else 1)
    c=rng.normal(size=(n,kc));z=rng.normal(size=(n,ki))
    e=(.03 if case=='weak' else .8)*z[:,:ke]+.15*c+.4*rng.normal(size=(n,ke))
    levels=[max(20,n//120),max(15,n//220),31]
    groups=[rng.integers(0,levels[j],n) for j in range(q)]
    off=.5*rng.normal(size=n) if case=='offset' else np.zeros(n)
    eta=.1*c[:,0]+.22*e[:,0]+off+.2
    for j,g in enumerate(groups):eta+=rng.normal(scale=.12,size=levels[j])[g]
    y=rng.poisson(np.exp(eta)).astype(float)
    weights=np.exp(.25*rng.normal(size=n)) if case=='offset' else np.ones(n)
    return dict(y=y,exog=c,endog=e,instruments=z,absorb=groups,offset=off,weights=weights)


def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=1_000_000)
    p.add_argument('--case',default='crossed3',choices=['crossed2','crossed3','many_instruments','overidentified','weak','offset'])
    p.add_argument('--mode',default='moment',choices=['legacy','moment','moment_stable'])
    p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=20261028)
    p.add_argument('--output',required=True);a=p.parse_args()
    cfg=IVPPMLConfig(outer_solver='legacy' if a.mode=='legacy' else 'moment',
        outer_linear_solver='stable' if a.mode=='moment_stable' else 'auto',
        separation=('fe',),singleton_mode='unified',engine='optimized',
        tolerance=1e-9,target_inner_tol=1e-10,max_iter=150)
    exe=ExecutionConfig(threads=a.threads,memory_budget_mb=2048)
    import econhdfe.models.ppml_iv.estimator as estimator_module
    import econhdfe.models.ppml_iv.irls as irls_module
    meter = dict(outer_seconds=0.0, legacy_linear_seconds=0.0, legacy_linear_calls=0)
    original_outer, original_linear = estimator_module.fit_iv_irls, irls_module.weighted_2sls
    def timed_outer(*args, **kwargs):
        start=time.perf_counter()
        try: return original_outer(*args, **kwargs)
        finally: meter['outer_seconds'] += time.perf_counter()-start
    def timed_linear(*args, **kwargs):
        start=time.perf_counter()
        try: return original_linear(*args, **kwargs)
        finally:
            meter['legacy_linear_seconds'] += time.perf_counter()-start
            meter['legacy_linear_calls'] += 1
    estimator_module.fit_iv_irls, irls_module.weighted_2sls = timed_outer, timed_linear
    t=time.perf_counter();ivppmlhdfe(**data(100000,a.case,a.seed+1),config=cfg,execution_config=exe)
    warm=time.perf_counter()-t
    d=data(a.n,a.case,a.seed)
    digest=hashlib.sha256()
    for key in ['y','exog','endog','instruments','offset','weights']:digest.update(d[key].tobytes())
    for g in d['absorb']:digest.update(g.tobytes())
    baseline_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    for key in meter: meter[key]=0
    t=time.perf_counter()
    try:
        r=ivppmlhdfe(**d,config=cfg,execution_config=exe,vce='robust')
        elapsed=time.perf_counter()-t
        result=dict(status='ok',seconds=elapsed,iterations=r.iterations,nobs=r.nobs,
                    coefficients=r.params.tolist(),vcov=r.vcov.tolist(),
                    deviance=r.deviance,absorb_iterations=r.diagnostics['absorb_iterations'],
                    outer=r.diagnostics.get('outer',{}),raw_moments=r.moment_conditions.tolist() if hasattr(r,'moment_conditions') else None)
        np.savez(str(a.output)+'.npz',params=r.params,vcov=r.vcov,mu=r.mu,eta=r.eta,sample=r.sample_mask)
    except Exception as ex:
        result=dict(status='error',seconds=time.perf_counter()-t,error=repr(ex))
    result.update(meter)
    result.update(case=a.case,n=a.n,mode=a.mode,threads=a.threads,seed=a.seed,
                  data_sha256=digest.hexdigest(),warmup_seconds=warm,
                  peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  rss_highwater_before_fit_mb=baseline_rss,python=sys.version,numpy=np.__version__,
                  platform=platform.platform(),cpu_count=os.cpu_count())
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('outer','vcov','raw_moments')},indent=2),flush=True)
    if result['status']!='ok':raise SystemExit(1)
if __name__=='__main__':main()
