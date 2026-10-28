"""Reproducible over-ID performance and fixed-dimensional inference experiments."""
from __future__ import annotations
import argparse,json,os,platform,resource,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from econhdfe import IVPPMLConfig,ivppmlhdfe,ExecutionConfig


def draw(n,seed,*,misspec=0.,large=False):
    rng=np.random.default_rng(seed);c=rng.normal(size=n);z=rng.normal(size=(n,3));v=rng.normal(size=n)
    sigma=.55;rho=.6;be=.25;bc=.1
    e=.8*z[:,0]+.15*c+sigma*v
    levels=[max(5,n//120),max(4,n//220),17] if large else [4]
    groups=[rng.integers(0,L,n) for L in levels]
    fe=sum(rng.normal(scale=.12,size=L)[g] for L,g in zip(levels,groups))
    # E[y-exp(.2+FE+bc*c+be*e)|z,c,FE]=0 when misspec=0,
    # although v induces genuine endogeneity. For Gaussian independent z2,
    # misspec*z2 shifts the population FE intercept by misspec**2/2,
    # while the projected-IV pseudo-true slopes remain (bc,be).
    mean=np.exp(.2+fe+bc*c+be*e+rho*v-be*sigma*rho-.5*rho*rho+misspec*z[:,1])
    y=rng.poisson(mean).astype(float)
    return dict(y=y,exog=c,endog=e,instruments=z,absorb=groups)


def config(mode,inference,history=False):
    return IVPPMLConfig(outer_solver=mode,outer_inference=inference,
        outer_overid_test=(inference=='estimating_equation'),outer_record_history=history,
        engine='optimized',singleton_mode='unified',tolerance=1e-9,target_inner_tol=1e-10,max_iter=100)


def benchmark(a):
    exe=ExecutionConfig(threads=a.threads,memory_budget_mb=2048)
    cfg=config(a.mode,a.inference)
    ivppmlhdfe(**draw(20000,33,large=True),config=cfg,execution_config=exe)
    d=draw(a.n,a.seed,large=True);t=time.perf_counter()
    r=ivppmlhdfe(**d,config=cfg,execution_config=exe,vce='robust')
    result=dict(n=a.n,seed=a.seed,mode=a.mode,inference=a.inference,
        seconds=time.perf_counter()-t,iterations=r.iterations,params=r.params.tolist(),vcov=r.vcov.tolist(),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        outer=r.diagnostics['outer'],inference_diagnostics=r.diagnostics['outer_inference'],
        overid=r.diagnostics['overidentification'],platform=platform.platform(),python=sys.version,
        threads=a.threads,numpy=np.__version__)
    return result


def simulation(a):
    records=[];t=time.perf_counter();exe=ExecutionConfig(threads=1,memory_budget_mb=1024)
    for delta in [0.,.7]:
        for seed in range(a.reps):
            d=draw(a.n,a.seed+seed,misspec=delta)
            try:
                full=ivppmlhdfe(**d,config=config('profile','estimating_equation'),execution_config=exe,vce='robust')
                standard=ivppmlhdfe(**d,config=config('profile','standard'),warm_start=full,execution_config=exe,vce='robust')
                records.append(dict(delta=delta,rep=seed,params=full.params[:2].tolist(),
                    se_full=full.stderr[:2].tolist(),se_standard=standard.stderr[:2].tolist(),
                    p_overid=full.diagnostics['overidentification']['pvalue']))
            except Exception as exc:records.append(dict(delta=delta,rep=seed,error=repr(exc)))
        print('simulation',delta,'completed',flush=True)
    out=[]
    for delta in [0.,.7]:
        selected=[x for x in records if x['delta']==delta and 'error' not in x]
        b=np.array([x['params'] for x in selected]);err=b-np.array([.1,.25])
        s1=np.array([x['se_full'] for x in selected]);s0=np.array([x['se_standard'] for x in selected])
        out.append(dict(delta=delta,success=len(selected),failed=a.reps-len(selected),
            mean_coef=b.mean(axis=0).tolist(),empirical_sd=b.std(axis=0,ddof=1).tolist(),
            rms_se_full=np.sqrt((s1*s1).mean(axis=0)).tolist(),rms_se_standard=np.sqrt((s0*s0).mean(axis=0)).tolist(),
            coverage_full=(abs(err)<=1.95996398454*s1).mean(axis=0).tolist(),
            coverage_standard=(abs(err)<=1.95996398454*s0).mean(axis=0).tolist(),
            reject_overid=float(np.mean([x['p_overid']<.05 for x in selected if x['p_overid'] is not None]))))
    return dict(n=a.n,reps=a.reps,seconds=time.perf_counter()-t,summary=out,records=records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['benchmark','simulation'],default='benchmark')
    p.add_argument('--n',type=int,default=1000000);p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--mode',choices=['moment','profile'],default='profile')
    p.add_argument('--inference',choices=['standard','estimating_equation'],default='estimating_equation')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--reps',type=int,default=300)
    p.add_argument('--output',required=True);a=p.parse_args()
    try:result=benchmark(a) if a.task=='benchmark' else simulation(a)
    except Exception as exc:
        import traceback;traceback.print_exc();result=dict(error=repr(exc),arguments=vars(a))
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','outer']},indent=2))
