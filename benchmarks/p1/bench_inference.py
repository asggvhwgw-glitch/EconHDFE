"""Measure per-null inference, NOT end-to-end point estimation."""
from __future__ import annotations
import sys,json,time,resource,os,platform,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from econhdfe.inference.iv_poisson import PreparedPanelS,RestrictedPoissonS

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=1_000_000)
 p.add_argument('--kind',choices=['panel','restricted'],default='panel');p.add_argument('--output',required=True)
 a=p.parse_args();n=a.n;rng=np.random.default_rng(20261028)
 z=rng.choice([-1.,1.],(n,2));v=rng.choice([-1.,1.],n);x=.6*(z[:,0]+.4*z[:,1])+.7*v
 c=rng.uniform(-1,1,n);b=.5
 results=[]
 if a.kind=='panel':
  T=20;assert n%T==0;g=np.repeat(np.arange(n//T),T);h=np.tile(np.arange(T)>=T//2,n//T)
  alpha=np.log(5.)+.1*np.tanh(rng.normal(size=n//T))[g]
  rate=np.exp(alpha+b*x+.12*c)+.3*np.exp(alpha)*(1+.8*z[:,0])*v;y=rng.poisson(rate).astype(float)
  for method in ['profile','spj','cross_product']:
   tic=time.perf_counter();model=PreparedPanelS(y,x,z,g,method=method,offset=.12*c,split=h,
                           assume_long_panel=True,assume_cross_time_independence=True)
   construction=time.perf_counter()-tic
   model.test([b]);ts=[];stats=[]
   for _ in range(3):
    tic=time.perf_counter();r=model.test([b]);ts.append(time.perf_counter()-tic);stats.append(r.statistic)
   results.append(dict(method=method,construction_seconds=construction,times=ts,
                       median_seconds=float(np.median(ts)),statistics=stats,
                       scope='one_null_evaluation_on_prepared_data_not_estimation',
                       peak_rss_mib=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)))
   del model
 else:
  gs=[rng.integers(0,max(4,n//200),n),rng.integers(0,40,n),rng.integers(0,12,n)]
  alpha=np.full(n,np.log(5.))
  for g in gs:alpha+=.1*np.tanh(rng.normal(size=g.max()+1))[g]
  rate=np.exp(alpha+b*x+.12*c)+.3*np.exp(alpha)*(1+.8*z[:,0])*v;y=rng.poisson(rate).astype(float)
  tic=time.perf_counter();model=RestrictedPoissonS(y,x,z,exog=c,absorb=gs);construction=time.perf_counter()-tic
  ts=[];diagnostics=[]
  # Small warm-up before full inference, not benchmarked.
  small=RestrictedPoissonS(y[:2000],x[:2000],z[:2000],exog=c[:2000],absorb=[g[:2000] for g in gs[1:]])
  small.test([b]);del small
  for _ in range(3):
   tic=time.perf_counter();r=model.test([b]);ts.append(time.perf_counter()-tic);diagnostics.append(r.to_dict())
  results.append(dict(method='restricted_S_3FE_regular_nuisance',construction_seconds=construction,
                       times=ts,median_seconds=float(np.median(ts)),diagnostics=diagnostics,
                       scope='one_null_nuisance_fit_plus_S_test_not_IV_point_estimation',
                       peak_rss_mib=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)))
 out=dict(n=n,kind=a.kind,results=results,platform=platform.platform(),python=platform.python_version(),
          numpy=np.__version__,numba_threads=os.environ.get('NUMBA_NUM_THREADS'),
          blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
          benchmark_does_not_establish_statistical_validity=True)
 Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
