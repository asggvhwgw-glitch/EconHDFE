from __future__ import annotations
import argparse,json,time
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from econhdfe import reghdfe

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=200000); ap.add_argument('--controls',type=int,default=8); ap.add_argument('--threads',default='4'); ap.add_argument('--reps',type=int,default=3); ap.add_argument('--out',required=True); a=ap.parse_args()
 thread_arg = 'auto' if str(a.threads).lower() == 'auto' else int(a.threads)
 rng=np.random.default_rng(20260913); levels=(min(20_000,max(500,a.n//10)),min(5_000,max(200,a.n//40)),min(1_000,max(80,a.n//200)),min(200,max(20,a.n//1000)))
 G=[rng.integers(0,L,a.n,dtype=np.int32) for L in levels]
 X=rng.normal(size=(a.n,a.controls)); beta=np.linspace(-.5,.7,a.controls); y=X@beta
 for codes,L in zip(G,levels): y += rng.normal(scale=.35,size=L)[codes]
 y += rng.normal(scale=.5,size=a.n)
 def fit():
  return reghdfe(None,y=y,x=X,absorb=G,vce='robust',projection_backend='auto',absorb_threads=thread_arg,pool_size='auto',memory_budget_mb=256,tol=1e-8,collinearity='drop')
 r=fit(); vals=[]
 for _ in range(a.reps):
  t=time.perf_counter(); r=fit(); vals.append(time.perf_counter()-t)
 out=dict(n=a.n,controls=a.controls,threads=thread_arg,times=vals,median_seconds=float(np.median(vals)),params=np.asarray(r.params,float).tolist(),stderr=np.asarray(r.stderr,float).tolist(),vcov=np.asarray(r.vcov,float).tolist(),nobs=int(r.nobs),rank=int(r.rank),df_absorbed=int(r.df_absorbed),df_resid=float(r.df_resid),iterations=int(r.iterations),solver=(r.absorb_info or {}).get('method'),execution=(r.absorb_info or {}).get('execution_plan'),max_abs_beta_error=float(np.max(np.abs(r.params-beta))))
 Path(a.out).write_text(json.dumps(out,indent=2)); print(a.out)
if __name__=='__main__': main()
