"""Two additional hold-out designs; no tuning or modification of planner rules.

Fully independent random, unweighted three-way FE use scipy LSMR as an
independent reference. Dense repeated cells use an exact paired residual.
One worker process per method, no shared solver or warm factorization.
"""
import os
for name in ['OMP_NUM_THREADS','RAYON_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name]='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,subprocess,sys,time,resource
from pathlib import Path
import numpy as np
from econhdfe_planner import PlannedAbsorber
from econhdfe_planner.structure import compact_codes
from econhdfe_dual import DualAbsorber
from _econhdfe_within import NativeSolver
from fixtures import forward_error
ROOT=Path(__file__).resolve().parent


def data(case,n,k):
    rng=np.random.default_rng(71031)
    if case=='iid_three':
        levels=(n//10,n//12,n//16)
        cols=[]
        for l in levels:
            c=(np.arange(n,dtype=np.uint32)%l);rng.shuffle(c);cols.append(c)
        g=np.column_stack(cols)
        y=rng.normal(size=(n,k))
        for col,l in zip(cols,levels):y+=rng.normal(scale=.3,size=(l,k))[col]
        offsets=np.cumsum([0,*levels]);total=offsets[-1]
        from scipy.sparse.linalg import LinearOperator,lsmr
        def mv(a):
            z=np.zeros(n)
            for j,c in enumerate(cols):z+=a[offsets[j]+c]
            return z
        def rm(v):return np.concatenate([np.bincount(c,weights=v,minlength=l) for c,l in zip(cols,levels)])
        op=LinearOperator((n,total),matvec=mv,rmatvec=rm,dtype=np.float64)
        truth=np.empty_like(y);logs=[]
        for j in range(k):
            r=lsmr(op,y[:,j],atol=1e-13,btol=1e-13,maxiter=3000)
            assert r[1] in (0,1,2),r[1:4]
            truth[:,j]=y[:,j]-op.matvec(r[0])
            score=np.linalg.norm(op.rmatvec(truth[:,j]))/max(1.,np.linalg.norm(op.rmatvec(y[:,j])))
            assert score<1e-9
            logs.append(dict(stop=int(r[1]),iterations=int(r[2]),normal_residual=float(score)))
        return g,y,np.ones(n),truth,dict(type='independent_scipy_lsmr',atol=1e-13,btol=1e-13,columns=logs)
    m=n//16
    base=np.column_stack([rng.integers(l,size=m,dtype=np.uint32) for l in (m//4,m//6,32)])
    g=np.repeat(base,16,axis=0);g,_=compact_codes(g)
    w=np.repeat(np.exp(rng.normal(scale=.5,size=m)),16)
    e=rng.normal(size=(n//2,k));truth=np.empty((n,k));truth[::2]=e;truth[1::2]=-e
    y=truth.copy()
    for col in g.T:y+=rng.normal(scale=.3,size=(int(col.max())+1,k))[col]
    order=rng.permutation(n)
    return g[order],y[order],w[order],truth[order],dict(type='exact_paired_residual',repetition=16)


def worker(case,method,n,k):
    g,y,w,truth,reference=data(case,n,k)
    t=time.perf_counter()
    if method=='legacy_auto':s=NativeSolver(g,weights=w,strategy='auto',threads=2,workspace_bytes=1024<<20)
    elif method=='v01_auto':s=DualAbsorber(g,weights=w,backend='auto',threads=2,scratch_bytes=1024<<20,allow_fallback=False)
    else:s=PlannedAbsorber(g,weights=w,backend='auto',threads=2,scratch_bytes=1024<<20,allow_fallback=False)
    ctor=time.perf_counter()-t
    result=dict(case=case,n=n,rhs=k,seed=71031,method=method,constructor_seconds=ctor,reference=reference,rounds=[])
    for step in range(3):
        t=time.perf_counter()
        if method=='legacy_auto':
            z,ok,it,cert=s.solve(y,tol=1e-10,max_iter=6000)
            info=dict(accepted=all(ok),backend=s.route,certificate=cert)
        else:z,info=s.residualize(y,tol=1e-10,max_iter=6000,return_info=True)
        elapsed=time.perf_counter()-t
        err=forward_error(z,truth,w)
        assert info['accepted'] and err<2e-8,(case,method,err)
        result['rounds'].append(dict(step=step,seconds=elapsed,forward_error=err,route=info['backend'],diagnostics=info))
    result.update(valid=True,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (ROOT/'results'/f'supplement-{case}-{method}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({key:result[key] for key in ['case','method','valid','constructor_seconds']} | {'times':[v['seconds'] for v in result['rounds']],'route':s.backend if method!='legacy_auto' else s.route}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('--case');ap.add_argument('--method');ap.add_argument('--wait',action='store_true');a=ap.parse_args()
    if a.worker:worker(a.case,a.method,1000000,2 if a.case=='iid_three' else 8)
    else:
        if a.wait:
            while json.loads((ROOT/'results/suite-status.json').read_text())['completed']<32:time.sleep(3)
        for case in ['iid_three','dense_repeated']:
            for method in ['legacy_auto','v01_auto','planned']:
                subprocess.run([sys.executable,__file__,'--worker','--case',case,'--method',method],check=True,timeout=480)
        print('SUPPLEMENT_COMPLETE 6',flush=True)
