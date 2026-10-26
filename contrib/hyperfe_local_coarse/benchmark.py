"""Isolated million-row validation. Same inputs and audit for every solver."""
from __future__ import annotations
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','RAYON_NUM_THREADS']:os.environ[key]='1'
import argparse,json,sys,time,traceback,threading,resource,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'src'))
import numpy as np
import psutil
from cases import make_case,metadata
from scalable_hyperfe import ProfiledSystem,StarInverse,StarRightFactor,anchors,pair_reference_mask,full_lsmr,ResourceGuard
from budget_preconditioner import *
from validation import audit

p=argparse.ArgumentParser()
p.add_argument('--case', choices=['joint4','equal_sparse','smooth_sparse','irregular','nonadditive','mobility3','hub4','weighted4','high5'], default='joint4');p.add_argument('--mode', choices=['balanced','block','additive','old_star','ic0','ilu','reference','native_adaptive','native_diagonal','native_additive'], default='balanced')
p.add_argument('--seed',type=int,default=20261026);p.add_argument('--rhs',type=int,default=3)
p.add_argument('--n',type=int,default=1_000_000);p.add_argument('--tol',type=float,default=1e-12)
p.add_argument('--maxiter',type=int,default=1200);p.add_argument('--tau',type=float,default=1e-3);p.add_argument('--relabel',action='store_true')
p.add_argument('--reuse',action='store_true');p.add_argument('--save',type=str,default=None)
p.add_argument('--reference',type=str,default=None);p.add_argument('--out',required=True)
a=p.parse_args()
if a.n<1_000_000:raise ValueError('Performance runs require >= 1 million observations')
if a.rhs < 3: raise ValueError('Benchmark requires >=3 RHS, including the oracle column')
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
rec={'request':vars(a),'status':'starting'}
try:
    import within
    rec['within_version']=__import__('importlib.metadata',fromlist=['version']).version('within-py')
    tj=time.perf_counter();warmup()
    for f in [__import__('scalable_hyperfe').triangular_unit_lower,__import__('scalable_hyperfe').triangular_unit_upper]:
        f(np.ones(1),np.array([0],np.int32),np.array([0,1],np.int32),np.ones(1))
    c,w,Y,truth=make_case(a.case,a.n,a.seed)
    if a.rhs>3:
        rng=np.random.default_rng(a.seed+913)
        Y=np.column_stack([Y[:,:2],rng.normal(size=(a.n,a.rhs-3)),Y[:,-1]])
    if a.relabel:
        rng=np.random.default_rng(a.seed+717)
        c=np.column_stack([rng.permutation(int(c[:,q].max())+1)[c[:,q]] for q in range(c.shape[1])]).astype(np.int32)
    anchors(c[:,0],w,int(c[:,0].max())+1)
    for j in range(1,c.shape[1]):pair_reference_mask(c[:,0],c[:,j],int(c[:,0].max())+1,int(c[:,j].max())+1)
    rec['generation_and_warmup_excluded']=time.perf_counter()-tj
    rec['design']=metadata(c,w)
    rec['input_sha256']=hashlib.sha256(c.tobytes()+w.tobytes()+Y.tobytes()).hexdigest()
    proc=psutil.Process();rss0=proc.memory_info().rss;peak=[rss0];stop=threading.Event()
    def monitor():
        while not stop.wait(.02):peak[0]=max(peak[0],proc.memory_info().rss)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    start=time.perf_counter()
    try:
        native=a.mode.startswith('native_')
        if native:
            config={'adaptive':within.PreconditionerConfig.Adaptive,'diagonal':within.PreconditionerConfig.Diagonal,'additive':within.PreconditionerConfig.Additive}[a.mode[7:]]()
            opts=within.LsmrOptions(tol=a.tol,maxiter=a.maxiter)
            t=time.perf_counter();cat=np.asfortranarray(c,dtype=np.uint32);resp=np.asfortranarray(Y)
            design=within.Design(cat);solver=within.Solver(design,weights=w,preconditioner=config)
            rec['setup_seconds']=time.perf_counter()-t
            t=time.perf_counter();sol=solver.solve_batch(resp,options=opts);R=np.asarray(sol.demeaned)
            rec['solve_seconds']=time.perf_counter()-t
            rec['iterations']=list(sol.iterations);rec['converged']=list(sol.converged)
            rec['preconditioner_seconds']=solver.preconditioner.build_duration_seconds
        elif a.mode=='reference':
            R,inf=full_lsmr(c,w,Y,rtol=a.tol,maxiter=a.maxiter)
            rec.update(setup_seconds=inf['setup_seconds'],solve_seconds=inf['solve_seconds'],iterations=[x['iterations'] for x in inf['rhs']],converged=[x['status'] in [1,2,4,5] for x in inf['rhs']])
        else:
            t=time.perf_counter();s=ProfiledSystem.build(c,w);rec['system_seconds']=time.perf_counter()-t
            if a.mode=='old_star':
                pre=StarInverse.build(s);root=StarRightFactor(pre);rec['preconditioner']=pre.stats
            else:
                T,scale,stats=build_star(s);rec['assembly']=stats
                if a.mode=='ic0':root=ICRight(T,scale)
                elif a.mode=='ilu':root=ILURight(T,scale,drop_tol=1e-4,fill_factor=3.)
                else:
                    root=ComponentRight(T,scale,tau=a.tau,max_block=256,eig_cut=0 if a.mode=='block' else 0.005,max_coarse=256)
                    rec['blocks']=root.stats
                    if a.mode in ['additive','balanced']:
                        root=CoarseRight(s,root,balanced=a.mode=='balanced');rec['coarse']=root.stats
                rec['preconditioner']=root.stats
            rec['setup_seconds']=time.perf_counter()-start
            R,inf=solve_rect(s,Y,root,tol=a.tol,maxiter=a.maxiter)
            rec['solve_seconds']=inf['solve_seconds'];rec['iterations']=[x['iterations'] for x in inf['rhs']];rec['converged']=[x['status'] in [1,2,4,5] for x in inf['rhs']]
        rec['total_seconds']=time.perf_counter()-start
    finally:
        peak[0]=max(peak[0],proc.memory_info().rss);stop.set();thread.join()
        rec['peak_rss_mib']=peak[0]/2**20;rec['extra_rss_mib']=(peak[0]-rss0)/2**20
    rec['audit']=audit(c,w,Y,R,truth)
    rec['passed']=bool(all(rec['converged']) and rec['audit']['projection_gate_passed'])
    if a.reference:
        ref=np.load(a.reference,mmap_mode='r');err=np.linalg.norm(np.sqrt(w)[:,None]*(R-ref),axis=0)/np.linalg.norm(np.sqrt(w)[:,None]*ref,axis=0)
        rec['reference_relative_errors']=err.tolist();rec['all_rhs_reference_passed']=bool(max(err)<1e-8)
        rec['reference_fwl']=audit(c,w,Y,ref,truth)
        rec['passed']=bool(rec['passed'] and rec['all_rhs_reference_passed'])
    if a.save:np.save(a.save,R)
    if a.reuse and a.mode!='reference':
        Y2=Y.copy();Y2[:,0]+=Y[:,1];Y2[:,-1]*=-1
        t=time.perf_counter()
        if native:
            rr=solver.solve_batch(np.asfortranarray(Y2),options=opts);R2=np.asarray(rr.demeaned)
            reuse_converged=list(rr.converged);reuse_iterations=list(rr.iterations)
        else:
            R2,inf=solve_rect(s,Y2,root,tol=a.tol,maxiter=a.maxiter)
            reuse_converged=[x['status'] in [1,2,4,5] for x in inf['rhs']]
            reuse_iterations=[x['iterations'] for x in inf['rhs']]
        elapsed=time.perf_counter()-t
        expect=R.copy();expect[:,0]+=R[:,1];expect[:,-1]*=-1
        reuse_audit=audit(c,w,Y2,R2,-truth)
        denominator=np.maximum(np.linalg.norm(np.sqrt(w)[:,None]*expect,axis=0),np.finfo(float).tiny)
        linearity_error=np.linalg.norm(np.sqrt(w)[:,None]*(R2-expect),axis=0)/denominator
        reuse_passed=bool(all(reuse_converged) and reuse_audit['projection_gate_passed']
                          and np.all(np.isfinite(linearity_error)) and np.all(linearity_error<1e-8))
        rec['reuse']={'solve_seconds':elapsed,'iterations':reuse_iterations,'converged':reuse_converged,
                      'audit':reuse_audit,'linearity_error':linearity_error.tolist(),'passed':reuse_passed}
        rec['passed']=bool(rec['passed'] and reuse_passed)
    rec['status']='completed'
except ResourceGuard as e:
    rec.update(status='budget_rejected',error=str(e),passed=False)
    rec['elapsed_to_rejection']=time.perf_counter()-start
except Exception as e:rec.update(status='error',error=str(e),traceback=traceback.format_exc(),passed=False)
finally:
    rec['process_maxrss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    out.write_text(json.dumps(rec,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
    print(json.dumps({'case':a.case,'mode':a.mode,'seed':a.seed,'rhs':a.rhs,'status':rec['status'],'passed':rec.get('passed'),'time':rec.get('total_seconds'),'error':rec.get('error')}),flush=True)
