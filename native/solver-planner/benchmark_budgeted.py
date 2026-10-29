"""Paid finite-horizon comparisons: all planner/IPC costs, independent checks."""
from __future__ import annotations
import argparse,json,os,platform,resource
from pathlib import Path
from time import perf_counter
import numpy as np
from fixtures import corpus,updated_weights,forward_error

def run(a):
    from econhdfe_planner import BudgetedAbsorber,PlannedAbsorber
    from econhdfe_planner.budgeted import _LegacyAuto
    if a.case in ('iid_three','dense_repeated'):
        from supplementary import data as extra_data
        g,y,w,truth,ref=extra_data(a.case,a.rows,a.rhs)
        units=np.arange(len(g),dtype=np.uint32)
    else:
        g,y,w,truth,units=corpus(a.case,a.rows,a.rhs,a.seed);ref={'type':'analytic_cycle_or_paired'}
    n=len(g)
    t=perf_counter()
    if a.method=='legacy': obj=_LegacyAuto(g,w,a.threads,1<<30)
    elif a.method=='v02': obj=PlannedAbsorber(g,weights=w,threads=a.threads,scratch_bytes=1<<30,workload='reweight' if a.reweight else 'fixed',allow_fallback=False)
    else: obj=BudgetedAbsorber(g,weights=w,threads=a.threads,scratch_bytes=1<<30,expected_calls=a.calls,workload='reweight' if a.reweight else 'fixed',allow_fallback=False,max_probe_seconds=a.budget,probe_max_iter=a.cap,max_overhead_fraction=a.overhead,exploration_fraction=a.explore,probe_rhs=a.pilot_rhs)
    constructor=perf_counter()-t; rounds=[]
    try:
        for j in range(a.calls):
            ww=updated_weights(w,units,j) if a.reweight else w
            scale=1+.05*np.sin(j+np.arange(a.rhs)); yy=y*scale; exact=truth*scale
            t=perf_counter()
            if a.reweight and j>0: obj.update_weights(ww)
            update=perf_counter()-t;t=perf_counter()
            if a.method=='legacy':z,info=obj.project(yy,1e-10,6000)
            else:z,info=obj.residualize(yy,tol=1e-10,max_iter=6000,return_info=True)
            elapsed=perf_counter()-t;gap=forward_error(z,exact,ww)
            accepted=bool(info.get('accepted') and np.isfinite(gap) and gap<2e-8)
            rounds.append(dict(call=j+1,update_seconds=update,solve_seconds=elapsed,seconds=update+elapsed,forward_error=gap,valid=accepted,info=info))
            print('CALL', a.case, a.method, j+1, round(update+elapsed,4),info.get('backend'), flush=True)
            if not accepted:break
    finally:
        if hasattr(obj,'close'):obj.close()
    return dict(case=a.case,n=n,rhs=a.rhs,seed=a.seed,reference=ref,method=a.method,horizon=a.calls,reweight=a.reweight,constructor_seconds=constructor,rounds=rounds,total_seconds=constructor+sum(x['seconds'] for x in rounds),peak_parent_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,peak_child_rss_mb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,memory_scope='separate process HWM, not simultaneous combined RSS',valid=len(rounds)==a.calls and all(x['valid'] for x in rounds),threads=a.threads,python=platform.python_version(),budget=a.budget,probe_cap=a.cap,max_overhead_fraction=a.overhead,exploration_fraction=a.explore,probe_rhs=a.pilot_rhs)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',required=True);p.add_argument('--rows',type=int,default=1000000)
    p.add_argument('--rhs',type=int,default=8);p.add_argument('--seed',type=int,default=31733)
    p.add_argument('--method',choices=['legacy','v02','v03'],required=True)
    p.add_argument('--calls',type=int,default=8);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--reweight',action='store_true');p.add_argument('--budget',type=float,default=3.)
    p.add_argument('--cap',type=int,default=128);p.add_argument('--overhead',type=float,default=.5)
    p.add_argument('--explore',type=float,default=.1)
    p.add_argument('--pilot-rhs',type=int,default=1)
    p.add_argument('--out',required=True);a=p.parse_args()
    result=run(a);Path(a.out).write_text(json.dumps(result,indent=2,allow_nan=False))
    plan=result['rounds'][-1]['info'].get('plan',{})
    print(json.dumps(dict(case=a.case,n=result['n'],rhs=a.rhs,method=a.method,horizon=a.calls,seconds=result['total_seconds'],valid=result['valid'],route=plan.get('selected'),reason=plan.get('reason'),statuses=[x['status'] for x in plan.get('trials',[])],max_error=max(x['forward_error'] for x in result['rounds']))),flush=True)
    if not result['valid']:raise SystemExit(2)
if __name__=='__main__':main()
