"""One process per method/configuration. All construction and pilot cost retained."""
from __future__ import annotations
import argparse,json,os,resource,sys,time,platform,traceback
from pathlib import Path
for name in ['OMP_NUM_THREADS','RAYON_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name]='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from fixtures import corpus,updated_weights,forward_error
from econhdfe_planner import PlannedAbsorber
from econhdfe_dual import DualAbsorber
from _econhdfe_within import NativeSolver


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',required=True,choices=['unique_one','unique_two','unique_three','weak_cycle','paired_three','repeated'])
    p.add_argument('--rows',type=int,default=1000000)
    p.add_argument('--rhs',type=int,default=8)
    p.add_argument('--seed',type=int,default=8411)
    p.add_argument('--method',required=True,choices=['planned','v01_auto','legacy_auto','fereg','within','compressed_within'])
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--reweight',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    g,y,w,truth,unit=corpus(a.case,a.rows,a.rhs,a.seed)
    result=dict(arguments=vars(a)|{'output':str(a.output)}, n=len(g),
                levels=[int(col.max())+1 for col in g.T],python=sys.version,
                platform=platform.platform(),threads=2,scope='all setup, copying, calibration, and certification included; oracle audit excluded',
                oracle='weighted four-edge cycle' if a.case.startswith('unique') or a.case=='weak_cycle' else 'paired tuple',
                rounds=[])
    try:
        t=time.perf_counter()
        if a.method=='legacy_auto':
            solver=NativeSolver(g,weights=w,strategy='auto',threads=2,workspace_bytes=1024<<20)
        elif a.method=='v01_auto':
            solver=DualAbsorber(g,weights=w,backend='auto',threads=2,scratch_bytes=1024<<20,allow_fallback=False)
        else:
            solver=PlannedAbsorber(g,weights=w,backend='auto' if a.method=='planned' else a.method,
                                  threads=2,scratch_bytes=1024<<20,allow_fallback=False,
                                  workload='reweight' if a.reweight else 'fixed')
        result['constructor_seconds']=time.perf_counter()-t
        for step in range(a.rounds):
            ww=updated_weights(w,unit,step) if a.reweight else w
            t=time.perf_counter()
            if step and a.reweight:solver.update_weights(ww)
            update=time.perf_counter()-t
            t=time.perf_counter()
            if a.method=='legacy_auto':
                z,ok,it,cert=solver.solve(y,tol=1e-10,max_iter=6000)
                info=dict(accepted=all(ok),backend=solver.route,iterations=it,certificate=cert)
            else:
                z,info=solver.residualize(y,tol=1e-10,max_iter=6000,return_info=True)
            sec=time.perf_counter()-t
            error=forward_error(z,truth,ww)
            valid=bool(info['accepted'] and np.isfinite(z).all() and error<=2e-8)
            result['rounds'].append(dict(step=step,seconds=sec,update_seconds=update,
                                         forward_error=error,valid=valid,route=info['backend'],diagnostics=info))
            if not valid:raise AssertionError(result['rounds'][-1])
        result['valid']=True
    except Exception as e:
        result.update(valid=False,error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
    result['peak_rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(case=a.case,n=len(g),rhs=a.rhs,method=a.method,seed=a.seed,
                          valid=result['valid'],constructor=result.get('constructor_seconds'),
                          times=[r['seconds']+r['update_seconds'] for r in result['rounds']],
                          route=[r['route'] for r in result['rounds']],error=result.get('error')),allow_nan=False),flush=True)
    if not result['valid']:sys.exit(2)

if __name__=='__main__':main()
