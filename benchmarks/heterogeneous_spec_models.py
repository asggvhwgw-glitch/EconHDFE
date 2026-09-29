from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

from econhdfe import (
    olshdfe, ivhdfe, ppmlhdfe, PPMLHDFE, PPMLConfig,
    ivppmlhdfe, IVPPMLHDFE, IVPPMLConfig, factor, reg_interaction,
)
from econhdfe.design import build_design, _compile_execution_design
from econhdfe.hdfe.plan import FEPlan


def medtime(fn, reps=3):
    out=[]; val=None
    for _ in range(reps):
        t=time.perf_counter(); val=fn(); out.append(time.perf_counter()-t)
    return float(np.median(out)), val


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n',type=int,default=36000)
    ap.add_argument('--groups',type=int,default=6)
    ap.add_argument('--local-slopes',type=int,default=6)
    ap.add_argument('--reps',type=int,default=2)
    ap.add_argument('--output',default='benchmarks/heterogeneous_spec_models.json')
    a=ap.parse_args()
    n=(a.n//a.groups)*a.groups; m=n//a.groups
    rng=np.random.default_rng(20260913)
    g=np.repeat(np.arange(a.groups),m)
    df=pd.DataFrame({'g':g})
    local=[]
    for j in range(a.local_slopes):
        x=rng.normal(size=n); df[f'x{j}']=x
        local.append(reg_interaction(factor('g',drop_base=False),f'x{j}',name=f'gx{j}'))
    for j in range(2): df[f'c{j}']=rng.normal(size=n)
    specs=local+['c0','c1']
    Xdense=build_design(df,specs,n,structural=False)
    plan=FEPlan.from_dataframe(df,['g'])
    execd=_compile_execution_design(df,specs,n,groups=plan.groups,expected_passes=2,memory_budget_mb=512,structural=False)

    # OLS
    b=np.linspace(-.25,.35,a.groups*a.local_slopes).reshape(a.groups,a.local_slopes)
    eta=np.zeros(n)
    for j in range(a.local_slopes): eta += b[g,j]*df[f'x{j}'].to_numpy()
    y_ols=eta+.15*df.c0.to_numpy()+rng.normal(size=n); df['y_ols']=y_ols
    t_ols_s, r_ols_s=medtime(lambda: olshdfe(df,y='y_ols',x=specs,absorb=['g'],vce='robust',collinearity='drop'),a.reps)
    def ols_dense():
        d=build_design(df,specs,n,structural=False)
        return olshdfe(df,y='y_ols',x=d.values,absorb=['g'],vce='robust',collinearity='drop')
    t_ols_d,r_ols_d=medtime(ols_dense,a.reps)

    # Linear IV: heterogeneous endogenous/instrument slopes plus shared controls.
    z=rng.normal(size=n); e=.8*z+.15*df.c0.to_numpy()+rng.normal(scale=.55,size=n)
    df['z']=z; df['e']=e
    end=[reg_interaction(factor('g',drop_base=False),'e',name='ge')]
    ins=[reg_interaction(factor('g',drop_base=False),'z',name='gz')]
    y_iv=np.linspace(.15,.45,a.groups)[g]*e+.1*df.c0.to_numpy()+rng.normal(size=n); df['y_iv']=y_iv
    t_iv_s,r_iv_s=medtime(lambda: ivhdfe(df,y='y_iv',exog=['c0','c1'],endog=end,instruments=ins,absorb=['g'],vce='robust',collinearity='drop'),a.reps)
    def iv_dense():
        C=build_design(df,['c0','c1'],n,structural=False).values
        E=build_design(df,end,n,structural=False).values
        Z=build_design(df,ins,n,structural=False).values
        return ivhdfe(df,y='y_iv',exog=C,endog=E,instruments=Z,absorb=['g'],vce='robust',collinearity='drop')
    t_iv_d,r_iv_d=medtime(iv_dense,a.reps)

    # PPML
    mu=np.exp(np.clip(.08*df.c0.to_numpy()+.25*eta,-2,2)); df['y_pp']=rng.poisson(mu)
    pcfg=PPMLConfig(separation=('fe',),standardize=False,tolerance=1e-8,target_inner_tol=1e-9,max_iter=200,engine='optimized')
    pm=PPMLHDFE(df,absorb=['g'],config=pcfg)
    t_pp_s,r_pp_s=medtime(lambda: pm.fit('y_pp',X=specs,vce='robust'),a.reps)
    def pp_dense():
        d=build_design(df,specs,n,structural=False)
        return ppmlhdfe(df.y_pp.to_numpy(),d.values,absorb=[g],vce='robust',config=pcfg)
    t_pp_d,r_pp_d=medtime(pp_dense,a.reps)

    # IV-PPML
    muiv=np.exp(np.clip(.05*df.c0.to_numpy()+np.linspace(.08,.25,a.groups)[g]*e,-2,2)); df['y_ivpp']=rng.poisson(muiv)
    icfg=IVPPMLConfig(separation=('fe',),standardize=False,tolerance=1e-8,target_inner_tol=1e-9,max_iter=250,engine='optimized')
    im=IVPPMLHDFE(df,absorb=['g'],config=icfg)
    t_ip_s,r_ip_s=medtime(lambda: im.fit('y_ivpp',exog=['c0','c1'],endog=end,instruments=ins,vce='robust'),a.reps)
    def ip_dense():
        C=build_design(df,['c0','c1'],n,structural=False).values
        E=build_design(df,end,n,structural=False).values
        Z=build_design(df,ins,n,structural=False).values
        return ivppmlhdfe(df.y_ivpp.to_numpy(),exog=C,endog=E,instruments=Z,absorb=[g],vce='robust',config=icfg)
    t_ip_d,r_ip_d=medtime(ip_dense,a.reps)

    def rec(td,ts,err):
        return {'dense_seconds':td,'structured_seconds':ts,'speedup':td/ts if ts else None,'max_coef_abs_diff':float(err)}
    out={
      'nobs':n,'groups':a.groups,'local_slopes_per_group':a.local_slopes,
      'design':{
        'ncols':Xdense.values.shape[1],
        'dense_payload_bytes':int(Xdense.values.nbytes),
        'planned_representation':execd.storage_plan.representation,
        'planned_payload_bytes':int(getattr(execd.values,'payload_bytes',execd.values.nbytes if hasattr(execd.values,'nbytes') else 0)),
      },
      'ols':rec(t_ols_d,t_ols_s,np.max(np.abs(r_ols_d.params-r_ols_s.params))),
      'linear_iv':rec(t_iv_d,t_iv_s,np.max(np.abs(r_iv_d.params-r_iv_s.params))),
      'ppml':rec(t_pp_d,t_pp_s,np.max(np.abs(r_pp_d.coef-r_pp_s.coef))),
      'ivppml':rec(t_ip_d,t_ip_s,np.max(np.abs(r_ip_d.coef-r_ip_s.coef))),
      'notes':[
        'Development microbenchmark; not a release performance claim.',
        'PPML/IV-PPML use FE-only separation so shared coefficients remain on the certified structured path.',
        'Linear-IV diagnostics retain dense role arrays but share projections and use bounded multi-RHS AP/SW batches.'
      ]
    }
    Path(a.output).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
