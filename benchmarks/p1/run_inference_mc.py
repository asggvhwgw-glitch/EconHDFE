"""Seeded P1 null-size experiments; count failures, never silently drop them."""
from __future__ import annotations
import sys,json,time,argparse,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import chi2
from econhdfe.inference.iv_poisson import RestrictedPoissonS,panel_s_test,circuit_s_test,invert_scalar_grid

def simulate(rng,G,T,pi,*,lag=0.):
 n=G*T;unit=np.repeat(np.arange(G),T)
 Z=rng.choice([-1.,1.],(n,2));V=rng.choice([-1.,1.],(G,T+1))
 v=V[:,1:].ravel();prev=V[:,:-1].ravel()
 X=pi*(Z[:,0]+.4*Z[:,1])+.7*v+lag*prev
 C=rng.uniform(-1,1,n);b=.5
 alpha=np.log(5.)+.25*np.tanh(rng.normal(size=G))[unit]
 mu=np.exp(alpha+b*X+.12*C)
 rate=mu+.3*np.exp(alpha)*(1+.8*Z[:,0])*v
 assert np.all(rate>0)
 y=rng.poisson(rate).astype(float)
 return y,X,Z,unit,C,b

def wilson(k,n):
 z=1.95996398454;p=k/n;den=1+z*z/n
 mid=(p+z*z/(2*n))/den;rad=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [float(mid-rad),float(mid+rad)]

def record(values,failures,reps):
 ok=[v for v in values if v is not None];k=sum(p<.05 for p in ok)
 return dict(repetitions=reps,completed=len(ok),failures=failures,rejected=k,
  rejection_rate=None if len(ok)!=reps else k/reps,
  rejection_rate_among_completed=None if not ok else k/len(ok),
  wilson_95=None if len(ok)!=reps else wilson(k,reps))

def finite_dim(reps):
 n=4000;out=[]
 for label,pi in [('strong',1.),('local_weak',4/np.sqrt(n)),('zero',0.)]:
  vals=[];fails=[];wrong=[];tic=time.perf_counter()
  for k in range(reps):
   y,e,z,g,c,b=simulate(np.random.default_rng(110_000+k),4,n//4,pi)
   try:
    r=RestrictedPoissonS(y,e,z,exog=c,absorb=[g]).test([b]);vals.append(r.pvalue)
    wrong.append(None if r.statistic is None else float(chi2.sf(r.statistic,1)))
    if r.pvalue is None:fails.append(dict(rep=k,error=r.status))
   except Exception as ex:vals.append(None);wrong.append(None);fails.append(dict(rep=k,error=str(ex)))
  item=dict(suite='fixed_nuisance',n=n,FE=4,strength=label,pi=float(pi),
   correct=record(vals,fails,reps),wrong_df_L_minus_K=record(wrong,fails,reps),
   seconds=time.perf_counter()-tic,pvalues=vals)
  out.append(item);print({k:v for k,v in item.items() if k!='pvalues'},flush=True)
 return out

def panel(reps):
 out=[]
 specs=[(1000,2,'strong',1.,0.),(1000,4,'strong',1.,0.),
  (1000,4,'local_weak',4/np.sqrt(4000),0.),(1000,4,'zero',0.,0.),
  (1000,16,'strong',1.,0.),(1000,64,'strong',1.,0.),
  (4000,4,'strong',1.,0.),(1000,4,'serial_violation',.8,.6)]
 for G,T,label,pi,lag in specs:
  methods=['profile','cross_product']+(['spj'] if T>=4 else [])
  vals={m:[] for m in methods};fails={m:[] for m in methods};tic=time.perf_counter()
  for k in range(reps):
   y,e,z,g,c,b=simulate(np.random.default_rng(210_000+k),G,T,pi,lag=lag)
   split=np.tile(np.arange(T)>=T//2,G)
   for method in methods:
    try:
     kw=dict(method=method,offset=.12*c)
     if method=='cross_product':kw['assume_cross_time_independence']=True
     else:kw['assume_long_panel']=True
     if method=='spj':kw['split']=split
     result=panel_s_test(y,e,z,g,[b],**kw);vals[method].append(result.pvalue)
     if result.pvalue is None:fails[method].append(dict(rep=k,error=result.status))
    except Exception as ex:vals[method].append(None);fails[method].append(dict(rep=k,error=str(ex)))
  item=dict(suite='growing_one_way_FE',G=G,T=T,strength=label,pi=float(pi),lag=lag,
   baseline_additive_IV_valid=True,cross_product_assumptions_valid=(lag==0),
   long_panel_methods_fixed_T_are_deliberate_stress_tests=True,
   methods={m:record(vals[m],fails[m],reps) for m in methods},
   seconds=time.perf_counter()-tic,pvalues=vals)
  out.append(item);print({k:v for k,v in item.items() if k!='pvalues'},flush=True)
 return out

def gravity(N=64,T=4):
 i,j,t=np.meshgrid(np.arange(N),np.arange(N),np.arange(T),indexing='ij')
 i,j,t=i.ravel(),j.ravel(),t.ravel();grid=np.arange(len(i)).reshape(N,N,T);pos=[];neg=[]
 for a in range(0,N,2):
  for b in range(0,N,2):
   for s in range(0,T,2):
    plus=[];minus=[]
    for di,dj,dt in itertools.product([0,1],repeat=3):
     (plus if (di+dj+dt)%2==0 else minus).append(grid[a+di,b+dj,s+dt])
    pos.append(plus);neg.append(minus)
 return [i*N+j,i*T+t,j*T+t],np.array(pos),np.array(neg)

def circuits(reps):
 gs,pos,neg=gravity();n=len(gs[0]);out=[]
 for label,pi in [('strong',1.),('local_weak',4/np.sqrt(n)),('zero',0.)]:
  vals=[];fails=[];energy=[];power=[];tic=time.perf_counter()
  for k in range(reps):
   rng=np.random.default_rng(310_000+k);z=rng.choice([-1.,1.],(n,2));v=rng.choice([-1.,1.],n)
   x=pi*(z[:,0]+.4*z[:,1])+.7*v;b=.5;alpha=np.full(n,np.log(5.))
   for g in gs:alpha+=.15*np.tanh(rng.normal(size=g.max()+1))[g]
   mu=np.exp(alpha+b*x);rate=mu+.3*np.exp(alpha)*(1+.8*z[:,0])*v
   assert np.all(rate>0);y=rng.poisson(rate).astype(float)
   try:
    r=circuit_s_test(y,x,z,[b],pos,neg,absorb=gs,assume_independent_rows=True,assume_conditional_iv=True)
    vals.append(r.pvalue);energy.append(r.diagnostics.get('max_unit_energy_share'))
    rr=circuit_s_test(y,x,z,[0.],pos,neg,absorb=gs,assume_independent_rows=True,assume_conditional_iv=True)
    power.append(rr.pvalue)
    if r.pvalue is None:fails.append(dict(rep=k,error=r.status))
   except Exception as ex:vals.append(None);power.append(None);fails.append(dict(rep=k,error=str(ex)))
  item=dict(suite='three_way_gravity_circuits',origins=64,destinations=64,T=4,n=n,
   FE_levels=[int(g.max()+1) for g in gs],independent_circuits=len(pos),strength=label,pi=float(pi),
   null=record(vals,fails,reps),alternative_beta_zero=record(power,fails,reps),
   median_max_energy_share=float(np.median(energy)) if energy else None,
   seconds=time.perf_counter()-tic,pvalues=vals)
  out.append(item);print({k:v for k,v in item.items() if k!='pvalues'},flush=True)
 return out

def inversion_examples():
 out=[];G=200;T=4;n=G*T
 for label,pi in [('strong',1.),('local_weak',4/np.sqrt(n)),('zero',0.)]:
  y,e,z,g,c,b=simulate(np.random.default_rng(5500),G,T,pi)
  def test(value):return panel_s_test(y,e,z,g,[value],method='cross_product',offset=.12*c,assume_cross_time_independence=True)
  inv=invert_scalar_grid(test,np.linspace(-3.5,4.5,161))
  out.append(dict(strength=label,pi=float(pi),true_beta=b,method='cross_product',grid=inv))
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--suite',choices=['fixed','panel','circuits','inversion'],required=True)
 p.add_argument('--reps',type=int,default=1000);p.add_argument('--output',required=True);a=p.parse_args()
 result={'fixed':finite_dim,'panel':panel,'circuits':circuits,'inversion':lambda _:inversion_examples()}[a.suite](a.reps)
 Path(a.output).write_text(json.dumps(dict(suite=a.suite,repetitions=a.reps,results=result),indent=2))
