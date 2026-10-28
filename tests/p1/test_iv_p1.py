from __future__ import annotations
import itertools
import numpy as np
import pytest
from scipy.optimize import root
from scipy.stats import chi2
from econhdfe.inference.iv_poisson import (
 RestrictedPoissonS,panel_null_scores,panel_s_test,moment_test,invert_scalar_grid,circuit_s_test)


def dgp(n=800,pi=.6,seed=7):
 r=np.random.default_rng(seed); z=r.choice([-1.,1.],size=(n,2));v=r.choice([-1.,1.],size=n)
 c=r.uniform(-1,1,n);e=pi*(z[:,0]+.4*z[:,1])+.7*v;g=np.arange(n)%4
 alpha=np.array([1.,1.2,1.4,1.6])[g];b=.5
 mu=np.exp(alpha+b*e+.1*c)
 y=r.poisson(mu+.2*np.exp(alpha)*v).astype(float)
 return y,e,z,c,g,b


@pytest.mark.parametrize('dims',[1,2,3])
def test_restricted_against_independent_dummy_root(dims):
 y,e,z,c,g,b=dgp();n=len(y);rng=np.random.default_rng(817)
 gs=[g]+[rng.integers(0,j+3,n) for j in range(dims-1)]
 model=RestrictedPoissonS(y,e,z,exog=c,absorb=gs,tol=1e-11,inner_tol=1e-12)
 scores,mu,zt,info=model.null_scores([b])
 U=np.column_stack([np.eye(4)[g],c]+[np.eye(int(gj.max())+1)[gj][:,1:] for gj in gs[1:]])
 def f(a):return U.T@(y-np.exp(b*e+U@a))
 def j(a):
  m=np.exp(b*e+U@a);return -U.T@(m[:,None]*U)
 ref=root(f,np.zeros(U.shape[1]),jac=j,tol=1e-11)
 assert np.max(np.abs(f(ref.x)))<1e-7
 mur=np.exp(b*e+U@ref.x)
 np.testing.assert_allclose(mu,mur,atol=2e-8,rtol=1e-9)
 Zt=z-U@np.linalg.solve(U.T@(mur[:,None]*U),U.T@(mur[:,None]*z))
 np.testing.assert_allclose(zt,Zt,atol=1e-8,rtol=1e-8)
 sr=Zt*(y-mur)[:,None]
 V=(sr-sr.mean(axis=0)).T@(sr-sr.mean(axis=0));m=sr.sum(axis=0)
 stat=float(m@np.linalg.solve(V,m))
 res=model.test([b])
 assert res.statistic==pytest.approx(stat,rel=1e-7,abs=1e-8)
 assert res.pvalue==pytest.approx(chi2.sf(stat,2),rel=1e-7,abs=1e-8)
 assert res.df==2


def test_zero_relevance_does_not_prevent_null_test():
 y,e,z,c,g,b=dgp(pi=0.)
 E=np.zeros_like(e)
 model=RestrictedPoissonS(y,E,z,absorb=[g])
 a=model.test([0.]);d=model.test([100.])
 assert a.pvalue is not None
 assert a.statistic==pytest.approx(d.statistic)
 assert a.diagnostics['target_jacobian_inverted'] is False


@pytest.mark.parametrize('cluster',[False,True])
def test_frequency_expansion(cluster):
 y,e,z,c,g,b=dgp(n=400);w=np.arange(len(y))%3+1;ind=np.repeat(np.arange(len(y)),w)
 cl=np.arange(len(y))//4 if cluster else None
 a=RestrictedPoissonS(y,e,z,exog=c,absorb=[g],weights=w,weight_kind='fweight',clusters=cl).test([b])
 d=RestrictedPoissonS(y[ind],e[ind],z[ind],exog=c[ind],absorb=[g[ind]],
                     clusters=None if cl is None else cl[ind]).test([b])
 assert a.statistic==pytest.approx(d.statistic,rel=1e-8)


def test_instrument_basis_and_units():
 y,e,z,c,g,b=dgp();A=np.array([[1.,3.],[2.,-1.]])
 a=RestrictedPoissonS(y,e,z,exog=c,absorb=[g]).test([b])
 d=RestrictedPoissonS(y,e*10,z@A,exog=c*.001,absorb=[g]).test([b/10])
 assert a.statistic==pytest.approx(d.statistic,rel=1e-8)


def test_rank_failure_no_fake_df():
 y,e,z,c,g,b=dgp();zz=np.column_stack([z[:,0],2*z[:,0]])
 res=RestrictedPoissonS(y,e,zz,absorb=[g]).test([b])
 assert res.pvalue is None and res.df==2
 assert res.status=='singular_or_indefinite_covariance'


@pytest.mark.parametrize('bad',['zero_y','zero_level','badweight','freq','badbeta','missing','constantcontrol'])
def test_restricted_failures(bad):
 y,e,z,c,g,b=dgp();kw=dict(absorb=[g])
 if bad=='zero_y':y[:]=0
 if bad=='zero_level':y[g==0]=0
 if bad=='badweight':kw['weights']=np.ones(len(y))*-1
 if bad=='freq':kw.update(weights=np.ones(len(y))*1.5,weight_kind='fweight')
 if bad=='missing':e[0]=np.nan
 if bad=='constantcontrol':kw['exog']=np.ones(len(y))
 with pytest.raises((ValueError,RuntimeError)):
  model=RestrictedPoissonS(y,e,z,**kw)
  model.test([b,b] if bad=='badbeta' else [b])


def panel_data(G=40,T=8):
 y,e,z,c,_,b=dgp(n=G*T);g=np.repeat(np.arange(G),T);h=np.tile(np.arange(T)>=T//2,G)
 return y,e,z,g,h,b


def test_cross_product_equals_quadratic_enumeration():
 y,e,z,g,h,b=panel_data();r=np.exp(e*b);T=8
 s,info=panel_null_scores(y,e,z,g,[b],method='cross_product',assume_cross_time_independence=True)
 brute=[]
 for j in np.unique(g):
  ids=np.flatnonzero(g==j);tmp=np.zeros(2)
  for it in ids:
   for js in ids:
    if it!=js:tmp+=z[it]*(y[it]*r[js]-y[js]*r[it])
  brute.append(tmp/(T*(T-1)))
 # production scores have a COMMON exp(-max eta) factor
 np.testing.assert_allclose(s,np.array(brute)*np.exp(-np.max(e*b)),atol=1e-12)


def test_spj_joint_cluster_score_combination():
 y,e,z,g,h,b=panel_data()
 s,info=panel_null_scores(y,e,z,g,[b],method='spj',split=h,assume_long_panel=True)
 f,_=panel_null_scores(y,e,z,g,[b],method='profile',assume_long_panel=True)
 a,_=panel_null_scores(y[~h],e[~h],z[~h],g[~h],[b],method='profile',assume_long_panel=True)
 d,_=panel_null_scores(y[h],e[h],z[h],g[h],[b],method='profile',assume_long_panel=True)
 np.testing.assert_allclose(s,2*f-.5*(a+d),atol=1e-12)
 test=panel_s_test(y,e,z,g,[b],method='spj',split=h,assume_long_panel=True)
 assert test.statistic==pytest.approx(moment_test(2*f-.5*(a+d)).statistic)


def test_cross_product_exact_mean_but_not_serial_endogeneity():
 out={}
 for lag in [0.,.6]:
  vals=[];raw=[]
  for z0,z1,vp,v0,v1 in itertools.product([-1.,1.],repeat=5):
   z=np.array([z0,z1]);v=np.array([v0,v1]);x=.8*z+.7*v+lag*np.array([vp,v0])
   r=np.exp(.5*x);y=r+.25*(1+.4*z)*v
   vals.append((r.sum()*(z*y).sum()-y.sum()*(z*r).sum())/2)
   raw.append(((z*y).sum()-y.sum()/r.sum()*(z*r).sum())/2)
  out[lag]=(float(np.mean(vals)),float(np.mean(raw)))
 assert abs(out[0.][0])<1e-14
 assert abs(out[0.][1])>1e-4
 assert abs(out[.6][0])>1e-4


def test_panel_no_zero_group_deletion():
 y,e,z,g,h,b=panel_data();y[g==0]=0
 res=panel_s_test(y,e,z,g,[b],method='cross_product',assume_cross_time_independence=True)
 assert res.diagnostics['all_zero_units_retained']==1 and res.n_units==40


@pytest.mark.parametrize('issue',['assumptions','split','balance','beta','single','nonfinite','method'])
def test_panel_validation(issue):
 y,e,z,g,h,b=panel_data();kw=dict(method='spj',split=h,assume_long_panel=True)
 if issue=='assumptions':kw=dict(method='cross_product')
 if issue=='split':kw.pop('split')
 if issue=='balance':h[0]=1
 if issue=='beta':b=[b,b]
 else:b=[b]
 if issue=='single':g[0]=100
 if issue=='nonfinite':y[0]=np.inf
 if issue=='method':kw['method']='whatever'
 with pytest.raises(ValueError):panel_s_test(y,e,z,g,b,**kw)


def test_grid_unknown_not_rejection_and_boundary_not_infinity():
 def t(b):
  if b==0:raise ValueError('trial failure')
  return moment_test(np.array([[1.],[-1.],[.2],[-.2]]))
 grid=invert_scalar_grid(t,[-2,-1,0,1,2])
 assert grid['unresolved_points']==1
 assert len(grid['accepted_grid_runs'])==2
 assert grid['accepted_grid_runs'][0]['left_search_boundary']
 assert not grid['continuum_set_certified']


def cube_indices(B=12):
 cube=np.array(list(itertools.product(range(2),repeat=3)))
 a=np.tile(cube,(B,1));block=np.repeat(np.arange(B),8)
 i,j,t=a.T
 gs=[4*block+2*i+j,4*block+2*i+t,4*block+2*j+t]
 pos=np.arange(B*8).reshape(B,8)[:,cube.sum(axis=1)%2==0]
 neg=np.arange(B*8).reshape(B,8)[:,cube.sum(axis=1)%2==1]
 return gs,pos,neg


def test_three_way_gravity_circuit():
 gs,pos,neg=cube_indices(200);n=len(gs[0]);y,e,z,c,_,b=dgp(n=n)
 r=circuit_s_test(y,e,z,[b],pos,neg,absorb=gs,
                  assume_independent_rows=True,assume_conditional_iv=True)
 assert r.pvalue is not None and r.n_units==200
 assert r.diagnostics['circuit_size']==8


@pytest.mark.parametrize('bad',['overlap','balance','assumptions'])
def test_circuit_guards(bad):
 gs,pos,neg=cube_indices(10);y,e,z,c,_,b=dgp(n=80)
 kw=dict(absorb=gs,assume_independent_rows=True,assume_conditional_iv=True)
 if bad=='overlap':pos[1,0]=pos[0,0]
 if bad=='balance':gs.append(np.arange(80))
 if bad=='assumptions':kw['assume_conditional_iv']=False
 with pytest.raises(ValueError):circuit_s_test(y,e,z,[b],pos,neg,**kw)


def test_statistic_common_scaling_and_rank():
 s=np.random.default_rng(1).normal(size=(500,3));a=moment_test(s)
 assert moment_test(s*np.array([1e-50,1e50,3.])).statistic==pytest.approx(a.statistic)
 bad=moment_test(np.column_stack([s[:,0],s[:,0]]))
 assert bad.pvalue is None and bad.df==2


def test_circuit_statistic_matches_unscaled_products():
 gs,pos,neg=cube_indices(120);n=len(gs[0]);y,e,z,c,_,b=dgp(n=n)
 r=np.exp(b*e)
 raw=(np.prod(y[pos],axis=1)*np.prod(r[neg],axis=1)
      -np.prod(y[neg],axis=1)*np.prod(r[pos],axis=1))
 scores=(z[pos].sum(axis=1)-z[neg].sum(axis=1))*raw[:,None]
 expected=moment_test(scores)
 actual=circuit_s_test(y,e,z,[b],pos,neg,absorb=gs,
                      assume_independent_rows=True,assume_conditional_iv=True)
 assert actual.statistic==pytest.approx(expected.statistic,rel=1e-10,abs=1e-10)


def test_three_way_product_moment_exact_conditional_mean():
 cube=np.array(list(itertools.product(range(2),repeat=3)))
 i,j,t=cube.T;pos=cube.sum(axis=1)%2==0;neg=~pos
 # Deliberately nonconstant FE, fixed instruments; independent endogenous shocks.
 alpha=np.array([.1,.2,.4,.5])[2*i+j]+np.array([.3,.1,.2,.4])[2*i+t]+np.array([.1,.3,.2,.5])[2*j+t]
 z=np.array([1.,-1.,1.,1.,-1.,1.,-1.,-1.])
 terms=[]
 for v in itertools.product([-1.,1.],repeat=8):
  v=np.array(v);x=.6*z+.7*v;r=np.exp(.5*x)
  y=np.exp(alpha)*(r+.2*(1+.5*z)*v)
  terms.append(np.prod(y[pos])*np.prod(r[neg])-np.prod(y[neg])*np.prod(r[pos]))
 assert abs(np.mean(terms))<1e-10


def test_statistic_does_not_drop_singular_directions():
 result=moment_test(np.zeros((25,3)))
 assert result.pvalue is None and result.df==3


def test_null_scores_cluster_covariance_independent_reference():
 y,e,z,c,g,b=dgp(n=800);cluster=np.arange(len(y))//8
 model=RestrictedPoissonS(y,e,z,exog=c,absorb=[g],clusters=cluster)
 scores,_,_,_=model.null_scores([b])
 sums=scores.reshape(-1,8,2).sum(axis=1)
 ref=moment_test(sums);result=model.test([b])
 assert result.statistic==pytest.approx(ref.statistic,rel=1e-9)


def test_grid_retains_nonconvex_acceptance():
 from econhdfe.inference.iv_poisson.statistic import MomentTestResult
 def t(b):
  accepted=b<-1 or b>1
  return MomentTestResult(0. if accepted else 10.,.8 if accepted else .001,1,'ok',100)
 result=invert_scalar_grid(t,[-3.,-2.,-1.,0.,1.,2.,3.])
 assert len(result['accepted_grid_runs'])==2
 assert result['accepted_grid_runs'][0]['last_grid_point']==-2.
 assert result['accepted_grid_runs'][1]['first_grid_point']==2.


@pytest.mark.parametrize('method',['profile','spj','cross_product'])
@pytest.mark.parametrize('beta',[-.5,.5,1.5])
def test_prepared_panel_exact_reference_parity(method,beta):
 from econhdfe.inference.iv_poisson import PreparedPanelS
 y,e,z,g,h,b=panel_data();kw=dict(method=method,split=h,assume_long_panel=True,assume_cross_time_independence=True)
 p=PreparedPanelS(y,e,z,g,**kw);a=p.test([beta]);r=panel_s_test(y,e,z,g,[beta],**kw)
 assert a.statistic==pytest.approx(r.statistic,rel=2e-12,abs=2e-12)


def test_prepared_copies_original_design():
 from econhdfe.inference.iv_poisson import PreparedPanelS
 y,e,z,g,h,b=panel_data();p=PreparedPanelS(y,e,z,g,assume_cross_time_independence=True)
 a=p.test([b]);e[:]=100.;z[:]=100.;y[:]=0.
 assert p.test([b]).statistic==pytest.approx(a.statistic)
