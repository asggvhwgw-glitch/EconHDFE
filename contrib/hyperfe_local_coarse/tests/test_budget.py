import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import pytest
from budget_preconditioner import *
from scalable_hyperfe import ProfiledSystem,dummy,ResourceGuard

@pytest.mark.parametrize('kind',['ic0','ilu','block'])
@pytest.mark.parametrize('seed',range(4))
def test_right_adjoint(kind,seed):
 rng=np.random.default_rng(seed);A=rng.normal(size=(9,9));T=sp.csc_matrix(A.T@A+np.eye(9));scale=np.exp(rng.normal(size=9))
 r=ICRight(T,scale) if kind=='ic0' else ILURight(T,scale) if kind=='ilu' else ComponentRight(T,scale,tau=0,max_block=20,eig_cut=0)
 x=rng.normal(size=9);y=rng.normal(size=9)
 assert np.allclose(r.forward(x)@y,x@r.adjoint(y),rtol=1e-12,atol=1e-12)
 R=np.column_stack([r.forward(x) for x in np.eye(9)])
 assert np.linalg.matrix_rank(R)==9

@pytest.mark.parametrize('seed',range(4))
def test_full_pattern_ic_is_exact(seed):
 rng=np.random.default_rng(seed);A=rng.normal(size=(12,12));T=sp.csc_matrix(A.T@A+np.eye(12))
 r=ICRight(T,np.ones(12),ordering='none')
 assert r.stats['shift']==0
 assert np.allclose((r.L@r.L.T).toarray(),T.toarray(),atol=1e-12)
 assert r.L.nnz<=sp.tril(T).nnz

@pytest.mark.parametrize('seed',range(4))
@pytest.mark.parametrize('balanced',[False,True])
def test_coarse_formula_and_adjoint(seed,balanced):
 rng=np.random.default_rng(seed);n=60;m=8;k=3;g=np.repeat(np.arange(12),5);w=np.exp(rng.normal(size=n))
 H=sp.csr_matrix(rng.normal(size=(n,m)));delta=np.bincount(g,weights=w)
 center=lambda y:y-(np.bincount(g,weights=w*y)/delta)[g]
 sys=SimpleNamespace(H=H,w=w,groups=g,delta=delta,center=center)
 A=np.column_stack([np.sqrt(w)*center(H[:,j].toarray().ravel()) for j in range(m)])
 S=A.T@A;T=sp.csc_matrix(S+np.eye(m));base=ICRight(T,np.ones(m),ordering='none')
 Z=rng.normal(size=(m,k));base.Z=sp.csr_matrix(Z)
 root=CoarseRight(sys,base,balanced)
 R=np.column_stack([root.forward(x) for x in np.eye(m+k)])
 R0=np.column_stack([base.forward(x) for x in np.eye(m)])
 E=Z@np.linalg.solve(Z.T@S@Z,Z.T)
 target=E+(np.eye(m)-E@S)@(R0@R0.T)@(np.eye(m)-S@E) if balanced else E+R0@R0.T
 assert np.allclose(R@R.T,target,rtol=1e-11,atol=1e-11)
 x=rng.normal(size=m+k);y=rng.normal(size=m)
 assert np.allclose(root.forward(x)@y,x@root.adjoint(y),rtol=1e-12,atol=1e-12)
 if balanced:assert np.allclose((R@R.T)@S@Z,Z,rtol=1e-10,atol=1e-10)
 Y=rng.normal(size=(n,2));res,info=solve_rect(sys,Y,root,tol=1e-13)
 target_r=[]
 for j in range(2):
  yy=np.sqrt(w)*center(Y[:,j]);beta=np.linalg.lstsq(A,yy,rcond=None)[0]
  target_r.append(center(Y[:,j]-H@beta))
 assert np.allclose(res,np.column_stack(target_r),atol=1e-10)

@pytest.mark.parametrize('bad_weight',[0.,-1.,np.inf,np.nan])
def test_invalid_weight_rejected(bad_weight):
 rng=np.random.default_rng(5);c=rng.integers(0,3,size=(100,3),dtype=np.int32);w=np.ones(100);w[0]=bad_weight
 with pytest.raises(ValueError):ProfiledSystem.build(c,w)

def test_block_budget_rejected():
 T=sp.csc_matrix(np.ones((12,12))+np.eye(12))
 with pytest.raises(ResourceGuard):ComponentRight(T,np.ones(12),tau=1e-3,max_block=4)

def test_coarse_budget_rejected():
 B=np.array([[1.,-.999],[-.999,1.]])
 T=sp.block_diag([B]*5,format='csc')
 with pytest.raises(ResourceGuard):ComponentRight(T,np.ones(10),eig_cut=.01,max_coarse=3)

def test_component_permutation_equivariance():
 rng=np.random.default_rng(72)
 blocks=[]
 for n in [3,5,7]:
  A=rng.normal(size=(n,n));blocks.append(A.T@A+np.eye(n))
 T=sp.block_diag(blocks,format='csc');scale=np.ones(15)
 a=ComponentRight(T,scale,tau=1e-9,eig_cut=0)
 p=rng.permutation(15);ip=np.argsort(p);b=ComponentRight(T[p,:][:,p],scale,tau=1e-9,eig_cut=0)
 x=rng.normal(size=15)
 assert np.allclose(a.forward(a.adjoint(x)),b.forward(b.adjoint(x[p]))[ip],atol=1e-11)

def test_preflight_uniform_bypass():
 c=np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]],dtype=np.int32)
 assert not preflight(c,np.ones(4))['eligible']

def test_support_partition_does_not_change_input():
 c=np.array([[0,0,0],[0,1,1],[1,2,2],[1,3,3],[0,2,3]],dtype=np.int32)
 w=np.array([1,1,1,1,1e-6]);oldc=c.copy();oldw=w.copy()
 r=preflight(c,w,max_block=8)
 assert r['eligible'] and r['components']==2 and r['max_other_levels']==4
 assert np.array_equal(c,oldc) and np.array_equal(w,oldw)
