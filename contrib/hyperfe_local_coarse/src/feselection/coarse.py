"""Same balanced coarse correction as frozen PR #27, with allocation admission."""
from __future__ import annotations
from time import perf_counter
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from .allocation import product_row_counts,group_union_counts,sparse_bytes,CandidateRejected
from scalable_hyperfe import memory_bytes

class GuardedCoarseRight:
    def __init__(self,sys,base,ledger,retained_bytes=0):
        t=perf_counter();self.base=base;Z=base.Z.tocsr();self.Z=Z;self.m,self.k=Z.shape
        if self.k==0:raise CandidateRejected('empty_coarse_space')
        n=sys.H.shape[0];g=sys.groups;ng=len(sys.delta)
        # Admit metadata before symbolic support passes, never allocate N*k dense.
        fixed=retained_bytes+memory_bytes(sys.H)+memory_bytes(Z)+memory_bytes(base.L)+memory_bytes(base.U)
        ledger.admit('coarse_symbolic_before_allocation',fixed+32*n+128*(self.m+ng+self.k))
        cap=max(1,(ledger.budget.limit-fixed)//(16*4))
        counts,nz,work,status=product_row_counts(sys.H.indptr,sys.H.indices,Z.indptr,Z.indices,self.k,
                                               cap,ledger.budget.max_symbolic_work)
        if status:raise CandidateRejected('coarse_symbolic_budget',nnz=int(nz),work=int(work),status=int(status))
        ledger.admit('HZ_before_multiply',fixed+3*sparse_bytes(nz,n)+16*n+16*ng+64*self.k)
        HZ=(sys.H@Z).tocsr();HZ.eliminate_zeros()
        per_group=group_union_counts(g,HZ.indptr,HZ.indices,ng,self.k)
        means_nz=int(per_group.sum());expanded_nz=int(np.sum(per_group[g],dtype=np.int64))
        # The centered support is a subset of the expanded group-mean support.
        # Counts ignore algebraic cancellation, hence they are safe upper bounds.
        Cbound=expanded_nz
        upper=(fixed+3*memory_bytes(HZ)+sparse_bytes(means_nz,ng)
               +5*sparse_bytes(Cbound,n)+64*n+64*ng
               +4*8*self.m*self.k+6*8*self.k*self.k)
        ledger.admit('coarse_centering_before_allocation',upper,centered_support_upper=Cbound,
                     HZ_nnz=int(HZ.nnz),coarse_dimension=self.k,dense_SZ_bytes=8*self.m*self.k)
        F0=sp.csr_matrix((np.ones(n),g,np.arange(n+1,dtype=np.int32)),shape=(n,ng))
        means=(sp.diags(1/sys.delta)@(F0.T@HZ.multiply(sys.w[:,None]))).tocsr()
        C=HZ-means[g];C.eliminate_zeros()
        E=(C.T@C.multiply(sys.w[:,None])).toarray();E=(E+E.T)*.5
        self.Echol=la.cholesky(E,lower=True,check_finite=False)
        self.SZ=(sys.H.T@C.multiply(sys.w[:,None])).toarray()
        self.stats=dict(kind='balanced_coarse',coarse_dimension=self.k,coarse_seconds=perf_counter()-t,
                        centered_nnz=int(C.nnz),centered_nnz_upper=Cbound,coarse_min_eigenvalue=float(la.eigvalsh(E)[0]))
    def forward(self,x):
        fine=self.base.forward(x[self.k:])
        coarse=la.solve_triangular(self.Echol.T,x[:self.k],lower=False,check_finite=False)
        coarse-=la.cho_solve((self.Echol,True),self.SZ.T@fine,check_finite=False)
        return fine+self.Z@coarse
    def adjoint(self,y):
        zy=np.asarray(self.Z.T@y).ravel()
        coarse=la.solve_triangular(self.Echol,zy,lower=True,check_finite=False)
        y=y-self.SZ@la.cho_solve((self.Echol,True),zy,check_finite=False)
        return np.r_[coarse,self.base.adjoint(y)]
