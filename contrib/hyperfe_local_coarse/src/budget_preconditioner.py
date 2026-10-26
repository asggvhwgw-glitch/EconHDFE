"""Bounded-fill experimental right preconditioners, NOT penalties to the target.
No spectral bound from the exact star inverse is claimed for incomplete factors.
"""
from __future__ import annotations
from time import perf_counter
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import spilu, LinearOperator, lsmr
from numba import njit
from scalable_hyperfe import anchors, ResourceGuard

@njit(cache=True)
def ic0_factor(values, indices, ptr, shift, min_pivot=1e-12):
    """Row-oriented IC(0). Lower CSR sorted, diagonal last in every row."""
    out=np.zeros_like(values)
    n=len(ptr)-1
    for i in range(n):
        begin=ptr[i]; end=ptr[i+1]-1
        for pos in range(begin,end):
            j=indices[pos]; s=values[pos]; ia=begin; jb=ptr[j]; je=ptr[j+1]-1
            while ia<pos and jb<je:
                ka=indices[ia]; kb=indices[jb]
                if ka==kb:s-=out[ia]*out[jb];ia+=1;jb+=1
                elif ka<kb:ia+=1
                else:jb+=1
            out[pos]=s/out[je]
        p=values[end]+shift
        for pos in range(begin,end):p-=out[pos]*out[pos]
        if not np.isfinite(p) or p<=min_pivot:return out,i,p
        out[end]=np.sqrt(p)
    return out,-1,1.

@njit(cache=True)
def lower_solve(data, indices, ptr,b):
    x=b.copy()
    for i in range(len(x)):
        end=ptr[i+1]-1
        for j in range(ptr[i],end):x[i]-=data[j]*x[indices[j]]
        x[i]/=data[end]
    return x

@njit(cache=True)
def upper_solve(data, indices,ptr,b):
    x=b.copy()
    for i in range(len(x)-1,-1,-1):
        begin=ptr[i]
        for j in range(begin+1,ptr[i+1]):x[i]-=data[j]*x[indices[j]]
        x[i]/=data[begin]
    return x


def build_star(sys, max_gram_nnz=20_000_000):
    t=perf_counter();g=sys.groups;H=sys.H;w=sys.w
    aa=anchors(g,w,len(sys.delta))
    B=H-H[aa[g]];B.eliminate_zeros()
    B=B.multiply(np.sqrt(w)[:,None]).tocsr()
    T=(B.T@B).tocsc();T.sum_duplicates();T.eliminate_zeros()
    if T.nnz>max_gram_nnz:raise ResourceGuard('star Gram nnz budget exceeded')
    d=T.diagonal()
    if np.any(d<=0):raise ValueError('unidentified star column')
    scale=1/np.sqrt(d)
    Ts=(sp.diags(scale)@T@sp.diags(scale)).tocsc()
    return Ts,scale,dict(star_nnz=int(B.nnz),gram_nnz=int(T.nnz),dimension=T.shape[0],assembly_seconds=perf_counter()-t)

class ICRight:
    def __init__(self,T,scale,ordering='rcm',max_shift=1.):
        t=perf_counter();n=T.shape[0];self.scale=scale
        self.order=reverse_cuthill_mckee(T,symmetric_mode=True) if ordering=='rcm' else np.arange(n)
        self.inv=np.argsort(self.order)
        K=sp.tril(T[self.order,:][:,self.order],format='csr');K.sort_indices()
        self.trials=[];failed=0
        for shift in [0.,1e-6,1e-5,1e-4,1e-3,.01,.1,1.]:
            if shift>max_shift:break
            data,failed,pivot=ic0_factor(K.data,K.indices,K.indptr,shift)
            self.trials.append(dict(shift=shift,failed_row=int(failed),pivot=float(pivot)))
            if failed<0:break
        if failed>=0:raise ValueError('IC0 failed within bounded shift budget')
        self.L=sp.csr_matrix((data,K.indices,K.indptr),shape=K.shape)
        self.U=self.L.T.tocsr();self.U.sort_indices()
        self.stats=dict(kind='ic0',ordering=ordering,shift=shift,trials=self.trials,
                        factor_nnz=int(self.L.nnz+self.U.nnz),factor_seconds=perf_counter()-t)
    def forward(self,x):
        y=upper_solve(self.U.data,self.U.indices,self.U.indptr,x)
        return self.scale*y[self.inv]
    def adjoint(self,x):
        y=(self.scale*x)[self.order]
        return lower_solve(self.L.data,self.L.indices,self.L.indptr,y)

class ILURight:
    """ILUT upper factor used as a right factor, NOT a symmetric ILU inverse.
    R = diag(scale) P_c U^-1 sqrt(abs(diag U)); exact transpose implemented.
    No assertion that U=D L^T and no exact-Cholesky spectrum claim.
    """
    def __init__(self,T,scale,drop_tol=1e-3,fill_factor=2.,max_nnz=5_000_000):
        t=perf_counter();self.scale=scale
        lu=spilu(T,drop_tol=drop_tol,fill_factor=fill_factor,
                 drop_rule='basic,area',permc_spec='MMD_AT_PLUS_A',
                 diag_pivot_thresh=0,options={'SymmetricMode':True})
        self.U=lu.U.tocsr();self.U.sort_indices();self.L=self.U.T.tocsr();self.L.sort_indices()
        if self.U.nnz+self.L.nnz>max_nnz:raise ResourceGuard('ILU right factor storage budget exceeded')
        self.p=lu.perm_c;self.ip=np.argsort(self.p);d=self.U.diagonal()
        if np.any(~np.isfinite(d)) or np.any(np.abs(d)<1e-15):raise ValueError('ILU pivot is non-finite or numerically zero')
        self.ds=np.sqrt(np.abs(d))
        self.stats=dict(kind='ilu_right',drop_tol=drop_tol,fill_factor=fill_factor,
                factor_nnz=int(self.U.nnz+self.L.nnz),ilu_both_nnz=int(lu.U.nnz+lu.L.nnz),
                negative_pivots=int(np.sum(d<0)),factor_seconds=perf_counter()-t)
    def forward(self,x):
        z=upper_solve(self.U.data,self.U.indices,self.U.indptr,self.ds*x)
        return self.scale*z[self.p]
    def adjoint(self,x):
        z=lower_solve(self.L.data,self.L.indices,self.L.indptr,(self.scale*x)[self.ip])
        return self.ds*z


def solve_rect(sys,Y,root,tol=1e-12,maxiter=1000):
    sw=np.sqrt(sys.w);g=sys.groups;w=sys.w;delta=sys.delta;H=sys.H
    def mv(x):return sw*sys.center(H@root.forward(x))
    def rmv(z):
        v=sw*z
        v-=w*(np.bincount(g,weights=v,minlength=len(delta))/delta)[g]
        return root.adjoint(np.asarray(H.T@v).ravel())
    A=LinearOperator((H.shape[0],H.shape[1]+getattr(root,"k",0)),matvec=mv,rmatvec=rmv,dtype=float)
    t=perf_counter();rs=[];info=[]
    for j in range(Y.shape[1]):
        ts=perf_counter()
        b=sw*sys.center(Y[:,j]);v=lsmr(A,b,atol=tol,btol=tol,conlim=1e14,maxiter=maxiter)
        rs.append(sys.center(Y[:,j]-H@root.forward(v[0])))
        info.append(dict(iterations=int(v[2]),status=int(v[1]),seconds=perf_counter()-ts))
    return np.column_stack(rs),dict(solve_seconds=perf_counter()-t,rhs=info)


def warmup():
    T=sp.csc_matrix([[2.,-.5],[-.5,2.]])
    a=ICRight(T,np.ones(2));a.forward(np.ones(2));a.adjoint(np.ones(2))
    b=ILURight(T,np.ones(2));b.forward(np.ones(2));b.adjoint(np.ones(2))

class ComponentRight:
    """Dense solves only inside small strong-edge components of normalized T.
    The threshold selects a preconditioner partition, never filters target rows.
    Refuses large components; bounded block size is a hard structural budget.
    """
    def __init__(self,T,scale,tau=1e-3,max_block=256,eig_cut=0.005,max_coarse=256):
        from scipy.sparse.csgraph import connected_components
        import scipy.linalg as la
        t=perf_counter();self.scale=scale
        graph=T.copy();graph.data=(np.abs(graph.data)>tau).astype(float);graph.eliminate_zeros()
        nc,labels=connected_components(graph,directed=False)
        sizes=np.bincount(labels)
        if sizes.max()>max_block:raise ResourceGuard(f'largest strong component {sizes.max()} > block budget {max_block}')
        self.order=np.argsort(labels,kind='stable');self.inv=np.argsort(self.order)
        blocks=[];zi=[];zj=[];zv=[];ne=0;local_eig=[];start=0
        for j,sz in enumerate(sizes):
            ids=self.order[start:start+sz];block=T[ids,:][:,ids].toarray();block=(block+block.T)/2
            L=la.cholesky(block,lower=True,check_finite=False);blocks.append(sp.csr_matrix(L))
            if eig_cut>0:
                vals,vec=la.eigh(block,check_finite=False)
                take=vals < eig_cut*vals[-1]
                keep=vec[:,take];nkeep=keep.shape[1]
                local_eig.append(dict(size=int(sz),minimum=float(vals[0]),maximum=float(vals[-1]),selected=nkeep))
                if ne+nkeep>max_coarse:raise ResourceGuard('coarse dimension budget exceeded')
                if nkeep:
                    zi.extend(np.repeat(ids,nkeep));zj.extend(np.tile(np.arange(ne,ne+nkeep),sz));zv.extend((scale[ids,None]*keep).ravel());ne+=nkeep
            start+=sz
        self.L=sp.block_diag(blocks,format='csr');self.L.eliminate_zeros();self.L.sort_indices()
        self.U=self.L.T.tocsr();self.U.sort_indices()
        self.Z=sp.coo_matrix((zv,(zi,zj)),shape=(T.shape[0],ne)).tocsr()
        self.stats=dict(kind='strong_components',threshold=tau,components=int(nc),max_block=int(sizes.max()),
                        factor_nnz=int(self.L.nnz+self.U.nnz),coarse_candidates=ne,
                        factor_seconds=perf_counter()-t,local_eigenvalues=local_eig)
    def forward(self,x):
        z=upper_solve(self.U.data,self.U.indices,self.U.indptr,x)
        return self.scale*z[self.inv]
    def adjoint(self,x):
        return lower_solve(self.L.data,self.L.indices,self.L.indptr,(self.scale*x)[self.order])

class CoarseRight:
    """Rectangular right factor of additive/balanced two-level inverse.

    Rectangular redundancy is allowed: range(R)=R^m. LSMR solves the ORIGINAL
    rectangular profiled least squares. A correct adjoint is essential.
    Z is discovered from the matrix; no generator community labels are passed.
    """
    def __init__(self,sys,base,balanced=True):
        import scipy.linalg as la
        t=perf_counter();self.base=base;self.balanced=balanced;Z=base.Z
        self.Z=Z;self.m,self.k=Z.shape
        if not self.k:raise ValueError('no coarse vectors selected')
        # Construct exact coarse energy from centered observation-space columns,
        # sparse throughout. Avoid subtracting near equal coarse Gram matrices.
        HZ=(sys.H@Z).tocsr()
        g=sys.groups;n=len(g);k0=len(sys.delta)
        F0=sp.csr_matrix((np.ones(n),g,np.arange(n+1,dtype=np.int32)),shape=(n,k0))
        means=(sp.diags(1/sys.delta)@(F0.T@HZ.multiply(sys.w[:,None]))).tocsr()
        C=HZ-means[g];C.eliminate_zeros()
        E=(C.T@C.multiply(sys.w[:,None])).toarray();E=(E+E.T)/2
        self.Echol=la.cholesky(E,lower=True,check_finite=False)
        self.SZ=(sys.H.T@C.multiply(sys.w[:,None])).toarray() if balanced else None
        self.stats=dict(coarse_dimension=self.k,balanced=balanced,coarse_seconds=perf_counter()-t,
                        coarse_centered_nnz=int(C.nnz),coarse_energy_min=float(la.eigvalsh(E)[0]))
    def forward(self,x):
        import scipy.linalg as la
        fine=self.base.forward(x[self.k:])
        coarse=la.solve_triangular(self.Echol.T,x[:self.k],lower=False,check_finite=False)
        if self.balanced:
            coarse-=la.cho_solve((self.Echol,True),self.SZ.T@fine,check_finite=False)
        return fine+self.Z@coarse
    def adjoint(self,y):
        import scipy.linalg as la
        zy=np.asarray(self.Z.T@y).ravel()
        coarse=la.solve_triangular(self.Echol,zy,lower=True,check_finite=False)
        if self.balanced:y=y-self.SZ@la.cho_solve((self.Echol,True),zy,check_finite=False)
        return np.r_[coarse,self.base.adjoint(y)]

@njit(cache=True)
def support_partition(codes,weights,cutoff,sizes):
    """Union-find on retained-weight hyperedges. Only a routing heuristic."""
    offsets=np.empty(len(sizes)+1,np.int64);offsets[0]=0
    for j in range(len(sizes)):offsets[j+1]=offsets[j]+sizes[j]
    parent=np.arange(offsets[-1]);rank=np.zeros(offsets[-1],np.int32)
    for i in range(len(weights)):
        if weights[i]<cutoff:continue
        a=codes[i,0]
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        for q in range(1,codes.shape[1]):
            b=offsets[q]+codes[i,q]
            while parent[b]!=b:parent[b]=parent[parent[b]];b=parent[b]
            if a!=b:
                if rank[a]<rank[b]:a,b=b,a
                parent[b]=a
                if rank[a]==rank[b]:rank[a]+=1
    counts=np.zeros(offsets[-1],np.int32)
    for v in range(offsets[1],offsets[-1]):
        a=v
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        counts[a]+=1
    return counts[counts>0]


def preflight(codes,w,max_block=256,ratio_cutoff=1e-3):
    """Cheap, conservative eligibility test; never deletes data from the target.
    Pure unit weights bypass this specialized weighted-community path.
    This is an empirical routing heuristic, not a theorem on solver difficulty.
    """
    t=perf_counter()
    if np.min(w)==np.max(w):
        return {'eligible':False,'reason':'uniform_weight_bypass','seconds':perf_counter()-t}
    sizes=np.array([int(codes[:,j].max())+1 for j in range(codes.shape[1])],np.int64)
    cutoff=ratio_cutoff*float(np.median(w))
    counts=support_partition(codes,w,cutoff,sizes)
    eligible=counts.max(initial=0)<=max_block and len(counts)>1
    return {'eligible':bool(eligible),'reason':'small_strong_support_components' if eligible else 'large_strong_support_component',
            'cutoff':cutoff,'components':len(counts),'max_other_levels':int(counts.max(initial=0)),
            'seconds':perf_counter()-t}
