"""Sparse million-row research backend. Not a production econhdfe/within release.

The target is always the original positive-weight categorical FE projection.
Approximation is confined to star-Schur preconditioning. Sparse LU is guarded;
no ridge is added to the target, and no large dense Gram matrix is constructed.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, lsmr, cg, splu
from numba import njit

@njit(cache=True)
def anchors(groups, weights, k):
    out=np.full(k,-1,np.int64)
    for i in range(len(groups)):
        g=groups[i]
        if out[g]<0 or weights[i]>weights[out[g]]: out[g]=i
    return out


def dummy(codes, sizes, drop_first=False):
    """Build CSR in O(N*Q); remove one reference in each selected FE block."""
    n,q=codes.shape
    if drop_first:
        offsets=np.r_[0,np.cumsum(np.array(sizes)-1)]
        ii=np.broadcast_to(np.arange(n,dtype=np.int32)[:,None],(n,q)).ravel()
        cc=(codes+offsets[:-1]-1).ravel()
        mask=(codes.ravel()!=0)
        out=sp.coo_matrix((np.ones(mask.sum()),(ii[mask],cc[mask])),shape=(n,int(offsets[-1]))).tocsr()
    else:
        offsets=np.r_[0,np.cumsum(sizes)]
        indices=(codes+offsets[:-1]).ravel().astype(np.int32)
        indptr=np.arange(0,n*q+1,q,dtype=np.int32)
        out=sp.csr_matrix((np.ones(n*q),indices,indptr),shape=(n,int(offsets[-1])))
    return out


def memory_bytes(a):
    return a.data.nbytes+a.indices.nbytes+a.indptr.nbytes


class ResourceGuard(RuntimeError): pass

@dataclass
class ProfiledSystem:
    groups: np.ndarray
    w: np.ndarray
    sizes: list
    H: sp.csr_matrix
    delta: np.ndarray
    diagonal: np.ndarray
    setup_seconds: float

    @classmethod
    def build(cls,codes,w):
        t=perf_counter()
        if codes.ndim!=2 or not np.issubdtype(codes.dtype,np.integer) or np.any(codes<0):
            raise ValueError('nonnegative dense-coded categorical IDs required')
        if len(w)!=len(codes) or np.any(~np.isfinite(w)) or np.any(w<=0):
            raise ValueError('weights must be finite and strictly positive')
        sizes=[int(codes[:,j].max())+1 for j in range(codes.shape[1])]
        g=codes[:,0].astype(np.int32,copy=False)
        delta=np.bincount(g,weights=w,minlength=sizes[0])
        if np.any(delta==0):raise ValueError('unobserved first-FE level')
        H,references=profiled_dummy(codes,sizes)
        # Exact profiled column norm: diag(H'WH)-diag(C' Delta^-1 C).
        first=sp.csr_matrix((np.ones(len(g)),g,np.arange(len(g)+1,dtype=np.int32)),shape=(len(g),sizes[0]))
        cross=(first.T @ H.multiply(w[:,None])).tocsr()
        hd=np.asarray(H.T@w).ravel()
        corr=np.asarray(cross.multiply(cross).T@(1/delta)).ravel()
        diagonal=hd-corr
        # Do not silently delete nearly absorbed columns or manufacture a ridge.
        if np.any(diagonal<=0):
            raise ValueError('zero/negative profiled variance: extra nesting or cancellation; canonicalization required')
        obj=cls(g,w,sizes,H,delta,diagonal,perf_counter()-t)
        obj.reference_counts=references
        return obj

    def center(self,z):
        if z.ndim==1:
            return z-(np.bincount(self.groups,weights=self.w*z,minlength=len(self.delta))/self.delta)[self.groups]
        return np.column_stack([self.center(z[:,j]) for j in range(z.shape[1])])

    def matvec(self,x):
        return np.asarray(self.H.T @ (self.w*self.center(self.H@x))).ravel()

    def solve(self,Y,pre=None,rtol=1e-11,maxiter=2000):
        t=perf_counter();cols=[];infos=[]
        m=self.H.shape[1]
        S=LinearOperator((m,m),matvec=self.matvec,dtype=float)
        M=LinearOperator((m,m),matvec=(lambda z:z/self.diagonal) if pre is None else pre.solve,dtype=float)
        for j in range(Y.shape[1]):
            rhs=np.asarray(self.H.T@(self.w*self.center(Y[:,j]))).ravel()
            niter=[0]
            def cb(x):niter[0]+=1
            ts=perf_counter()
            x,info=cg(S,rhs,M=M,rtol=rtol,atol=0,maxiter=maxiter,callback=cb)
            # Reconstruct from the original y, not approximate/preconditioner rows.
            r=self.center(Y[:,j]-self.H@x)
            cols.append(r)
            infos.append(dict(status=int(info),iterations=niter[0],seconds=perf_counter()-ts))
        return np.column_stack(cols),dict(solve_seconds=perf_counter()-t,rhs=infos)


@dataclass
class StarInverse:
    lu: object
    scale: np.ndarray
    V: np.ndarray|None
    small_factor: object|None
    stats: dict

    def base(self,z):
        return self.scale*self.lu.solve(self.scale*z)

    def solve(self,z):
        a=self.base(z)
        if self.V is not None:
            a=a+self.V@la.cho_solve(self.small_factor,self.V.T@z,check_finite=False)
        return a

    @classmethod
    def build(cls,sys,*,rho_cap=None,max_corrected=32,max_dimension=6500,max_gram_nnz=18_000_000):
        t=perf_counter();H=sys.H;w=sys.w;g=sys.groups;m=H.shape[1]
        aa=anchors(g,w,len(sys.delta));rho=sys.delta/w[aa]
        selected=np.where(rho>rho_cap*(1+1e-12))[0] if rho_cap is not None else np.array([],int)
        stats=dict(schur_dimension=m,raw_rho_max=float(rho.max()),
                   rho_quantiles=np.quantile(rho,[.5,.9,.99,1]).tolist(),
                   correction_requested=int(len(selected)),correction_applied=0,
                   rho_cap=rho_cap)
        if len(selected)>max_corrected:
            raise ResourceGuard(f'{len(selected)} corrections exceeds rank budget {max_corrected}; dense Woodbury deliberately refused')
        if m>max_dimension:
            raise ResourceGuard(f'Schur dimension {m} exceeds sparse direct budget {max_dimension}; use matrix-free fallback')
        # Sparse star rows contain at most 2*(Q-1) entries.
        B=H-H[aa[g]];B.eliminate_zeros()
        stats['star_nnz']=int(B.nnz)
        B=B.multiply(np.sqrt(w)[:,None]).tocsr()
        T=(B.T@B).tocsc();T.sum_duplicates();T.eliminate_zeros();del B
        stats['gram_nnz']=int(T.nnz);stats['gram_storage_bytes']=memory_bytes(T)
        if T.nnz>max_gram_nnz:raise ResourceGuard('sparse Gram nnz exceeds memory budget')
        scale=1/np.sqrt(T.diagonal())
        if np.any(~np.isfinite(scale)):raise ValueError('star has zero diagonal')
        Ts=sp.diags(scale)@T@sp.diags(scale);Ts=Ts.tocsc();del T
        stats['assembly_seconds']=perf_counter()-t
        tf=perf_counter()
        # LU is used only as inverse preconditioner of an SPD gauged matrix.
        lu=splu(Ts,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0,
                options={'SymmetricMode':True})
        stats['factor_seconds']=perf_counter()-tf
        stats['factor_nnz']=int(lu.L.nnz+lu.U.nnz)
        stats['factor_storage_bytes']=memory_bytes(lu.L)+memory_bytes(lu.U)
        obj=cls(lu,scale,None,None,stats)
        tu=perf_counter()
        if len(selected):
            U=np.empty((m,len(selected)))
            for j,v in enumerate(selected):
                mask=g==v
                mean=np.asarray(H[mask].T@w[mask]).ravel()/sys.delta[v]
                anchor=H[aa[v]].toarray().ravel()
                U[:,j]=np.sqrt(sys.delta[v])*(anchor-mean)
            V=scale[:,None]*lu.solve(scale[:,None]*U)
            K=np.eye(len(selected))-U.T@V;K=(K+K.T)/2
            ev=la.eigvalsh(K)
            stats['woodbury_min_eigenvalue']=float(ev[0])
            if ev[0]<=1e-12:raise ValueError('Woodbury downdate unresolved numerically; refusing unstable inverse')
            obj.V=V;obj.small_factor=la.cho_factor(K,lower=True);obj.correction_U=U
            stats['correction_applied']=len(selected)
        stats['correction_seconds']=perf_counter()-tu
        stats['preconditioner_seconds']=perf_counter()-t
        stats['effective_rho_bound']=float(max(1,np.max(np.delete(rho,selected),initial=1)))
        return obj


def full_lsmr(codes,w,Y,rtol=1e-11,maxiter=2000):
    t=perf_counter()
    sizes=[int(codes[:,j].max())+1 for j in range(codes.shape[1])]
    D=dummy(codes,sizes,False);sw=np.sqrt(w)
    scale=1/np.sqrt(np.asarray(D.T@w).ravel())
    A=LinearOperator(D.shape,matvec=lambda x:sw*(D@(scale*x)),
        rmatvec=lambda z:scale*(D.T@(sw*z)),dtype=float)
    setup=perf_counter()-t;t=perf_counter();cols=[];infos=[]
    for j in range(Y.shape[1]):
        ts=perf_counter();v=lsmr(A,sw*Y[:,j],atol=rtol,btol=rtol,conlim=1e14,maxiter=maxiter)
        cols.append(Y[:,j]-D@(scale*v[0]));infos.append(dict(status=int(v[1]),iterations=int(v[2]),seconds=perf_counter()-ts))
    return np.column_stack(cols),dict(setup_seconds=setup,solve_seconds=perf_counter()-t,rhs=infos)


def audit(codes,w,Y,R,truth=None):
    sw=np.sqrt(w);norm=np.maximum(np.linalg.norm(sw[:,None]*Y,axis=0),1e-300)
    ng=np.zeros(Y.shape[1])
    for j in range(codes.shape[1]):
        ids=codes[:,j];de=np.bincount(ids,weights=w)
        for k in range(Y.shape[1]):
            gr=np.bincount(ids,weights=w*R[:,k])/np.sqrt(de)
            ng[k]+=float(gr@gr)
    out={'relative_scaled_normal_residual':(np.sqrt(ng)/norm).tolist(),
         'weighted_residual_norm':np.linalg.norm(sw[:,None]*R,axis=0).tolist()}
    if truth is not None:
        out['oracle_relative_projection_error']=float(np.linalg.norm(sw*(R[:,-1]-truth))/norm[-1])
    # One-regressor FWL; CR0, cluster in first FE. No small-sample correction claim.
    x=R[:,1];y=R[:,0];xx=np.dot(w*x,x);beta=float(np.dot(w*x,y)/xx)
    u=y-beta*x;score=np.bincount(codes[:,0],weights=w*x*u)
    out['fwl_beta']=beta;out['cluster_cr0_se']=float(np.sqrt(np.dot(score,score))/xx)
    return out

@njit(cache=True)
def symmetric_sweep(v, ids, sw, deltas):
    """Symmetric Kaczmarz in whitened observation space, one reusable buffer."""
    x=v.copy();q,n=ids.shape
    for jj in range(2*q-1):
        j=jj if jj<q else 2*q-2-jj
        delta=deltas[j];sums=np.zeros(len(delta))
        for i in range(n):sums[ids[j,i]]+=sw[i]*x[i]
        for h in range(len(delta)):sums[h]/=delta[h]
        for i in range(n):x[i]-=sw[i]*sums[ids[j,i]]
    return x


def map_cg(codes,w,Y,rtol=1e-11,maxiter=1600):
    """Standard symmetric MAP accelerated with CG; not Stata or native within."""
    from numba.typed import List
    t=perf_counter();ids=np.ascontiguousarray(codes.T);sw=np.sqrt(w)
    ds=List()
    for j in range(ids.shape[0]):ds.append(np.bincount(ids[j],weights=w))
    def op(v):return v-symmetric_sweep(v,ids,sw,ds)
    L=LinearOperator((len(w),len(w)),matvec=op,dtype=float)
    setup=perf_counter()-t;t=perf_counter();cols=[];infos=[]
    for j in range(Y.shape[1]):
        yy=sw*Y[:,j];b=op(yy);it=[0]
        def cb(_):it[0]+=1
        ts=perf_counter();f,status=cg(L,b,rtol=rtol,atol=0,maxiter=maxiter,callback=cb)
        cols.append((yy-f)/sw);infos.append(dict(status=int(status),iterations=it[0],seconds=perf_counter()-ts))
    return np.column_stack(cols),dict(setup_seconds=setup,solve_seconds=perf_counter()-t,rhs=infos)

@njit(cache=True)
def triangular_unit_lower(data, indices, indptr, b):
    x=b.copy()
    for i in range(len(x)):
        for k in range(indptr[i],indptr[i+1]):
            j=indices[k]
            if j<i:x[i]-=data[k]*x[j]
    return x

@njit(cache=True)
def triangular_unit_upper(data, indices, indptr, b):
    x=b.copy()
    for i in range(len(x)-1,-1,-1):
        for k in range(indptr[i],indptr[i+1]):
            j=indices[k]
            if j>i:x[i]-=data[k]*x[j]
    return x

class StarRightFactor:
    """Obtain a sparse inverse square-root factor from symmetric SuperLU LDL'.

    Check the permutation and LDL identity before using the factors. Low-rank
    downdates are applied via a small Gram eigendecomposition, never dense m*m.
    """
    def __init__(self,pre):
        self.scale=pre.scale;lu=pre.lu
        if not np.array_equal(lu.perm_r,lu.perm_c):raise ValueError('non-symmetric LU pivoting: no LDL right factor')
        self.p=lu.perm_c;self.ip=np.argsort(self.p)
        L=lu.L.tocsr();self.L=L;self.Lt=L.T.tocsr()
        d=lu.U.diagonal()
        if np.any(d<=0):raise ValueError('non-positive LDL pivot')
        err=lu.U-sp.diags(d)@lu.L.T
        relative=np.linalg.norm(err.data)/max(np.linalg.norm(lu.U.data),1e-300)
        if relative>1e-9:raise ValueError(f'LDL consistency error {relative}')
        self.isqrt=1/np.sqrt(d);self.F=None;self.C=None
        pre.stats['ldl_relative_identity_error']=float(relative)
        U=getattr(pre,'correction_U',None)
        if U is not None:
            F=np.column_stack([self.adjoint_base(U[:,j]) for j in range(U.shape[1])])
            vals,V=la.eigh((F.T@F+(F.T@F).T)/2)
            if np.max(vals)>=1-1e-12:raise ValueError('square-root downdate too close to singular')
            rt=np.sqrt(1-vals);self.F=F@V;self.C=1/(rt*(1+rt))

    def forward_base(self,x):
        z=triangular_unit_upper(self.Lt.data,self.Lt.indices,self.Lt.indptr,self.isqrt*x)
        return self.scale*z[self.p]

    def adjoint_base(self,x):
        z=triangular_unit_lower(self.L.data,self.L.indices,self.L.indptr,(self.scale*x)[self.ip])
        return self.isqrt*z

    def mix(self,x):
        return x if self.F is None else x+self.F@(self.C*(self.F.T@x))

    def forward(self,x):return self.forward_base(self.mix(x))
    def adjoint(self,x):return self.mix(self.adjoint_base(x))


def profiled_lsmr(sys,Y,pre=None,rtol=1e-12,maxiter=1600):
    t=perf_counter();sw=np.sqrt(sys.w)
    if pre is None:
        s=1/np.sqrt(sys.diagonal);forward=lambda x:s*x;adjoint=lambda x:s*x
    else:
        root=StarRightFactor(pre);forward=root.forward;adjoint=root.adjoint
    def mv(x):return sw*sys.center(sys.H@forward(x))
    def rmv(z):
        v=sw*z
        v=v-sys.w*(np.bincount(sys.groups,weights=v,minlength=len(sys.delta))/sys.delta)[sys.groups]
        return adjoint(np.asarray(sys.H.T@v).ravel())
    A=LinearOperator(sys.H.shape,matvec=mv,rmatvec=rmv,dtype=float)
    # Test exact transpose pair on a deterministic vector pair without consuming RNG.
    # The full numeric identity is covered by unit tests, not run on each million-row fit.
    extra=perf_counter()-t;t=perf_counter();cols=[];infos=[]
    for j in range(Y.shape[1]):
        ts=perf_counter();b=sw*sys.center(Y[:,j])
        res=lsmr(A,b,atol=rtol,btol=rtol,conlim=1e14,maxiter=maxiter)
        cols.append(sys.center(Y[:,j]-sys.H@forward(res[0])))
        infos.append(dict(status=int(res[1]),iterations=int(res[2]),seconds=perf_counter()-ts))
    return np.column_stack(cols),dict(right_factor_seconds=extra,solve_seconds=perf_counter()-t,rhs=infos)

@njit(cache=True)
def pair_reference_mask(first,other,k0,k1):
    """One second-factor reference per bipartite component; exact span reduction."""
    parent=np.arange(k0+k1);size=np.ones(k0+k1,np.int64)
    for i in range(len(first)):
        a=first[i];b=k0+other[i]
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        while parent[b]!=b:
            parent[b]=parent[parent[b]];b=parent[b]
        if a!=b:
            if size[a]<size[b]:a,b=b,a
            parent[b]=a;size[a]+=size[b]
    drop=np.zeros(k1,np.bool_);seen=np.zeros(k0+k1,np.bool_)
    for j in range(k1):
        a=k0+j
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        if not seen[a]:drop[j]=True;seen[a]=True
    return drop


def profiled_dummy(codes,sizes):
    n,q=codes.shape;cols=[];counts=[];offset=0
    for j in range(1,q):
        mask=pair_reference_mask(codes[:,0],codes[:,j],sizes[0],sizes[j])
        mapping=np.arange(sizes[j])-np.cumsum(mask);mapping[mask]=-1
        col=mapping[codes[:,j]];good=col>=0;col[good]+=offset
        cols.append(col);counts.append(int(mask.sum()));offset+=sizes[j]-mask.sum()
    cc=np.column_stack(cols).ravel();ii=np.repeat(np.arange(n,dtype=np.int32),q-1)
    ok=cc>=0
    H=sp.coo_matrix((np.ones(ok.sum()),(ii[ok],cc[ok])),shape=(n,int(offset))).tocsr()
    return H,counts
