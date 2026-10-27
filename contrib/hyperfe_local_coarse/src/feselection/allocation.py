"""Pre-allocation accounting for candidate-owned arrays, not a process RSS cap.

Native within / BLAS allocator internals are explicitly outside this contract.
Counts are conservative structural bounds; symbolic passes use O(dimension)
workspace and abort before constructing a numerical sparse product.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
from numba import njit
from scalable_hyperfe import ProfiledSystem, anchors, memory_bytes

class CandidateRejected(RuntimeError):
    def __init__(self, reason: str, **details):
        self.reason, self.details = reason, details
        super().__init__(reason)

@dataclass(frozen=True)
class ResourceBudget:
    workspace_mib: float = 1024.0
    max_reduced_dimension: int = 30000
    max_gram_nnz: int = 5_000_000
    max_symbolic_work: int = 120_000_000
    max_block: int = 256
    max_coarse: int = 256
    def __post_init__(self):
        if not np.isfinite(self.workspace_mib) or self.workspace_mib <= 0:
            raise ValueError('workspace_mib must be finite and positive')
        for name in ('max_reduced_dimension','max_gram_nnz','max_symbolic_work','max_block','max_coarse'):
            v = getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v < 1:
                raise ValueError(f'{name} must be a positive integer')
    @property
    def limit(self): return int(self.workspace_mib * 2**20)

@dataclass
class AllocationLedger:
    budget: ResourceBudget
    events: list = field(default_factory=list)
    def admit(self, stage: str, estimated_live_bytes: int, **details):
        # This is a per-stage live bound, not a sum of dead allocations.
        event = dict(stage=stage, estimated_live_bytes=int(estimated_live_bytes),
                     limit_bytes=self.budget.limit, accepted=bool(estimated_live_bytes<=self.budget.limit), **details)
        self.events.append(event)
        if not event['accepted']:
            raise CandidateRejected('candidate_workspace_budget', **event)

@njit(cache=True)
def product_row_counts(ap, aj, bp, bj, ncols, nnz_limit, work_limit):
    """Exact symbolic nnz(A@B) per row, ignoring cancellation, no value allocation."""
    marker=np.full(ncols,-1,np.int64); counts=np.zeros(len(ap)-1,np.int64)
    total=0; work=0
    for i in range(len(counts)):
        ct=0
        for p in range(ap[i],ap[i+1]):
            j=aj[p]
            for s in range(bp[j],bp[j+1]):
                work+=1
                if work>work_limit:return counts,total,work,2
                k=bj[s]
                if marker[k]!=i:marker[k]=i;ct+=1
        counts[i]=ct;total+=ct
        if total>nnz_limit:return counts,total,work,1
    return counts,total,work,0

@njit(cache=True)
def group_union_counts(g, p, j, ng, k):
    """Exact support union size of HZ rows inside each eliminated FE group."""
    count=np.zeros(ng,np.int64)
    for x in g:count[x]+=1
    gp=np.zeros(ng+1,np.int64)
    for x in range(ng):gp[x+1]=gp[x]+count[x]
    rows=np.empty(len(g),np.int32);pos=gp[:-1].copy()
    for i in range(len(g)):rows[pos[g[i]]]=i;pos[g[i]]+=1
    marker=np.full(k,-1,np.int64);out=np.zeros(ng,np.int64)
    for v in range(ng):
        for r in range(gp[v],gp[v+1]):
            i=rows[r]
            for z in range(p[i],p[i+1]):
                c=j[z]
                if marker[c]!=v:marker[c]=v;out[v]+=1
    return out


def sparse_bytes(nnz: int, rows: int) -> int:
    # int64 indexing upper bound, even though this prototype often uses int32.
    return int(nnz)*16 + (int(rows)+1)*8


def inspect(codes: np.ndarray, w: np.ndarray, *, with_masses=False):
    """O(NQ) summaries, never use sampled connectivity/rank as a certificate."""
    sizes=[]; summaries=[]; masses=[]
    for q in range(codes.shape[1]):
        g=codes[:,q]; m=int(g.max())+1
        counts=np.bincount(g,minlength=m)
        if np.any(counts==0):raise ValueError('codes must be dense observed integers in every FE')
        weight=np.bincount(g,weights=w,minlength=m)
        if np.any(weight<=0) or not np.all(np.isfinite(weight)):
            raise ValueError('invalid group total weight')
        sizes.append(m)
        if with_masses: masses.append(weight)
        summaries.append(dict(levels=m,min_rows=int(counts.min()),max_rows=int(counts.max()),
            singleton_levels=int(np.sum(counts==1)),rows_quantiles=np.quantile(counts,[.5,.9,.99]).tolist(),
            weight_min=float(weight.min()),weight_max=float(weight.max())))
    profile = dict(n=len(codes),q=codes.shape[1],levels=sizes,groups=summaries,
                equal_weights=bool(np.all(w==w[0])),weight_ratio=float(w.max()/w.min()))
    if with_masses:
        return profile, np.r_[0, np.cumsum(sizes)].astype(np.int64), np.concatenate(masses) if masses else np.empty(0)
    return profile


def elimination_candidates(profile: dict, budget: ResourceBudget, count: int=2):
    """Cheap dimension-first shortlist; not a fill-in oracle."""
    ks=profile['levels']; total=sum(ks); candidates=[]
    for q,k in enumerate(ks):
        m=total-k-(len(ks)-1)  # upper bound; additional references may shrink it
        candidates.append(dict(eliminate=q,reduced_upper=m,max_group=profile['groups'][q]['max_rows'],
                               admitted=m<=budget.max_reduced_dimension))
    candidates.sort(key=lambda d:(d['reduced_upper'],d['max_group'],d['eliminate']))
    return candidates[:count]


def prepare_system(codes,w,eliminate,ledger,retained_bytes=0):
    n,q=codes.shape; m=sum(int(codes[:,j].max())+1 for j in range(q) if j!=eliminate)
    if m-(q-1)>ledger.budget.max_reduced_dimension:
        raise CandidateRejected('reduced_dimension_budget',upper=m-(q-1),limit=ledger.budget.max_reduced_dimension)
    # Covers dense COO index temporaries, H/cross/F0 and weighted sparse copies.
    upper=retained_bytes+n*(128+96*(q-1))+m*96
    ledger.admit('profile_before_allocation',upper,remaining_columns_upper=m)
    order=[eliminate]+[j for j in range(q) if j!=eliminate]
    cc=codes if eliminate==0 else np.ascontiguousarray(codes[:,order])
    try:
        sys=ProfiledSystem.build(cc,w)
    except ValueError as e:
        if str(e).startswith('zero/negative profiled variance'):
            raise CandidateRejected('profiled_exact_or_numerical_dependency') from e
        raise
    return sys,order


def guarded_star(sys,ledger,retained_bytes=0):
    H,w,g=sys.H,sys.w,sys.groups;n,m=H.shape;l=max(1,len(sys.sizes)-1)
    ledger.admit('star_rows_before_allocation',retained_bytes+memory_bytes(H)+n*(80+96*l)+96*m)
    aa=anchors(g,w,len(sys.delta))
    B=(H-H[aa[g]]).tocsr();B.eliminate_zeros();B.sort_indices()
    # B and its transpose's CSR copy are admitted before creating the latter.
    ledger.admit('star_symbolic_before_allocation',retained_bytes+memory_bytes(H)+3*memory_bytes(B)+64*m+16*n)
    BT=B.T.tocsr()
    counts,total,work,status=product_row_counts(BT.indptr,BT.indices,B.indptr,B.indices,m,
                        ledger.budget.max_gram_nnz,ledger.budget.max_symbolic_work)
    if status:
        raise CandidateRejected('star_symbolic_work_budget' if status==2 else 'star_gram_nnz_budget',
                                symbolic_nnz=int(total),work=int(work))
    # product output, conversion, scaled matrix, candidate graph and allocator scratch.
    upper=retained_bytes+memory_bytes(H)+2*memory_bytes(B)+6*sparse_bytes(total,m)+96*m+24*n
    ledger.admit('star_gram_before_multiply',upper,symbolic_nnz=int(total),symbolic_work=int(work))
    B.data*=np.repeat(np.sqrt(w),np.diff(B.indptr))
    del BT
    T=(B.T@B).tocsc();T.sum_duplicates();T.eliminate_zeros()
    d=T.diagonal()
    if np.any(~np.isfinite(d)) or np.any(d<=0):raise CandidateRejected('unidentified_star_column')
    scale=1/np.sqrt(d)
    T=(sp.diags(scale)@T@sp.diags(scale)).tocsc();T.sort_indices()
    return T,scale,dict(star_nnz=int(B.nnz),gram_nnz=int(T.nnz),symbolic_nnz=int(total),symbolic_work=int(work))
