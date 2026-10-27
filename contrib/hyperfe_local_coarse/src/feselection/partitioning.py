"""Structure-only finite threshold ladder. No DGP names, labels, y or oracle."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from numba import njit
from .allocation import CandidateRejected, sparse_bytes

@njit(cache=True)
def _critical_threshold(i,j,s,m,cap):
    """Smallest graph cut threshold before a strong component exceeds cap.

    Edges arrive in decreasing strength. Returning the current strength means
    ALL tied edges are excluded by the strict > comparison, not a label-based
    arbitrary subset of a tie. Thus output partitions are relabelling invariant.
    """
    parent=np.arange(m);size=np.ones(m,np.int64)
    for z in range(len(s)):
        a=i[z];b=j[z]
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        while parent[b]!=b:parent[b]=parent[parent[b]];b=parent[b]
        if a==b:continue
        if size[a]+size[b]>cap:return s[z]
        if size[a]<size[b]:a,b=b,a
        parent[b]=a;size[a]+=size[b]
    return 0.


def candidates(T,ledger,max_candidates=3,retained_bytes=0):
    m=T.shape[0]
    ledger.admit('partition_before_sort',retained_bytes+8*sparse_bytes(T.nnz,m)+128*m)
    U=sp.triu(T,k=1,format='coo');s=np.abs(U.data);order=np.argsort(-s,kind='stable')
    i=U.row[order];j=U.col[order];s=s[order]
    caps=sorted({max(2,ledger.budget.max_block//4),max(2,ledger.budget.max_block//2),ledger.budget.max_block})
    thresholds=sorted({_critical_threshold(i,j,s,m,c) for c in caps})
    out=[];seen=set();total_energy=float(np.dot(s,s))
    for tau in thresholds:
        G=T.copy();G.data=(np.abs(G.data)>tau).astype(float);G.eliminate_zeros()
        nc,lab=connected_components(G,directed=False);sz=np.bincount(lab)
        # Label-free partition signature via membership equality is expensive;
        # exact labels are deterministic for the same matrix ordering here.
        key=lab.tobytes()
        if key in seen:continue
        seen.add(key)
        maxb=int(sz.max(initial=0));sq=int(np.sum(sz.astype(np.int64)**2));cube=int(np.sum(sz.astype(np.int64)**3))
        cut=float(np.dot(s[lab[i]!=lab[j]],s[lab[i]!=lab[j]])/total_energy) if total_energy>0 else 0.
        singleton=float(np.sum(sz==1)/max(1,m))
        # Transparent structural heuristic only; convergence is not certified.
        score=cut+0.05*cube/max(1,m*ledger.budget.max_block**2)+0.02*sq/max(1,m*ledger.budget.max_block)
        out.append(dict(tau=float(tau),components=int(nc),max_block=maxb,sum_block_squares=sq,
                        sum_block_cubes=cube,cut_energy_fraction=cut,singleton_dof_fraction=singleton,
                        structural_score=float(score),admitted=bool(maxb<=ledger.budget.max_block),labels=lab))
    out.sort(key=lambda d:d['structural_score'])
    return out[:max_candidates]
