"""Bounded complete-group scout, used ONLY to route a full-data projection.

This is not a spectral sparsifier, a rank certificate, or a substituted target.
A sampled low/zero mode can be spurious. Insufficient support or any proxy
numerical difficulty returns 'unknown', not a claim about full-data rank.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from time import perf_counter
import numpy as np
import scipy.linalg as la
from .allocation import (AllocationLedger, CandidateRejected, ResourceBudget,
                         prepare_system, guarded_star, elimination_candidates)
from .partitioning import candidates
from .numerical_screen import preview_partition

@dataclass(frozen=True)
class EarlyScreenPolicy:
    enabled: bool = True
    max_rows: int = 131072
    max_groups: int = 16384
    minimum_groups: int = 64
    minimum_level_coverage: float = .90
    workspace_mib: float = 128.
    seed: int = 1487
    warm_confirmation_max_iteration_ratio: float = 2.
    def __post_init__(self):
        if not isinstance(self.enabled, bool): raise ValueError('enabled must be bool')
        for key in ('max_rows','max_groups','minimum_groups','seed'):
            v=getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<1:
                raise ValueError(f'{key} must be a positive integer')
        if self.minimum_groups>self.max_groups: raise ValueError('minimum_groups exceeds max_groups')
        if not np.isfinite(self.minimum_level_coverage) or not 0<self.minimum_level_coverage<=1:
            raise ValueError('minimum_level_coverage must be in (0,1]')
        if not np.isfinite(self.workspace_mib) or self.workspace_mib<=0: raise ValueError('workspace_mib must be positive')
        if not np.isfinite(self.warm_confirmation_max_iteration_ratio) or self.warm_confirmation_max_iteration_ratio<1:
            raise ValueError('warm_confirmation_max_iteration_ratio must be >=1')


def complete_group_rows(codes, eliminate, levels, policy):
    """Uniform group order, complete groups, bounded output, not PPS sampling.

    Group labels affect deterministic sampling, never the fitted target. Every
    selected group has all its original rows. Stop before an oversized group;
    do not truncate it or preferentially skip it to select smaller groups.
    """
    g=codes[:,eliminate]
    counts=np.bincount(g,minlength=levels)
    rng=np.random.default_rng(policy.seed)
    perm=rng.permutation(levels)[:policy.max_groups]
    cumulative=np.cumsum(counts[perm],dtype=np.int64)
    selected=perm[cumulative<=policy.max_rows]
    pick=np.zeros(levels,dtype=bool); pick[selected]=True
    rows=np.flatnonzero(pick[g])
    return rows,int(len(selected))


def scout(codes,w,profile,ledger,*,policy,screen_policy,eig_cut,max_candidates=3,
          retained_bytes=0):
    """Return tri-state routing evidence before ANY full profile/star allocation."""
    t=perf_counter()
    info=dict(stage='early_group_scout',verdict='unknown',sampled_only=True,
              target_rows_unchanged=True,assurance='routing heuristic only; no rank or speed certificate')
    def finish(verdict,reason):
        info.update(verdict=verdict,reason=reason,seconds=perf_counter()-t)
        return info
    if not policy.enabled:return finish('unknown','scout_disabled')
    shortlist=elimination_candidates(profile,ledger.budget)
    selected=next((d for d in shortlist if d['admitted']),None)
    if selected is None:return finish('unknown','no_admitted_elimination')
    q=selected['eliminate'];n,Q=codes.shape;ng=profile['levels'][q]
    info['eliminate']=q
    if ng<policy.minimum_groups:return finish('unknown','too_few_groups_for_scout')
    if n<=policy.max_rows:return finish('unknown','small_input_no_scout')
    sample_n=min(n,policy.max_rows)
    workspace=min(policy.workspace_mib*2**20,ledger.budget.limit-retained_bytes)
    preliminary=n*9+ng*40+sample_n*(32+8*Q)
    if workspace<preliminary:return finish('unknown','scout_workspace_unavailable')
    ledger.admit('early_sample_before_allocation',retained_bytes+preliminary)
    rows,ng_sample=complete_group_rows(codes,q,ng,policy)
    info.update(sample_rows=len(rows),sample_groups=ng_sample,seed=policy.seed)
    if ng_sample<policy.minimum_groups:return finish('unknown','sample_group_coverage_insufficient')
    order=[q]+[j for j in range(Q) if j!=q]
    cc=np.array(codes[rows,:][:,order],dtype=np.int32,order='F',copy=True)
    coverage=[]
    for j,original in enumerate(order):
        unique,inverse=np.unique(cc[:,j],return_inverse=True)
        cc[:,j]=inverse
        if j:coverage.append(len(unique)/profile['levels'][original])
    info['remaining_level_coverage']=coverage
    if min(coverage,default=1.)<policy.minimum_level_coverage:
        return finish('unknown','sample_level_coverage_insufficient')
    ww=w[rows].copy();del rows
    sb=replace(ledger.budget,workspace_mib=workspace/2**20,
               max_gram_nnz=min(ledger.budget.max_gram_nnz,1_000_000),
               max_symbolic_work=min(ledger.budget.max_symbolic_work,20_000_000))
    sl=AllocationLedger(sb)
    try:
        sys,_=prepare_system(cc,ww,0,sl,cc.nbytes+ww.nbytes)
        T,scale,stats=guarded_star(sys,sl,cc.nbytes+ww.nbytes)
        pool=candidates(T,sl,max_candidates)
        useful=[p for p in pool if p['admitted'] and p['singleton_dof_fraction']<.9]
        info['proxy_star']=stats
        info['partitions']=[{key:v for key,v in p.items() if key!='labels'} for p in pool]
        if not useful:return finish('unknown','no_proxy_partition')
        p=useful[0]
        ev=preview_partition(T,p,sl,eig_cut=eig_cut,policy=screen_policy)
        info['preview']=ev
        if ev['accepted']:
            return finish('promising','proxy_local_structure_evidence')
        if ev['reason']=='large_cut_without_sampled_spectral_evidence' and ev['inspected_blocks']:
            return finish('reject','proxy_large_cut_without_spectral_evidence')
        return finish('unknown','proxy_inconclusive')
    except (CandidateRejected,la.LinAlgError,ValueError) as exc:
        info['proxy_issue']=str(exc)
        return finish('unknown','proxy_not_certifiable')
    finally:
        info['allocation_events']=sl.events
