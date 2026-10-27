"""Bounded, pre-factor numerical evidence; never a rank/forward-error certificate.

A large cut with no sampled low-energy block directions is a poor bet for this
particular block/coarse builder. Reject it before all Cholesky/eigen/coarse work.
Small-cut block-only candidates remain eligible; a nonempty coarse space is NOT
mandatory. Sampling may miss a profitable direction, so this is conservative
routing, not a proof that rejected designs are hard or unsolvable.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import scipy.linalg as la
from .allocation import CandidateRejected, sparse_bytes


@dataclass(frozen=True)
class NumericalScreenPolicy:
    enabled: bool = True
    max_cut_without_spectral_evidence: float = .10
    max_preview_blocks: int = 8
    max_preview_cubic_work: int = 64_000_000

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('enabled must be bool')
        if (not np.isfinite(self.max_cut_without_spectral_evidence) or
                not 0 <= self.max_cut_without_spectral_evidence <= 1):
            raise ValueError('invalid cut evidence threshold')
        for name in ('max_preview_blocks', 'max_preview_cubic_work'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
                raise ValueError(f'{name} must be a positive integer')


def preview_partition(T, partition, ledger, *, eig_cut, policy, retained_bytes=0):
    """Inspect normalized T before constructing local factors.

    Uses exact eigvalues of at most eight small principal blocks, not full-system
    eigenvalues. The largest-work blocks are inspected first, and a weak-mode
    witness permits (does not guarantee profitability of) a later useful-column
    test. All large target arrays remain untouched.
    """
    start = perf_counter()
    cut = float(partition['cut_energy_fraction'])
    info = dict(stage='prefactor_numerical_screen', cut_energy_fraction=cut,
                cut_limit=policy.max_cut_without_spectral_evidence,
                eig_cut=float(eig_cut), inspected_blocks=[], preview_cubic_work=0,
                assurance='bounded routing evidence, not a spectral equivalence or speed certificate')
    def finish(accepted, reason):
        info.update(accepted=bool(accepted), reason=reason, seconds=perf_counter()-start)
        return info
    if not np.isfinite(cut) or not 0 <= cut <= 1+1e-12:
        raise CandidateRejected('invalid_partition_energy')
    if not policy.enabled:
        return finish(True, 'screen_disabled_for_ablation')
    if cut <= policy.max_cut_without_spectral_evidence:
        return finish(True, 'small_cut_block_only_or_coarse_candidate')
    if eig_cut <= 0:
        return finish(False, 'large_cut_coarse_disabled')

    lab = np.asarray(partition['labels'])
    m = T.shape[0]
    if lab.shape != (m,):
        raise ValueError('partition labels must match T')
    sizes = np.bincount(lab)
    max_block = int(sizes.max(initial=0))
    if max_block > ledger.budget.max_block:
        raise CandidateRejected('preview_block_budget', max_block=max_block)
    # Admit temporary ordering, dense block and LAPACK workspace before toarray.
    ledger.admit('prefactor_preview_before_allocation',
                 retained_bytes+sparse_bytes(T.nnz, m)+80*m+64*max_block**2)
    order = np.argsort(lab, kind='stable')
    offsets = np.r_[0, np.cumsum(sizes)]
    chosen = np.argsort(-sizes, kind='stable')
    work = 0
    for j in chosen:
        size = int(sizes[j])
        if size < 2:
            break
        if len(info['inspected_blocks']) >= policy.max_preview_blocks:
            break
        if work + size**3 > policy.max_preview_cubic_work:
            info['preview_budget_exhausted'] = True
            break
        ids = order[offsets[j]:offsets[j+1]]
        block = T[ids, :][:, ids].toarray()
        block = (block + block.T)*.5
        if not np.all(np.isfinite(block)):
            raise CandidateRejected('nonfinite_preview_block')
        vals = la.eigh(block, eigvals_only=True, check_finite=False)
        low, high = float(vals[0]), float(vals[-1])
        work += size**3
        info['preview_cubic_work'] = work
        if low <= 0 or high <= 0:
            return finish(False, 'nonpositive_preview_block')
        ratio = low/high
        info['inspected_blocks'].append(dict(size=size, minimum=low, maximum=high,
                                              relative_minimum=ratio))
        if ratio < eig_cut:
            return finish(True, 'sampled_local_low_energy_direction')
    # No complete-rank/no-coarse claim: only the bounded inspected blocks were
    # examined. Ties in which blocks are sampled can affect decisions, not fits.
    return finish(False, 'large_cut_without_sampled_spectral_evidence')
