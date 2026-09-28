"""Exact independent Fraction oracles and adversarial structural identities.
Finite enumeration is regression evidence, not a mathematical proof."""
from fractions import Fraction
from itertools import product
from unittest.mock import patch
import importlib
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import structural_rank
from econhdfe.hdfe.rank import categorical_rank, _properly_connected
from econhdfe.hdfe.numerical_core import build_numerical_core
from econhdfe.hdfe.absorber import HDFEAbsorber
from econhdfe.hdfe.structure import canonicalize_fixed_effects
from econhdfe.design_structure import (StructuralTerm, plan_structural_collinearity,
                                       _component_dependency_closure)
from econhdfe.compute.block_design import BlockDesign, DenseDesignBlock
from econhdfe.compute.partitioned_lstsq import partitioned_weighted_lstsq, compress_partitioned_wls


def exact_rank(a):
    a = np.asarray(a)
    rows = [[Fraction(int(v)) for v in row] for row in a]
    r = 0
    for j in range(a.shape[1]):
        pivot = next((i for i in range(r, len(rows)) if rows[i][j]), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        p = rows[r][j]
        rows[r] = [v / p for v in rows[r]]
        for i in range(r + 1, len(rows)):
            c = rows[i][j]
            if c:
                rows[i] = [v - c*u for v, u in zip(rows[i], rows[r])]
        r += 1
        if r == len(rows):
            break
    return r


def dummies(groups):
    gs = [np.unique(g, return_inverse=True)[1] for g in groups]
    return np.column_stack([np.eye(int(g.max())+1, dtype=int)[g] for g in gs])


def test_all_nonempty_subsets_of_2_by_2_by_3_rank():
    edges = np.array(list(product(range(2), range(2), range(3))))
    for mask in range(1, 1 << len(edges)):
        ix = [j for j in range(len(edges)) if mask & (1 << j)]
        gs = list(edges[ix].T)
        assert categorical_rank(gs, backend='native') == exact_rank(dummies(gs)), mask


@pytest.mark.parametrize('ways', range(1, 7))
def test_random_multihypergraphs_exact_integer_rank(ways):
    rng = np.random.default_rng(6300 + ways)
    for _ in range(35):
        gs = [rng.integers(0, rng.integers(1, 6), size=18) for _ in range(ways)]
        ref = exact_rank(dummies(gs))
        assert categorical_rank(gs, backend='native') == ref
        assert categorical_rank([np.r_[g, g[:4]] for g in gs], backend='native') == ref
        assert categorical_rank([g[::-1] for g in gs[::-1]], backend='native') == ref


def test_bad_prime_must_not_be_accepted_as_characteristic_zero_rank():
    # All four rows have even parity: rank GF(2)=3 but rank Q=4.
    edges = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
    mod = importlib.import_module('econhdfe.hdfe.rank')
    assert exact_rank(dummies(edges.T)) == 4
    assert not _properly_connected(edges)
    assert categorical_rank(list(edges.T), backend='native') == 4
    # Force both modular certificate attempts to fail to reach the upper bound.
    with patch.object(mod, '_gf2_bitset_rank', return_value=0), \
         patch.object(mod, '_modular_sparse_rank', return_value=0):
        assert categorical_rank(list(edges.T), backend='native') == 4


def test_proper_connectivity_is_sufficient_but_not_necessary():
    edges = np.array(list(product(range(2), range(2), range(3))))
    assert _properly_connected(edges)
    assert categorical_rank(list(edges.T)) == 7-3+1
    # Previous parity example attains V-G+1 without proper connectivity.
    nonproper = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
    assert not _properly_connected(nonproper)
    assert categorical_rank(list(nonproper.T)) == 6-3+1


def test_generic_sparsity_rank_is_not_dummy_design_rank():
    D = dummies(np.array(list(product(range(2), range(2)))).T)
    assert structural_rank(csr_matrix(D)) == 4
    assert exact_rank(D) == 3


@pytest.mark.parametrize('codes', [[0,2,2], [.25,.25,1.25], [0,np.nan,1],
                                   [0,np.inf,1], [0,-1,1], [0,2**63,1]])
@pytest.mark.parametrize('ways', [1,2,3])
def test_exact_rank_validates_observed_dense_integer_codes(codes, ways):
    with pytest.raises(ValueError):
        categorical_rank([np.array(codes)]*ways, assume_dense=True)


@pytest.mark.parametrize('ways', [3,4,5])
@pytest.mark.parametrize('method', ['map','lsmr'])
def test_residual_core_matches_independent_weighted_dense_projection(ways, method):
    rng = np.random.default_rng(194+ways)
    # The core includes repeated tuples with DIFFERENT outcomes and weights.
    core = np.array(list(product(*([range(2)]*ways))), dtype=np.int32)
    core = np.r_[core, core[:4]]
    leaves = np.array([[2+j]*ways for j in range(6)], dtype=np.int32)
    edges = np.r_[core, leaves]
    gs = list(edges.T)
    plan = build_numerical_core(gs)
    assert plan.peeled == 6
    Y = rng.normal(size=(len(edges),3))
    w = rng.uniform(.2,2,len(edges))
    D = dummies(gs); sw = np.sqrt(w)
    B = np.linalg.lstsq(D*sw[:,None], Y*sw[:,None], rcond=None)[0]
    expected = Y - D@B
    engine = HDFEAbsorber(gs, weights=w, tol=1e-11, method=method,
                          core_reduction='on', absorb_threads=1)
    got = engine.residualize(Y)
    np.testing.assert_allclose(got, expected, atol=2e-9, rtol=1e-8)
    np.testing.assert_array_equal(got[-6:], np.zeros((6,3)))
    w2 = rng.uniform(.2,2,len(edges)); engine.update_weights(w2)
    B2 = np.linalg.lstsq(D*np.sqrt(w2)[:,None], Y*np.sqrt(w2)[:,None], rcond=None)[0]
    np.testing.assert_allclose(engine.residualize(Y), Y-D@B2, atol=2e-9, rtol=1e-8)


def test_zero_weight_invalidates_positive_support_core_certificate():
    gs = [np.array([0,0,1,1,2])]*3
    a = HDFEAbsorber(gs, weights=np.ones(5), core_reduction='on')
    assert a._core_plan is not None
    a.update_weights(np.array([1.,1,1,1,0]))
    assert a._core_plan is None


def term(index, name, codes, active, signature=()):
    return StructuralTerm(index,name,np.array(codes,dtype=np.int32),len(active),
                          np.array(active,dtype=bool),tuple(f'{name}{j}' for j in range(len(active))),
                          signature, 'factor', (name,), (np.array(codes,dtype=np.int32),))


def test_structural_reductions_preserve_exact_span_with_all_reference_masks():
    fine = np.repeat(np.arange(4), 2); coarse = fine//2
    for fmask in range(1, 16):
        for cmask in range(1, 4):
            for sig, multiplier in [((),np.ones(8,dtype=int)), (('x',),np.array([0,2,-1,4,2,3,0,-2]))]:
                terms = [term(0,'coarse',coarse,[bool(cmask&(1<<j)) for j in range(2)],sig),
                         term(1,'fine',fine,[bool(fmask&(1<<j)) for j in range(4)],sig)]
                def matrix():
                    return np.column_stack([np.eye(t.n_levels,dtype=int)[t.codes][:,t.active]*multiplier[:,None] for t in terms])
                before = matrix()
                plan_structural_collinearity(terms)
                after = matrix()
                assert exact_rank(before) == exact_rank(after)


def test_different_slope_multiplier_cannot_use_pure_dummy_absorption():
    t = term(0,'g',np.repeat([0,1],4),[True,True],('x',))
    plan_structural_collinearity([t], absorbed_groups=[np.zeros(8,dtype=int)], absorbed_names=['constant'])
    assert t.active.all()


def test_equivalent_component_partitions_form_cycle_not_dag():
    a=np.array([0,0,1,1]); b=1-a
    reach,*_ = _component_dependency_closure([term(0,'a',a,[True,True]),term(1,'b',b,[True,True])])
    assert ('a','b') in reach and ('b','a') in reach


def test_fe_canonicalization_projection_span_and_row_subset_fd():
    fine=np.repeat(np.arange(5),3); coarse=fine//2
    gs=[coarse,fine]
    newgs,*_ = canonicalize_fixed_effects(gs,[None,None],[True,True],['coarse','fine'])
    assert exact_rank(dummies(newgs)) == exact_rank(dummies(gs))
    subset=np.array([0,2,3,7,8,11,13])
    # A true fine->coarse dependency survives arbitrary row deletion.
    for f in np.unique(fine[subset]):
        assert len(np.unique(coarse[subset][fine[subset]==f])) == 1


@pytest.mark.parametrize('shared', [False,True])
@pytest.mark.parametrize('deficient', [False,True])
@pytest.mark.parametrize('chunk', [5,1000])
def test_partitioned_qr_matches_global_minimum_norm_and_objective(shared, deficient, chunk):
    rng=np.random.default_rng(732)
    n=100; k=5 if shared else 4
    cols=[np.array([0,1,2]),np.array([0,3,4])] if shared else [np.array([0,1]),np.array([2,3])]
    X=np.zeros((n,k)); blocks=[]
    for j,(rows,c) in enumerate(zip(np.array_split(np.arange(n),2),cols)):
        vals=rng.normal(size=(len(rows),len(c)))
        if deficient: vals[:,-1]=vals[:,-2]
        X[np.ix_(rows,c)]=vals
        blocks.append(DenseDesignBlock(j,rows,c,vals))
    design=BlockDesign(n,k,tuple(blocks)); y=rng.normal(size=n);w=rng.uniform(.2,2,n);w[::17]=0
    b,resid,_=partitioned_weighted_lstsq(design,y,w,chunk_rows=chunk)
    ref=np.linalg.lstsq(X*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0]
    np.testing.assert_allclose(b,ref,atol=2e-10,rtol=1e-9)
    np.testing.assert_allclose(resid,y-X@ref,atol=2e-10,rtol=1e-9)
    Xc,yc,_=compress_partitioned_wls(design,y,w,chunk_rows=chunk)
    trial=rng.normal(size=k)
    np.testing.assert_allclose(np.linalg.norm(Xc@trial-yc)**2,np.sum(w*(X@trial-y)**2),rtol=1e-12)
