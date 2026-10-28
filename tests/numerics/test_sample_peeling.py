"""Independent brute-force oracle for queue-based categorical peeling."""
import numpy as np
import pytest
from econhdfe.hdfe.sample import peel_sample


def _oracle(groups, fw, positive, drop=True):
    keep = np.ones(len(groups[0]),dtype=bool)
    while True:
        remove = np.zeros(len(keep),dtype=bool)
        for g in groups:
            for level in np.unique(g[keep]):
                ids = np.flatnonzero(keep & (g==level))
                if (drop and sum(int(fw[i]) for i in ids)==1) or (positive is not None and not positive[ids].any()):
                    remove[ids]=True
        if not remove.any():return keep
        keep[remove]=False


@pytest.mark.parametrize('seed,use_positive', [(43,False),(711,True)])
def test_queue_matches_independent_fixed_point_and_permutations(seed,use_positive):
    rng=np.random.default_rng(seed);n=400
    groups=[rng.integers(0,k,n) for k in [170,183,125,20]]
    fw=rng.integers(1,4,n)
    positive=rng.random(n)>.65 if use_positive else None
    expected=_oracle(groups,fw,positive)
    r=peel_sample(groups,frequency_weights=fw,positive=positive)
    np.testing.assert_array_equal(r.keep,expected)
    assert not np.any(r.singletons&r.separated)
    np.testing.assert_array_equal(~r.keep,r.singletons|r.separated)
    if positive is not None:assert not r.separated[positive].any()
    p=rng.permutation(n)
    rr=peel_sample([g[p] for g in groups[::-1]],frequency_weights=fw[p],positive=None if positive is None else positive[p])
    np.testing.assert_array_equal(rr.keep,expected[p])


def test_large_frequency_and_tiny_positive_support_are_exact():
    r=peel_sample([np.array([0,0]),np.array([0,1])],frequency_weights=[2**54,1])
    assert r.keep.tolist()==[True,False]
    r=peel_sample([np.array([0,0])],positive=np.array([1e-300,0.])>0)
    assert r.keep.all()
    assert peel_sample([],nobs=4).keep.all()


@pytest.mark.parametrize('bad', [[1.,0.],[1.,1.5],[1.,np.nan],[1.]])
def test_bad_multiplicity_fails_before_peeling(bad):
    with pytest.raises(ValueError,match='positive integers'):
        peel_sample([np.array([0,1])],frequency_weights=bad)


def test_long_cascade_frequency_protection_and_disabled_singletons():
    n=30000
    groups=[np.arange(n)//2,(np.arange(n)+1)//2,np.arange(n)%29]
    assert not peel_sample(groups).keep.any()
    assert peel_sample(groups,frequency_weights=np.full(n,2)).keep.all()
    assert peel_sample(groups,drop_singletons=False).keep.all()
