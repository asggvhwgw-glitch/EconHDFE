import numpy as np
import pytest
from econhdfe.effects import NormalizationSpec, recover_fixed_effects, FixedEffectIdentificationError


def _lookup(term, values):
    mapping = {v: term.coefficients[i] for i, v in enumerate(term.levels.tolist())}
    return np.array([mapping[v] for v in values], dtype=float)


def _reconstruct(result, *groups):
    out = np.zeros(len(groups[0]), dtype=float)
    for term, g in zip(result.terms, groups, strict=False):
        out += _lookup(term, np.asarray(g))
    return out


def test_two_way_reference_normalization_and_renormalize_do_not_refit():
    worker = np.array([0, 0, 1, 1, 2, 2])
    firm = np.array([0, 1, 0, 1, 1, 2])
    target = np.array([1.1, 1.8, -0.4, 0.3, 0.5, 1.2])
    base = recover_fixed_effects(target, [worker, firm], names=["worker", "firm"])
    ref = base.renormalize(
        NormalizationSpec("reference", baseline="worker", references={"firm": 1})
    )
    assert abs(ref.term("firm").coefficients[ref.term("firm").index_of(1)]) < 1e-12
    assert np.max(np.abs(_reconstruct(base, worker, firm) - _reconstruct(ref, worker, firm))) < 1e-12
    assert ref.diagnostics.iterations == base.diagnostics.iterations


def test_three_way_connected_design_can_have_extra_nullity():
    edges = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 0]], dtype=int)
    g1, g2, g3 = edges.T
    target = np.array([1.0, 2.0, 3.0])
    with pytest.raises(FixedEffectIdentificationError) as exc:
        recover_fixed_effects(target, [g1, g2, g3], names=["a", "b", "c"], solver="lsmr")
    assert exc.value.code == "identification.fe_data_rank_deficiency"

    res = recover_fixed_effects(
        target, [g1, g2, g3], names=["a", "b", "c"], solver="lsmr",
        strict_identification=False,
    )
    assert res.identification.n_components == 1
    assert res.identification.nullity == 3
    assert res.identification.structural_shift_nullity == 2
    assert res.identification.extra_nullity == 1
    assert not res.diagnostics.normalization_complete
    assert np.max(np.abs(_reconstruct(res, g1, g2, g3) - target)) < 1e-8


def test_partial_component_salvage_recovers_good_block_and_marks_bad_levels_nan():
    # Component 0 is a fully crossed 3-way block (ordinary K-1 nullity only).
    good = np.array(np.meshgrid(np.arange(2), np.arange(2), np.arange(2), indexing="ij")).reshape(3, -1).T
    # Component 1 is connected but has one extra unidentified direction.
    bad = np.array([[2, 2, 2], [2, 2, 3], [3, 3, 2]], dtype=int)
    edges = np.vstack([good, bad])
    g1, g2, g3 = edges.T
    a = np.array([0.2, -0.3, 1.1, 2.2])
    b = np.array([0.1, 0.4, -0.7, 0.9])
    c = np.array([-0.2, 0.5, 1.5, -1.0])
    target = a[g1] + b[g2] + c[g3]

    res = recover_fixed_effects(target, [g1, g2, g3], names=["worker", "firm", "year"], solver="lsmr")

    assert res.identification.n_components == 2
    assert res.identification.extra_nullity == 1
    assert tuple(x.extra_nullity for x in res.identification.components) == (0, 1)
    assert res.identified_components == (0,)
    assert res.unidentified_components == (1,)
    assert res.diagnostics.n_recovered_obs == len(good)
    assert res.diagnostics.n_total_obs == len(edges)
    assert res.diagnostics.n_recovered_components == 1
    assert res.diagnostics.n_unidentified_components == 1

    obs_component = res.identification.component_by_term[0][g1]
    good_obs = obs_component == 0
    rebuilt = _reconstruct(res, g1, g2, g3)
    np.testing.assert_allclose(rebuilt[good_obs], target[good_obs], atol=1e-9, rtol=1e-9)
    assert np.all(np.isnan(rebuilt[~good_obs]))

    for term in res.terms:
        assert np.all(np.isfinite(term.coefficients[term.identified]))
        assert np.all(np.isnan(term.coefficients[~term.identified]))

    issue = next(x for x in res.diagnosis.issues if x.code == "identification.fe_partial_component_rank_deficiency")
    assert issue.severity == "warning"
    assert issue.details["component_failure_code"] == "identification.fe_data_rank_deficiency"
    assert issue.details["identified_components"] == (0,)
    assert issue.details["unidentified_components"] == (1,)
    assert issue.details["component_extra_nullity"] == {0: 0, 1: 1}
