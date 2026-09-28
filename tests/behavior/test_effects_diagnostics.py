import itertools
import numpy as np
import pytest

from econhdfe import ppmlhdfe
from econhdfe.effects import diagnose_fe_recovery, raise_for_fe_identification, FixedEffectIdentificationError
from econhdfe.hdfe.plan import FEPlan
from econhdfe.models.ppml import PPMLConfig


def _codes(edges):
    return [edges[:, j] for j in range(edges.shape[1])]


def test_singleton_pruning_is_reported_without_false_causal_rank_claim():
    worker = np.array([0, 0, 1, 1, 2])
    firm = np.array([0, 1, 0, 1, 2])
    plan = FEPlan.from_arrays([worker, firm])
    singleton = plan.singleton_mask()
    assert singleton.tolist() == [False, False, False, False, True]

    diag = diagnose_fe_recovery(
        [worker, firm], names=["worker", "firm"],
        final_mask=~singleton, singleton_mask=singleton,
    )
    assert diag.estimable
    assert any(x.code == "fe_recovery.singleton_pruned" for x in diag.issues)
    assert not any(x.code == "identification.fe_singleton_induced" for x in diag.issues)


def test_ppml_separation_can_create_additional_unidentified_direction():
    edges = np.array(list(itertools.product(range(2), repeat=3)), dtype=int)
    separation = np.zeros(len(edges), dtype=bool)
    separation[[0, 1, 6, 7]] = True
    final = ~separation

    diag = diagnose_fe_recovery(
        _codes(edges), names=["origin", "destination", "year"],
        final_mask=final, singleton_mask=np.zeros(len(edges), dtype=bool),
        separation_mask=separation, separation_by_method={"fe": 4}, estimator="ppml",
    )
    assert not diag.estimable
    assert diag.raw_identification.extra_nullity == 0
    assert diag.post_singleton_identification.extra_nullity == 0
    assert diag.final_identification.extra_nullity == 1
    issue = diag.primary_error()
    assert issue.code == "identification.fe_ppml_separation_induced"
    assert issue.details["n_separated"] == 4
    with pytest.raises(FixedEffectIdentificationError) as exc:
        raise_for_fe_identification(diag)
    assert exc.value.code == "identification.fe_ppml_separation_induced"


def test_realized_data_rank_failure_gets_data_specific_error():
    edges = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 0]], dtype=int)
    diag = diagnose_fe_recovery(_codes(edges), names=["worker", "firm", "year"])
    assert not diag.estimable
    assert diag.primary_error().code == "identification.fe_data_rank_deficiency"
    assert diag.primary_error().details["extra_nullity"] == 1


def test_nested_support_is_named_explicitly():
    firm = np.array([0, 0, 1, 1, 2, 2])
    industry = np.array([0, 0, 0, 0, 1, 1])
    diag = diagnose_fe_recovery([firm, industry], names=["firm", "industry"])
    issue = next(x for x in diag.issues if x.code == "fe_recovery.nested_or_redundant")
    assert ("firm", "industry") in issue.details["relations"]


def test_ppml_no_finite_sample_error_reports_separation_reason():
    y = np.zeros(6)
    fe = np.array([0, 0, 1, 1, 2, 2])
    with pytest.raises(Exception) as exc:
        ppmlhdfe(
            y, np.empty((6, 0)), absorb=[fe],
            config=PPMLConfig(separation=("fe",)),
        )
    err = exc.value
    assert getattr(err, "code", None) == "identification.no_finite_estimate"
    assert err.details["reason"] == "ppml_separation_no_finite_mle"
    assert err.details["n_separated"] == 6
    assert err.details["separation_by_method"]["fe"] == 6


def test_ppml_empty_after_singletons_reports_singleton_reason():
    y = np.ones(5)
    unique_fe = np.arange(5)
    with pytest.raises(Exception) as exc:
        ppmlhdfe(
            y, np.empty((5, 0)), absorb=[unique_fe],
            config=PPMLConfig(separation=("fe",)),
        )
    err = exc.value
    assert getattr(err, "code", None) == "input.empty_after_singletons"
    assert err.details["reason"] == "all_rows_pruned_as_singletons"
    assert err.details["n_singletons"] == 5


def test_partial_component_rank_failure_is_warning_not_global_error():
    good = np.array(np.meshgrid(np.arange(2), np.arange(2), np.arange(2), indexing="ij")).reshape(3, -1).T
    bad = np.array([[2, 2, 2], [2, 2, 3], [3, 3, 2]], dtype=int)
    edges = np.vstack([good, bad])
    diag = diagnose_fe_recovery(_codes(edges), names=["worker", "firm", "year"])

    assert diag.estimable
    assert diag.primary_error() is None
    issue = next(x for x in diag.issues if x.code == "identification.fe_partial_component_rank_deficiency")
    assert issue.severity == "warning"
    assert issue.details["component_failure_code"] == "identification.fe_data_rank_deficiency"
    assert issue.details["identified_components"] == (0,)
    assert issue.details["unidentified_components"] == (1,)
    assert issue.details["component_extra_nullity"] == {0: 0, 1: 1}


def test_ppml_separation_partial_failure_preserves_other_identified_component():
    good0 = np.array(np.meshgrid(np.arange(2), np.arange(2), np.arange(2), indexing="ij")).reshape(3, -1).T
    good1 = good0 + 2
    edges = np.vstack([good0, good1])
    separation = np.zeros(len(edges), dtype=bool)
    # In component 1 retain only [(2,2,2), (2,2,3), (3,3,2)], a connected
    # 3-way support with one extra null direction; component 0 remains intact.
    keep_second = {(2, 2, 2), (2, 2, 3), (3, 3, 2)}
    for i in range(len(good0), len(edges)):
        if tuple(edges[i]) not in keep_second:
            separation[i] = True
    final = ~separation

    diag = diagnose_fe_recovery(
        _codes(edges), names=["origin", "destination", "year"],
        final_mask=final, singleton_mask=np.zeros(len(edges), dtype=bool),
        separation_mask=separation, separation_by_method={"fe": int(separation.sum())},
        estimator="ppml",
    )
    assert diag.estimable
    assert diag.raw_identification.extra_nullity == 0
    assert diag.final_identification.extra_nullity == 1
    issue = next(x for x in diag.issues if x.code == "identification.fe_partial_component_rank_deficiency")
    assert issue.details["component_failure_code"] == "identification.fe_ppml_separation_induced"
    assert len(issue.details["identified_components"]) == 1
    assert len(issue.details["unidentified_components"]) == 1
