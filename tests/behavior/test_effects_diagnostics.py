import numpy as np
from econhdfe.effects import diagnose_fe_recovery


def _codes(edges):
    return [edges[:, j] for j in range(edges.shape[1])]


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
