from __future__ import annotations
import numpy as np
from econhdfe.planner import PlanCertificate, RepresentationCandidate, choose_representation


def test_certificate_blocks_uncertified_fast_path_even_when_cheaper():
    dense = RepresentationCandidate(
        "dense", PlanCertificate.exact_yes("dense"), payload_bytes=1_000,
    )
    unsafe = RepresentationCandidate(
        "unsafe", PlanCertificate.no("unsafe", "no_exact_certificate"), payload_bytes=10,
    )
    plan = choose_representation((dense, unsafe), expected_passes=20)
    assert plan.representation == "dense"
    by_name = {c.name: c for c in plan.candidates}
    assert not by_name["unsafe"].eligible
    assert np.isinf(by_name["unsafe"].score)
