from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import numpy as np
from econhdfe.design_structure import StructuralTerm, plan_structural_collinearity


def _term(index, name, codes):
    codes = np.asarray(codes, dtype=np.int32)
    nlev = int(codes.max()) + 1
    return StructuralTerm(
        index=index,
        name=name,
        codes=codes,
        n_levels=nlev,
        active=np.ones(nlev, dtype=bool),
        level_names=tuple(f"{name}[{j}]" for j in range(nlev)),
        continuous_signature=(),
        kind="factor",
        categorical_components=(f"column:{name}",),
        component_codes=(codes,),
    )


def main(n=2_000_000, seed=7404):
    rng = np.random.default_rng(seed)
    city = rng.integers(0, 2400, n, dtype=np.int32)
    province = city // 40
    region = province // 10
    country = region // 6
    terms = [
        _term(0, "country", country),
        _term(1, "region", region),
        _term(2, "province", province),
        _term(3, "city", city),
    ]
    t0 = time.perf_counter()
    p = plan_structural_collinearity(terms)
    sec = time.perf_counter() - t0
    print(f"N={n:,}")
    print(f"planning_seconds={sec:.3f}")
    print(f"direct_component_edges={len(p.component_dependencies)}")
    print(f"closure_edges={len(p.closure_dependencies)}")
    print(f"full_component_checks={p.component_checks}")
    print(f"closure_inferences={p.closure_inferences}")
    print("closure=" + ",".join(f"{d.coarse}->{d.fine}" for d in p.closure_dependencies))


if __name__ == "__main__":
    main()
