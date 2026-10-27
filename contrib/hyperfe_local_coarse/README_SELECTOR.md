# FE selector v0.5 — experimental opt-in research workflow

This folder extends the standalone hypergraph local/coarse FE projector.
No changes are made to `econhdfe` estimator APIs, defaults, samples,
singleton rules, absorbed DoF, or covariance conventions.

**Requirements:** Python 3.13; `within-py==0.4.0`; SciPy 1.17.0.

```sh
python -m pip install -r selector_requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  python -m pytest tests -q
```

```python
import sys
sys.path.insert(0, 'src')
from feselection import FESelector, SelectionConfig

# codes (N,Q) observed dense categorical IDs, weights (N,) strictly >0;
# Y (N,K) float64. This computes ONLY the FE projection.
p = FESelector(SelectionConfig(mode='auto')).prepare(
    codes, weights, expected_rhs=Y.shape[1]
)
result = p.residualize(Y)
R = result.residuals
trace = result.diagnostics
```

Use `mode='observe'` to collect diagnostic evidence without attempting
experimental preconditioning, or `mode='native'` to force native within.
The default selector mode is observe. A negative sample-only scout is
`unknown`, not a hard veto; pilot timeouts are never treated as convergence.

Million-row reproducible example (run each method in a separate process):
```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  python benchmark_selector_v05.py --case smooth_sparse --seed 93517 --rhs 3 \
  --method selector --out /tmp/fe_selector_v05.json
```

Switch `--method` to `native_adaptive`, `native_diagonal`, or
`native_additive` for direct upstream comparisons. The standalone runner
excludes data generation and initial JIT costs but includes selection,
setup, failed probes, reconstruction and audit. Use identical seeds and
check output audits; a single timing is not a general speed claim.

`evidence/` carries frozen quantitative summaries and rejected/slow cases.
Other tests and full source archive are preserved in the previous research
delivery; the PR does not commit gigabyte-scale synthetic inputs.

**Important:** `src/feselection/monitored_lsmr.py` adapts upstream SciPy
code under BSD-3-Clause. See THIRD_PARTY_NOTICES.md and third_party/.
No production integration, publication, or historical novelty claim.
