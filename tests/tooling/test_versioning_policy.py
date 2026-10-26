from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_workqueue_parallel_bootstrap_in_fresh_process():
    # A regression aborts the child, never the entire test runner. These are the
    # original weighted/unweighted serial-versus-parallel numerical assertions.
    env = dict(os.environ, NUMBA_THREADING_LAYER="workqueue", NUMBA_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUTF8="1")
    command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/behavior/test_cluster_inference.py::test_weighted_wild_bootstrap_parallel_reproducibility_and_distributions",
        "tests/behavior/test_linear_models.py::test_parallel_wild_bootstrap_shape_and_reproducibility",
    ]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
