"""Reference compatibility, replay, input boundaries and CPU-only import tests."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from boundedqr import bootstrap, fn_fit, solve_perturbations
from boundedqr.scores import cluster_scores, encode_clusters, mammen_block


DATA = Path(__file__).resolve().parents[1] / "experiments" / "data" / "n2000_p6_b5"


@pytest.fixture(scope="module")
def frozen_r():
    names = ("x", "y", "cluster", "base_beta", "psi", "v", "W", "r_fn", "r_pwy")
    return {name: np.load(DATA / f"{name}.npy") for name in names}


@pytest.mark.parametrize("batch_size", [1, 3])
def test_every_replicate_matches_fixed_r_scores_and_multipliers(frozen_r, batch_size):
    d = frozen_r
    before = {name: hashlib.sha256((DATA / f"{name}.npy").read_bytes()).hexdigest()
              for name in ("psi", "v", "W")}
    result = bootstrap(d["x"], d["y"], d["cluster"], quantiles=[.5], reps=5,
                       base_coefficients=d["base_beta"], reference_psi=d["psi"],
                       multipliers=d["v"], batch_size=batch_size, threads=1)
    assert result.bootstrap_coefficients.shape == (5, 1, 6)
    for j in range(5):
        np.testing.assert_allclose(result.bootstrap_coefficients[j, 0], d["r_fn"][j],
                                   atol=2e-8, rtol=1e-8)
        np.testing.assert_allclose(result.bootstrap_coefficients[j, 0], d["r_pwy"][j],
                                   atol=2e-8, rtol=1e-8)
    assert result.diagnostics["quantiles"][0]["score_source"] == "supplied_reference"
    assert np.isfinite(result.coefficient_radii).all()
    for name in before:
        assert hashlib.sha256((DATA / f"{name}.npy").read_bytes()).hexdigest() == before[name]


def test_unsorted_cluster_score_reduction_matches_observation_level_oracle():
    cluster = np.array([9, -4, 9, 2, -4, 2, 9, 2])
    X = np.arange(24., dtype=float).reshape(8, 3) - 11
    psi = np.array([.8, -.2, .8, -.2, -.2, .8, .8, -.2])
    labels, codes = encode_clusters(cluster, len(X))
    exact = np.array([[math.fsum(float(X[i, k] * psi[i]) for i in range(len(X)) if cluster[i] == g)
                       for k in range(3)] for g in sorted(set(cluster))])
    for block_rows in (1, 3, 100):
        score = cluster_scores(X, psi, codes, len(labels), block_rows=block_rows)
        np.testing.assert_allclose(score, exact, atol=2e-15, rtol=1e-15)
        V = np.array([[-.5, 1.], [2., -.5], [-1., 2.]])
        expected_W = X.T @ (psi[:, None] * V[codes])
        np.testing.assert_allclose(score.T @ V, expected_W, atol=2e-14, rtol=1e-14)


def test_frozen_cluster_aggregation_reconstructs_original_r_W(frozen_r):
    d = frozen_r
    labels, codes = encode_clusters(d["cluster"], len(d["x"]))
    reconstructed = cluster_scores(d["x"], d["psi"], codes, len(labels)).T @ d["v"]
    np.testing.assert_allclose(reconstructed, d["W"], atol=1e-11, rtol=1e-12)


def test_mammen_indexed_replay_across_blocks_order_and_repeated_indices():
    whole = mammen_block(37, range(11), seed=214)
    split = np.column_stack((mammen_block(37, range(3), 214),
                             mammen_block(37, range(3, 8), 214),
                             mammen_block(37, range(8, 11), 214)))
    np.testing.assert_array_equal(whole, split)
    np.testing.assert_array_equal(mammen_block(37, [7, 1, 7], 214), whole[:, [7, 1, 7]])
    assert len(np.unique(whole)) == 2
    values = np.unique(whole)
    probability_low = (np.sqrt(5) + 1) / np.sqrt(20)
    assert abs(probability_low * values[0] + (1 - probability_low) * values[1]) < 1e-15
    assert abs(probability_low * values[0]**2 + (1 - probability_low) * values[1]**2 - 1) < 1e-15


def test_quantile_process_replays_same_draws_as_separate_calls():
    rng = np.random.default_rng(938)
    X = np.column_stack((np.ones(80), rng.normal(size=(80, 2))))
    y = X @ [.5, 1., -1.] + rng.normal(size=80)
    cluster = np.tile([8, -1, 3, 2, 6, 7, 0, 5], 10)
    taus = [.2, .8]
    fits = [fn_fit(X, y, tau, tol=1e-8) for tau in taus]
    base = np.array([fit.beta for fit in fits])
    psi = np.array([(fit.residual < 0) - tau for fit, tau in zip(fits, taus)])
    together = bootstrap(X, y, cluster, quantiles=taus, reps=4, seed=17, batch_size=3,
                         threads=1, base_coefficients=base, reference_psi=psi)
    for j, tau in enumerate(taus):
        separate = bootstrap(X, y, cluster, quantiles=[tau], reps=4, seed=17,
                             batch_size=1, threads=1, base_coefficients=base[j], reference_psi=psi[j])
        np.testing.assert_allclose(together.bootstrap_coefficients[:, j],
                                   separate.bootstrap_coefficients[:, 0], atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("labels", [
    [1, np.nan, "x"], np.array([1, np.nan, 2], dtype=object),
    [1, None, "x"], [1, np.inf, "x"],
    np.array(["2024-01-01", "NaT", "2024-01-03"], dtype="datetime64[D]"),
])
def test_missing_or_infinite_cluster_labels_are_never_silently_encoded(labels):
    with pytest.raises(ValueError):
        encode_clusters(labels, 3)


def test_mixed_large_integer_cluster_labels_are_not_silently_merged():
    # np.asarray normally promotes these to float and loses the first two labels.
    labels = [2**53, 2**53 + 1, 0.5]
    try:
        unique, codes = encode_clusters(labels, 3)
    except ValueError:
        return  # Explicitly rejecting unsupported mixed types is acceptable.
    assert len(unique) == 3
    assert len(set(codes)) == 3


@pytest.mark.parametrize("option,value", [
    ("batch_size", 1.5), ("batch_size", True), ("batch_size", 0),
    ("threads", 1.5), ("threads", True), ("threads", 0),
    ("seed", 1.5), ("seed", True), ("seed", -1),
])
def test_invalid_controls_rejected_even_with_supplied_multipliers(option, value):
    X = np.column_stack((np.ones(8), np.arange(8.)))
    kwargs = {"reps": 2, "threads": 1, "base_coefficients": [[0., 0.]],
              "reference_psi": np.full((1, 8), -.5), "multipliers": np.ones((2, 2)),
              option: value}
    with pytest.raises(ValueError):
        bootstrap(X, np.arange(8.)**2, np.tile([0, 1], 4), **kwargs)


@pytest.mark.parametrize("where", ["X", "y", "W", "beta"])
def test_nonfinite_solver_inputs_are_explicit(where):
    X = np.column_stack((np.ones(8), np.arange(8.)))
    y, W, beta = np.arange(8.)**2, np.ones((2, 1)), np.zeros(2)
    arrays = {"X": X, "y": y, "W": W, "beta": beta}
    arrays[where].flat[0] = np.nan
    with pytest.raises(ValueError):
        solve_perturbations(X, y, .5, W, beta, threads=1)


def test_cpu_user_path_runs_when_gpu_and_R_are_unavailable(tmp_path):
    script = r'''
import sys, importlib.abc, numpy as np
class NoGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cupy','torch','rpy2','pyfixest'}:
            raise AssertionError('CPU path tried optional runtime: '+fullname)
sys.meta_path.insert(0, NoGPU())
from boundedqr import bootstrap
rng=np.random.default_rng(75)
X=np.column_stack((np.ones(80),rng.normal(size=80)))
y=X@np.array([1.,-.5])+rng.normal(size=80)
result=bootstrap(X,y,np.arange(80)%8,reps=3,threads=1,batch_size=2)
assert result.bootstrap_coefficients.shape==(3,1,2)
assert np.isfinite(result.bootstrap_coefficients).all()
assert result.diagnostics['quantiles'][0]['base_source']=='pyfixest_fn_0.60.0'
assert not any(k.split('.')[0] in {'cupy','torch','rpy2','pyfixest'} for k in sys.modules)
print('independent CPU fit and bootstrap passed')
'''
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=45)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "independent CPU fit and bootstrap passed" in out.stdout
