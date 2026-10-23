"""Fresh-process PanelOLS checks for the isolated linearmodels overlay.

Both processes fit complete models and explicitly access deferred covariance.
The installed source hash must be identical before and after every run.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from tools.run_optimized import PROJECT_ROOT, optimized_environment, prepare_overlay


PANEL_WORKER = r'''
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import linearmodels
from linearmodels import PanelOLS
from linearmodels.datasets import wage_panel
import linearmodels.shared.covariance as covariance_module

def serialize(result):
    covariance = result.cov
    return {"params": result.params.to_numpy().tolist(),
            "cov": covariance.to_numpy().tolist(),
            "stderr": result.std_errors.to_numpy().tolist(),
            "nobs": result.nobs, "df_resid": result.df_resid,
            "columns": result.params.index.tolist()}

rng = np.random.default_rng(775921)
nentity, ntime = 24, 9
index = pd.MultiIndex.from_product([range(nentity), range(ntime)],
                                  names=["entity", "time"])
entity = index.get_level_values(0).to_numpy()
time = index.get_level_values(1).to_numpy()
exog = pd.DataFrame({"const": 1.0, "x1": rng.normal(size=len(index)),
                     "x2": rng.normal(size=len(index))}, index=index)
noise = rng.normal(size=len(index)) * np.linspace(0.5, 2.0, len(index))
endog = pd.Series(0.3 + 0.6*exog.x1 - 0.2*exog.x2 +
                  rng.normal(size=nentity)[entity] + 0.1*time + noise,
                  index=index, name="outcome")
weights = pd.Series(np.geomspace(0.2, 3.0, len(index)), index=index)
keep = rng.uniform(size=len(index)) > 0.18
output = {}
for unbalanced in (False, True):
    mask = keep if unbalanced else np.ones(len(index), dtype=bool)
    for weighted in (False, True):
        for group_debias in (False, True):
            model = PanelOLS(endog[mask], exog[mask],
                weights=weights[mask] if weighted else None,
                entity_effects=True, time_effects=True)
            result = model.fit(cov_type="clustered", cluster_entity=True,
                               cluster_time=True, group_debias=group_debias)
            key = f"synthetic_unbalanced={unbalanced}_weights={weighted}_debias={group_debias}"
            output[key] = serialize(result)

data = wage_panel.load().set_index(["nr", "year"])
exog_public = data[["expersq", "union", "married"]].copy()
exog_public.insert(0, "const", 1.0)
for group_debias in (False, True):
    result = PanelOLS(data.lwage, exog_public, entity_effects=True,
                      time_effects=True).fit(cov_type="clustered",
                      cluster_entity=True, cluster_time=True,
                      group_debias=group_debias)
    output[f"wage_panel_debias={group_debias}"] = serialize(result)

source = Path(covariance_module.__file__)
print(json.dumps({"cases": output, "package_file": linearmodels.__file__,
                  "source_file": str(source),
                  "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}))
'''


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def panel_results():
    overlay = prepare_overlay()
    baseline_path = Path(overlay["baseline_path"])
    original_hash = source_hash(baseline_path)
    baseline_environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT),
                                PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    optimized = optimized_environment(overlay["overlay_root"], baseline_environment)
    runs = {}
    for name, environment in (("baseline", baseline_environment), ("optimized", optimized)):
        result = subprocess.run([sys.executable, "-c", PANEL_WORKER],
                                cwd=PROJECT_ROOT, env=environment, capture_output=True,
                                text=True, check=True, timeout=90)
        runs[name] = json.loads(result.stdout)
        assert source_hash(baseline_path) == original_hash
    return overlay, runs


def test_fresh_processes_select_distinct_validated_source(panel_results):
    overlay, runs = panel_results
    assert Path(runs["baseline"]["source_file"]).resolve() == Path(overlay["baseline_path"]).resolve()
    assert runs["baseline"]["source_sha256"] == overlay["baseline_sha256"]
    optimized_source = Path(runs["optimized"]["source_file"]).resolve()
    assert optimized_source.is_relative_to(Path(overlay["overlay_root"]).resolve())
    assert runs["optimized"]["source_sha256"] == overlay["replacement_sha256"]
    assert source_hash(overlay["baseline_path"]) == overlay["baseline_sha256"]


@pytest.mark.parametrize("name", [
    f"synthetic_unbalanced={unbalanced}_weights={weighted}_debias={debiased}"
    for unbalanced in (False, True) for weighted in (False, True) for debiased in (False, True)
] + [f"wage_panel_debias={debiased}" for debiased in (False, True)])
def test_full_panel_covariance_and_standard_errors_match(panel_results, name):
    _, runs = panel_results
    expected = runs["baseline"]["cases"][name]
    actual = runs["optimized"]["cases"][name]
    assert actual["columns"] == expected["columns"]
    assert actual["nobs"] == expected["nobs"]
    assert actual["df_resid"] == expected["df_resid"]
    # Coefficients must be exactly the same: only covariance aggregation changes.
    np.testing.assert_array_equal(actual["params"], expected["params"])
    np.testing.assert_allclose(actual["cov"], expected["cov"], rtol=2e-10, atol=2e-13)
    np.testing.assert_allclose(actual["stderr"], expected["stderr"],
                               rtol=2e-10, atol=2e-13, equal_nan=True)


def test_launcher_forwards_script_arguments_without_editing_installation():
    overlay = prepare_overlay()
    baseline_hash = source_hash(overlay["baseline_path"])
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory(prefix="panel-launcher-", dir=PROJECT_ROOT / ".cache") as temporary:
        temporary_path = Path(temporary).resolve()
        assert temporary_path.is_relative_to(PROJECT_ROOT)
        script = temporary_path / "verify_overlay.py"
        script.write_text("import json, sys, linearmodels\n"
                          "print(json.dumps({'file': linearmodels.__file__, 'args': sys.argv[1:]}))\n",
                          encoding="utf-8")
        result = subprocess.run([sys.executable, str(PROJECT_ROOT / "tools" / "run_optimized.py"),
                                str(script), "--model-label", "policy example"],
                                cwd=PROJECT_ROOT, env=environment, capture_output=True,
                                text=True, check=True, timeout=45)
    payload = json.loads(result.stdout)
    assert Path(payload["file"]).resolve().is_relative_to(Path(overlay["overlay_root"]).resolve())
    assert payload["args"] == ["--model-label", "policy example"]
    assert source_hash(overlay["baseline_path"]) == baseline_hash
