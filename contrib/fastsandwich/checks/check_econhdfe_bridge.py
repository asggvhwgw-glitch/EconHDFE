import importlib.util
from pathlib import Path
import pytest
pytest.importorskip("econhdfe")

def test_explicit_bandwidth_and_normalization():
    path = Path(__file__).resolve().parents[1] / "examples/econhdfe_scores.py"
    spec = importlib.util.spec_from_file_location("score_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.compare()
