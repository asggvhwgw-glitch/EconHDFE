from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from econhdfe import (
    DataTypeError, EconHDFEError, InputError, UnderidentifiedError,
    VariableRole, olshdfe, preflight_dataframe,
)
from econhdfe.resampling import run_replicates


def test_preflight_flags_numeric_text_but_allows_string_fe():
    df = pd.DataFrame({"y": [1.0, 2.0], "x": ["1.2", "2.3"], "firm": ["a", "b"]})
    report = preflight_dataframe(df, {"y": "outcome", "x": "regressor", "firm": "fixed_effect"})
    assert not report.ok
    assert [x.code for x in report.errors] == ["input.non_numeric"]
    assert report.errors[0].variable == "x"


def test_public_ols_reports_structured_non_numeric_regressor():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x": ["1", "2", "3", "4"], "firm": ["a", "a", "b", "b"]})
    with pytest.raises(DataTypeError) as err:
        olshdfe(df, y="y", x=["x"], absorb=["firm"], drop_singletons=False)
    payload = err.value.to_dict()
    assert payload["code"] == "input.non_numeric"
    assert payload["stage"] == "frontend"
    assert payload["details"]["variable"] == "x"


def test_structured_error_is_backward_compatible_value_error():
    e = InputError("bad", code="input.test", details={"x": 1})
    assert isinstance(e, ValueError)
    assert isinstance(e, EconHDFEError)
    assert e.to_dict()["details"] == {"x": 1}


def test_ivppml_underidentification_has_reason_code():
    from econhdfe import ivppmlhdfe
    n = 20
    y = np.ones(n)
    endog = np.ones((n, 2))
    z = np.ones((n, 1))
    with pytest.raises(UnderidentifiedError) as err:
        ivppmlhdfe(y, endog=endog, instruments=z, config=None)
    assert err.value.code == "identification.underidentified"
    assert err.value.details["n_endog"] == 2


def test_resampling_engine_counts_declared_failures_by_code():
    from econhdfe.errors import ConvergenceError
    seeds = np.random.SeedSequence(4).spawn(5)
    calls = {"i": 0}
    def worker(ss):
        calls["i"] += 1
        if calls["i"] in {2, 4}:
            raise ConvergenceError("nope", code="convergence.test")
        return calls["i"]
    batch = run_replicates(worker, seeds, n_jobs=1)
    assert batch.completed == 3 and batch.failed == 2
    assert batch.failure_counts == {"convergence.test": 2}


def test_resampling_does_not_hide_programming_errors():
    seeds = np.random.SeedSequence(4).spawn(1)
    def worker(ss):
        raise AttributeError("bug")
    with pytest.raises(AttributeError):
        run_replicates(worker, seeds, n_jobs=1)
