from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from econhdfe import DataTypeError, UnderidentifiedError, olshdfe
from econhdfe.resampling import run_replicates


def test_public_ols_reports_structured_non_numeric_regressor():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x": ["1", "2", "3", "4"], "firm": ["a", "a", "b", "b"]})
    with pytest.raises(DataTypeError) as err:
        olshdfe(df, y="y", x=["x"], absorb=["firm"], drop_singletons=False)
    payload = err.value.to_dict()
    assert payload["code"] == "input.non_numeric"
    assert payload["stage"] == "frontend"
    assert payload["details"]["variable"] == "x"


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


def test_resampling_does_not_hide_programming_errors():
    seeds = np.random.SeedSequence(4).spawn(1)
    def worker(ss):
        raise AttributeError("bug")
    with pytest.raises(AttributeError):
        run_replicates(worker, seeds, n_jobs=1)
