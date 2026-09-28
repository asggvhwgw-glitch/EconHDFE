from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from econhdfe import IVHDFESession, OLSHDFESession, SpecificationError, preflight_dataframe


def _tiny_data():
    return pd.DataFrame({
        "y": np.arange(8, dtype=float),
        "x": np.linspace(0.0, 1.0, 8),
        "z": np.linspace(1.0, 2.0, 8),
        "e": np.linspace(0.5, 1.5, 8),
        "firm": [0, 0, 1, 1, 2, 2, 3, 3],
    })


def test_ols_session_bad_spec_is_structured_error():
    session = OLSHDFESession(_tiny_data())
    with pytest.raises(SpecificationError) as exc:
        session.fit_many([{"x": ["x"], "absorb": ["firm"]}])
    assert exc.value.code == "specification.session_spec"
    assert exc.value.stage == "specification"
    assert exc.value.details["missing"] == ["y"]


def test_iv_session_bad_spec_is_structured_error():
    session = IVHDFESession(_tiny_data())
    with pytest.raises(SpecificationError) as exc:
        session.fit_many([{"y": "y", "endog": ["e"]}])
    assert exc.value.code == "specification.session_spec"
    assert exc.value.details["missing"] == ["instruments"]


def test_preflight_invalid_level_uses_structured_error():
    with pytest.raises(SpecificationError) as exc:
        preflight_dataframe(_tiny_data(), {"y": "outcome"}, level="aggressive")
    assert exc.value.code == "specification.preflight_level"
    assert exc.value.stage == "frontend"
