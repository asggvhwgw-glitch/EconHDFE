from __future__ import annotations

import ast
from pathlib import Path
import numpy as np
import pandas as pd


def test_legacy_linear_module_is_a_thin_compatibility_layer():
    from pyreghdfe.linear import fit_ols, fit_iv_2sls
    from econhdfe.models.ols import _fit_ols
    from econhdfe.models.linear_iv.estimators import fit_iv_2sls as new_fit_iv_2sls

    assert fit_ols is _fit_ols
    assert fit_iv_2sls is new_fit_iv_2sls
