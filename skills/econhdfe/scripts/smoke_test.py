#!/usr/bin/env python3
"""Small deterministic installed-package smoke tests for econhdfe."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd


def main() -> int:
    try:
        from econhdfe import ivhdfe, olshdfe, ppmlhdfe

        n = 80
        i = np.arange(n)
        firm = i % 10
        year = (i // 10) % 8
        x = np.sin(i / 7.0) + 0.2 * (firm == 0)
        z = np.cos(i / 9.0) + 0.25 * (year == 0)
        endog = 0.8 * z + 0.3 * x + np.sin(i / 13.0)
        y = 1.5 * x + 0.7 * endog + 0.1 * firm - 0.05 * year + np.cos(i / 11.0)
        df = pd.DataFrame({"y": y, "x": x, "e": endog, "z": z, "firm": firm, "year": year})

        ols = olshdfe(df, y="y", x=["x"], absorb=["firm", "year"], vce="robust")
        iv = ivhdfe(
            df, y="y", exog=["x"], endog=["e"], instruments=["z"],
            absorb=["firm", "year"], vce="robust",
        )

        X = np.column_stack([np.ones(n), x])
        eta = 0.2 + 0.1 * x + 0.03 * (firm - firm.mean())
        count_y = np.maximum(0.0, np.round(np.exp(eta))).astype(float)
        ppml = ppmlhdfe(y=count_y, x=X, absorb=[firm, year], vce="robust", names=["const", "x"])

        checks = {
            "ols_finite": bool(np.all(np.isfinite(np.asarray(ols.params)))),
            "iv_finite": bool(np.all(np.isfinite(np.asarray(iv.params)))),
            "ppml_finite": bool(np.all(np.isfinite(np.asarray(ppml.params)))),
            "ols_nobs": int(ols.nobs),
            "iv_nobs": int(iv.nobs),
            "ppml_nobs": int(ppml.nobs),
        }
        if not all(checks[k] for k in ("ols_finite", "iv_finite", "ppml_finite")):
            raise RuntimeError("non-finite coefficient in smoke fit")
        print(json.dumps({"status": "pass", "checks": checks}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "fail", "error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
