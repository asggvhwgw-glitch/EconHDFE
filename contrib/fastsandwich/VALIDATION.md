# Python contribution validation — 2026-10-02

Windows, Python 3.12, NumPy 2.5.3, SciPy 1.18.1, statsmodels 0.15.0, linearmodels 7.0; BLAS limited to one thread. 382 companion checks passed. EconHDFE 0.7.0 full suite: 1,068 passed with all companions installed in a fresh environment.

## Real public data

Seven-repeat medians after warmup; macro kernels use 100 calls per repeat. Data hashes and estimates are in [baseline.json](validation/baseline.json) and [optimized.json](validation/optimized.json).

| Data and timed operation | Baseline | FastSandwich | Result |
|---|---:|---:|---|
| US macrodata, 202 growth observations, HAC lag 4 | 23.5 µs | 127.3 µs | Slower for this short bandwidth |
| Same macrodata, HAC lag 100 | 359.2 µs | 128.6 µs | About 2.8× faster |
| wage_panel, 4,360 observations, two-way FE and clustering; full fit plus covariance | 27.97 ms | 16.79 ms | About 1.67× faster |

Macro covariance differences were at most 2.61e-18. Wage-panel coefficients were identical and covariance differed by at most 5.56e-17. These data are real bundled public datasets, not synthetic macro-shaped arrays. They illustrate descriptive associations, not causal treatment effects. Timing is machine-specific and does not establish speedups for EconHDFE's default kernels.

Reproduce from this directory:

```bash
python -m pip install ".[test]"
python validation/real_data.py --output .cache/baseline.json
python tools/run_optimized.py validation/real_data.py --output .cache/optimized.json
python -m pytest -q
```

The overlay verifies linearmodels 7.0 source and copies it privately; it does not modify the installed package. All contributed computation is Python with NumPy/SciPy. No R or Stata runtime is needed.
