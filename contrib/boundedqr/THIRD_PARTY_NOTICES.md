# Third-party code and algorithm credits

BoundedQR project code is GPL-3.0-or-later. The unmodified PyFixest 0.60.0 kernel `src/boundedqr/_vendor/pyfixest_fn.py` is Copyright (c) 2022 pyfixest authors, MIT licensed, with its complete notice in `PYFIXEST_LICENSE`. Original SHA256: 54cb36544e902b549d344637698d82e3686eb99059df32073cf89d32eee50d38. Source: https://github.com/py-econometrics/pyfixest

The interface follows quantreg 6.1, maintained by Roger Koenker and its listed contributors (GPL >= 2), installed separately from CRAN. The finite pseudo-observation convention, baseline residual sign and bootstrap multipliers are preserved for fixed-input replay. Source: https://cran.r-project.org/package=quantreg

Established methods include Portnoy and Koenker (1997), Chernozhukov, Fernandez-Val and Melly (2022; online 2020) preprocessing, Hagemann (2017) clustered wild-gradient bootstrap, Moreau smoothing, semismooth Newton steps and primal-dual optimization. This contribution claims an implementation/integration, not invention of these estimators or methods.

NumPy, SciPy/HiGHS, threadpoolctl, optional CuPy, R and quantreg remain separate dependencies under their own licenses. No CUDA runtime, proprietary software, ACS data, or HPR-LP code is included.
