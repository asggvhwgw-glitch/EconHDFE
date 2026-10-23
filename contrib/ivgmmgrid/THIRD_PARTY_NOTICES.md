# Code and data provenance

The Python ivgmmgrid implementation is Copyright (c) 2026 So-mysterious, GPL-3.0-only; see LICENSE. It ports the statistical contract and moment-cache approach of the contributed Stata/Mata ivgmmgrid study. NumPy and SciPy are separately installed dependencies. No Stata binary or proprietary solver is distributed or required by the runtime.

Real-data validation is based on Quy-Toan Do and Hanan G. Jacoby (2024), *Optimal Pricing of a New Utility Service: The Case of Piped Water in Vietnam*, Zenodo v2, DOI https://doi.org/10.5281/zenodo.10965745, licensed [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Frozen household maps, sample keys and oracle results in validation/water are derived validation materials under CC-BY-4.0. Changes comprise selected bootstrap draws and stored estimates. Attribution does not imply endorsement. The full raw dataset is downloaded separately. Raw data SHA256: 13f63946ea39f462f1f43057ab3d79ecc5149d2a77367af042626fe6d9539142.

`validation/make_water_oracle.do` adapts the deposited CC-BY-4.0 bootstrap.do and Table2 preparation for validation. Its added exports capture Python replay inputs and oracle outputs. `validation/make_contract.do` generates synthetic test data and reference estimates. ivreghdfe 1.1.4, reghdfe 6.14.1, ivreg2 4.1.12, ftools, moremata and their dependencies remain separately installed upstream tools under their own licenses; use a licensed Stata installation only when regenerating oracles.

The frozen synthetic contract fixture contains no empirical microdata. The water example's preselected differencing lags and instruments are recorded explicitly, so it makes no claim to reimplement the deposited LASSO preselection pipeline.
