# External validation

The local Python suite is necessary but not sufficient for release certification.

1. Run `python validation/generate_fixture.py`.
2. On a licensed Stata installation with current `ppmlhdfe`, `reghdfe`, and `ftools`, run:
   `do validation/stata/generate_golden.do validation`
3. Run `python validation/compare_golden.py` from an environment containing this source tree (installed as `econhdfe`).

The golden matrix covers no-FE, one-FE, two-FE clustered, genuine three-FE two-way-clustered, and exposure specifications. The Stata script also exports the official five-observation ReLU-primer separation mask.

For 3+ FEs, raw `df_a` is reported but is not treated as a universal equality gate because exact structural rank and `reghdfe`'s pairwise DoF convention can differ while coefficients and score equations remain identical. Coefficients, VCE/SE, estimation sample, and separated-observation count remain hard gates.

## Upstream 17-dataset separation suite

On an internet-enabled host:

```bash
python validation/validate_separation_suite.py --download
```

The script downloads the official `01.csv` ... `17.csv` files from the upstream
`guides/separation_datasets/` directory and requires exact observation-level
agreement with the supplied `separated` indicator.
