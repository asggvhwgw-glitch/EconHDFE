# Licenses and external dependencies

Original ivgmmgrid source is GPL-3.0-only, Copyright (c) 2026 ivgmmgrid contributors. See LICENSE. The main Mata implementation is shipped as source, not an opaque compiled binary.

Native comparison requires Stata 18+ (licensed separately) and ivreghdfe with its dependency stack. None of those binaries or upstream sources is redistributed here. The checked native stack is documented in VALIDATION.md. Install using the upstream instructions at https://github.com/sergiocorreia/ivreghdfe ; pin versions for research replication.

ivreghdfe carries Sergio Correia's MIT notice and embeds ivreg2-derived code. ivreg2 and ranktest have GPL-v3 provenance; their authors include Christopher F. Baum, Mark E. Schaffer, Steven Stillman, Frank Kleibergen and Frank Windmeijer as applicable. reghdfe, ftools and require carry their own MIT notices. The MIT badge on one repository does not relicense inherited ivreg2 code. Preserve original notices when obtaining or redistributing those dependencies. This companion does not claim to resolve upstream rights history.

Two-step efficient GMM, clustered covariance and within-FE transformation are established econometric methods. This code reuses their computation and checks against native results; it does not claim a new estimator.
