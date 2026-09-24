{smcl}
{* *! version 0.1.0 16sep2026}{...}
{vieweralsosee "ivreghdfe" "help ivreghdfe"}{...}
{title:Title}

{p 4 4 2}
{bf:ivgmmgrid} {hline 2} Evaluate stored candidate regressors with clustered
two-step GMM and shared fixed-effect preparation

{title:Syntax}

{p 8 12 2}
{cmd:ivgmmgrid} {it:depvar} {ifin},
{cmd:exog(}{it:varlist}{cmd:)}
{cmd:candidates(}{it:varlist}{cmd:)}
{cmd:instruments(}{it:varlist}{cmd:)}
{cmd:absorb(}{it:varname}{cmd:)}
{cmd:cluster(}{it:varname}{cmd:)}
[{cmd:verifyall}]

{p 4 4 2}
Stata 18 or later and an installed, working {cmd:ivreghdfe} stack are required.
Place {cmd:ivgmmgrid.ado}, this help file, and {cmd:water_gmm_moments.mata}
together on the ado-path. No installation or network access occurs when the
command runs.

{title:Description}

{p 4 4 2}
For each variable {it:c} listed in {cmd:candidates()}, the command evaluates the
model

{p 8 12 2}
{cmd:ivreghdfe} {it:depvar exog} {cmd:(}{it:c}{cmd:=}{it:instruments}{cmd:)}
{ifin}, {cmd:absorb(}{it:absorb}{cmd:)}
{cmd:cluster(}{it:cluster}{cmd:)} {cmd:gmm2s}

{p 4 4 2}
Each candidate is a separate model's single endogenous regressor.
The supplied candidate columns are read as stored. The command does not
construct a gamma grid, combine lag variables, optimize an objective, select a
winner, or construct bootstrap confidence intervals. In particular, it does
not replace a sequence of float-storage {cmd:replace} operations with a
double-precision polynomial evaluation.

{p 4 4 2}
The first model is always estimated by native {cmd:ivreghdfe}. Its
{cmd:e(sample)}, degrees of freedom, ranks, coefficients, covariance and RMSE
anchor the prepared calculation. Eligible models share one within-FE
transformation and global and cluster cross-products. Every candidate still
gets a newly computed initial 2SLS estimate, first-step clustered moment
covariance, and efficient second-step GMM estimate.

{title:Options and supported domain}

{phang}
{cmd:exog(}{it:varlist}{cmd:)} specifies one or more fixed included exogenous
variables. These also enter the instrument set. They must not overlap with
{cmd:candidates()}; overlap is rejected with error 198.

{phang}
{cmd:candidates(}{it:varlist}{cmd:)} specifies physical numeric columns in
the desired model order. Candidate names do not supply parameter values.
Maintain a separate parameter vector if the columns represent a grid.

{phang}
{cmd:instruments(}{it:varlist}{cmd:)} specifies excluded instruments.
{cmd:absorb()} and {cmd:cluster()} each take one numeric variable.
All lists must use physical numeric variable names: expand factor or
time-series expressions beforehand.

{phang}
{cmd:verifyall} additionally estimates every model with native
{cmd:ivreghdfe} and compares coefficients, all covariance elements, RSS, RMSE,
Hansen J, sample masks and inference metadata. Any incompatibility returns
the entire native grid. This option is for validation and is not included in
default performance claims.

{p 4 4 2}
The prepared engine covers unweighted, full-rank, one-way uncentered clustered
two-step GMM, one numeric intercept FE, and one endogenous variable per model.
It uses the efficient-GMM covariance based on first-step residuals and native
small-sample corrections. The command does not accept weights, multiple FEs,
multiple cluster variables, slopes, factor-variable expressions, multiple
simultaneous endogenous regressors, HAC, centered moments, LIML or CUE.
Unsupported syntax is rejected.

{title:Fallback and numeric checks}

{p 4 4 2}
Missing-value patterns are compared only on the domain where the fixed model
inputs and if/in restriction are available. Candidates with different missing
patterns are estimated separately by native {cmd:ivreghdfe}. The command never
forces every candidate onto the intersection of candidate-complete samples.

{p 4 4 2}
Native omissions or incompatible ranks, missing kernel availability, a kernel
failure, or a failed native comparison select the native engine for the whole
batch. Each native model uses its own sample. Native estimation errors stop
the helper and propagate their return code; unsuccessful models are not
silently dropped, regularized, replaced by 2SLS, or assigned invented results.

{p 4 4 2}
The coefficient/covariance/statistic comparison uses
{cmd:abs(fast-native) <= 1e-10 + 1e-8*abs(native)}.
Row counts, cluster counts and inference degrees of freedom must agree
exactly. Compared RMSE values must also agree after Stata {cmd:float()}
rounding. The default checks only the first candidate against native output.
It is {bf:not a certificate} of every candidate's rounded objective or of an
unchanged near-tie decision. {cmd:verifyall} performs the additional pointwise
native comparison.

{p 4 4 2}
Returned matrix values remain double precision. Reproduce the application's
original storage type, parameter vector, objective exclusions and tie rule
when selecting a winner. Always refit the chosen model with native
{cmd:ivreghdfe} to obtain the final {cmd:e()} results and postestimation support.

{title:Stored results}

{p 4 4 2}
{cmd:ivgmmgrid} is r-class. It restores the input data, variable storage types
and observation order. Internal {cmd:e()} results are cleared on success and
on errors during execution, so the first anchor cannot be mistaken for the
selected model. Save or refit any earlier estimates that you need.

{synoptset 25 tabbed}{...}
{synopt:{cmd:r(grid)}}one row per candidate, in input order{p_end}
{synopt:{cmd:r(sample_counts)}}one-column matrix of each model's native or matched N{p_end}
{synopt:{cmd:r(engine)}}{cmd:moments} or {cmd:stock}{p_end}
{synopt:{cmd:r(fallback)}}0 for moments, 1 for the native grid{p_end}
{synopt:{cmd:r(fallback_reason)}}diagnostic string; empty for the moments engine{p_end}
{synopt:{cmd:r(sample_common)}}1 when actual model sample masks are common, otherwise 0{p_end}
{synopt:{cmd:r(sample_n)}}common sample N; missing if actual masks differ, even if their counts coincide{p_end}
{synopt:{cmd:r(candidates)}}number of evaluated candidate columns{p_end}
{synopt:{cmd:r(verifyall)}}1 if the option was requested; it is not a standalone certification flag{p_end}

{p 4 4 2}
Let k = 1 + the number of included exogenous variables.
{cmd:r(grid)} has 1+k+k*k+8 columns:

{p 8 12 2}
{cmd:candidate}: one-based candidate index{break}
{cmd:b_candidate}, then {cmd:b_}{it:exog}: coefficients, endogenous first{break}
{cmd:V_1_1 V_2_1 ... V_k_1 V_1_2 ... V_k_k}:
the complete covariance matrix stacked by columns{break}
{cmd:rss rmse N N_clust df_r hansen_j cond_omega rss_direct}

{p 4 4 2}
{cmd:cond_omega} describes the first-step clustered moment covariance; the
native path may return missing if that matrix cannot be read.
{cmd:rss_direct}=1 identifies direct residual RSS.
With two exogenous controls there are 21 columns: b in columns 2-4, V in 5-13,
RSS in 14 and RMSE in 15. Prefer column names to hardcoded positions.
Long exogenous names may require a truncated {cmd:b_} column label on the
native fallback path; retain the original exogenous list as the mapping.

{title:Example}

{p 4 4 2}
After adding the package directory to the ado-path and installing the native
dependencies:

{phang2}{cmd:ivgmmgrid y, exog(w) candidates(c1 c2 c3) instruments(z1 z2) absorb(t) cluster(id) verifyall}{p_end}
{phang2}{cmd:matrix G = r(grid)}{p_end}
{phang2}{cmd:matrix list G}{p_end}

{p 4 4 2}
See {cmd:contrib/ivgmmgrid/examples/ivgmmgrid_example.do} for an example that generates
its own data, evaluates a grid, applies an explicitly specified demonstration
tie rule and refits the winner. Its selection rule is illustrative, not a
replacement for a replication package's rule.

{title:Provenance}

{p 4 4 2}
This is an experimental computational helper for existing {cmd:ivreghdfe}
two-step GMM, not a new estimator. The initial implementation was checked
against ivreghdfe 1.1.4, commit
bfb5577a6dbdfb029ab4ab6a7e93f7257a827b42.
Different dependency versions require renewed parity checks.
