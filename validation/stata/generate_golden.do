* econhdfe Stata golden-result generator.
* Run from the repository root after installing pinned reghdfe/ivreghdfe.
version 18
clear all
set more off

local data_path "validation/fixtures/golden_data.csv"
local out_path  "validation/fixtures/golden_results_stata.csv"
local env_path  "validation/fixtures/golden_environment_stata.log"

log using "`env_path'", text replace name(econhdfe_env)
about
which reghdfe
which ivreghdfe
which ivreg2
which ranktest
capture which ftools
log close econhdfe_env

import delimited using "`data_path'", clear varnames(1) numericcols(_all)
xtset firm year

tempname P
tempfile R
postfile `P' str40 model double bx sex bw sew covxw bcheck secheck N dfa widstat idstat using "`R'", replace

capture program drop post_current
program define post_current
    args handle model checkvar
    tempname dfa wid ids cov bc sec
    scalar `dfa' = .
    scalar `wid' = .
    scalar `ids' = .
    scalar `cov' = .
    scalar `bc' = .
    scalar `sec' = .
    capture scalar `dfa' = e(df_a)
    capture scalar `wid' = e(widstat)
    capture scalar `ids' = e(idstat)
    capture matrix __V = e(V)
    capture scalar `cov' = __V[colnumb(__V,"x"), colnumb(__V,"w")]
    if "`checkvar'" != "" {
        capture scalar `bc' = _b[`checkvar']
        capture scalar `sec' = _se[`checkvar']
    }
    post `handle' ("`model'") (_b[x]) (_se[x]) (_b[w]) (_se[w]) (`cov') (`bc') (`sec') (e(N)) (`dfa') (`wid') (`ids')
end

* Historical OLS/IV/weights/VCE corpus.
quietly reghdfe y x w, absorb(firm year)
post_current `P' ols_iid

quietly reghdfe y x w, absorb(firm year) vce(robust)
post_current `P' ols_robust

quietly reghdfe y x w, absorb(firm##c.trend year) vce(robust)
post_current `P' ols_slope_robust

quietly reghdfe y x w, absorb(firm year) vce(cluster c1 c2 c3)
post_current `P' ols_cluster3

quietly reghdfe y x w, absorb(firm year) vce(dkraay 4)
post_current `P' ols_dk

quietly reghdfe y x w [fweight=fw], absorb(firm year) vce(robust)
post_current `P' ols_fweight_robust

quietly reghdfe y x w [aweight=aw], absorb(firm year) vce(robust)
post_current `P' ols_aweight_robust

quietly reghdfe y x w [pweight=pw], absorb(firm year)
post_current `P' ols_pweight

quietly ivreghdfe y w (x=z1 z2), absorb(firm year) robust
post_current `P' iv_2sls_robust

quietly ivreghdfe y w (x=z1 z2), absorb(firm year) cluster(c1 c2)
post_current `P' iv_2sls_cluster2

quietly ivreghdfe y w (x=z1 z2), absorb(firm year) liml robust
post_current `P' iv_liml_robust

quietly ivreghdfe y w (x=z1 z2), absorb(firm year) gmm2s robust
post_current `P' iv_gmm2s_robust

quietly ivreghdfe y w (x=z1 z2) [fweight=fw], absorb(firm year) robust
post_current `P' iv_fweight_robust

quietly ivreghdfe y w (x=z1 z2) [aweight=aw], absorb(firm year) robust
post_current `P' iv_aweight_robust

quietly ivreghdfe y w (x=z1 z2) [pweight=pw], absorb(firm year)
post_current `P' iv_pweight

* v0.8 canonicalization: year and province#year are spanned by city#year.
quietly reghdfe y x w, absorb(firm year province#year city#year) vce(robust)
post_current `P' ols_canonical_hierarchy_robust

* Same estimand as ordinary two-way HDFE; Python must route method=auto to
* its specialized two-way Schur/PCG implementation.
quietly reghdfe y x w, absorb(firm year) vce(robust)
post_current `P' ols_twoway_auto_robust

* Explicit user omission from an exactly collinear x + w + x3 basis.
* Stata receives the algebraically equivalent active basis x + w.
quietly reghdfe y x w, absorb(firm year) vce(robust)
post_current `P' ols_user_omit_x3_robust

* Structural/post-absorption collinearity: year indicators are already in
* the span of city#year and must not perturb the identified x/w coefficients.
quietly reghdfe y x w i.year, absorb(firm city#year) vce(robust)
post_current `P' ols_structural_factor_robust

* Event-study reference. event_code=2 corresponds to event time -1.
* Record event_code=3 (event time 0) as an extra nontrivial coefficient check.
quietly reghdfe y x w ib2.event_code#c.ever, absorb(firm year) vce(robust)
post_current `P' ols_event_ref_robust "3.event_code#c.ever"

* Group-level outcomes with individual FEs. Unknown ivreghdfe options are
* forwarded to reghdfe's HDFE object in current ivreghdfe.
import delimited using "validation/fixtures/golden_group_individual.csv", clear varnames(1) numericcols(_all)

quietly reghdfe y x w, absorb(year inventor) group(patent) individual(inventor) aggregation(mean) vce(robust)
post_current `P' ols_group_ind_mean

quietly reghdfe y x w, absorb(year inventor) group(patent) individual(inventor) aggregation(sum) vce(robust)
post_current `P' ols_group_ind_sum

quietly ivreghdfe y w (x=z1 z2), absorb(year inventor) group(patent) individual(inventor) aggregation(mean) robust
post_current `P' iv_group_ind_mean

postclose `P'
use "`R'", clear
export delimited using "`out_path'", replace
noi di as result "Wrote Stata golden results to `out_path'"
