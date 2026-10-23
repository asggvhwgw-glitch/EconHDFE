* Validation adaptation of Do and Jacoby (2024), Zenodo v2, CC-BY-4.0.
* See THIRD_PARTY_NOTICES.md. Run in a fresh scratch output directory.
args data_root vendor_root seed
version 18
clear all
set more off
assert "`c(type)'"=="float"
set processors 1
capture log close _all
log using "benchmark.log", text replace
adopath ++ "`vendor_root'/require_upstream/source/src"
adopath ++ "`vendor_root'/ftools_upstream/source/src"
adopath ++ "`vendor_root'/reghdfe_upstream/source/src"
adopath ++ "`vendor_root'/moremata_upstream/source"
adopath ++ "`vendor_root'/ivreg2_ssc"
adopath ++ "`vendor_root'/ranktest_ssc"
adopath ++ "`vendor_root'/ivreghdfe_upstream/source/src"
adopath ++ "`vendor_root'/lassopack_ssc"
adopath ++ "`vendor_root'/pdslasso_ssc"
adopath ++ "`vendor_root'/avar_ssc"
adopath ++ "`data_root'"
global WATER_GRID_ENGINE "original"
global WATER_GRID_AUDIT "1"
mata: mata mlib index
which ivreghdfe
which ivreg2
which ivlasso
which rlasso
which ranktest
* Original Table2 preparation, including default float storage and original order.
use "`data_root'/Usage_July15_May18.dta", clear
replace Nd = Nd/1000
keep id t lnC lnC_1 p Nd lnP
forvalues j=2(1)4 {
    qui bys id: gen lnC_`j'=lnC[_n-`j']
}
xtset id t
save temp_monthly, replace
global reps "1"
set seed `seed'

postfile __study_final int draw double Lstar double lag double gamma ///
    double N double G double rmse double rss double df_r ///
    double b_x double b_p double b_nd double v_xx double v_xp double v_xnd ///
    double v_pp double v_pnd double v_ndnd using "final_audit.dta", replace

postfile __study_grid int draw int n double gamma double N double G ///
    double rmse double rss double df_r double b_x double b_p double b_nd ///
    double v_xx double v_xp double v_xnd double v_pp double v_pnd double v_ndnd ///
    using "grid_audit.dta", replace

timer clear

timer on 1


/**********************************************

Bootstrap for CIs in Table 2

Includes:

L* selection
IV LASSO selection
tau selection
linear IV estimation (alpha, beta, omega)
NLLS estimation (gamma)

BC CIs for alpha, beta, gamma, omega
Bootstap p-value for test of SR elast = LR elast
**********************************************/

****************************************************************************************
***CODE FOR GAMMA ESTIMATION
****************************************************************************************


capture program drop distlag_bs
program distlag_bs, eclass

args n m

local iv = "iv`m'"

tempvar X a1 a2
qui {
gen `X'=0
gen `a1'=0
gen `a2'=0

local j=1
forvalues k=0.05(.01).7 {
replace `X'= DlnC_1
replace `X'= `X'+`k'*DlnC_2 if `n'>=2
replace `X'= `X'+`k'^2*DlnC_3 if `n'>=3
replace `X'= `X'+`k'^3*DlnC_4 if `n'>=4

qui ivreghdfe DlnC Dp DNd (`X'= $`iv'), cluster(bsid) ab(t) gmm2s

replace `a1'=`k' if _n==`j'
replace `a2'=e(rmse) if _n==`j'
matrix __study_b=e(b)
matrix __study_v=e(V)
post __study_grid ($STUDY_DRAW) (`n') (`k') (e(N)) (e(N_clust)) ///
    (e(rmse)) (e(rss)) (e(df_r)) ///
    (__study_b[1,1]) (__study_b[1,2]) (__study_b[1,3]) ///
    (__study_v[1,1]) (__study_v[1,2]) (__study_v[1,3]) ///
    (__study_v[2,2]) (__study_v[2,3]) (__study_v[3,3])

local j = `j' +1
}
su `a2' if `a2'>0
su `a1' if `a2'==r(min)
local kk = r(mean)
scalar gam = `kk'
replace `X'= DlnC_1
replace `X'= `X'+`kk'*DlnC_2 if `n'>=2
replace `X'= `X'+`kk'^2*DlnC_3 if `n'>=3
replace `X'= `X'+`kk'^3*DlnC_4 if `n'>=4
}

qui ivreghdfe DlnC Dp DNd (`X'= $`iv'), cluster(bsid) ab(t)   gmm2s

end


capture program drop grid_tg_bs
program grid_tg_bs, eclass

tempvar gg ll r2
gen `gg'=.
gen `ll'=.
gen `r2'=.

preserve

forvalues n=1(1)4 {
distlag_bs `n' 0

qui replace `ll'=`n' if _n==`n'
qui replace `r2'=e(rmse)  if _n==`n'
qui replace `gg'=scalar(gam) if _n==`n'
}


** gamma estimation
qui{
su `r2'
su `ll' if `r2'==r(min)
local lag = r(max)
su `r2'
su `gg' if `r2'==r(min)
local kk = r(max)


scalar gam = `kk'
scalar T = `lag'
gen X= DlnC_1
replace X= X+`kk'*DlnC_2 if `lag'>=2
replace X= X+`kk'^2*DlnC_3 if `lag'>=3
replace X= X+`kk'^3*DlnC_4 if `lag'>=4

}

** for correct sample
qui ivreghdfe DlnC Dp DNd (X= $iv0) if DlnC_4~=.,  cluster(bsid) ab(t) gmm2s

matrix __study_b=e(b)
matrix __study_v=e(V)
post __study_final ($STUDY_DRAW) ($STUDY_LSTAR) (T) (gam) ///
    (e(N)) (e(N_clust)) (e(rmse)) (e(rss)) (e(df_r)) ///
    (__study_b[1,1]) (__study_b[1,2]) (__study_b[1,3]) ///
    (__study_v[1,1]) (__study_v[1,2]) (__study_v[1,3]) ///
    (__study_v[2,2]) (__study_v[2,3]) (__study_v[3,3])
export delimited id bsid t if e(sample) using "sample_draw$STUDY_DRAW.csv", replace

end

****************************************************************************************
***BOOTSTRAP EXECUTION
****************************************************************************************
set more 1


***************************************************************************
** SET BS REPS

local R = $reps

** initialize BS output datasets
preserve
 clear
 set obs 1
 foreach x in a b o L {
 gen double `x'star = .
 }

 save BS_full_linear, replace
 restore

 preserve
 clear
 set obs 1
 foreach x in a b g o L t {
 gen double `x'star = .
 }

 save BS_full_NL, replace
 restore


forvalues r=1(1)`R' {
di "`r'"
qui{

use temp_monthly, clear
bsample, cluster(id) idcluster(bsid)
global STUDY_DRAW `r'
export delimited id bsid t using "draw_map$STUDY_DRAW.csv", replace
timer on 2

gen F = .
gen lag=.


*** Basic IVs
* loop through lag lengths

forvalues t=2(1)22 {
capture drop D*
foreach v of varlist lnC lnC_1 p Nd lnP {
qui bys bsid (t): gen D`v'= `v' - `v'[_n-`t']
}

dis `t'

replace lag = `t' if _n==`t'

xtset bsid t
ivreg2 DlnC Dp DNd (DlnC_1 = L(1).Dp  L(1).DlnP) i.t, r cluster(bsid) gmm2s
local IVfinal = e(exexog)
F_eff
replace F = r(F_eff) if _n==`t'
}



** L* selection-- basic IVs

 egen maxF = max(F) if F~=.
 su F
 local mF = r(max)
 su lag if maxF==F&F~=.
 local Lstar = r(mean)

**noisily di "`Lstar'" " Fstat =  `mF'"

**  LASSO @L* w/ k =2 lags

capture drop D*
foreach v of varlist lnC lnC_1 p Nd lnP {
qui bys bsid (t): gen D`v'= `v' - `v'[_n-`Lstar']
}

ivlasso DlnC Dp DNd  (DlnC_1 =L(1/2).Dp   L(1/2).DlnP ) i.t, r cluster(bsid) nocons
local IVs = e(xzselected_dend)


** now check redundancy of L2
** skip if no IVs selected
if  "`IVs'"~="." {

local IV2 = "L2.Dp L2.DlnP"
local IV1: list IVs & IV2   /** pick out L2 IVs in lasso chosen list **/

**noisily di "`IV1'"

if "`IV1'"~="" {

ivreg2 DlnC Dp DNd (DlnC_1 =`e(xzselected_dend)') i.t, r cluster(bsid) gmm2s redundant(`IV1')
local pp = e(redp)

F_eff

replace F=.
drop maxF
** redo lag selection if new IVs give better fit
** NB: only if we get bigger F-stat AND L2 IVs are NOT redundant do we go  a 2nd round

if (r(F_eff) > `mF')&(`pp' < .05) {

forvalues t=2(1)22 {
capture drop D*
foreach v of varlist lnC lnC_1 p Nd lnP {
qui bys bsid (t): gen D`v'= `v' - `v'[_n-`t']
}

dis `t'

replace lag = `t' if _n==`t'

xtset bsid t
ivreg2 DlnC Dp DNd (DlnC_1 = `IVs')  i.t, r cluster(bsid) gmm2s
local IVfinal = "`IVs'"
F_eff
replace F = r(F_eff) if _n==`t'
  }

 ** L* selection  (only change if above conditions are met)

 egen maxF = max(F) if F~=.
 su F
 local mF = r(max)
 su lag if maxF==F&F~=.
 local Lstar = r(mean)

}
}
} /* close first if */

 ** Now go to gamma/tau estimation with L* and final IV set

capture drop D*

foreach v of varlist lnC lnC_* p Nd lnP  {
qui bys bsid (t): gen D`v'= `v' - `v'[_n-`Lstar']
}

global iv0  "`IVfinal'"


** output for tau=1 model

ivreghdfe DlnC Dp DNd (DlnC_1 =`IVfinal'),  ab(t) cluster(bsid)  gmm2s

preserve
gen bstar = _b[DlnC_1]
gen astar = _b[Dp]
gen ostar = _b[DNd]


gen Lstar = `Lstar'

keep *star
keep if _n==1
append using BS_full_linear
save BS_full_linear, replace
restore

** output for tau=1 model


timer off 2
global STUDY_LSTAR `Lstar'

preserve
tsrevar $iv0
local __cols `r(varlist)'
local __i=0
foreach __v of local __cols {
    local ++__i
    gen double iv_`__i' = `__v'
}
keep id bsid t DlnC DlnC_1 DlnC_2 DlnC_3 DlnC_4 Dp DNd iv_*
save "python_input$STUDY_DRAW.dta", replace
restore
file open __meta using "python_input$STUDY_DRAW.txt", write replace
file write __meta "Lstar=$STUDY_LSTAR" _n "instruments=$iv0" _n
file close __meta

timer on 3
grid_tg_bs
timer off 3


gen bstar = _b[X]
gen astar = _b[Dp]
gen ostar = _b[DNd]
gen gstar = gam
gen Lstar= `Lstar'
gen tstar= T


keep *star
keep if _n==1
append using BS_full_NL
save BS_full_NL, replace
}
}



timer off 1
postclose __study_final
postclose __study_grid
file open __rng using "rngstate.txt", write replace
file write __rng "`c(rngstate)'" _n
file close __rng

timer list
tempname f
file open `f' using "timings.csv", write replace
file write `f' "engine,replications,seed,processors,seconds,preselection_s,grid_s" _n
file write `f' "original,1,1,1," (r(t1)) "," (r(t2)) "," (r(t3)) _n
file close `f'
use BS_full_NL, clear
list, noobs
assert _N == 2
assert !missing(astar,bstar,gstar,Lstar,tstar) if _n<_N
display "STUDY_CONFIRMATORY_OK"
log close
exit, clear
