* Synthetic native-parity and error/sample/data preservation checks.
* Requires native dependencies and this companion on adopath.
version 18
clear all
set more off
set type float
set processors 1
mata: mata mlib index

set obs 400
gen long rowkey = _n
gen int bsid = ceil(_n/10)
gen byte t = mod(_n-1,10)+1
gen double z1 = sin(rowkey*.17) + cos(bsid*.61)
gen double z2 = cos(rowkey*.29) + sin(bsid*.37)
gen double w = sin(rowkey*.71) + cos(t*.83)
gen double u = cos(rowkey*1.11) + sin(bsid*.43)
gen float x1 = .8*z1 - .5*z2 + .3*w + u
gen float x2 = x1 + .3*sin(rowkey*.53)
gen float x3 = x1
gen double y = .7*x1 + .2*w + .3*u + .1*cos(rowkey*1.31) + .05*t

* G01: if and in are intersected; verifyall returns native-compatible records.
quietly ivreghdfe y w (x1=z1 z2) if mod(rowkey,3)!=0 in 11/390, ///
    absorb(t) cluster(bsid) gmm2s
scalar native_n = e(N)
scalar native_g = e(N_clust)
scalar native_rmse = e(rmse)
quietly datasignature
local sig_before "`r(datasignature)'"
unab vars_before : _all
quietly ivgmmgrid y if mod(rowkey,3)!=0 in 11/390, ///
    exog(w) candidates(x1 x2 x3) instruments(z1 z2) absorb(t) cluster(bsid) verifyall
matrix GRID = r(grid)
matrix COUNTS = r(sample_counts)
assert r(verifyall) == 1
assert r(candidates) == 3
assert r(sample_common) == 1
assert r(sample_n) == native_n
assert inlist("`r(engine)'","moments","stock")
local ncol = colnumb(GRID,"N")
local gcol = colnumb(GRID,"N_clust")
local rmsecol = colnumb(GRID,"rmse")
assert rowsof(GRID) == 3
assert rowsof(COUNTS) == 3 & colsof(COUNTS) == 1
assert GRID[1,`ncol'] == native_n
assert GRID[1,`gcol'] == native_g
assert float(GRID[1,`rmsecol']) == float(native_rmse)
forvalues j=1/3 {
    assert COUNTS[`j',1] == GRID[`j',`ncol']
}
unab vars_after : _all
assert "`vars_before'" == "`vars_after'"
quietly datasignature
assert "`sig_before'" == "`r(datasignature)'"
assert rowkey == _n

* G02: different sample sizes must return all individual stock samples.
gen float xm1 = x1
replace xm1 = . in 1
quietly ivgmmgrid y, exog(w) candidates(x1 xm1) instruments(z1 z2) ///
    absorb(t) cluster(bsid) verifyall
matrix GRID = r(grid)
assert "`r(engine)'" == "stock"
assert r(fallback) == 1
assert r(sample_common) == 0
assert missing(r(sample_n))
local ncol = colnumb(GRID,"N")
assert GRID[1,`ncol'] == 400
assert GRID[2,`ncol'] == 399

* G03: equal N is insufficient: these masks omit different row IDs.
gen float xm2 = x2
replace xm2 = . in 2
quietly ivgmmgrid y, exog(w) candidates(xm1 xm2) instruments(z1 z2) ///
    absorb(t) cluster(bsid) verifyall
matrix GRID = r(grid)
assert "`r(engine)'" == "stock"
assert r(fallback) == 1
assert r(sample_common) == 0
assert missing(r(sample_n))
local ncol = colnumb(GRID,"N")
assert GRID[1,`ncol'] == 399 & GRID[2,`ncol'] == 399

* G04: fixed effects nested in clusters use native absorption DoF.
quietly ivreghdfe y w (x1=z1 z2), absorb(bsid) cluster(bsid) gmm2s
scalar nested_rmse = e(rmse)
matrix NATIVE_V = e(V)
quietly ivgmmgrid y, exog(w) candidates(x1 x2) instruments(z1 z2) ///
    absorb(bsid) cluster(bsid) verifyall
matrix GRID = r(grid)
assert r(sample_common) == 1
local rmsecol = colnumb(GRID,"rmse")
assert float(GRID[1,`rmsecol']) == float(nested_rmse)
forvalues a=1/2 {
    forvalues b=1/2 {
        local vcol = colnumb(GRID,"V_`a'_`b'")
        assert abs(GRID[1,`vcol']-NATIVE_V[`a',`b']) <= ///
            1e-10 + 1e-7*abs(NATIVE_V[`a',`b'])
    }
}

* G05: a late singular candidate triggers stock fallback, never pinv success.
* Native may omit a redundant column successfully or may error. Match that path.
gen double x_collinear = w
capture quietly ivreghdfe y w (x_collinear=z1 z2), absorb(t) cluster(bsid) gmm2s
local native_rc = _rc
if !`native_rc' {
    scalar singular_native_rmse = e(rmse)
}
quietly datasignature
local sig_before "`r(datasignature)'"
unab vars_before : _all
capture noisily ivgmmgrid y, exog(w) candidates(x1 x_collinear) ///
    instruments(z1 z2) absorb(t) cluster(bsid) verifyall
local candidate_rc = _rc
assert `candidate_rc' == `native_rc'
if !`candidate_rc' {
    matrix GRID = r(grid)
    assert "`r(engine)'" == "stock"
    assert r(fallback) == 1
    local rmsecol = colnumb(GRID,"rmse")
    assert float(GRID[2,`rmsecol']) == float(singular_native_rmse)
}
unab vars_after : _all
assert "`vars_before'" == "`vars_after'"
quietly datasignature
assert "`sig_before'" == "`r(datasignature)'"
assert rowkey == _n

* G06: empty if/in sample propagates native failure and cleans every tempvar.
capture quietly ivreghdfe y w (x1=z1 z2) if rowkey<0, absorb(t) cluster(bsid) gmm2s
local native_rc = _rc
assert `native_rc' != 0
quietly datasignature
local sig_before "`r(datasignature)'"
unab vars_before : _all
capture noisily ivgmmgrid y if rowkey<0, exog(w) candidates(x1 x2) ///
    instruments(z1 z2) absorb(t) cluster(bsid) verifyall
local candidate_rc = _rc
assert `candidate_rc' == `native_rc'
unab vars_after : _all
assert "`vars_before'" == "`vars_after'"
quietly datasignature
assert "`sig_before'" == "`r(datasignature)'"
assert rowkey == _n

* G07: API documents clearing internal e() rather than leaving stale e(sample).
assert "`e(cmd)'" == ""
display "IVGMMGRID_GENERIC_CONTRACT_OK"
