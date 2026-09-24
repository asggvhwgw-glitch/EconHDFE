* Self-contained data example; native dependencies must already be installed.
* Add package/stata to adopath before running this file.
* This example does not download data or install any software.
version 18.0
clear all
set more off
set seed 314159

capture which ivreghdfe
if _rc {
    display as error "Install the native ivreghdfe dependency stack before this example."
    exit 499
}
capture which ivgmmgrid
if _rc {
    display as error "Add this package's stata directory to adopath before this example."
    exit 499
}

set obs 500
generate long id = ceil(_n/10)
generate byte t = mod(_n-1,10)+1
generate double z1 = rnormal()
generate double z2 = rnormal()
generate double w = rnormal()
generate double cluster_u = rnormal()
bysort id (t): replace cluster_u = cluster_u[1]
generate double u = .3*cluster_u + rnormal()
generate double xbase = .9*z1 + .5*z2 + .3*w + .4*u + rnormal()
generate double lagterm = .2*z1 - .1*z2 + rnormal()
generate double y = 1.4*xbase + .7*w + .08*t + u

* Physical float candidates are intentionally generated with stored operations.
generate float c1 = xbase
generate float c2 = xbase
replace c2 = c2 + .25*lagterm
generate float c3 = xbase
replace c3 = c3 + .50*lagterm

ivgmmgrid y, exog(w) candidates(c1 c2 c3) instruments(z1 z2) ///
    absorb(t) cluster(id) verifyall

* Copy r() values before using commands that may overwrite them.
matrix G = r(grid)
local engine "`r(engine)'"
local fallback = r(fallback)
display "Engine: `engine'; fallback: `fallback'"
matrix list G

* Demonstration rule ONLY: lowest float-stored RMSE, earliest candidate on ties.
* A replication adapter must instead implement its own original tie rule.
local rmsecol = colnumb(G,"rmse")
local best 1
scalar best_rmse = float(G[1,`rmsecol'])
forvalues j=2/3 {
    if float(G[`j',`rmsecol']) < scalar(best_rmse) {
        local best `j'
        scalar best_rmse = float(G[`j',`rmsecol'])
    }
}
local candidate_list "c1 c2 c3"
local chosen : word `best' of `candidate_list'
display "Selected demonstration candidate: `chosen'"

* ivgmmgrid clears internal e(); obtain final e() explicitly from native code.
ivreghdfe y w (`chosen'=z1 z2), absorb(t) cluster(id) gmm2s
