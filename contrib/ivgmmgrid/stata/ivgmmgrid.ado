*! ivgmmgrid 0.1.0 16sep2026
* One-FE, one-cluster, unweighted two-step GMM over stored candidate columns.
* r(grid): candidate, b_candidate, b_<exog>, column-major V_i_j,
*          rss rmse N N_clust df_r hansen_j cond_omega rss_direct.
* r(sample_n) is missing when native sample masks differ across candidates.
* Execution errors restore input data and clear internal e() results.

program define ivgmmgrid, rclass
    version 18.0
    syntax varname(numeric) [if] [in],                         ///
        EXog(varlist numeric) CANDidates(varlist numeric)     ///
        INSTruments(varlist numeric) ABSorb(varname numeric)  ///
        CLuster(varname numeric) [VERIFYAll]

    tempname held_grid held_counts
    preserve
    capture noisily _ivgmmgrid_run `varlist' `if' `in',        ///
        exog(`exog') candidates(`candidates')                 ///
        instruments(`instruments') absorb(`absorb')           ///
        cluster(`cluster') `verifyall'
    local run_rc = _rc
    if !`run_rc' {
        matrix `held_grid' = r(grid)
        matrix `held_counts' = r(sample_counts)
        local held_engine "`r(engine)'"
        local held_reason "`r(fallback_reason)'"
        local held_fallback = r(fallback)
        local held_sample_n = r(sample_n)
        local held_sample_common = r(sample_common)
        local held_candidates = r(candidates)
        local held_verifyall = r(verifyall)
    }
    restore
    ereturn clear
    if `run_rc' exit `run_rc'

    return clear
    return matrix grid = `held_grid'
    return matrix sample_counts = `held_counts'
    return local engine "`held_engine'"
    return local fallback_reason "`held_reason'"
    return scalar fallback = `held_fallback'
    return scalar sample_n = `held_sample_n'
    return scalar sample_common = `held_sample_common'
    return scalar candidates = `held_candidates'
    return scalar verifyall = `held_verifyall'
end

program define _ivgmmgrid_run, rclass
    version 18.0
    syntax varname(numeric) [if] [in],                         ///
        EXog(varlist numeric) CANDidates(varlist numeric)     ///
        INSTruments(varlist numeric) ABSorb(varname numeric)  ///
        CLuster(varname numeric) [VERIFYAll]

    local y "`varlist'"
    local overlap : list candidates & exog
    if "`overlap'"!="" {
        display as error "ivgmmgrid: candidates() and exog() must not overlap: `overlap'"
        exit 198
    }
    local first : word 1 of `candidates'
    local k = 1 + wordcount("`exog'")
    local q = wordcount("`instruments'") + wordcount("`exog'")
    local nj : word count `candidates'
    local basecol = 1 + `k' + `k'*`k'
    local rmsecol = `basecol' + 2
    local ncol = `basecol' + 3
    local verified = ("`verifyall'"!="")

    tempvar base common anchor_sample
    tempname anchor stockrow fastgrid stockgrid finalgrid samplecounts compatible
    marksample base, novarlist
    quietly generate byte `common' = `base'
    quietly markout `common' `y' `exog' `instruments' `absorb' `cluster'

    * Do not mark out all candidates: preserve their individual native samples.
    local same_missing 1
    foreach c of local candidates {
        quietly count if `common' & (missing(`c') != missing(`first'))
        if r(N)>0 local same_missing 0
    }

    quietly ivreghdfe `y' `exog' (`first'=`instruments') if `base', ///
        absorb(`absorb') cluster(`cluster') gmm2s
    quietly generate byte `anchor_sample' = e(sample)
    local anchorn = e(N)
    local dminus = e(dofminus)
    local sdminus = e(sdofminus)
    local fullrank = (e(rankxx)==`k' & e(rankzz)==`q' & ///
                      e(rankS)==`q' & e(rankV)==`k')
    quietly _ivgmmgrid_stockrow, candidate(`first') exog(`exog') index(1)
    matrix `anchor' = r(row)

    local fast = `same_missing' & `fullrank'
    local reason ""
    if !`same_missing' local reason "candidate_missing_patterns"
    else if !`fullrank' local reason "native_rank_or_omission"

    if `fast' {
        capture mata: wgm_require(1,"")
        if _rc {
            capture quietly findfile water_gmm_moments.mata
            if _rc {
                local fast 0
                local reason "kernel_unavailable"
            }
            else {
                local kernelpath "`r(fn)'"
                capture quietly do "`kernelpath'"
                if _rc {
                    local fast 0
                    local reason "kernel_load_error"
                }
            }
        }
    }
    if `fast' {
        capture noisily mata: wgm_stata("`y'","`exog'","`candidates'", ///
            "`instruments'","`anchor_sample'","`absorb'","`cluster'", ///
            `dminus',`sdminus',1,1,"`fastgrid'")
        if _rc {
            local fast 0
            local reason "kernel_domain_or_rank"
        }
        else {
            mata: st_numscalar("`compatible'",                       ///
                _ivgg_close(st_matrix("`fastgrid'"),1,               ///
                            st_matrix("`anchor'"),`k'))
            if !scalar(`compatible') {
                local fast 0
                local reason "anchor_numeric_mismatch"
            }
            else if float(`fastgrid'[1,`rmsecol']) != float(`anchor'[1,`rmsecol']) {
                local fast 0
                local reason "anchor_float_rmse_mismatch"
            }
        }
    }

    local sample_common 1
    * verifyall computes a native oracle that can be reused on a failed check.
    if !`fast' | `verified' {
        local j 0
        foreach c of local candidates {
            local ++j
            if `j'==1 {
                matrix `stockrow' = `anchor'
            }
            else {
                quietly ivreghdfe `y' `exog' (`c'=`instruments') if `base', ///
                    absorb(`absorb') cluster(`cluster') gmm2s
                quietly count if e(sample) != `anchor_sample'
                if r(N)>0 local sample_common 0
                quietly _ivgmmgrid_stockrow, candidate(`c') exog(`exog') index(`j')
                matrix `stockrow' = r(row)
            }
            matrix `stockgrid' = nullmat(`stockgrid') \ `stockrow'
            if `fast' & `verified' {
                mata: st_numscalar("`compatible'",                       ///
                    _ivgg_close(st_matrix("`fastgrid'"),`j',             ///
                                st_matrix("`stockrow'"),`k'))
                if !scalar(`compatible') {
                    local fast 0
                    local reason "verifyall_numeric_mismatch"
                }
                else if float(`fastgrid'[`j',`rmsecol']) != float(`stockrow'[1,`rmsecol']) {
                    local fast 0
                    local reason "verifyall_float_rmse_mismatch"
                }
                else if !`sample_common' {
                    local fast 0
                    local reason "verifyall_sample_mismatch"
                }
            }
        }
    }

    if `fast' {
        matrix `finalgrid' = `fastgrid'
        local engine "moments"
        local fallback 0
        local sample_common 1
        local samplen = `anchorn'
    }
    else {
        matrix `finalgrid' = `stockgrid'
        local engine "stock"
        local fallback 1
        local samplen .
        if `sample_common' local samplen = `anchorn'
    }
    matrix `samplecounts' = `finalgrid'[1...,`ncol']
    matrix colnames `samplecounts' = N
    return matrix grid = `finalgrid'
    return matrix sample_counts = `samplecounts'
    return local engine "`engine'"
    return local fallback_reason "`reason'"
    return scalar fallback = `fallback'
    return scalar sample_n = `samplen'
    return scalar sample_common = `sample_common'
    return scalar candidates = `nj'
    return scalar verifyall = `verified'
end

program define _ivgmmgrid_stockrow, rclass
    version 18.0
    syntax , CANDidate(varname numeric) EXog(varlist numeric) INDEX(integer)
    tempname bb vv row omega_condition
    matrix `bb' = e(b)
    matrix `vv' = e(V)
    local coefvars "`candidate' `exog'"
    local k : word count `coefvars'
    local width = 1 + `k' + `k'*`k' + 8
    matrix `row' = J(1,`width',.)
    matrix `row'[1,1] = `index'
    local positions
    local names "candidate b_candidate"
    local i 0
    foreach v of local coefvars {
        local ++i
        local pos = colnumb(`bb',"`v'")
        if missing(`pos') | `pos'<=0 local pos = colnumb(`bb',"o.`v'")
        if missing(`pos') | `pos'<=0 {
            display as error "ivgmmgrid: native coefficient `v' cannot be mapped to the grid schema"
            exit 498
        }
        local positions "`positions' `pos'"
        matrix `row'[1,1+`i'] = `bb'[1,`pos']
        if `i'>1 {
            local bn = substr("b_`v'",1,32)
            local names "`names' `bn'"
        }
    }
    local col = 1 + `k'
    forvalues j=1/`k' {
        local pj : word `j' of `positions'
        forvalues i=1/`k' {
            local pi : word `i' of `positions'
            local ++col
            matrix `row'[1,`col'] = `vv'[`pi',`pj']
            local names "`names' V_`i'_`j'"
        }
    }
    scalar `omega_condition' = .
    capture mata: st_numscalar("`omega_condition'",cond(st_matrix("e(S)")))
    foreach stat in rss rmse N N_clust df_r j {
        local ++col
        matrix `row'[1,`col'] = e(`stat')
    }
    local ++col
    matrix `row'[1,`col'] = scalar(`omega_condition')
    local ++col
    matrix `row'[1,`col'] = 1
    local names "`names' rss rmse N N_clust df_r hansen_j cond_omega rss_direct"
    matrix colnames `row' = `names'
    return matrix row = `row'
end

mata:
real scalar _ivgg_close(real matrix grid, real scalar j, real matrix native,
                       real scalar k)
{
    real rowvector f, n
    real scalar last, i, b
    f = grid[j,.]
    n = native[1,.]
    if (cols(f)!=cols(n)) return(0)
    b = 1+k+k*k
    last = b+2
    if (hasmissing(f[2..last]) | hasmissing(n[2..last])) return(0)
    for (i=2; i<=last; i++) {
        if (abs(f[i]-n[i]) > 1e-10+1e-8*abs(n[i])) return(0)
    }
    for (i=b+3; i<=b+5; i++) {
        if (f[i]!=n[i]) return(0)
    }
    i = b+6
    if (missing(f[i]) | missing(n[i])) return(f[i]==n[i])
    if (abs(f[i]-n[i]) > 1e-10+1e-8*abs(n[i])) return(0)
    return(1)
}
end
