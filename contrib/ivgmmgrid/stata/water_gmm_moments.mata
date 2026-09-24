version 15.1
mata:

/*
    Prepared one-way-clustered two-step GMM over exact stored candidate columns.
    Each model has X = [candidate_j, fixed_exog], Z = [excluded, fixed_exog].
    Candidate construction, e(sample), FE singleton removal and final selection
    belong to the caller. No polynomial approximation is used.
*/

struct wgm_state {
    real scalar N, G, p, J, q, dofminus, sdofminus, small
    real matrix y, E, C, H, zy, ze, zc, EE, EC, Ey, Cy, cc
    real matrix cm, Hinvzy
    real scalar yy
    string rowvector enames, cnames
}

void wgm_require(real scalar ok, string scalar message)
{
    if (!ok) {
        errprintf("water_gmm_moments: %s\n", message)
        exit(198)
    }
}

real matrix wgm_fullsolve(real matrix A, real matrix B, string scalar label)
{
    real matrix Ai, answer
    Ai = invsym(makesymmetric(A))
    if (hasmissing(Ai) | diag0cnt(Ai)>0) {
        errprintf("water_gmm_moments: rank failure in %s; use native fallback\n", label)
        exit(506)
    }
    answer = cholsolve(makesymmetric(A), B)
    // A full-rank symmetric check and Cholesky must both succeed.
    // A QR fallback can accept a generalized solution under a different
    // numerical rank threshold; delegate that case to native ivreghdfe.
    if (hasmissing(answer)) {
        errprintf("water_gmm_moments: solve failure in %s; use native fallback\n", label)
        exit(506)
    }
    return(answer)
}

real matrix wgm_demean(real matrix D, real colvector fe)
{
    real colvector ord
    real matrix info, work
    real rowvector mu
    real scalar g, first, last
    ord = order(fe, 1)
    work = D[ord,.]
    info = panelsetup(fe[ord], 1)
    for (g=1; g<=rows(info); g++) {
        first = info[g,1]
        last = info[g,2]
        wgm_require(last>first, "touse retains an FE singleton; use native e(sample)")
        mu = quadcolsum(work[first..last,.]) / (last-first+1)
        work[first..last,.] = work[first..last,.] :- mu
    }
    D[ord,.] = work
    return(D)
}

struct wgm_state scalar wgm_prepare(
    string scalar yname,
    string scalar exog,
    string scalar candidates,
    string scalar excluded,
    string scalar touse,
    string scalar fevar,
    string scalar clustvar,
    real scalar dofminus,
    real scalar sdofminus,
    real scalar small)
{
    struct wgm_state scalar s
    real matrix D, Z, B, info, W
    real colvector fe, cl, ord
    real scalar ecount, wcount, ccount, g, first, last, lo, hi

    s.enames = tokens(exog)
    s.cnames = tokens(candidates)
    ecount = cols(s.enames)
    wcount = cols(tokens(excluded))
    ccount = cols(s.cnames)
    wgm_require(cols(tokens(yname))==1, "one dependent variable is required")
    wgm_require(ccount>0 & wcount>0, "candidate and excluded-IV lists cannot be empty")
    wgm_require(cols(tokens(clustvar))==1, "one numeric cluster variable is required")
    wgm_require(fevar=="" | cols(tokens(fevar))==1, "only one numeric FE is supported")
    wgm_require(touse!="", "explicit common-sample marker is required")
    wgm_require(dofminus>=0 & sdofminus>=0 & !hasmissing((dofminus,sdofminus)),
                "nonnegative native dofminus and sdofminus are required")
    wgm_require(small==0 | small==1, "small must be zero or one")

    D = st_data(., (tokens(yname), s.enames, tokens(excluded), s.cnames), touse)
    cl = st_data(., tokens(clustvar), touse)
    wgm_require(rows(D)>0, "empty sample")
    wgm_require(!hasmissing(D) & !hasmissing(cl), "missing data inside touse; no silent sample changes")
    if (fevar!="") {
        fe = st_data(., tokens(fevar), touse)
        wgm_require(!hasmissing(fe), "missing FE inside touse")
        D = wgm_demean(D, fe)
    }

    s.N = rows(D)
    s.p = ecount
    s.J = ccount
    s.q = ecount+wcount
    s.dofminus = dofminus
    s.sdofminus = sdofminus
    s.small = small

    s.y = D[,1]
    s.E = J(s.N,0,.)
    if (ecount>0) s.E = D[,2..(1+ecount)]
    W = D[,(2+ecount)..(1+ecount+wcount)]
    s.C = D[,(2+ecount+wcount)..cols(D)]
    Z = W,s.E
    B = s.y,s.E,s.C
    D = J(0,0,.)

    s.H = quadcross(Z,Z)/s.N
    s.zy = quadcross(Z,s.y)/s.N
    s.ze = quadcross(Z,s.E)/s.N
    s.zc = quadcross(Z,s.C)/s.N
    s.Hinvzy = wgm_fullsolve(s.H,s.zy,"Z'Z")
    s.EE = quadcross(s.E,s.E)
    s.EC = quadcross(s.E,s.C)
    s.Ey = quadcross(s.E,s.y)
    s.Cy = quadcross(s.C,s.y)
    s.cc = quadcolsum(s.C:^2)
    s.yy = quadcross(s.y,s.y)

    ord = order(cl,1)
    info = panelsetup(cl[ord],1)
    s.G = rows(info)
    wgm_require(s.G>1, "at least two bootstrap clusters are required")
    wgm_require(s.q>=1+s.p, "model is underidentified")
    wgm_require(s.N-(1+s.p)-dofminus-sdofminus>0, "nonpositive residual degrees of freedom")
    wgm_require(s.N-(1+s.p)-sdofminus>0, "nonpositive small-VCE denominator")

    Z = Z[ord,.]
    B = B[ord,.]
    s.cm = J(s.G*s.q,cols(B),.)
    for (g=1; g<=s.G; g++) {
        first = info[g,1]
        last = info[g,2]
        lo = (g-1)*s.q+1
        hi = g*s.q
        s.cm[lo..hi,.] = quadcross(Z[first..last,.],B[first..last,.])
    }
    return(s)
}

/*
    Columns:
      candidate_index, b[1..k], vec(V)'[1..k*k],
      rss, rmse, N, N_clust, df_r, hansen_j, cond_omega, rss_direct.
    Coefficients use native endogenous-first ordering.
    vec(V) stacks columns, including all off-diagonal elements.

    directrss=1 evaluates residuals on the already-demeaned original candidate
    column. This is the compatibility-first default used by the Stata entry point.
    directrss=0 uses cached quadratic moments with an explicit cancellation guard.
*/
real matrix wgm_evaluate(struct wgm_state scalar s, real scalar directrss)
{
    real scalar j, g, k, lo, hi, rss, mag, useddirect, rmse, hJ, dfres, qsmall
    real matrix answer, QXZ, aux1, A1, b1, scores, flat, omega, A2, b2, V
    real matrix xx, xy, er, gb, cv
    wgm_require(directrss==0 | directrss==1, "directrss must be zero or one")
    k = 1+s.p
    answer = J(s.J,1+k+k*k+8,.)
    qsmall = 1
    if (s.small) qsmall = (s.N-1)/(s.N-k-s.sdofminus)*s.G/(s.G-1)
    dfres = s.N-s.dofminus
    if (s.small) dfres = s.N-k-s.dofminus-s.sdofminus

    for (j=1; j<=s.J; j++) {
        QXZ = (s.zc[,j],s.ze)'
        aux1 = wgm_fullsolve(s.H,QXZ',"Z'Z")
        A1 = makesymmetric(QXZ*aux1)
        b1 = wgm_fullsolve(A1,QXZ*s.Hinvzy,"first-step identified regressor Gram")

        flat = s.cm[,1] - s.cm[,1+s.p+j]*b1[1]
        if (s.p>0) flat = flat-s.cm[,2..(1+s.p)]*b1[2..k]
        scores = J(s.G,s.q,.)
        for (g=1; g<=s.G; g++) {
            lo = (g-1)*s.q+1
            hi = g*s.q
            scores[g,.] = flat[lo..hi]'
        }
        omega = makesymmetric(quadcross(scores,scores)/s.N)
        cv = wgm_fullsolve(omega,(QXZ',s.zy),"first-step clustered moment covariance")
        A2 = makesymmetric(QXZ*cv[,1..k])
        b2 = wgm_fullsolve(A2,QXZ*cv[,k+1],"second-step identified regressor Gram")
        V = wgm_fullsolve(A2,I(k),"efficient GMM variance")/s.N*qsmall

        useddirect = directrss
        if (directrss) {
            er = s.y - s.C[,j]*b2[1]
            if (s.p>0) er = er-s.E*b2[2..k]
            rss = quadcross(er,er)
        }
        else {
            xx = (s.cc[j],s.EC[,j]') \ (s.EC[,j],s.EE)
            xy = s.Cy[j] \ s.Ey
            rss = s.yy-2*b2'*xy+b2'*xx*b2
            mag = abs(s.yy)+2*abs(b2'*xy)+abs(b2'*xx*b2)
            if (hasmissing(rss) | rss<0 | abs(rss)<=1e-10*mag) {
                er = s.y-s.C[,j]*b2[1]
                if (s.p>0) er = er-s.E*b2[2..k]
                rss = quadcross(er,er)
                useddirect = 1
            }
        }
        rmse = sqrt(rss/dfres)
        gb = s.zy-QXZ'*b2
        hJ = 0
        if (s.q>k) hJ = s.N*gb'*wgm_fullsolve(omega,gb,"Hansen-J covariance")

        answer[j,.] = j,b2',vec(V)',rss,rmse,s.N,s.G,s.G-1,hJ,cond(omega),useddirect
    }
    return(answer)
}

void wgm_stata(
    string scalar yname,
    string scalar exog,
    string scalar candidates,
    string scalar excluded,
    string scalar touse,
    string scalar fevar,
    string scalar clustvar,
    real scalar dofminus,
    real scalar sdofminus,
    real scalar small,
    real scalar directrss,
    string scalar outmatrix)
{
    struct wgm_state scalar s
    real matrix result
    string rowvector names
    string matrix stripe
    real scalar k, i, j
    s = wgm_prepare(yname,exog,candidates,excluded,touse,fevar,clustvar,
                    dofminus,sdofminus,small)
    result = wgm_evaluate(s,directrss)
    k = 1+s.p
    names = "candidate","b_candidate"
    for (i=1; i<=s.p; i++) names = names,"b_"+s.enames[i]
    for (j=1; j<=k; j++) {
        for (i=1; i<=k; i++) {
            names = names,"V_"+strofreal(i)+"_"+strofreal(j)
        }
    }
    names = names,("rss","rmse","N","N_clust","df_r","hansen_j","cond_omega","rss_direct")
    stripe = J(cols(names),1,""),names'
    st_matrix(outmatrix,result)
    st_matrixcolstripe(outmatrix,stripe)
}

end
