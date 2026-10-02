args vendor_root
version 18
clear all
set more off
set processors 1
adopath ++ "`vendor_root'/require_upstream/source/src"
adopath ++ "`vendor_root'/ftools_upstream/source/src"
adopath ++ "`vendor_root'/reghdfe_upstream/source/src"
adopath ++ "`vendor_root'/moremata_upstream/source"
adopath ++ "`vendor_root'/ivreg2_ssc"
adopath ++ "`vendor_root'/ranktest_ssc"
adopath ++ "`vendor_root'/ivreghdfe_upstream/source/src"
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


save "fixture.dta", replace
postfile result str12 model double b1 double b2 double v11 double v12 double v22 double rss double rmse double j double n double g double df double sdof using "oracle.dta", replace
ivreghdfe y w (x1=z1 z2), absorb(t) cluster(bsid) gmm2s
matrix b=e(b)
matrix v=e(V)
post result ("base") (b[1,1]) (b[1,2]) (v[1,1]) (v[1,2]) (v[2,2]) (e(rss)) (e(rmse)) (e(j)) (e(N)) (e(N_clust)) (e(df_r)) (e(sdofminus))
ivreghdfe y w (x1=z1 z2), absorb(bsid) cluster(bsid) gmm2s
matrix b=e(b)
matrix v=e(V)
post result ("nested") (b[1,1]) (b[1,2]) (v[1,1]) (v[1,2]) (v[2,2]) (e(rss)) (e(rmse)) (e(j)) (e(N)) (e(N_clust)) (e(df_r)) (e(sdofminus))
gen double xcopy=w
ivreghdfe y w (xcopy=z1 z2), absorb(t) cluster(bsid) gmm2s
matrix b=e(b)
matrix v=e(V)
post result ("omitted") (b[1,1]) (b[1,2]) (v[1,1]) (v[1,2]) (v[2,2]) (e(rss)) (e(rmse)) (e(j)) (e(N)) (e(N_clust)) (e(df_r)) (e(sdofminus))
postclose result
display "ORACLE_OK"
exit, clear
