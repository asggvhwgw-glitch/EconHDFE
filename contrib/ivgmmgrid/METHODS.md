# Mathematical and implementation contract

For a candidate's own complete sample, remove singleton FE levels and demean y, X and Z by the single FE. X contains the endogenous candidate followed by exogenous controls; Z contains excluded instruments followed by controls. Retain independent columns in deterministic input order and scale them to unit Euclidean norm for the solves, transforming coefficients and covariance back afterwards.

Let H=Z'Z/N, Q=Z'X/N and q=Z'y/N. The first estimate is b1=(Q'H^-1 Q)^-1 Q'H^-1 q. For cluster g, form s_g=Z_g'(y_g-X_g b1); Omega=sum(s_g s_g')/N. Then b2=(Q'Omega^-1 Q)^-1 Q'Omega^-1 q. Covariance is (Q'Omega^-1 Q)^-1/N, multiplied by (N-1)/(N-k-a) times G/(G-1). Hansen J is N times (q-Q b2)'Omega^-1(q-Q b2); its degrees of freedom are retained instruments minus retained regressors.

The cached route shares Z_g'[y, controls, candidates] and derives first-step cluster scores algebraically. The reference route directly accumulates Z_g'u1. Both routes use Cholesky solves, fail explicitly on rank-deficient moments, and compute final residual RSS directly. They share model semantics; their agreement alone is not independent validation. Synthetic Stata oracles and 780 real-data Stata fits provide the separate implementation comparison.

Single-FE small-sample convention: a equals the retained FE level count unless all FE levels nest inside cluster labels, in which case a=1. This reproduces the pinned ivreghdfe fixtures. More complex FE graphs, weights, multiple endogenous regressors and multiway clustering are outside scope. Numerical column independence uses a configurable relative rank_tol (default 1e-10); boundary cases may differ from Stata's internal omission choices and are not universally certified.

Prior art: standard two-step efficient GMM, within projection, cluster score covariance and reuse of sufficient cross-products. The contribution is their Python implementation, bounded public contract, and reproducible tests; it does not claim a new estimator.
