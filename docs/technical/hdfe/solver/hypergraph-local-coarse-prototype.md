# Experimental hypergraph local/coarse projection

Status: standalone research prototype in `contrib/hyperfe_local_coarse/`, not a
registered EconHDFE backend. Originally based on 0.9.0 and reviewed against
0.9.1; no public API/default/dependency change.

## Mathematical target and assumptions

Let D=[F,H] be a pure categorical intercept design and W a finite strictly
positive diagonal matrix. F is the first FE. Profiling F gives
M=I-F(F'WF)^(-1)F'W and the original rectangular problem
min_b ||sqrt(W) M (Y-Hb)||_F^2. Its Schur matrix is S=H'WMH.
Reference removal preserves the fit space for the identified pairwise components,
but is not a complete certificate for arbitrary higher-order dependencies.

For each F level v, let s=sum_i w_i and hbar=sum_i w_i h_i/s. Choose a
maximum-weight anchor j. The exact local Schur contribution and star are
S_v=sum_i w_i(h_i-hbar)(h_i-hbar)' and
T_v=sum_i w_i(h_i-h_j)(h_i-h_j)'. Expanding around hbar gives
T_v=S_v+s(h_j-hbar)(h_j-hbar)'. Weighted Cauchy-Schwarz gives
S_v <= T_v <= (s/w_j) S_v, hence equal kernels. These bounds concern the
exact star matrix, not an arbitrary incomplete or block inverse.

## Local and global correction

Normalize T=sum_v T_v by its diagonal, threshold off-diagonal magnitudes, and
use its connected components as local blocks. A block may mix levels from more
than two FE groups. Enforce block and coarse-dimension budgets. Small eigenvectors
from each block provide candidates Z; this empirical selection is not claimed
to satisfy every assumption of GenEO theory.

Given a nonsingular coarse energy K=Z'SZ, set E=Z K^(-1) Z' and let B0=R0 R0'.
The balanced inverse is B=E+(I-ES)B0(I-SE), so B S Z=Z. If L L'=K, the
rectangular right factor R=[Z L^(-T), (I-ES)R0] satisfies R R'=B.
`CoarseRight` implements R and its exact adjoint; `solve_rect` runs LSMR on
sqrt(W) M H R, still targeting the original projection. K is assembled from
weighted centered observation-space columns to avoid subtractive cancellation.

Tests compare R R' and B, the adjoint identity, B S Z=Z, and weighted residuals
with independent dense least squares. No uniform condition-number bound for all
FE, complexity guarantee, new hypergraph Laplacian solver, or priority is claimed.

## Known failures and research boundary

Fixed thresholds are seed-sensitive. Equal/continuous-weight successful families
still contain the planted additive relation. An irregular nonadditive design
can select no coarse vectors or be far slower; large remaining systems may be
rejected. Sparse assembly may exceed memory before an nnz check, and budgeted
blocks do not impose a process-RSS cap. No automatic threshold selector is in
this PR. Native within remains the production comparison, not replaced by a
self-written pairwise baseline.

This contribution freezes the prototype and evidence; integration with OLS/IV,
PPML, sessions, structural DoF and user-facing configuration is deliberately
separate. Existing `method="within"` work in PR #25 is not modified.

References: Spillane, GenEO spectral coarse spaces, arXiv:2104.00280;
Fong and Saunders, LSMR, SIAM J. Sci. Comput. 33(5), 2011;
within 0.4.0 project documentation, py-econometrics/within.
