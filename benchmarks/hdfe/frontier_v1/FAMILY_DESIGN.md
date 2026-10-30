# FE frontier v1: data families and projection oracle

This file specifies benchmark inputs. It does not assert a new universal FE
algorithm, a lower bound, or a finite-precision solver certificate. The two
October 2026 research notes cited below are working manuscripts, not externally
reviewed theorems. Their local constructions are reproduced here and checked
independently on small instances.

## Public API and scaling variables

```python
case = make_case(
    family="rigidity_grid", size=256, rhs=2, seed=0,
    repeat=1, weight_ratio=1.0, rhs_kind="mixed",
)
```

`case.groups` is an `int64` matrix of shape `(N, 3)`. Labels in each dimension
run from zero to `2*m-1`; they are local to that dimension. `case.y` and
`case.truth` have shape `(N, rhs)` even when `rhs=1`. `case.weights` is `None`
for unit weights and otherwise a positive length-`N` vector.

`case.coefficients` contains three arrays of shape `(2*m, rhs)`. These are an
independent, inexpensive feasibility witness: summing the selected coefficient
in each dimension reconstructs `case.y - case.truth`, to rounding error. Large
arrays are deliberately outside `case.metadata`, which is JSON serializable.

| Family | Meaning of `size` | Modules `m` | Base `N` | `G` | Unique tuples `U` | Unit-weight condition bound |
|---|---|---:|---:|---:|---:|---|
| `kernel_grid` | periodic grid side `ell >= 3` | `ell**2` | `16*m` | `6*m` | `16*m` | `kappa_FE <= 12*m` |
| `rigidity_grid` | periodic grid side `ell >= 4` | `ell**2` | `20*m` | `6*m` | `16*m` | `kappa_FE <= 12*m` |
| `rigidity_grid_unique` | periodic grid side `ell >= 4` | `ell**2` | `16*m` | `6*m` | `16*m` | `kappa_FE <= 24*m` |
| `kernel_cycle` | cycle vertices `m >= 3` | `size` | `12*m` | `6*m` | `12*m` | `kappa_FE <= 6*m**2` |
| `easy_crossed` | graph vertices `m >= 3` | `size` | `8*m + 4*|E| <= 20*m` | `6*m` | `N` | no certified constant-gap claim |

Here `kappa_FE` is the ratio of the largest to smallest positive eigenvalue of
`H = Delta**(-1/2) D.T W D Delta**(-1/2)`, with
`Delta = diag(D.T W D)`. Its largest eigenvalue is three. Every family has
normal rank `rank(D) = 6*m - 2`. The matrices are genuinely three-way; every
module contains a full non-anchor rectangle for every choice of anchor.

The principal scaling experiment should hold `rhs`, `repeat`, weight ratio,
and tolerance fixed, and increase `size`. Thus `G = Theta(N)` throughout.
For the main rigidity sequence `ell = 32, 64, 128, 256`, actual observation
counts are `20,480`, `81,920`, `327,680`, and `1,310,720`.

`repeat` repeats the complete base row set without adding categories. It is a
separate compression experiment. Increasing only `repeat` does **not** probe
rank-growing complexity; with unit weights it leaves the canonical spectrum
unchanged. The original rigidity family has repeated axial rows even at
`repeat=1`, so the unique variant and `kernel_grid` are essential controls.

## Common local construction and exact residual oracle

Module `i` has categories `(a_i0,a_i1,b_i0,b_i1,c_i0,c_i1)` and all eight
observations `(a_ix,b_iy,c_iz)` for `x,y,z in {0,1}`. The local design `T`
has rank four, positive Gram eigenvalues `4,4,4,12`, and a two-dimensional
kernel:

```
a_i0 = a_i1 = u_i
b_i0 = b_i1 = v_i
c_i0 = c_i1 = -u_i-v_i
```

Three legal connector types turn these local kernel coefficients into
directional differences:

| FE observation | Value on the local kernel |
|---|---|
| `(a_i0,b_j0,c_j0)` | `u_i-u_j` |
| `(a_j0,b_i0,c_j0)` | `v_i-v_j` |
| `(a_i0,b_i0,c_j0)` | `(u_i+v_i)-(u_j+v_j)` |

For an exact residual oracle, use the four local Walsh interactions

```
(-1)**(x+y), (-1)**(x+z), (-1)**(y+z), (-1)**(x+y+z).
```

Each contrast sums to zero within each category in every dimension. Choose
independent random combinations per module and RHS, place them on core rows,
and put zero on connector rows, obtaining `h` with `D.T @ h = 0`.
For actual positive row weights set `r = h / w`, and normalize each residual
column by one scalar. Consequently `D.T @ W @ r = 0` remains an algebraic
identity. Repeated base copies preserve the same cancellation even when their
weights differ.

The target is constructed as `y = D @ alpha + r`. Orthogonality and feasibility
together prove that `r` is the exact mathematical projection residual. No CG,
LSMR, package absorber, Fourier inversion, or dense pseudoinverse is used to
construct the oracle. Small tests compare it against an independent dense
weighted least-squares solution. Large verification can check both identities
using the returned coefficients and group reductions. Floating-point arrays
approximate the algebraic construction, so report their checked numerical
defects, not a claim of exact bit arithmetic.

Both the fit and residual are normalized to weighted RMS one. They therefore
each contribute half the target's squared weighted norm. A residual tolerance
is never made artificially easy by adding a huge already-orthogonal target.

## Slow and mixed right-hand sides

`rhs_kind` is one of `slow`, `mixed`, `broadband`, and `random`.

For `slow`, the coefficients lie in the local cube kernels above. The two
fields `u` and `v` mix multiple real Fourier frequencies, with independently
drawn phases and signs. The frequency set contains the smallest nonzero
frequency and a range of powers-of-two scales, plus several intermediate
frequencies; up to 24 spatial modes are used. The actual frequencies are in
`metadata['rhs']['spatial_frequencies']`.

For a grid mode `(p,q)`, define

```
tau = 4*sin(pi*p/ell)**2 + 4*sin(pi*q/ell)**2.
```

Each coefficient mode is multiplied by `1/sqrt(tau)` before aggregation. A
connector differentiates the coefficient field, so this compensates for the
small gradient at low frequencies. The slowest fitted components therefore
remain visible in the observation-space target. For a cycle the same rule
uses `tau = 4*sin(pi*p/m)**2`. These are local-kernel trial fields, **not** a
single supplied eigenvector of the complete FE system. Their broad frequency
support prevents the intended stress test from reducing to one-eigenvalue,
one-step Krylov convergence. However, its support is still at most 24 spatial
frequencies. For a translation-invariant grid this limits its effective
Krylov spectral support independently of `N`. **Pure `slow` is a diagnostic,
not a worst-case RHS scaling experiment.** At very small sizes only a few
distinct spatial frequencies exist, as expected.

For `mixed`, form a random feasible fit, orthogonalize it against the slow fit
in the actual weighted observation norm, normalize both, and combine them with
coefficients `sqrt(0.9)` and `sqrt(0.1)`. Exactly 90% of the fitted energy is in
the supplied slow component, up to rounding. Since the fit is half the target
energy, the intended component remains 45% of total target energy. For the
random graph control, Fourier order of vertices does not imply a slow graph
mode, and the metadata explicitly avoids that claim.

`random` uses independent random categorical coefficients. Random RHS success
alone is not evidence of a worst-case operator guarantee.

For `broadband`, start from independent real white-noise fields `u` and `v`,
FFT them, set the zero frequency to zero, divide every other Fourier
coefficient by `sqrt(tau)`, and inverse FFT. Every nonzero spatial frequency
has nonzero random support with probability one in exact arithmetic, and
the support dimension grows with `m`. The filtered fields are local-kernel
coefficients. Orthogonalize their fitted vector against the targeted
multi-frequency fit and combine the two with 50%/50% weighted fitted energy.
Thus targeted low-frequency content remains 25% of the full target energy,
while the other quarter spans the growing set of nonzero spatial frequencies.
Use this alongside the default `mixed` path for a stronger scaling check.
Neither construction is advertised as an adversarial worst-case RHS for
every possible solver. Arbitrary row weights break the unweighted Fourier
diagonalization; the oracle remains valid, but exact weighted eigenmode
alignment is not claimed.

## Families and their spectral scope

### Kernel grid

Place `m=ell**2` modules on a periodic square grid. For every directed axial
edge, including both orientations, add both `u` and `v` connector types.
There are `8*m` core and `8*m` connector observations, all unique.
The two coarse fields are connected scalar graph systems; the **original**
FE model is still three-way with rank `6*m-2`.

This is the family of *Implicit Kernel Coupling for Multiway Fixed Effects*
(28 October 2026), Section 5. Its normalized Gram has
`lambda_min_positive >= 1/(4*m)`, so `kappa_FE <= 12*m < N`.
The local-kernel trial spaces also give `kappa_FE = Theta(m)` and many small
positive eigenvalues. This is not an isolated slow eigenvector family.

### Rigidity grid

For each periodic cell with vertices `(0,1,2,3)` in the order lower-left,
right, up, upper-right, add both orientations of:

```
u_0-u_1, u_2-u_3
v_0-v_2, v_1-v_3
(u_0+v_0)-(u_3+v_3), (u_1+v_1)-(u_2+v_2)
```

Each individual edge has rank-one directional energy. For even `ell >= 6`,
the `u` graph has `ell` components, the `v` graph has `ell` components, and
the `u+v` graph has two. Nonetheless the complete coupled system has only
the two global gauges. Thus independently solving each direction graph does
not describe the coupled coarse problem.

The six-row patch has positive spectrum `2,2,3-sqrt(5),3-sqrt(5),
3+sqrt(5),3+sqrt(5)`. The original patch multiset contains repeated axial
connectors. Its frequency vector per module is `(16,4,16,4,16,4)` and its
canonical bound is `kappa_FE <= 12*m < N`. This is exactly the row multiset
and ordering of *Rigidity Patches and Single-Call Coarse Frames for Multiway
Fixed Effects* (28 October 2026), Section 6. Coordinate labels are supplied
by the generator; discovering coordinates from an arbitrary FE input is not
part of this benchmark family.

### Unique rigidity grid

Remove one of each repeated axial connector, retaining both orientations of
each distinct axial and diagonal edge. Now all `16*m` tuples are unique,
the per-module frequencies are `(12,4,12,4,12,4)`, and the row space and rank
are unchanged.

Let `A_o,Delta_o` be the original Gram and diagonal, and `A_u,Delta_u` those
after duplicate removal. Then

```
A_u <= A_o <= 2*A_u,     Delta_u <= Delta_o <= 2*Delta_u.
```

Consequently the positive canonical gap decreases by at most a factor two.
For example, `H_u >= (1/2) S H_o S`, where
`S = Delta_u**(-1/2) Delta_o**(1/2)` has smallest singular value at least one.
The ordered positive eigenvalues of `S H_o S` are no smaller than those of
`H_o`, by their shared nonzero spectrum with `H_o**(1/2) S**2 H_o**(1/2)`.
Thus `kappa_FE <= 24*m`. The matching order `Theta(m)` also survives these
constant-factor comparisons. The looser bound **does not itself certify**
`kappa_FE <= N`; it certifies a constant multiple of `N`.

### Kernel cycle

Place modules on a cycle and put both `u` and `v` connectors on both
orientations of each edge. The number of observations is `12*m`, with all
tuples unique. Both coarse graphs are connected, giving normal rank. Unlike
the grid family this has `kappa_FE = Theta(m**2)`, so it tests a more ill
conditioned regime than `kappa_FE <= N`.

Here is an elementary conservative bound, independent of a numerical fit.
For a cycle frequency let `tau=2-2*cos(2*pi*p/m) in [0,4]`. The raw six by
six Gram symbol is

```
R(tau) = T.T @ T + I0 @ E(tau) @ I0.T
E(tau) = [[4, 4-2*tau, 4-tau],
          [4-2*tau, 4, 4-tau],
          [4-tau, 4-tau, 4]]
Delta = diag(8,4,8,4,8,4),
```

where `I0` selects the representative coordinates. The leading principal
minors of the integer matrix `8*R(4)-Delta` are exactly

```
56, 1568, 66304, 914432, 35020800, 74088448.
```

They are positive, so `R(4) >= Delta/8`. Since `R(0)` is positive semidefinite
and the symbol is affine, `R(tau) >= (tau/32)*Delta`. At zero frequency the
positive gap is at least `4/8=1/2`. At other frequencies
`tau >= 4*sin(pi/m)**2 >= 16/m**2`. Hence the canonical positive gap is at
least `1/(2*m**2)`, giving `kappa_FE <= 6*m**2`. A nonconstant local-kernel
trial field at the smallest frequency supplies the matching order. Tests
verify the displayed integer minors and independently assembled small Gram
spectra; those numerical tests do not replace the analytic argument.

### Easy crossed control

Take the union of three Hamiltonian cycles, one fixed and two independently
permuted, and remove parallel undirected edges. Put both `u` and `v` connector
types on both orientations. The fixed cycle certifies connectivity. Degree
is bounded by six, `|E| <= 3*m`, and all tuples are unique.

This is an empirical easy-spectrum control, not a proved expander family.
No constant spectral gap or asymptotic runtime is asserted. Report the
observed small-instance spectrum separately from analytic family bounds.

## Weights, replication, and what remains certified

For `weight_ratio > 1`, independent log-spread weights lie in
`[1, weight_ratio]`, with the endpoints inserted explicitly. Topology, target,
and weight random streams are separate. The weighted truth is rebuilt as
above; changing weights never silently reuses an unweighted oracle.

If `w_max/w_min <= R`, normalized Gram comparison gives

```
lambda_min_positive(H_weighted) >= lambda_min_positive(H_unit)/R,
kappa_FE(weighted) <= R*kappa_FE(unit).
```

The same argument applies after uniform row replication, because the
aggregate weight of each original row remains in the same ratio interval.
Metadata reports the resulting conservative analytic upper bound. The
unweighted `kappa_FE <= N` promise must **not** be carried unchanged into
arbitrarily weighted cases. For unit weights, uniform replication leaves
the spectrum invariant. With non-unit weights the generator draws independent
weights on each row copy, so numerical spectra may differ from a separately
generated `repeat=1` case. Only the comparison bound, not spectral invariance,
is asserted in that case.

## Sources

- *Implicit Kernel Coupling for Multiway Fixed Effects*, research continuation,
  28 October 2026, Sections 4–6. Original file:
  `FE_Implicit_Kernel_Coupling_2026.pdf`; original implementation:
  `kernel_projector.py`.
- *Rigidity Patches and Single-Call Coarse Frames for Multiway Fixed Effects*,
  research continuation, 28 October 2026, Sections 3 and 6. Original file:
  `FE_Rigidity_Patch_Continuation_2026.pdf`; original implementation:
  `rigid_patch_projector.py`.

The unique-tuple comparison, cycle bound, oracle construction, and RHS energy
controls are documented explicitly here. No external implementation is used
as an unexplained projection oracle.


## Additional large-scale cross-topology stress design (planned)

A separate research workflow compares four additional graph structures at roughly
5.2M and 10.4M rows, with one RHS, unit weights, broadband targets, and the same
independent forward-error gate. This section specifies the design only; it is
not a report of measured results.

| Family | Size parameters | Approximate N | Spectral scope |
|---|---:|---:|---|
| `kernel_grid` | side 572 / 806 | 5.235M / 10.394M | Analytic `kappa_FE <= 12m < N`; periodic 2D grid coupling |
| `rigidity_grid_unique` | side 572 / 806 | 5.235M / 10.394M | Unique tuples; analytic `kappa_FE <= 24m = 1.5N` bound |
| `kernel_cycle` | modules 432,000 / 864,000 | 5.184M / 10.368M | Analytic `kappa_FE = Theta(m^2)`; deliberately harder conditioning, outside the `kappa_FE <= N` regime |
| `easy_crossed` | modules 262,144 / 524,288 | about 5.24M / 10.49M | Three-cycle connected random control; no certified spectral-gap or runtime claim |

The three native backends (`within`, `native_auto`, `fereg`) are run in fresh
workers, cold then same-solver reuse, with native diagnostics enabled. Each
worker retains its generated N/G/U, analytic condition bound when available,
forward error, stationarity, full-call time, whole-worker peak RSS, preconditioner
route and actual operator-application counts. One trial per configuration is
an exploratory topology screen, not a timing benchmark with statistical
precision. Failures, timeouts and rejected routes are retained rather than
silently dropped.
