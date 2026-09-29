# Exact categorical FE rank backends

**Status:** Current exact-rank backend documentation  
**Scope:** Certification order, optional exact-arithmetic backends, fallback behavior, and resource boundaries.  
**Claim boundary:** Mathematical rank certificates and backend program correctness are separate claims; no floating-point rank is used on the exact path.
`rank.py` keeps the exact 3+ FE DoF engine dependency-safe.  The public
`categorical_rank(..., backend="auto")` path is:

1. exact structural reductions already used by the HDFE engine;
2. duplicate-edge removal and exact degree-one peeling;
3. connected residual-core decomposition;
4. proper-connectivity certificate when applicable;
5. a GF(2) bitset rank certificate.  If it reaches the deterministic
   characteristic-zero upper bound, the rank is proven exactly because the
   corresponding minor is nonzero over the integers;
6. optional python-flint exact rank for bounded dense cores;
7. optional SymPy sparse `DomainMatrix` exact rank;
8. the dependency-free native modular + rational fallback.

No floating-point numerical rank is used anywhere in the exact path.

## Optional libraries

The module has no mandatory dependency on either library.

- `python-flint`: `fmpz_mat.rank()` is a compiled exact integer rank routine.
  Its matrix API is dense, so automatic use is limited by `flint_max_cells`
  (default 4,000,000 cells).  Pass `backend="flint"` to require it explicitly.
- `sympy`: sparse `DomainMatrix` over `GF(p)`/`ZZ` is used without densifying
  the core. Pass `backend="sympy"` to require it explicitly.

`available_rank_backends()` reports which optional backends can currently be
imported.

LinBox was evaluated but is not wired into this Python module: its sparse
Wiedemann/elimination algorithms are a strong fit for much larger cores, but
there is no comparably stable standard Python binding to depend on here.

## Local targeted validation (2026-09-11)

Environment: CPython 3.13, NumPy/SciPy/Numba from the current econhdfe test
image, SymPy 1.14.  python-flint was not installable in this sandbox because
external package download is disabled; its adapter follows the documented
`fmpz_mat(m, n, entries).rank()` API and was exercised with a compatible test
stub.

Correctness checks run for this module revision:

- known 3-FE collective-dependency counterexample: exact rank 4;
- parity example with rank over Q greater than rank over GF(2): correctly
  falls through the GF(2) certificate and returns the Q-rank;
- all 255 nonempty 2x2x2 three-way hypergraphs, against rational elimination,
  for `auto`, `native`, and `sympy` backends;
- 200 random 3-FE/4-FE small designs against rational elimination;
- requested `backend="flint"` dispatch and memory-cap behavior with a compatible
  stub;
- one-million-row leaf-rich design: exact rank path completed with the core
  fully peeled.

Warm-path timings in this environment (illustrative, not release guarantees):

- 5,000 random unique edges / ~1,500 levels: ~0.06 s after JIT warm-up;
- 10,000 / ~3,000: ~0.24 s;
- 20,000 / ~6,000: ~0.91 s;
- 1,000,000 leaf-rich rows: ~0.29 s in the measured run.

The main remaining hard case is a very large residual core whose true rank is
below the structural upper bound and for which the GF(2) certificate therefore
cannot terminate the calculation. SymPy/FLINT improve the fallback story, but
a future LinBox-style sparse black-box backend can still be worthwhile for
that regime.
