"""Optional GPU FP64 envelope fusion, selected explicitly as ``gpu_fused``.

Based on the independently reviewed prototype SHA256
1130e0fee04c259ff9ce2f825b09ef3442c88fa85d640e99378da80e6981086b.
The original CPU-vector envelope remains available in ``batched_radius``.

Inherits the frozen auditor's data snapshots, clipping and shared products.
Only its O(n) per-row envelope expressions move to a CUDA kernel. This first
version reuploads the already returned host dot products; optimizing that
transfer separately must not be confused with the envelope mechanism.

Every arithmetic stage follows the frozen expression order. Explicit CUDA
round-to-nearest-even FP64 intrinsics prevent contraction/reassociation, and
bit-neighbour functions implement nextafter toward +/-infinity, including
signed zero, subnormals, infinity and NaN. Fast math and FMA are disabled.
FTZ=false is also explicit, but NVRTC documents FTZ as a single-precision
option, not an additional guarantee about these FP64 operations.
Positive GPU summation retains the same gamma_(n+1) and underflow allowance;
its reduction order may change the bound, not its stated FP64 model.
Zero-preserving row helpers inherit the frozen arithmetic: an individual
underflowed term need not enclose its exact positive value. The complete
enclosure relies on the final global underflow budget as well.

This remains model-qualified, not formal interval arithmetic. Importing this
module initializes no CUDA device. Hardware regression tests exercise
implementation behavior; they do not prove every model assumption.

Primary documentation:
https://docs.nvidia.com/cuda/nvrtc/
https://docs.nvidia.com/cuda/archive/12.9.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html
"""
from __future__ import annotations

import time
import hashlib
import numpy as np

from .batched_radius import RadiusAuditor
from .certification import (
    RadiusCertificate, _U, _up, _down, _gamma, _underflow,
    _add_up, _mul_up, _nonnegative_up, _positive_reduction_upper,
    _norm_upper, _sum_upper, _residual_envelope, _verified_kappa,
)


KERNEL_OPTIONS = ('--std=c++11', '--fmad=false', '--ftz=false',
                  '--prec-div=true', '--prec-sqrt=true')

ROW_KERNEL = r'''
__device__ __forceinline__ bool fp_nan(double x) {
    return ((unsigned long long)__double_as_longlong(x) & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL;
}
__device__ __forceinline__ bool fp_finite(double x) {
    return ((unsigned long long)__double_as_longlong(x) & 0x7fffffffffffffffULL) < 0x7ff0000000000000ULL;
}
__device__ __forceinline__ double fp_abs(double x) {
    return __longlong_as_double(__double_as_longlong(x) & 0x7fffffffffffffffLL);
}
__device__ __forceinline__ double fp_inf() { return __longlong_as_double(0x7ff0000000000000LL); }
__device__ __forceinline__ double fp_up(double x) {
    unsigned long long b = (unsigned long long)__double_as_longlong(x);
    unsigned long long a = b & 0x7fffffffffffffffULL;
    if (a > 0x7ff0000000000000ULL || b == 0x7ff0000000000000ULL) return x;
    if (a == 0ULL) return __longlong_as_double(1LL);
    b = (b & 0x8000000000000000ULL) ? b - 1ULL : b + 1ULL;
    return __longlong_as_double((long long)b);
}
__device__ __forceinline__ double fp_down(double x) {
    unsigned long long b = (unsigned long long)__double_as_longlong(x);
    unsigned long long a = b & 0x7fffffffffffffffULL;
    if (a > 0x7ff0000000000000ULL || b == 0xfff0000000000000ULL) return x;
    if (a == 0ULL) return __longlong_as_double((long long)0x8000000000000001ULL);
    b = (b & 0x8000000000000000ULL) ? b + 1ULL : b - 1ULL;
    return __longlong_as_double((long long)b);
}
__device__ __forceinline__ double nn_up(double x) { return x == 0.0 ? 0.0 : fp_up(x); }
__device__ __forceinline__ double add_up(double a, double b) { return nn_up(__dadd_rn(a,b)); }
__device__ __forceinline__ double mul_up(double a, double b) { return nn_up(__dmul_rn(a,b)); }
__device__ __forceinline__ double nonnegative(double x) { return fp_nan(x) ? x : (x > 0.0 ? x : 0.0); }
__device__ __forceinline__ double minimum(double a, double b) {
    if (fp_nan(a)) return a;
    if (fp_nan(b)) return b;
    return a < b ? a : b;
}
extern "C" __global__ void row_envelope(
    const long long n, const double* y, const double* dot,
    const double* absdot, const double* used_a,
    const double tau, const double low, const double high,
    const double gp, const double ufp, const double denom_p, const double subtraction_factor,
    const double pseudo_residual, const double pseudo_error,
    double* margins, double* residual, double* error, double* terms) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i > n) return;
    double r, re;
    if (i == n) {
        r = pseudo_residual;
        re = pseudo_error;
    } else {
        r = __dadd_rn(y[i], -dot[i]);
        double magnitude = gp >= 1.0 ? fp_inf() : fp_up(__ddiv_rn(add_up(absdot[i],ufp),denom_p));
        double dot_error = mul_up(gp,magnitude);
        double subtraction_error = mul_up(subtraction_factor,fp_abs(r));
        re = add_up(add_up(dot_error,subtraction_error),ufp);
    }
    double a = used_a[i];
    margins[i] = nonnegative(minimum(fp_down(__dadd_rn(a,-high)),fp_down(__dadd_rn(tau,-a))));
    residual[i] = r;
    error[i] = re;
    double positive = nonnegative(fp_up(__dadd_rn(r,re)));
    double negative = nonnegative(fp_up(__dadd_rn(-r,re)));
    double pc = nn_up(__dadd_rn(tau,-a));
    double nc = nn_up(__dadd_rn(a,-low));
    terms[i] = add_up(mul_up(positive,pc),mul_up(negative,nc));
}
extern "C" __global__ void neighbour_probe(const long long n, const double* x, double* up, double* down) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) { up[i] = fp_up(x[i]); down[i] = fp_down(x[i]); }
}
extern "C" __global__ void positive_partials(const long long n, const double* terms,
    const double* error, double* sums, double* maxima, int* invalid) {
    __shared__ double s[256];
    __shared__ double e[256];
    __shared__ int bad[256];
    int t = threadIdx.x;
    long long i = (long long)blockIdx.x * 256LL + t;
    bool good = i >= n || fp_finite(terms[i]);
    s[t] = i < n && good ? terms[i] : 0.0;
    e[t] = i < n ? error[i] : 0.0;
    bad[t] = good ? 0 : 1;
    __syncthreads();
    for (int offset=128; offset>0; offset/=2) {
        if (t < offset) {
            s[t] = __dadd_rn(s[t],s[t+offset]);
            double a=e[t], b=e[t+offset];
            e[t] = fp_nan(a) ? a : (fp_nan(b) ? b : (a>b ? a : b));
            bad[t] |= bad[t+offset];
        }
        __syncthreads();
    }
    if (t == 0) { sums[blockIdx.x]=s[0]; maxima[blockIdx.x]=e[0]; invalid[blockIdx.x]=bad[0]; }
}
'''


_KERNEL_CACHE = {}


def compiled_kernels(cp):
    """Compile directly: CuPy RawKernel appends -ftz=true in this installation.

    Only an in-process module cache is used. No global compiler/allocator
    monkeypatch and no production CuPy behavior is changed.
    """
    from cupy.cuda import compiler, function
    device = cp.cuda.Device()
    arch = device.compute_capability
    key = (device.id, arch, hashlib.sha256(ROW_KERNEL.encode()).hexdigest(), KERNEL_OPTIONS)
    if key not in _KERNEL_CACHE:
        binary, _ = compiler.compile_using_nvrtc(ROW_KERNEL, options=KERNEL_OPTIONS,
                                                arch=arch, cache_in_memory=True)
        module = function.Module()
        module.load(binary)
        names = ('row_envelope', 'neighbour_probe', 'positive_partials')
        functions = {name: module.get_function(name) for name in names}
        _KERNEL_CACHE[key] = (module, functions)
    return _KERNEL_CACHE[key][1]


def row_reference(y, dot, absolute_dot, used_a, tau, low, high, p, r0, r0_error):
    """CPU expression contract for tiny tests; not an optimized CPU auditor."""
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        r = np.asarray(y)-dot
        magnitude = _positive_reduction_upper(absolute_dot, p)
        dot_error = _mul_up(_gamma(p), magnitude)
        subtraction_error = _mul_up(_up(_U/(1-_U)), np.abs(r))
        r_error = _add_up(_add_up(dot_error, subtraction_error), _underflow(p))
        residual, error = np.r_[r, r0], np.r_[r_error, r0_error]
        margins = np.maximum(0., np.minimum(_down(used_a-high), _down(tau-used_a)))
        positive = np.maximum(0., _up(residual+error))
        negative = np.maximum(0., _up(-residual+error))
        terms = _add_up(_mul_up(positive, _nonnegative_up(tau-used_a)),
                        _mul_up(negative, _nonnegative_up(used_a-low)))
    return dict(margins=margins, residual=residual, error=error, terms=terms)


def select_compact(indices, interior_margins, limit):
    """Same deterministic selection as _select_rows, without an n-vector upload.

    interior_margins is exactly margins[indices], in the same sorted-index
    order. All argpartition/spread/unique operations match the frozen helper.
    """
    if len(indices) <= limit:
        return indices
    strong_count = limit//2
    strong = (indices[np.argpartition(interior_margins, -strong_count)[-strong_count:]]
              if strong_count else indices[:0])
    spread = indices[np.linspace(0, len(indices)-1, limit-strong_count, dtype=int)]
    selected = np.unique(np.r_[strong, spread])
    if selected.size < limit:
        remaining = np.setdiff1d(indices, selected, assume_unique=True)
        selected = np.r_[selected, remaining[:limit-selected.size]]
    return np.sort(selected)


class FusedRadiusAuditor(RadiusAuditor):
    """GPU row envelopes with CPU or GPU shared products and CPU inverse checks.

    ``backend`` retains the inherited shared-product meaning. The per-row
    envelope always uses CUDA; backend='cpu' is not a CPU-only execution mode.
    """
    def __init__(self, X, y, tau, **kwargs):
        setup_started = time.perf_counter()
        backend = kwargs.pop('backend', 'gpu')
        super().__init__(X, y, tau, backend=backend, **kwargs)
        if self.cp is None:
            from boundedqr.gpu_pdhg import _prepare_windows_import_paths
            _prepare_windows_import_paths()
            import cupy as cp
            self.cp = cp
        self.device_y = self.cp.asarray(self.y, dtype=self.cp.float64)
        self.cp.cuda.get_current_stream().synchronize()
        self.kernels = compiled_kernels(self.cp)
        self.row_kernel = self.kernels['row_envelope']
        self.diagnostics['persistent_device_array_bytes'] += self.device_y.nbytes
        self.diagnostics['gamma_model_assumed_for'] += '; CUDA FP64 RN row operations and FP64 positive reduction'
        self.diagnostics['fused_envelope'] = dict(
            implementation='optional_gpu_fused_v2', calls=0, cumulative_seconds=0.,
            kernel_options=list(KERNEL_OPTIONS), explicit_RN_intrinsics=True,
            outward_neighbour='IEEE binary64 integer-bit successor/predecessor',
            reduction_model='Explicit RN 256-thread positive tree plus CPU FP64 sum, unchanged gamma_(n+1)/underflow allowance',
            compiler_route='compile_using_nvrtc directly; avoids RawKernel appended -ftz=true',
            extra_per_column_device_budget_bytes=8*10*(self.n+1),
            budget_scope='Conservative additional explicit arrays/selection, not pool reservation or measured peak')
        self.diagnostics['inherited_product_setup_seconds'] = self.diagnostics['setup_seconds']
        self.diagnostics['setup_seconds'] = time.perf_counter()-setup_started
        self.diagnostics['fused_envelope']['additional_setup_seconds'] = (
            self.diagnostics['setup_seconds']-self.diagnostics['inherited_product_setup_seconds'])

    def row_arrays(self, used_a, dot, absolute_dot, r0, r0_error):
        """Expose device arrays for small adversarial tests; caller releases them."""
        cp = self.cp
        arrays = [cp.asarray(value, dtype=cp.float64) for value in (dot, absolute_dot, used_a)]
        output = [cp.empty(self.n+1, dtype=cp.float64) for _ in range(4)]
        gp = _gamma(self.p)
        self.row_kernel(((self.n+1+255)//256,), (256,),
            (np.int64(self.n), self.device_y, *arrays,
             np.float64(self.tau), np.float64(self.low), np.float64(self.high),
             np.float64(gp), np.float64(_underflow(self.p)), np.float64(_down(1-gp)),
             np.float64(_up(_U/(1-_U))), np.float64(r0), np.float64(r0_error), *output))
        # Keep all temporary input arrays alive through the launch completion;
        # do not rely on implicit allocator/stream lifetime behavior here.
        cp.cuda.get_current_stream().synchronize()
        return dict(zip(('margins', 'residual', 'error', 'terms'), output))

    def _certificate(self, w, beta, original_a, used_a, dot, absolute_dot,
                     ordinary_e, ordinary_absolute_e):
        started = time.perf_counter()
        cp = self.cp
        result = RadiusCertificate()
        result.dual_lower_endpoint_lower, result.dual_lower_endpoint_upper = self.low, self.high
        result.clipping_adjustment_max = float(np.max(np.abs(used_a-original_a)))
        z0 = w/self.tau
        if not np.isfinite(self.t0) or not np.all(np.isfinite(z0)):
            result.reason = 'nonfinite_augmented_data'
            return result, 0.
        r0, r0_error = _residual_envelope(z0[None, :], np.array([self.t0]), beta)
        device = self.row_arrays(used_a, dot, absolute_dot, r0[0], r0_error[0])
        interior_gpu = cp.flatnonzero(device['margins'] > self.interior_tol)
        indices = cp.asnumpy(interior_gpu)
        interior_margins = cp.asnumpy(device['margins'][interior_gpu])
        selected = select_compact(indices, interior_margins, self.max_candidates)
        result.interior_count, result.selected_count = len(indices), len(selected)
        result.selected_indices = tuple(int(i) for i in selected)
        blocks = (self.n+1+255)//256
        partial = cp.empty(blocks, dtype=cp.float64)
        maximum = cp.empty(blocks, dtype=cp.float64)
        invalid = cp.empty(blocks, dtype=cp.int32)
        self.kernels['positive_partials']((blocks,), (256,),
            (np.int64(self.n+1), device['terms'], device['error'], partial, maximum, invalid))
        sums_cpu, maxima_cpu, invalid_cpu = [cp.asnumpy(value) for value in (partial, maximum, invalid)]
        result.residual_error_max = float(np.max(maxima_cpu))
        if not np.any(invalid_cpu):
            total = float(np.sum(sums_cpu, dtype=np.float64))
            result.gap_upper = float(_positive_reduction_upper(total, self.n+1))
        else:
            result.gap_upper = np.inf
        selected_gpu = cp.asarray(selected, dtype=cp.int64)
        margins = cp.asnumpy(device['margins'][selected_gpu])
        residual = cp.asnumpy(device['residual'][selected_gpu])
        residual_error = cp.asnumpy(device['error'][selected_gpu])
        # All n+1 terms have been accounted for. Release per-row device arrays;
        # the process-global pool may reserve their blocks for later reuse.
        del device, interior_gpu, selected_gpu, partial, maximum, invalid
        envelope_seconds = time.perf_counter()-started
        fused = self.diagnostics['fused_envelope']
        fused['calls'] += 1
        fused['cumulative_seconds'] += envelope_seconds
        fused['last_seconds'] = envelope_seconds

        # The remaining p-sized equality envelope and selected-row left-inverse
        # verification follow the frozen implementation unchanged.
        e = ordinary_e+z0*used_a[-1]
        absolute_e = ordinary_absolute_e+np.abs(z0*used_a[-1])
        magnitude_upper = _positive_reduction_upper(absolute_e, self.n+1)
        e_error = _add_up(_mul_up(_gamma(self.n+1), magnitude_upper), _underflow(self.n+1))
        result.stationarity_norm_upper = _norm_upper(_add_up(np.abs(e), e_error))
        selected_residual = _add_up(np.abs(residual), residual_error)
        result.interior_residual_upper = _sum_upper(_mul_up(margins, selected_residual))
        if not np.isfinite(result.gap_upper+result.stationarity_norm_upper+result.interior_residual_upper):
            result.reason = 'nonfinite_forward_error_envelope'
            return result, 0.
        if selected.size < self.p:
            result.reason = 'fewer_selected_interior_rows_than_coefficients'
            return result, 0.
        zrows = np.empty((len(selected), self.p), dtype=np.float64)
        ordinary = selected < self.n
        zrows[ordinary] = self.x[selected[ordinary]]
        zrows[~ordinary] = z0
        inverse_started = time.perf_counter()
        kappa, eta, inverse_norm, verification = _verified_kappa(zrows, margins)
        inverse_seconds = time.perf_counter()-inverse_started
        result.kappa_lower, result.inverse_residual_frobenius_upper = kappa, eta
        result.left_inverse_frobenius_upper = inverse_norm
        denominator = float(_down(kappa-result.stationarity_norm_upper))
        result.denominator_lower = max(0., denominator)
        if verification != 'verified':
            result.reason = verification
            return result, inverse_seconds
        if denominator <= 0:
            result.reason = 'stationarity_envelope_not_below_sharpness_bound'
            return result, inverse_seconds
        numerator = float(_add_up(result.gap_upper, result.interior_residual_upper))
        result.radius = float(_up(numerator/denominator))
        if not np.isfinite(result.radius):
            result.reason = 'radius_overflow'
            return result, inverse_seconds
        result.status, result.reason = 'finite', 'model_qualified_enclosure'
        return result, inverse_seconds
