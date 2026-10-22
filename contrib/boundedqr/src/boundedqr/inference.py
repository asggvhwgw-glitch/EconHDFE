"""Inference summaries and deterministic propagation of coefficient enclosures.

These summaries do not add econometric validity to a bootstrap; clustering and
model assumptions must be appropriate. Numerical enclosures concern optimization
of fixed draws, not Monte Carlo error or sampling uncertainty.

The SD enclosure assumes round-to-nearest IEEE binary64 with gradual underflow
and correctly rounded scalar arithmetic/sqrt, and a NumPy reduction satisfying
the classical gamma_B summation model. This is model-qualified FP64 error
analysis, not a formal interval proof or a check of the installed BLAS/runtime.
It includes arithmetic error in the reported sample SD separately from error
induced by the supplied coefficient radii. Covariance and confidence-band point
summaries do not thereby acquire complete floating-point certificates.
"""
import numpy as np
from fractions import Fraction


def _nonnegative_up(value):
    """Round a nonnegative result outwards, preserving exact zero."""
    return np.where(value == 0, 0., np.nextafter(value, np.inf))


def _positive_sum_bounds(values, gamma):
    """Bound the exact sum of nonnegative represented values along axis zero.

    At most B rounded additions occur. Subnormal additions of nonnegative
    binary64 operands are exact while their result is subnormal; products and
    divisions are separately enclosed by nextafter in the caller.
    """
    total = np.sum(values, axis=0)
    denominator_hi = np.nextafter(1. + gamma, np.inf)
    denominator_lo = np.nextafter(1. - gamma, -np.inf)
    lower = np.maximum(0., np.nextafter(total / denominator_hi, -np.inf))
    upper = _nonnegative_up(total / denominator_lo)
    return lower, upper


def _square_interval(lower, upper):
    """Pointwise square of an interval, including intervals crossing zero."""
    small = np.where(lower > 0, lower, np.where(upper < 0, -upper, 0.))
    large = np.maximum(np.abs(lower), np.abs(upper))
    lo = np.maximum(0., np.nextafter(small * small, -np.inf))
    hi = _nonnegative_up(large * large)
    # If a nonzero product rounds to zero, its true square needs a subnormal
    # upper endpoint. The exact-zero case alone may keep upper == zero.
    hi = np.where((large != 0) & (hi == 0), np.nextafter(0., np.inf), hi)
    return lo, hi


def _sd_enclosure(draws):
    """Enclose exact sample SDs of represented draws in O(B*Q*p) work.

    Use ANY represented centre c. For t_i=(x_i-c)/scale, the exact identity is
    SSE(x)=scale**2 * (sum(t_i**2)-sum(t_i)**2/B). The centre need not equal the
    exact mean. Scaling before squaring avoids an unnecessary underflow floor
    on tiny but nonzero standard deviations. No accuracy of np.std is assumed.
    """
    count = len(draws)
    unit = np.finfo(np.float64).eps / 2
    bu = count * unit
    if bu >= .5:
        raise ValueError('Too many draws for the declared FP64 summation bound')
    gamma = np.nextafter(bu / np.nextafter(1. - bu, -np.inf), np.inf)
    center = np.mean(draws, axis=0)
    same = np.all(draws == draws[0], axis=0)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        difference = draws - center[None, :, :]
        # Constant columns have exact SD zero even if a rounded mean differs
        # from their common value. Avoid needless extreme intermediate values.
        difference = np.where(same[None, :, :], 0., difference)
        difference_lo = np.nextafter(difference, -np.inf)
        difference_hi = np.nextafter(difference, np.inf)
        scale = np.maximum(np.max(np.abs(difference_lo), axis=0),
                           np.max(np.abs(difference_hi), axis=0))
        scale = np.where(same, 1., scale)
        lower = np.nextafter(difference_lo / scale[None, :, :], -np.inf)
        upper = np.nextafter(difference_hi / scale[None, :, :], np.inf)

        square_lo, square_hi = _square_interval(lower, upper)
        q_lo, _ = _positive_sum_bounds(square_lo, gamma)
        _, q_hi = _positive_sum_bounds(square_hi, gamma)
        # Signed sums need absolute-sum error bounds; treating a cancelled sum
        # as having only relative error would invalidate the enclosure.
        _, abs_lo = _positive_sum_bounds(np.abs(lower), gamma)
        _, abs_hi = _positive_sum_bounds(np.abs(upper), gamma)
        error_lo = np.nextafter(gamma * abs_lo, np.inf)
        error_hi = np.nextafter(gamma * abs_hi, np.inf)
        sum_lo = np.nextafter(np.sum(lower, axis=0) - error_lo, -np.inf)
        sum_hi = np.nextafter(np.sum(upper, axis=0) + error_hi, np.inf)
        s2_lo, s2_hi = _square_interval(sum_lo, sum_hi)
        correction_lo = np.maximum(0., np.nextafter(s2_lo / count, -np.inf))
        correction_hi = np.nextafter(s2_hi / count, np.inf)
        sse_lo = np.maximum(0., np.nextafter(q_lo - correction_hi, -np.inf))
        sse_hi = np.maximum(0., np.nextafter(q_hi - correction_lo, np.inf))
        variance_lo = np.maximum(0., np.nextafter(sse_lo / (count - 1), -np.inf))
        variance_hi = np.nextafter(sse_hi / (count - 1), np.inf)
        root_lo = np.maximum(0., np.nextafter(np.sqrt(variance_lo), -np.inf))
        root_hi = _nonnegative_up(np.sqrt(variance_hi))
        sd_lo = np.maximum(0., np.nextafter(root_lo * scale, -np.inf))
        sd_hi = np.nextafter(root_hi * scale, np.inf)
    return np.where(same, 0., sd_lo), np.where(same, 0., sd_hi)

def _quantile(a, probabilities):
    ordered=np.sort(a,axis=0)
    values=[]
    for probability in probabilities:
        index=(len(ordered)-1)*probability
        left=int(np.floor(index))
        weight=index-left
        if weight==0:
            values.append(ordered[left])
        else:
            values.append((1-weight)*ordered[left]+weight*ordered[left+1])
    return np.asarray(values)

def _quantile_enclosure(a, probabilities, direction):
    ordered=np.sort(a,axis=0)
    target=-np.inf if direction=='lower' else np.inf
    out=[]
    for prob in probabilities:
        h=Fraction.from_float(float(prob))*(len(a)-1)
        left=h.numerator//h.denominator
        exact_weight=h-left
        lo=ordered[left]
        if exact_weight==0:
            out.append(lo)
            continue
        hi=ordered[left+1]
        weight=float(exact_weight)
        if ((direction=='lower' and Fraction.from_float(weight)>exact_weight) or
            (direction=='upper' and Fraction.from_float(weight)<exact_weight)):
            weight=np.nextafter(weight,target)
        with np.errstate(invalid='ignore',over='ignore'):
            delta=np.nextafter(hi-lo,target)
            q=np.nextafter(lo+np.nextafter(weight*delta,target),target)
        q=np.where(np.isneginf(lo),-np.inf,q)
        q=np.where(np.isposinf(hi),np.inf,q)
        out.append(q)
    return np.asarray(out)

def summarize_process(beta, draws, *, level=.95, radii=None):
    beta=np.asarray(beta,dtype=float)
    draws=np.asarray(draws,dtype=float)
    if beta.ndim!=2 or draws.ndim!=3 or draws.shape[1:]!=beta.shape or len(draws)<2:
        raise ValueError('Expected beta(Q,p), draws(B,Q,p), B>=2')
    if not 0<level<1 or not np.isfinite(beta).all() or not np.isfinite(draws).all():
        raise ValueError('Finite inputs and 0<level<1 required')
    alpha=1-level
    try:
        with np.errstate(over='raise',invalid='raise'):
            se=draws.std(axis=0,ddof=1)
            deviation=draws-beta[None,:,:]
            covariance=np.cov(draws.reshape(len(draws),-1),rowvar=False)
    except FloatingPointError as exc:
        raise ValueError('Inference summary overflows FP64; rescale the input coefficients') from exc
    quantile=np.quantile(deviation,[alpha/2,1-alpha/2],axis=0,method='linear')
    basic=np.stack([beta-quantile[1],beta-quantile[0]],axis=-1)
    # Zero standard errors are explicit; never turn them into finite t-values.
    good=se>0
    stats=np.max(np.divide(np.abs(deviation),se[None,:,:],
                          out=np.full_like(deviation,np.inf),where=good[None,:,:]),axis=(1,2))
    critical=float(np.quantile(stats,level,method='linear')) if np.isfinite(stats).all() else float('inf')
    band=(np.stack([beta-critical*se,beta+critical*se],axis=-1) if np.isfinite(critical)
          else np.broadcast_to([-np.inf,np.inf],beta.shape+(2,)).copy())
    result=dict(standard_errors=se,covariance=np.atleast_2d(covariance),basic_intervals=basic,
                uniform_band=band,uniform_critical_value=critical,level=level,
                quantile_method='linear',zero_standard_error=~good)
    if radii is not None:
        radius=np.asarray(radii,dtype=float)
        if radius.shape!=draws.shape[:2] or np.any(radius<0) or np.isnan(radius).any():
            raise ValueError('radii must have shape(B,Q), with nonnegative values')
        lower=np.nextafter(draws-radius[:,:,None],-np.inf)
        upper=np.nextafter(draws+radius[:,:,None],np.inf)
        # Coordinatewise monotonicity of empirical quantiles gives enclosures
        # for the FIXED finite-draw quantiles, including linear interpolation.
        out_lo=_quantile_enclosure(lower,[alpha/2,1-alpha/2],'lower')
        out_hi=_quantile_enclosure(upper,[alpha/2,1-alpha/2],'upper')
        # sqrt(B/(B-1))*max radius bounds each sample-SD error, conditional
        # on all radii enclosing the corresponding exact coefficient vector.
        ratio=np.nextafter(len(draws)/(len(draws)-1),np.inf)
        factor=np.nextafter(np.sqrt(ratio),np.inf)
        se_optimization=_nonnegative_up(factor*radius.max(axis=0)[:,None])
        try:
            sd_lower,sd_upper=_sd_enclosure(draws)
        except FloatingPointError as exc:
            raise ValueError('SD enclosure overflows FP64; rescale the input coefficients') from exc
        se_arithmetic=_nonnegative_up(np.maximum(np.abs(se-sd_lower),np.abs(se-sd_upper)))
        se_optimization=np.broadcast_to(se_optimization,se.shape).copy()
        se_error=_nonnegative_up(se_optimization+se_arithmetic)
        result.update(bootstrap_quantile_lower=out_lo,bootstrap_quantile_upper=out_hi,
                      standard_error_optimization_bound=se_optimization,
                      standard_error_arithmetic_bound=se_arithmetic,
                      standard_error_numerical_bound=se_error,
                      standard_error_bound_scope='Coefficient perturbation plus FP64 arithmetic under the gamma_B/IEEE model; conditional on fixed input draws and supplied radii')
    return result
