"""Cluster-level score reduction and reproducible bounded multiplier blocks."""
import numpy as np
import numbers

def encode_clusters(cluster, n):
    # Preserve Python labels before any NumPy mixed-type coercion. Converting
    # [1, nan, 'x'] to strings would turn missingness into a valid cluster.
    values=(cluster if isinstance(cluster,np.ndarray) else np.asarray(cluster,dtype=object))
    if values.ndim!=1 or len(values)!=n:
        raise ValueError('cluster must have one label for every observation')
    if values.dtype.kind in 'fc' and not np.isfinite(values).all():
        raise ValueError('cluster labels must not be missing or infinite')
    if values.dtype.kind=='O':
        numeric=all(isinstance(v,numbers.Real) for v in values)
        strings=all(isinstance(v,(str,np.str_)) for v in values)
        if not (numeric or strings):
            raise ValueError('Cluster labels must be homogeneous numeric or string values without missingness')
        if numeric and any(v!=v or v in (float('inf'),-float('inf')) for v in values):
            raise ValueError('cluster labels must not be missing or infinite')
    elif values.dtype.kind not in 'biufUS':
        raise ValueError('Unsupported cluster label type')
    labels,codes=np.unique(values,return_inverse=True)
    if len(labels)<2:
        raise ValueError('Cluster bootstrap requires at least two clusters')
    return labels,codes

def cluster_scores(X, psi, codes, groups, *, block_rows=16384):
    """Compute S[g,:]=sum_{i in g} X[i,:]*psi[i], without n-by-B arrays.

    psi is explicitly supplied so a reference solver's exact residual-sign
    convention can be preserved. KKT dual scores are not substituted for psi.
    """
    x=np.asarray(X,dtype=np.float64)
    psi=np.asarray(psi,dtype=np.float64)
    codes=np.asarray(codes)
    if x.ndim!=2 or psi.shape!=(len(x),) or codes.shape!=(len(x),):
        raise ValueError('Incompatible X, psi, and cluster codes')
    if block_rows<1 or groups<1 or codes.dtype.kind not in 'iu':
        raise ValueError('Invalid block size or integer cluster codes')
    if codes.min(initial=0)<0 or codes.max(initial=0)>=groups:
        raise ValueError('Cluster codes are out of range')
    if not np.isfinite(x).all() or not np.isfinite(psi).all():
        raise ValueError('X and psi must be finite')
    out=np.zeros((groups,x.shape[1]))
    for first in range(0,len(x),block_rows):
        last=min(first+block_rows,len(x))
        np.add.at(out,codes[first:last],x[first:last]*psi[first:last,None])
    return out

def mammen_block(groups, replicate_indices, seed):
    """Draw indexed multipliers; changing block size preserves each draw.

    Each replicate has its own SeedSequence stream. The same seed/index across
    quantiles preserves process dependence without storing G-by-B weights.
    This RNG is not bit-identical to R; supply matched weights for R comparison.
    """
    if groups<1 or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError('Require a positive group count and nonnegative integer seed')
    indices=list(replicate_indices)
    low=-(np.sqrt(5)-1)/2
    high=(np.sqrt(5)+1)/2
    probability=(np.sqrt(5)+1)/np.sqrt(20)
    out=np.empty((groups,len(indices)))
    for j,index in enumerate(indices):
        if not isinstance(index,(int,np.integer)) or index<0:
            raise ValueError('Replicate indices must be nonnegative integers')
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(index)]))
        out[:,j]=np.where(rng.random(groups)<probability,low,high)
    return out
