"""Cluster score covariance using established grouped linear algebra."""

from __future__ import annotations


def cluster_meat(x, groups, method="auto"):
    """Return the cluster covariance meat, normalized by observation count.

    Parameters
    ----------
    x : array_like, shape (n, k) or (n,)
        Finite real score rows. A vector is interpreted as one score column.
        Scores are converted to float64 and are **not** implicitly demeaned.
    groups : array_like, shape (n,)
        Non-missing, mutually orderable labels. Strings, signed or sparse
        integer identifiers, and floating-point labels are supported. Labels
        are never treated directly as array positions.
    method : {"auto", "reduceat", "bincount"}
        ``auto`` selects the general grouped-reduction implementation.
        ``reduceat`` avoids sorting labels already in nondecreasing order.
        ``bincount`` factorizes labels then aggregates each score column.

    Returns
    -------
    ndarray, shape (k, k)
        ``sum_g outer(sum_i_in_g x_i, sum_i_in_g x_i) / n``. Apply estimator
        bread and any small-sample adjustment separately. Input arrays are
        never modified. Different reductions can change floating-point order.

    Notes
    -----
    Singleton and one-group shortcuts are verified from observed labels.
    There is no group-plan cache or hidden setup excluded from a call.
    This is an optimization of an existing statistic, not a new estimator.
    """
    import numpy as np

    if method not in ("auto", "reduceat", "bincount"):
        raise ValueError("method must be 'auto', 'reduceat', or 'bincount'")
    raw = np.asarray(x)
    if raw.dtype.kind not in "buif":
        raise ValueError("x must contain real numeric scores")
    scores = np.asarray(raw, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    if scores.ndim != 2 or min(scores.shape) == 0:
        raise ValueError("x must be a nonempty vector or N-by-K array")
    if not np.isfinite(scores).all():
        raise ValueError("x must contain only finite scores")
    labels = np.asarray(groups)
    if not isinstance(groups, np.ndarray) and labels.dtype.kind == "f":
        # NumPy can infer float64 for Python integer sequences spanning signed
        # and unsigned 64-bit ranges, or mixed large integers and floats. Keep
        # original scalar identities when a large integer would lose precision.
        object_labels = np.asarray(groups, dtype=object)
        if any(isinstance(label, (int, np.integer)) and abs(int(label)) > 2**53
               for label in object_labels.flat):
            labels = object_labels
            # Python numeric comparison retains exact integer/float equality;
            # NumPy float scalar comparison can coerce a large Python integer.
            for index, label in enumerate(labels.flat):
                if isinstance(label, np.generic):
                    labels.flat[index] = label.item()
    if labels.dtype.kind in "SU" and not isinstance(groups, np.ndarray):
        # A mixed Python sequence such as ["a", np.nan] must not silently
        # stringify the missing value (or merge numeric 1 with string "1").
        object_labels = np.asarray(groups, dtype=object)
        if any(not isinstance(label, (str, bytes)) for label in object_labels.flat):
            labels = object_labels
    n, k = scores.shape
    if labels.ndim != 1 or labels.shape[0] != n:
        raise ValueError("groups must be one-dimensional with one label per score row")
    if labels.dtype.kind in "fc" and np.isnan(labels).any():
        raise ValueError("groups must not contain missing labels")
    if labels.dtype.kind in "mM" and np.isnat(labels).any():
        raise ValueError("groups must not contain missing labels")
    if labels.dtype.kind in "cV":
        raise ValueError("groups must contain mutually orderable real or string labels")
    if labels.dtype.kind == "O":
        # Numeric/string arrays avoid this Python check. Object arrays require
        # explicit handling of None, NaN, NaT and nullable scalar sentinels.
        for label in labels:
            try:
                missing = label is None or bool(label != label)
            except (TypeError, ValueError) as exc:
                raise ValueError("groups must contain non-missing scalar labels") from exc
            if missing:
                raise ValueError("groups must not contain missing labels")

    if method == "bincount":
        try:
            values, codes = np.unique(labels, return_inverse=True)
        except TypeError as exc:
            raise ValueError("groups must contain mutually orderable labels") from exc
        ngroups = values.size
        if ngroups == n:
            return (scores.T @ scores) / n
        if ngroups == 1:
            total = scores.sum(axis=0)
            return np.outer(total, total) / n
        aggregate = np.empty((ngroups, k), dtype=np.float64)
        for column in range(k):
            aggregate[:, column] = np.bincount(
                codes, weights=scores[:, column], minlength=ngroups
            )
    else:
        try:
            is_sorted = bool(np.all(labels[1:] >= labels[:-1]))
            order = None if is_sorted else np.argsort(labels)
        except TypeError as exc:
            raise ValueError("groups must contain mutually orderable labels") from exc
        ordered_labels = labels if order is None else labels[order]
        starts = np.r_[0, np.flatnonzero(ordered_labels[1:] != ordered_labels[:-1]) + 1]
        if starts.size == n:
            return (scores.T @ scores) / n
        if starts.size == 1:
            total = scores.sum(axis=0)
            return np.outer(total, total) / n
        ordered_scores = scores if order is None else scores[order]
        aggregate = np.add.reduceat(ordered_scores, starts, axis=0)
    return (aggregate.T @ aggregate) / n
