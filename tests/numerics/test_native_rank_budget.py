"""Native exact-rank resource guards; no finite-field lower bound is a result."""

import importlib
import numpy as np
import pytest


rank = importlib.import_module("econhdfe.hdfe.rank")


def parity_edges():
    return np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])


def test_public_native_failure_never_returns_bad_prime_lower_bound(monkeypatch):
    monkeypatch.setattr(rank, "_EXACT_PRIME", 2)
    monkeypatch.setattr(rank, "_NATIVE_MAX_WORK", 1)
    with pytest.raises(RuntimeError, match="no exact rank returned"):
        rank.categorical_rank(list(parity_edges().T), backend="native")
