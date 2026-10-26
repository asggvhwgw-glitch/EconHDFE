"""Backing-array accounting for disposable execution and session caches."""
from dataclasses import fields, is_dataclass
import numpy as np
import pandas as pd


def array_owners(value):
    """Count shared ndarray storage once; do not walk arbitrary user objects."""
    owners, seen = {}, set()
    def visit(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            owner = item
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            owners[id(owner)] = int(owner.nbytes)
        elif isinstance(item, pd.DataFrame):
            # Existing block storage; column conversion may allocate copies.
            for block in item._mgr.blocks:
                visit(block.values)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif type(item).__module__.startswith(("econhdfe.", "scipy.sparse")):
            if is_dataclass(item):
                for field in fields(item):
                    visit(getattr(item, field.name))
            elif hasattr(item, "__dict__"):
                visit(vars(item))
            else:
                for name in getattr(type(item), "__slots__", ()):
                    if hasattr(item, name):
                        visit(getattr(item, name))
    visit(value)
    return owners


def retained_array_bytes(cache, inputs):
    excluded = array_owners(inputs)
    return sum(size for key, size in array_owners(cache).items() if key not in excluded)
