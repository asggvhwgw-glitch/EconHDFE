"""Private, disposable trial worker. Protocol: JSON lines and private .npy files.

Never fork an initialized native thread pool. This module is started by a fresh
Python interpreter. A killed trial shares no mutable solver/lock with its parent.
Only the private parent writes request files. No user-provided code is executed.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
from .planner import _Route


def main():
    root = Path(sys.argv[1]).resolve()
    route = None
    for line in sys.stdin:
        try:
            req = json.loads(line)
            op = req['op']
            if op == 'close':
                break
            timing = {}
            if route is None:
                if op != 'project':
                    raise RuntimeError('uninitialized worker')
                t = perf_counter()
                codes = np.load(root/'codes.npy', mmap_mode='r', allow_pickle=False)
                weights = np.load(root/'weights.npy', mmap_mode='r', allow_pickle=False)
                route = _Route(req['backend'], codes, weights, req['threads'], req['scratch_bytes'])
                timing['construction_seconds'] = perf_counter()-t
            if op == 'update' or req.get('refresh', False):
                t = perf_counter()
                weights = np.load(root/'weights.npy', mmap_mode='r', allow_pickle=False)
                route.update_weights(weights)
                timing['update_seconds'] = perf_counter()-t
            if op == 'update':
                reply = dict(ok=True, timing=timing)
            elif op == 'project':
                data = np.load(root/'input.npy', mmap_mode='r', allow_pickle=False)
                t = perf_counter()
                out, info = route.project(data, req['tol'], req['max_iter'])
                timing['solve_seconds'] = perf_counter()-t
                if not info.get('accepted') or not np.isfinite(out).all():
                    raise RuntimeError('uncertified result')
                np.save(root/'output.npy', out, allow_pickle=False)
                reply = dict(ok=True, info=info, timing=timing)
            else:
                raise ValueError('unknown worker operation')
        except (RuntimeError, ValueError, MemoryError, OSError) as exc:
            reply = dict(ok=False, error=f'{type(exc).__name__}: {exc}')
        print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
