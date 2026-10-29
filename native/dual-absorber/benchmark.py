"""Same-machine, same-input prototype vs existing NativeSolver baseline.

Independent known weighted residuals provide a forward-error check. Every tuple
is repeated at least twice to construct that oracle; these tests are therefore
NOT a replacement for fully unique-row or genuine AKM benchmarks.
"""
from __future__ import annotations
import argparse, gc, json, os, platform, resource, time
from pathlib import Path
import numpy as np


def corpus(case, n, rhs, seed=20261029):
    rng = np.random.default_rng(seed)
    n -= n % 2
    u = n//2
    repeat = 8 if case == 'repeated' else 1
    cells = u//repeat
    u = cells*repeat
    n = 2*u
    if case == 'one_big':
        base = np.column_stack([np.arange(cells)%max(2, n//8), rng.integers(48, size=cells), rng.integers(16, size=cells)])
    elif case == 'three_big':
        base = np.column_stack([rng.integers(max(2, n//l), size=cells) for l in (10, 12, 16)])
    elif case == 'weak':
        communities = 32
        a_per = max(2, n//24//communities)
        b_per = max(2, n//32//communities)
        c = np.arange(cells)%communities
        a = c*a_per + (np.arange(cells)//communities)%a_per
        b = c*b_per + rng.integers(b_per, size=cells)
        bridge = rng.random(cells)<.0005
        b[bridge] = rng.integers(communities*b_per, size=bridge.sum())
        base = np.column_stack([a,b])
    else:
        base = np.column_stack([rng.integers(max(2, min(n//12, cells//4)), size=cells),
                                rng.integers(max(2, min(n//20, cells//6)), size=cells),
                                rng.integers(16, size=cells)])
    unique = np.repeat(base, repeat, axis=0)
    codes = np.repeat(unique, 2, axis=0).astype(np.uint32)
    w = np.repeat(np.exp(rng.normal(scale=.5, size=u)), 2)
    error = rng.normal(size=(u, rhs))
    truth = np.empty((n, rhs))
    truth[::2] = error
    truth[1::2] = -error
    y = truth.copy()
    for col in codes.T:
        y += rng.normal(scale=.5, size=(int(col.max())+1, rhs))[col]
    # Shuffle pairs (not individual members), preserving the exact weighted oracle.
    pair_order = rng.permutation(u)
    order = np.column_stack([2*pair_order, 2*pair_order+1]).ravel()
    return codes[order], y[order], w[order], truth[order]


def execute(case, n, rhs, repeat, threads, methods):
    from econhdfe_dual import DualAbsorber
    from _econhdfe_within import NativeSolver
    codes, y, w, truth = corpus(case, n, rhs)
    n = len(y)
    records = []
    reference = None
    for method in methods:
        gc.collect()
        t0 = time.perf_counter()
        if method == 'legacy_auto':
            s = NativeSolver(codes, weights=w, strategy='auto', threads=threads)
        else:
            s = DualAbsorber(codes, weights=w, backend=method, threads=threads, allow_fallback=False)
        constructor = time.perf_counter()-t0
        rounds = []
        for step in range(repeat):
            wi = w if case != 'reweight' else w*np.repeat(np.exp(.12*np.sin(np.arange(n//2)*(.031+.007*step))), 2)
            t0 = time.perf_counter()
            if step > 0 and case == 'reweight': s.update_weights(wi)
            elif step == 0 and case == 'reweight': s.update_weights(wi)
            update = time.perf_counter()-t0
            t0 = time.perf_counter()
            if method == 'legacy_auto':
                out, ok, iterations, certificate = s.solve(y, tol=1e-10, max_iter=6000)
                info = dict(backend=s.route, accepted=all(ok), iterations=iterations,
                            certificate=certificate, setup_seconds=0.0)
            else:
                out, info = s.residualize(y, tol=1e-10, max_iter=6000, return_info=True)
            elapsed = time.perf_counter()-t0
            err = np.sqrt(np.sum(wi[:,None]*(out-truth)**2, axis=0)/np.sum(wi[:,None]*truth**2, axis=0)).max()
            valid = bool(info['accepted'] and np.isfinite(out).all() and err <= 2e-8)
            row = dict(step=step, seconds=elapsed, update_seconds=update,
                       kernel_setup_seconds=info.get('setup_seconds'),
                       forward_error=float(err), valid=valid, route=info['backend'],
                       iterations=info['iterations'], certificate=info['certificate'],
                       details=info if method != 'legacy_auto' else None)
            rounds.append(row)
            if not valid: raise AssertionError((case, method, row))
        record = dict(case=case, n=n, rhs=rhs, method=method, constructor_seconds=constructor,
                      rounds=rounds, threads=threads, valid=all(v['valid'] for v in rounds),
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      memory_scope='process cumulative HWM; NOT isolated per-method memory',
                      data_scope='paired identical FE tuples; known weighted residual oracle')
        print('PROTOTYPE_RESULT '+json.dumps(record, allow_nan=False), flush=True)
        records.append(record)
        del s, out
    return records


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case', choices=['one_big','two_big','three_big','weak','repeated','reweight'], required=True)
    p.add_argument('--rows', type=int, default=1000000)
    p.add_argument('--rhs', type=int, default=8)
    p.add_argument('--repeat', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output', type=Path)
    a=p.parse_args()
    results=execute(a.case,a.rows,a.rhs,a.repeat,a.threads,['within','fereg','auto','legacy_auto'])
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(dict(arguments=vars(a)|{'output':str(a.output)},
                                              platform=platform.platform(),results=results),indent=2))

if __name__=='__main__': main()
