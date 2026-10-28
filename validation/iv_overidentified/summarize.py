"""Rebuild this checkpoint's compact summary from the retained raw JSON files."""
from pathlib import Path
import hashlib, json, statistics
import numpy as np
ROOT = Path(__file__).resolve().parent

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

raw=[]
for p in sorted((ROOT/'formal').glob('*.json')):
    if p.name=='ledger.json':
        continue
    r=json.loads(p.read_text())
    if 'error' in r:
        raise RuntimeError(f'Failed final benchmark: {p.name}: {r["error"]}')
    r['_file']=p.relative_to(ROOT).as_posix(); raw.append(r)
rows=[]
for key in sorted({(r['n'],r['mode'],r['inference']) for r in raw}):
    rs=[r for r in raw if (r['n'],r['mode'],r['inference'])==key]
    seconds=[r['seconds'] for r in rs]
    rows.append(dict(n=key[0],mode=key[1],inference=key[2],repetitions=len(rs),
        median_seconds=statistics.median(seconds),min_seconds=min(seconds),max_seconds=max(seconds),
        median_peak_rss_mib=statistics.median([r['peak_rss_mib'] for r in rs]),
        outer_iterations=sorted({r['iterations'] for r in rs}),
        overid_test_included=(key[2]=='estimating_equation'),files=[r['_file'] for r in rs]))
parity=[]
for n in sorted({r['n'] for r in raw}):
    rs=[r for r in raw if r['n']==n]
    ref=next(r for r in rs if r['mode']=='moment' and r['inference']=='standard')
    row=dict(n=n,max_abs_coefficient_difference=max(float(np.max(np.abs(np.array(r['params'])-ref['params']))) for r in rs))
    for inf in ['standard','estimating_equation']:
        a=[r for r in rs if r['mode']=='moment' and r['inference']==inf]
        b=[r for r in rs if r['mode']=='profile' and r['inference']==inf]
        row[inf+'_covariance_parity']=None if not a or not b else dict(max_abs=float(np.max(np.abs(np.array(a[0]['vcov'])-b[0]['vcov']))))
    parity.append(row)
sim=[]
for name in ['simulation.json','simulation_12k.json']:
    r=json.loads((ROOT/name).read_text())
    assert 'error' not in r and len(r['records'])==r['reps']*2
    sim.append(dict(n=r['n'],repetitions_per_specification=r['reps'],summary=r['summary'],source=name))
checks={}
for name,expected in [('full-tests-final.log','589 passed, 1 skipped'),('installed-tests.log','116 passed'),('numerics_final.log','64 passed')]:
    txt=(ROOT/name).read_text()
    checks[name]=dict(expected_marker=expected,verified=expected in txt,sha256=digest(ROOT/name))
    assert expected in txt, name
summary=dict(checkpoint='overidentified follow-on to PR #36; local dev1; not rebased onto main',
    parent_github_commit='76523437e7e3f2ab5911f17d802c74a9b7041420',
    main_observed='85e5ff90fec2847e3dc8d5e4cf223a560f47a152',
    defaults_unchanged=True,new_overid_cases=28,full_tests=dict(passed=589,skipped=1),
    installed_wheel_tests=116,benchmarks=rows,parameter_and_vcov_parity=parity,simulations=sim,
    source_tests=checks,
    limitations=['5M timings have one run per configuration, not repeated speed certification.',
     'EE timing includes the optional overidentification test; standard timing does not.',
     '2048 MiB is the requested workspace admission budget, not a process RSS limit.',
     'Full profile changes numerical path, not the projected-IV estimating equation.',
     'No weak-IV, arbitrary growing-FE or finite-sample nominal coverage guarantee.',
     'Installed wheel uses host dependencies; not isolated build or clean dependency-resolution certification.',
     'PR #35/main integration and public version reconciliation remain pending.'])
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['source_tests','simulations']},indent=2))
