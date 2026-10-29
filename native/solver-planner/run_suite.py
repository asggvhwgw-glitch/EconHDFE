"""Sequential, process-isolated benchmark suite; fail on any missing run."""
import subprocess, sys, json, os, platform, time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
out=ROOT/'results';out.mkdir(exist_ok=True)
jobs=[]
for case in ['unique_one','unique_two','unique_three','paired_three','repeated','weak_cycle']:
    for method in ['legacy_auto','v01_auto','planned']:
        jobs.append((case,1000000,8,8411,method,case=='unique_two'))
for case in ['unique_two','paired_three','repeated']:
    for method in ['legacy_auto','planned']:
        jobs.append((case,1000000,2,8411,method,case=='unique_two'))
for case in ['weak_cycle','repeated']:
    for method in ['legacy_auto','planned']:
        jobs.append((case,5000000,2,8411,method,False))
for case in ['unique_three','paired_three']:
    for method in ['legacy_auto','planned']:
        jobs.append((case,250000,8,17323,method,False))
log=[]
source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py') if 'build/' not in str(p)}
for case,n,k,seed,method,rw in jobs:
    key=f'{case}-{n}-{k}-{seed}-{method}'+('-w' if rw else '')
    target=out/(key+'.json')
    cmd=[sys.executable,str(ROOT/'benchmark.py'),'--case',case,'--rows',str(n),'--rhs',str(k),'--seed',str(seed),'--method',method,'--rounds','3','--output',str(target)]
    if rw:cmd+=['--reweight']
    print('START',key,flush=True)
    t=time.perf_counter()
    try:
        p=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=480)
        print(p.stdout,flush=True)
        if p.returncode:print(p.stderr[-1500:],flush=True)
        log.append(dict(key=key,exit=p.returncode,elapsed=time.perf_counter()-t,stderr=p.stderr))
    except subprocess.TimeoutExpired as e:
        log.append(dict(key=key,exit=124,elapsed=time.perf_counter()-t,error='480 second per process limit'))
        print('TIMEOUT',key,flush=True)
    (out/'suite-status.json').write_text(json.dumps(dict(jobs=len(jobs),completed=len(log),runs=log,source_sha256=source_hashes),indent=2))
assert len(log)==len(jobs) and all(x['exit']==0 for x in log), log
print('SUITE_COMPLETE',len(log),flush=True)
