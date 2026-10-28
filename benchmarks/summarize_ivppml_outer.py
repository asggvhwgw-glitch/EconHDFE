"""Summarize every completed benchmark and independently compare saved arrays.

Run only after all timed fits finish. The report lists failures rather than
silently excluding them. The gate is empirical on this suite, not universal.
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
import numpy as np


def stats(values):
    return {'median': float(statistics.median(values)), 'min': float(min(values)),
            'max': float(max(values)), 'runs': len(values)}


def compare(old, new):
    with np.load(str(old)+'.npz') as a, np.load(str(new)+'.npz') as b:
        same = bool(np.array_equal(a['sample'], b['sample']))
        errors = dict(sample_identical=same)
        for key in ('params', 'vcov', 'eta', 'mu'):
            x, y = a[key], b[key]
            finite = np.isfinite(x) & np.isfinite(y)
            errors[key+'_nonfinite_pattern_identical'] = bool(np.array_equal(np.isfinite(x),np.isfinite(y)))
            errors[key+'_max_abs'] = float(np.max(np.abs(x[finite]-y[finite]), initial=0.0))
            errors[key+'_max_scaled'] = float(np.max(np.abs(x[finite]-y[finite])/(1.+np.abs(x[finite])), initial=0.0))
            del x,y,finite
        errors['accuracy_pass'] = (same and all(errors[k+'_nonfinite_pattern_identical'] for k in ('params','vcov','eta','mu'))
            and errors['params_max_scaled'] <= 1e-6 and errors['vcov_max_scaled'] <= 1e-7
            and errors['eta_max_scaled'] <= 1e-5 and errors['mu_max_scaled'] <= 1e-5)
        return errors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arrays-directory',type=Path,default=None)
    args=parser.parse_args(); arrays=args.arrays_directory or args.directory
    groups={};failures=[]
    for path in sorted(args.directory.glob('*.json')):
        if path.name=='ledger.json':continue
        data=json.loads(path.read_text())
        if 'mode' not in data:continue
        if data.get('status')!='ok':failures.append({'path':path.name,'data':data});continue
        groups.setdefault((data['case'],data['n']),{}).setdefault(data['mode'],[]).append((path,data))
    cases=[]
    for (case,n), modes in groups.items():
        row=dict(case=case,n=n,measurements={},comparisons=[])
        hashes={d['data_sha256'] for runs in modes.values() for _,d in runs}
        row['identical_input_hashes']=len(hashes)==1;row['data_sha256']=sorted(hashes)
        for mode,runs in modes.items():
            info={key:stats([d[key] for _,d in runs]) for key in ('seconds','outer_seconds','peak_rss_mb','iterations')}
            info['linear_seconds']=stats([d['legacy_linear_seconds'] if mode=='legacy' else d['outer']['timings']['linear_solve'] for _,d in runs])
            if mode!='legacy':
                info['moment_error_max']=max(d['outer']['moment_error'] for _,d in runs)
                info['step_error_max']=max(d['outer']['step_error'] for _,d in runs)
                info['max_accepted_forcing']=max(d['outer']['max_accepted_forcing'] for _,d in runs)
                info['backtracks']=stats([d['outer']['line_search_backtracks'] for _,d in runs])
                info['stage_seconds']={key:stats([d['outer']['timings'][key] for _,d in runs]) for key in ('projection','linear_solve','moments','line_search')}
            row['measurements'][mode]=info
        for mode in ('moment','moment_stable'):
            if mode not in modes:continue
            ref=modes.get('legacy',modes.get('moment',modes[mode]))[0][0]
            for path,_ in modes[mode]:
                row['comparisons'].append(dict(reference=ref.name,candidate=path.name,**compare(arrays/ref.name,arrays/path.name)))
        if 'legacy' in modes and 'moment' in modes:
            old,new=row['measurements']['legacy'],row['measurements']['moment']
            row['full_speedup']=old['seconds']['median']/new['seconds']['median']
            row['outer_speedup']=old['outer_seconds']['median']/new['outer_seconds']['median']
            row['peak_rss_reduction_fraction']=1.-new['peak_rss_mb']['median']/old['peak_rss_mb']['median']
            row['performance_pass']=new['seconds']['median']<=old['seconds']['median']*1.05
        row['accuracy_pass']=row['identical_input_hashes'] and all(c['accuracy_pass'] for c in row['comparisons'])
        cases.append(row)
    ledger=json.loads((args.directory/'ledger.json').read_text())
    expected = {(c,n,mode):3 for c,n in [('crossed3',1000000),('overidentified',1000000),
        ('many_instruments',1000000),('offset',1000000),('crossed3',5000000)]
        for mode in ('legacy','moment')}
    expected.update({('weak',1000000,mode):1 for mode in ('legacy','moment')})
    expected['many_instruments',1000000,'moment_stable']=3
    completeness = [dict(case=c,n=n,mode=mode,expected=count,
        observed=len(groups.get((c,n),{}).get(mode,[])))
        for (c,n,mode),count in expected.items()
        if len(groups.get((c,n),{}).get(mode,[])) != count]
    complete = (not completeness and len(ledger)==sum(expected.values())
                and len({r['name'] for r in ledger})==len(ledger)
                and (args.directory/'DONE').exists())
    summary=dict(scope='Empirical same-host gate; no universal speedup or convergence guarantee',
        accuracy_tolerances=dict(params_scaled=1e-6,vcov_scaled=1e-7,eta_scaled=1e-5,mu_scaled=1e-5),
        performance_tolerance='new median full-estimator seconds <= 1.05 * legacy median',
        completed_runs=len(ledger),suite_complete=complete,missing_or_extra_counts=completeness,failures=failures,cases=cases,
        passed=complete and bool(cases) and not failures and all(r.get('returncode')==0 for r in ledger)
            and all(r['accuracy_pass'] and r.get('performance_pass',True) for r in cases))
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    for r in cases: print(r['case'],r['n'],round(r.get('full_speedup',0),3),r['accuracy_pass'])
    print('PASSED:',summary['passed'])
    if not summary['passed']:raise SystemExit(1)

if __name__=='__main__':main()
