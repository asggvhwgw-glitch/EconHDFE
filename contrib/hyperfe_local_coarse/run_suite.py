"""Portable explicit-parameter runner; not an automatic threshold selector."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=['equal_sparse', 'smooth_sparse', 'irregular', 'nonadditive'], required=True)
    p.add_argument('--seed', type=int, default=20261028)
    p.add_argument('--tau', type=float, required=True)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--outdir', type=Path, default=Path('results/reproduce'))
    args = p.parse_args()
    if args.repetitions < 1 or args.timeout < 1 or not 0 < args.tau < 1:
        p.error('positive repetitions/timeout and 0 < tau < 1 required')
    args.outdir.mkdir(parents=True, exist_ok=True)
    benchmark = Path(__file__).resolve().with_name('benchmark.py')
    outcomes = []
    for rep in range(args.repetitions):
        modes = ['balanced', 'native_adaptive']
        if rep % 2:
            modes.reverse()
        for mode in modes:
            output = args.outdir / f'{args.case}_{mode}_{args.seed}_{rep}.json'
            command = [sys.executable, str(benchmark), '--case', args.case,
                       '--seed', str(args.seed), '--tau', str(args.tau),
                       '--mode', mode, '--n', '1000000', '--rhs', '3',
                       '--tol', '1e-12', '--out', str(output)]
            if output.exists():
                p.error(f'refusing to overwrite {output}; choose a fresh outdir')
            record = {'command': command}
            with output.with_suffix('.log').open('w') as log:
                try:
                    run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=args.timeout, check=False)
                    record['returncode'] = run.returncode
                    if output.exists():
                        result = json.loads(output.read_text())
                        record.update(status=result['status'], passed=result.get('passed'))
                    else:
                        record.update(status='missing_result', passed=False)
                except subprocess.TimeoutExpired:
                    record.update(status='timeout', passed=False)
            outcomes.append(record)
    (args.outdir / 'run_status.json').write_text(json.dumps(outcomes, indent=2) + '\n')
    return 0 if all(r.get('passed') is True for r in outcomes) else 1


if __name__ == '__main__':
    raise SystemExit(main())
