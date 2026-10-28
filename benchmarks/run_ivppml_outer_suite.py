"""Repeat the published same-host benchmark suite in sequential fresh processes.

The measured implementation does not change across cases. No run is discarded
based on speed or numerical outcome. Read ledger.json for failures/timeouts.
"""
import subprocess,sys,os,json,time, pathlib, traceback
root=pathlib.Path(__file__).resolve().parents[1]
import argparse
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=pathlib.Path,default=root/'validation/iv_outer/reproduction')
a=p.parse_args();dest=a.output_dir.resolve();dest.mkdir(parents=True,exist_ok=True)
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','NUMBA_NUM_THREADS':'4'}
tasks=[]
for case,n in [('crossed3',1000000),('overidentified',1000000),('many_instruments',1000000),('offset',1000000),('crossed3',5000000)]:
 for rep in range(1,4):
  for mode in (['legacy','moment'] if rep%2 else ['moment','legacy']): tasks.append((case,n,mode,rep))
for mode in ['legacy','moment']:tasks.append(('weak',1000000,mode,1))
for rep in range(1,4):tasks.append(('many_instruments',1000000,'moment_stable',rep))
ledger=[]
for case,n,mode,rep in tasks:
 name=f'{case}_{n}_{mode}_r{rep}'
 command=[sys.executable,'benchmarks/ivppml_outer.py','--n',str(n),'--case',case,'--mode',mode,'--output',str(dest/(name+'.json'))]
 print('START',name,flush=True)
 with (dest/(name+'.log')).open('w') as log:
  try:
   run=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
   returncode=run.returncode
  except subprocess.TimeoutExpired:
   returncode=124
   log.write('TIMEOUT: 240-second subprocess budget exceeded\n')
 record=dict(name=name,returncode=returncode)
 path=dest/(name+'.json')
 if path.exists():
  d=json.loads(path.read_text());record.update(status=d['status'],seconds=d['seconds'],iterations=d.get('iterations'),rss=d['peak_rss_mb'])
 ledger.append(record);(dest/'ledger.json').write_text(json.dumps(ledger,indent=2))
 print('DONE',record,flush=True)
(dest/'DONE').write_text('completed\n')
