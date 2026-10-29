"""One process per method/configuration, interleaved order and no cached timings."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=Path(sys.argv[1]).resolve(); OUT.mkdir(exist_ok=True,parents=True)
configs=[
 ('unique_one',1000000,8,1,31733,False),
 ('unique_two',1000000,2,3,31733,True),
 ('weak_cycle',1000000,8,8,31733,False),
 ('paired_three',250000,8,8,31733,False),
 ('unique_three',1000000,2,8,31733,False),
 ('unique_two',1000000,8,8,31733,True),
 ('repeated',1000000,8,8,31733,False),
 ('weak_cycle',5000000,2,8,31733,False),
 ('repeated',5000000,2,1,31733,False),
 ('iid_three',1000000,2,8,71031,False),
 ('dense_repeated',1000000,8,8,71031,False),
 ('unique_three',250000,8,8,89173,False),
]
env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='2',RAYON_NUM_THREADS='2')
status=[]
for ci,(case,n,rhs,h,seed,rw) in enumerate(configs):
 order=['legacy','v03','v02'] if ci%2==0 else ['v02','v03','legacy']
 for method in order:
  name=f'{case}-{n}-{rhs}-H{h}-s{seed}-{method}'
  target=OUT/(name+'.json'); log=OUT/(name+'.log')
  args=[sys.executable,str(ROOT/'benchmark_budgeted.py'),'--case',case,'--rows',str(n),'--rhs',str(rhs),'--calls',str(h),'--seed',str(seed),'--method',method,'--out',str(target)]
  if rw:args.append('--reweight')
  begin=time.time(); print('BEGIN',name,flush=True)
  with log.open('w') as f:
   try:code=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=420).returncode
   except subprocess.TimeoutExpired:code=124
  rec=dict(name=name,code=code,wall_seconds=time.time()-begin)
  if target.exists():
   v=json.loads(target.read_text());rec.update(valid=v['valid'],total_seconds=v['total_seconds'])
  status.append(rec);(OUT/'suite-status.json').write_text(json.dumps(status,indent=2))
  print('END',json.dumps(rec),flush=True)
(OUT/'suite-complete.json').write_text(json.dumps(dict(configs=len(configs),records=len(status),all_ok=all(r['code']==0 and r.get('valid') for r in status))))
