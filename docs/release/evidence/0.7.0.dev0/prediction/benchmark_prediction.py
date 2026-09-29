"""Synthetic, local chunk-size comparison; not an estimator speed claim."""
import json,sys,time,platform
from pathlib import Path
import numpy as np
import pandas as pd
from econhdfe import olshdfe, factor,reg_interaction
from econhdfe import prediction_api
rng=np.random.default_rng(441)
n=4000
df=pd.DataFrame({'g':np.arange(n)%40,'x':rng.normal(size=n),'w':rng.normal(size=n)})
df['y']=.5*df.x-.2*df.w+df.g*.1+rng.normal(size=n)
r=olshdfe(df,y='y',x=[reg_interaction(factor('g',drop_base=False),'x'),'w'],absorb=['g'])
new=pd.DataFrame({'g':np.arange(400000)%40,'x':rng.normal(size=400000),'w':rng.normal(size=400000)})
original=prediction_api._design_chunk
records=[]; reference=None
for chunk in (2048,32768):
 observed=[]
 def measured(*args,**kwargs):
  X,bad=original(*args,**kwargs); observed.append(X.nbytes); return X,bad
 prediction_api._design_chunk=measured
 times=[]
 for repeat in range(3):
  start=time.perf_counter();out=r.predict(new,kind='xb',chunk_size=chunk);times.append(time.perf_counter()-start)
 prediction_api._design_chunk=original
 if reference is None:reference=out
 np.testing.assert_allclose(out,reference,atol=1e-12,rtol=1e-12)
 records.append({'chunk_size':chunk,'seconds_all_repeats':times,'largest_design_chunk_bytes':max(observed),
                 'result_bytes':out.nbytes,'max_difference':float(np.max(np.abs(out-reference)))})
result={'n_new':len(new),'n_params':len(r.params),'hypothetical_full_design_bytes':len(new)*len(r.params)*8,
        'scope':'xb only; measured design block excludes other temporaries and source/output arrays; no peak RSS claim',
        'python':platform.python_version(),'platform':platform.platform(),'measurements':records}
print(json.dumps(result,indent=2));Path(__file__).with_name('benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
