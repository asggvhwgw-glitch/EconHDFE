import time
import numpy as np
from econhdfe import reghdfe

n=1_000_000
rng=np.random.default_rng(123)
f1=rng.integers(0,100_000,n,dtype=np.int32)
f2=rng.integers(0,2_000,n,dtype=np.int32)
X=rng.normal(size=(n,4))
a=rng.normal(size=100_000); b=rng.normal(size=2_000)
y=X@np.array([1.,-.5,.25,2.])+a[f1]+b[f2]+rng.normal(size=n)
t=time.perf_counter()
r=reghdfe(None,y=y,x=X,absorb=[f1,f2],vce="iid",pool_size=8,tol=1e-8)
print({"seconds":time.perf_counter()-t,"params":r.params,"iters":r.iterations,"n":r.nobs})
