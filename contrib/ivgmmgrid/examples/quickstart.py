"""Small runnable example; the separate water example uses real public data."""
import numpy as np
from ivgmmgrid import fit_grid

rng=np.random.default_rng(42)
n=1200
cluster=np.repeat(np.arange(120),10)
fe=np.tile(np.arange(10),120)
z=rng.normal(size=(n,2))
w=rng.normal(size=n)
x=z[:,0]+.5*z[:,1]+rng.normal(size=n)
y=.7*x+.2*w+rng.normal(size=10)[fe]+rng.normal(size=120)[cluster]+rng.normal(size=n)
candidates=np.column_stack([x,x+.1*w])
result=fit_grid(y,w,candidates,z,fe,cluster,verify=True)
for j,m in enumerate(result.models):
    print(f'candidate {j}: N={m.nobs}, clusters={m.nclusters}, b={m.coefficients}, se={np.sqrt(np.diag(m.covariance))}')
