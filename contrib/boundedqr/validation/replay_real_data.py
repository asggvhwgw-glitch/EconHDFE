"""Replay frozen ACS inputs with a supplied, separately computed QR reference."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from boundedqr import solve_perturbations

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--fixture', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--threads', type=int, default=1)
a = p.parse_args()
x,y,w,b0 = [np.load(a.fixture/f'{key}.npy') for key in ('x','y','W','base_beta')]
tau = json.loads((a.fixture/'meta.json').read_text(encoding='utf8')).get('tau', .5)
t = time.perf_counter()
beta,diag = solve_perturbations(x,y,tau,w,b0.ravel(),backend='cpu',audit_backend='cpu',
                              cpu_factor=.25,batch_size=16,threads=a.threads,verify_radius=True)
seconds = time.perf_counter()-t
reference = np.load(a.reference)
assert beta.shape == reference.shape and np.isfinite(beta).all() and np.isfinite(reference).all()
radii = np.array([f['radius_certificate']['radius'] for f in diag['fits']])
se = beta.std(axis=0,ddof=1)
ref_se = reference.std(axis=0,ddof=1)
row = dict(fixture=a.fixture.name,shape=[*x.shape,w.shape[1]],tau=tau,threads=a.threads,
           seconds=seconds,retained_draws=len(beta),finite_radius_count=int(np.isfinite(radii).sum()),
           infinite_radius_count=int(np.isinf(radii).sum()),all_draws_finite=True,
           max_coefficient_abs_difference=float(np.max(abs(beta-reference))),
           max_standard_error_abs_difference=float(np.max(abs(se-ref_se))),
           max_standard_error_relative_difference=float(np.max(abs(se-ref_se)/np.maximum(abs(ref_se),1e-12))),
           input_sha256={f:hashlib.sha256((a.fixture/f).read_bytes()).hexdigest()
                         for f in ('x.npy','y.npy','W.npy','base_beta.npy')},
           reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),
           scope='fixed-W perturbation solver only; excludes baseline, score construction and data loading')
a.output.mkdir(parents=True,exist_ok=True)
np.save(a.output/'beta.npy',beta)
np.save(a.output/'radii.npy',radii)
(a.output/'diagnostics.json').write_text(json.dumps(diag,indent=2),encoding='utf8')
(a.output/'summary.json').write_text(json.dumps(row,indent=2)+'\n',encoding='utf8')
print(json.dumps(row),flush=True)
