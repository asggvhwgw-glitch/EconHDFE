from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

if "--in-place" in sys.argv:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from econhdfe import (
    reghdfe, ivreghdfe, ppmlhdfe, ivppmlhdfe, PPMLConfig, IVPPMLConfig, factor, reg_interaction, fv, omit_level, __version__
)

p = argparse.ArgumentParser()
p.add_argument('--in-place', action='store_true')
args = p.parse_args()

rng = np.random.default_rng(6060)
n = 1800
f = rng.integers(0, 80, n)
t = rng.integers(0, 10, n)
x = rng.normal(size=(n, 3))
z = rng.normal(size=n)
v = rng.normal(size=n)
endog = .8*z + .2*x[:,0] + v
fe = rng.normal(size=80)[f] + rng.normal(size=10)[t]
y = x @ np.array([.4, -.7, 1.2]) + fe + rng.normal(scale=.2, size=n)
r = reghdfe(None, y=y, x=x, absorb=[f,t], cluster=[f,t], vce='cluster', projection_backend='indexed', absorb_threads=2)
assert r.converged
iv_y = 1.5*endog + .3*x[:,0] + fe + .5*v + rng.normal(size=n)
iv = ivreghdfe(None, y=iv_y, exog=x[:,0], endog=endog, instruments=z, absorb=[f,t], vce='robust', projection_backend='indexed', absorb_threads=2)
assert iv.converged
assert abs(iv.params[-1] - 1.5) < .12

# Explicit event-study reference must survive packaging and must not be
# confused with automatic collinearity omissions.
event_time = (t % 5) - 2
ever = (f % 4 != 0).astype(np.int8)
event = reg_interaction(
    factor('event_time', drop_base=False, name='event_time'),
    'ever',
    name='event_study',
)
event_data = {'y': y, 'x0': x[:, 0], 'f': f, 't': t, 'event_time': event_time, 'ever': ever}
es = reghdfe(
    event_data, y='y', x=['x0', event], absorb=['f', 't'],
    omit=[omit_level('event_time', -1, term='event_study')],
    collinearity='warn',
)
assert any(o['name'] == 'event_time[-1]#ever' for o in es.user_omitted_variables)

# Python-safe factor-variable DSL must survive wheel packaging and compile to
# the same canonical linear design engine as factor()/reg_interaction().
g = rng.integers(0, 3, n)
fv_data = {'y': y, 'x0': x[:, 0], 'g': g, 'f': f, 't': t}
fv_fit = reghdfe(
    fv_data, y='y', x=[fv('i(g)##c(x0)')], absorb=['f', 't'],
    collinearity='drop',
)
assert fv_fit.converged
assert fv_fit.names == ('g[1]', 'g[2]', 'x0', 'g[1]#x0', 'g[2]#x0')
# Integrated PPML-HDFE smoke: same shared FE runtime/solver package.
mu = np.exp(.15*x[:, 0] + .10*x[:, 1] + .15*fe)
y_count = rng.poisson(mu)
pp = ppmlhdfe(
    y=y_count, x=x[:, :2], absorb=[f, t], vce="model",
    config=PPMLConfig(engine="optimized", separation=(), tolerance=1e-8, target_inner_tol=1e-9),
)
assert pp.converged
assert np.max(np.abs(pp.params[:2] - np.array([.15, .10]))) < .08

# IV-PPML smoke: the nonlinear model must be present in the installed wheel and
# must use the same FE/runtime stack rather than a source-tree-only path.
iv_mu = np.exp(.12*x[:, 0] + .20*endog + .10*fe)
iv_count = rng.poisson(iv_mu)
ivpp = ivppmlhdfe(
    y=iv_count, exog=x[:, 0], endog=endog, instruments=z, absorb=[f, t],
    vce="robust",
    config=IVPPMLConfig(engine="optimized", separation=(), tolerance=1e-8, target_inner_tol=1e-9),
)
assert ivpp.converged
assert abs(ivpp.params[-2] - .20) < .12

print(
    f"econhdfe {__version__}: smoke PASS; OLS iter={r.iterations}; "
    f"IV beta={iv.params[-1]:.6f}; event-ref/fv PASS; PPML iter={pp.iterations}; IV-PPML iter={ivpp.iterations}"
)
