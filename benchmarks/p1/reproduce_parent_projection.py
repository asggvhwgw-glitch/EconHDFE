"""Diagnostic of the provided dev1 parent's MAP/CG range error; not main certification.

Observe BOTH weighted orthogonality and membership of the fitted component in
span(D). A convergence flag alone is insufficient. P1 null fitting uses LSMR.
"""
from __future__ import annotations
import sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import logsumexp
from econhdfe.hdfe.plan import FEPlan
from econhdfe.models.ppml_iv.overidentified import residualize_scaled
from econhdfe.compute.wls import weighted_lstsq
from tests.p1.test_iv_p1 import dgp

y,e,z,c,g,b=dgp();n=len(y);rng=np.random.default_rng(817)
gs=[g]+[rng.integers(0,j+3,n) for j in range(2)]
D=np.column_stack([np.eye(4)[g]]+[np.eye(int(gj.max())+1)[gj][:,1:] for gj in gs[1:]])
U=np.column_stack([D,c]);out={}
for method in ['map','lsmr']:
 projector=FEPlan.from_arrays(gs).projector(engine='optimized',method=method)
 eta=b*e+math.log(y.sum())-logsumexp(b*e);rows=[]
 for it in range(5):
  mu=np.exp(eta);w=mu;work=y/mu-1;block=np.column_stack([work,c/abs(c).max()]);orig=block.copy()
  block,info=residualize_scaled(projector,block,w,tol=1e-12,copy=False)
  ref=orig-D@np.linalg.solve(D.T@(w[:,None]*D),D.T@(w[:,None]*orig))
  dc,res=weighted_lstsq(block[:,1:],block[:,0],w);direction=work-res
  feasibility=direction-U@np.linalg.lstsq(U,direction,rcond=None)[0]
  rows.append(dict(iteration=it,max_projection_error=float(abs(block-ref).max()),
                   max_direction_range_error=float(abs(feasibility).max()),
                   reported_converged=bool(info.converged)))
  eta+=direction
 out[method]=rows
path=ROOT/'validation/iv_p1/parent_projection_diagnostic.json';path.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
