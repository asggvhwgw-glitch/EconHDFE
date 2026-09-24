"""Research pilot: residual-sign preprocessing with an exact reduced dual LP.

This imports established QR preprocessing ideas, not a novelty claim. It is the
strong bounded-memory CPU comparator for later shared-operator experiments.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import linprog

@dataclass
class Fit:
    beta: np.ndarray
    dual: np.ndarray
    gap: float
    equality_residual: float
    box_violation: float
    active_sizes: list
    seconds: float
    pseudo_residual: float
    fallback: bool

def _lp(x, y, tau, rhs, limit=60):
    sol = linprog(-y, A_eq=x.T, b_eq=rhs, bounds=(tau-1, tau),
                  method="highs-ds", options={"presolve":True,
                  "dual_feasibility_tolerance":1e-9,
                  "primal_feasibility_tolerance":1e-9,
                  "time_limit":limit})
    if not sol.success:
        raise RuntimeError(sol.message)
    # HiGHS reports d(min -y'a)/d(rhs), the negative QR coefficient.
    return -np.asarray(sol.eqlin.marginals), sol.x

def check(x, y, tau, w, beta, a, a0):
    # Sum local nonnegative Fenchel remainders; avoids cancellation against the
    # huge additive pseudo-observation constant in an objective subtraction.
    r = y-x@beta
    d = w/tau
    r0 = len(y)*np.max(np.abs(y))-d@beta
    loss = np.maximum(tau*r, (tau-1)*r)
    local = loss-a*r
    local0 = max(tau*r0,(tau-1)*r0)-a0*r0
    eq = x.T@a+d*a0
    gap = float(np.sum(np.maximum(local,0))+max(local0,0))
    # Near feasibility is reported, not promoted to a rigorous lower bound.
    return gap,float(np.max(np.abs(eq))),float(max(np.max(a-tau),np.max(tau-1-a),a0-tau,tau-1-a0,0)),float(r0)

def full_fit(x,y,tau,w=None):
    start=time.perf_counter()
    if w is None:
        w=np.zeros(x.shape[1])
    # Finite augmented QR remains bounded, even for an infeasible ideal tilt.
    z=np.vstack([x,w/tau])
    yy=np.r_[y,len(y)*np.max(np.abs(y))]
    b,a=_lp(z,yy,tau,np.zeros(x.shape[1]))
    gap,eq,box,r0=check(x,y,tau,w,b,a[:-1],a[-1])
    return Fit(b,a,gap,eq,box,[len(y)],time.perf_counter()-start,r0,False)

def reduced_fit(x,y,tau,w,beta0,factor=3.0,max_rounds=12):
    start=time.perf_counter()
    n,p=x.shape
    r=y-x@beta0
    # A leverage-normalized selection is used in mature CFM preprocessing;
    # this initial pilot uses row norms and records expansion failures.
    score=np.abs(r)/np.maximum(np.linalg.norm(x,axis=1),1e-12)
    m=min(n,max(4*p,int(factor*np.sqrt(n*p))))
    active=np.zeros(n,dtype=bool)
    active[np.argpartition(score,m-1)[:m]]=True
    basea=np.where(r<0,tau-1,tau)
    sizes=[]
    for _ in range(max_rounds):
        sizes.append(int(active.sum()))
        fixed=~active
        rhs=-w-x[fixed].T@basea[fixed]
        try:
            b,aa=_lp(x[active],y[active],tau,rhs)
        except RuntimeError:
            if active.all():
                break
            m=min(n,2*int(active.sum()))
            active[np.argpartition(score,m-1)[:m]]=True
            continue
        rnew=y-x@b
        wrong=fixed & (((basea==tau)&(rnew< -1e-9))|((basea==tau-1)&(rnew>1e-9)))
        if wrong.any():
            active[wrong]=True
            continue
        a=basea.copy()
        a[active]=aa
        gap,eq,box,r0=check(x,y,tau,w,b,a,tau)
        if r0>=0 and eq<1e-6 and gap<1e-6*(1+np.mean(np.abs(y))):
            return Fit(b,np.r_[a,tau],gap,eq,box,sizes,time.perf_counter()-start,r0,False)
        break
    out=full_fit(x,y,tau,w)
    out.fallback=True
    out.active_sizes=sizes+out.active_sizes
    out.seconds=time.perf_counter()-start
    return out
