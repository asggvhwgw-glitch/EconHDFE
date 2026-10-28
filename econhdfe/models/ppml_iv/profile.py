"""Exact profiled Newton solver for the SAME iterated projected-IV equation.

The categorical FE nuisance equations are solved at every trial beta. Trial
acceptance recomputes the effective instruments, so its merit is the actual
profiled equation, not the previous iteration's frozen-instrument surrogate.
This costs additional FE work and is opt-in, not a blanket speed improvement.
"""
from __future__ import annotations
from time import perf_counter
import math
import numpy as np
from scipy.special import logsumexp
from ...errors import ConvergenceError, NumericalError, ShapeError, UnderidentifiedError
from ...iv.step import solve_iv_step
from ...compute.wls import weighted_lstsq
from ..poisson import poisson_deviance
from .overidentified import profile_geometry,inverse_sensitivity,residualize_scaled


def _mean(eta):
    if not np.all(np.isfinite(eta)) or np.max(eta)>709. or np.min(eta)<-740.:
        raise NumericalError('profile trial mean is outside supported float64 range')
    mu=np.exp(eta)
    if not np.all(np.isfinite(mu)) or np.any(mu<=0):
        raise NumericalError('invalid profile trial mean')
    return mu


class FEProfiler:
    """Solve D'w0(y-mu)=0 with beta held fixed, reusing the existing projector."""
    def __init__(self,y,w0,plan,projector,tol,max_iter=80):
        self.y,self.w0,self.plan,self.projector=y,w0,plan,projector
        self.tol,self.max_iter=tol,max_iter
        mass=w0*(y+float(np.average(y,weights=w0)))
        self.scales=[np.bincount(g,weights=mass,minlength=L)
                     for g,L in zip(plan.groups,plan.levels)]
        if any(np.any(s<=0) for s in self.scales):
            raise NumericalError('nonpositive FE normalization mass')
        self.projections=0; self.iterations=0; self.seconds=0.; self.calls=0

    def moments(self,wr):
        return np.concatenate([np.bincount(g,weights=wr,minlength=L)/s
            for g,L,s in zip(self.plan.groups,self.plan.levels,self.scales)])

    def error(self,wr):
        return float(np.max(np.abs(self.moments(wr)),initial=0.))

    def fit(self,initial_eta):
        start=perf_counter(); self.calls+=1
        try:
            eta=np.array(initial_eta,dtype=float,copy=True);mu=_mean(eta)
            if not self.plan.groups: return eta,mu
            if len(self.plan.groups)==1:
                g=self.plan.groups[0]; L=self.plan.levels[0]
                sy=np.bincount(g,weights=self.w0*self.y,minlength=L)
                sm=np.bincount(g,weights=self.w0*mu,minlength=L)
                if np.any(sy<=0) or np.any(sm<=0):
                    raise ConvergenceError('FE profile has no finite zero-level solution')
                eta += (np.log(sy)-np.log(sm))[g];mu=_mean(eta)
                if self.error(self.w0*(self.y-mu))>max(self.tol,2e-13):
                    raise ConvergenceError('FE closed-form profile failed residual certificate')
                return eta,mu
            for _ in range(self.max_iter):
                wr=self.w0*(self.y-mu)
                if self.error(wr)<=self.tol:return eta,mu
                w=self.w0*mu;work=self.y/mu-1.
                if not (np.all(np.isfinite(work)) and np.all(np.isfinite(w)) and np.all(w>0)):
                    raise NumericalError('nonfinite FE profile working state')
                resid,info=residualize_scaled(self.projector,work[:,None],w,tol=self.tol,copy=True)
                self.projections+=1;self.iterations+=int(info.iterations)
                d=work-resid[:,0]
                f=self.moments(wr); phi0=.5*float(f@f)
                # Work in grouped moments: wr'd suffers cancellation near the
                # profile root because wr itself need not approach zero.
                derivative=-float(f@self.moments(w*d))
                del work,resid,w,wr
                if not derivative<0:
                    raise ConvergenceError('FE profile direction is not a descent direction')
                step=1.;accepted=False
                for _ in range(35):
                    try:
                        trial_mu=_mean(eta+step*d)
                        trial_f=self.moments(self.w0*(self.y-trial_mu))
                        phi=.5*float(trial_f@trial_f)
                    except NumericalError:
                        phi=math.inf
                    if np.isfinite(phi) and phi<=phi0+1e-4*step*derivative:
                        eta+=step*d;mu=trial_mu;accepted=True;break
                    step*=.5
                if not accepted:raise ConvergenceError('FE profile line search failed')
            raise ConvergenceError('FE profile exhausted nuisance iterations')
        finally:self.seconds+=perf_counter()-start


def fit_profile_irls(y,exog,endog,excluded,plan,true_w,offset,config,
                     *,projector=None,initial_eta=None):
    from .irls import IVIRLSState
    start=perf_counter()
    y,C,E,I,w0,offset=[np.asarray(a,dtype=float) for a in
                       (y,exog,endog,excluded,true_w,offset)]
    X=np.column_stack([C,E]);Z=np.column_stack([C,I]);n,k=X.shape;l=Z.shape[1]
    if not plan.groups:
        raise ShapeError('profile outer solver requires at least a constant FE')
    projector=projector if projector is not None else plan.projector(engine=config.engine)
    target=max(1e-12,min(config.target_inner_tol,.1*config.tolerance))
    nuisance_tol=max(1e-13,min(.1*target,.1*config.tolerance))
    profiler=FEProfiler(y,w0,plan,projector,nuisance_tol,config.outer_profile_max_iter)
    beta=np.zeros(k)
    eta=offset+math.log(float(w0@y))-float(logsumexp(offset+np.log(w0)))
    if initial_eta is not None:
        warm=np.asarray(initial_eta,dtype=float)
        if warm.shape!=y.shape or not np.all(np.isfinite(warm)):
            raise ShapeError('invalid profile warm start')
        initial,info=residualize_scaled(projector,np.column_stack([warm-offset,X]),w0,tol=target,copy=False)
        beta,resid=weighted_lstsq(initial[:,1:],initial[:,0],w0)
        eta=warm-resid
    reference=w0*(y+float(np.average(y,weights=w0)))
    # Fixed metric: no changing first-stage scales are hidden in line acceptance.
    scale=np.sqrt(np.einsum('ij,i,ij->j',X,reference,X))*math.sqrt(float(np.sum(reference)))
    if np.any(scale<=0) or not np.all(np.isfinite(scale)):
        raise NumericalError('invalid profiled score normalization')
    block=np.column_stack([X,Z])
    # The residualized design carries the raw design modulo the FE space.
    # Retaining redundant full raw X/Z here wastes N*(K+L) doubles.
    del X,Z,C,E,I,reference
    evaluations=0;projections=0;absorb_iters=0;geometry_seconds=0.;projection_seconds=0.
    def evaluate(start_eta):
        nonlocal block,evaluations,projections,absorb_iters,geometry_seconds,projection_seconds
        eta,mu=profiler.fit(start_eta);w=w0*mu;work=y/mu-1.
        tic=perf_counter()
        block,info=residualize_scaled(projector,block,w,tol=target,copy=False)
        projection_seconds+=perf_counter()-tic
        projections+=1;absorb_iters+=int(info.iterations)
        Xt,Zt=block[:,:k],block[:,k:]
        tic=perf_counter()
        solved=solve_iv_step(work,Xt,Zt,w,mode=config.outer_linear_solver,
                             chunk_rows=config.outer_chunk_rows,gram_rcond=config.outer_gram_rcond)
        solved.residuals=np.empty(0)  # unused increment residual; do not retain an N-vector
        H=Zt@solved.first_stage
        geometry=profile_geometry(Xt,Zt,H,w,w0*(y-mu),chunk_rows=config.outer_chunk_rows)
        geometry_seconds+=perf_counter()-tic;evaluations+=1
        return eta,mu,w,work,Xt,Zt,H,solved,geometry
    state=evaluate(eta);history=[];backtracks=0;confirmed=0
    for iteration in range(1,config.max_iter+1):
        eta,mu,w,work,Xt,Zt,H,solved,g=state
        candidate=None  # release the previous accepted trial alias
        inv,rcond=inverse_sensitivity(g.sensitivity)
        db=inv@g.score
        f=g.score/scale;error=float(np.max(np.abs(f)))
        step_error=float(np.max(np.abs(db)/(1.+np.abs(beta))))
        fe_error=profiler.error(w0*(y-mu))
        if error<=config.tolerance and step_error<=math.sqrt(config.tolerance) and fe_error<=max(nuisance_tol,2e-13):
            confirmed+=1
            if confirmed<config.min_ok:
                state=evaluate(eta);continue
            diagnostics=dict(solver='profile',target='legacy_projected_iv_equation',
                exact_identification=(l==k),moment_error=error,fe_moment_error=fe_error,
                step_error=step_error,history=history,line_search_backtracks=backtracks,
                actual_profile_merit=True,exact_profile_jacobian=True,
                correction_relative_norm=g.correction_relative_norm,
                sensitivity_rcond=rcond,projections=projections+profiler.projections,
                profile_evaluations=evaluations,fe_profile_calls=profiler.calls,
                final_state_consistent=True,global_guarantee='local_newton_under_regular_profile; no_global_root_guarantee',
                timings=dict(total=perf_counter()-start,fe_profile=profiler.seconds,
                             design_projection=projection_seconds,geometry=geometry_seconds))
            return IVIRLSState(beta.copy(),mu,eta,True,iteration,
                poisson_deviance(y,mu,w0,eta),error,absorb_iters+profiler.iterations,
                np.zeros(n,dtype=bool),block,Xt,Zt,work,H,w,diagnostics,solved)
        confirmed=0
        direction=Xt@db   # derivative of the profiled log mean wrt beta
        derivative=-float(f@(g.sensitivity@db/scale))
        if not np.isfinite(derivative) or derivative>=0:
            raise ConvergenceError('profile Newton direction has no verified descent')
        phi0=.5*float(f@f);step=1.;accepted=False
        correction_norm=g.correction_relative_norm
        # The line search needs eta/direction and small matrices, not the old
        # fitted instrument matrix, mean, weights, work vector or tilt vector.
        state=None
        del mu,w,work,Xt,Zt,H,solved,g
        for trial in range(config.outer_max_backtracks+1):
            candidate=None  # release a rejected trial before allocating its replacement
            try:
                candidate=evaluate(eta+step*direction)
                ft=candidate[-1].score/scale;phi=.5*float(ft@ft)
            except (NumericalError,ConvergenceError,UnderidentifiedError,FloatingPointError):
                phi=math.inf
            if np.isfinite(phi) and phi<=phi0+config.outer_armijo*step*derivative:
                accepted=True;break
            step*=.5
        if not accepted:
            raise ConvergenceError('actual profiled IV merit line search failed',
                code='convergence.ivppml_profile_search',details={'iteration':iteration,'moment_error':error})
        beta+=step*db;state=candidate;backtracks+=trial
        if config.outer_record_history:
            history.append(dict(iteration=iteration,step=step,backtracks=trial,
                moment_error=error,merit_before=phi0,merit_after=phi,
                directional_derivative=derivative,correction_relative_norm=correction_norm))
    raise ConvergenceError('profile IV solver exhausted outer iterations')
