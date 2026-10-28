"""Null-restricted additive-IV S test, with regular included controls and FE.

Only nuisance PPML scores are solved. There is no full IV point estimate, first
stage, or target-slope Jacobian inverse. Growing-FE bias is NOT corrected here.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import logsumexp
from ...hdfe.plan import FEPlan
from ...compute.wls import weighted_lstsq
from ...hdfe.scaled_projection import residualize_scaled
from ...errors import ConvergenceError, NumericalError
from .statistic import _quadratic
from .panel import _matrix


class RestrictedPoissonS:
    """Compile inputs for repeated null evaluations on a FIXED sample.

    endog: target slopes (every endogenous slope is fixed by .test(beta)).
    exog: regular nuisance controls; an intercept is absorbed automatically.
    instruments: excluded moment variables; no first-stage relevance is assumed.
    absorb: categorical intercept FE only. No dropping/reselection is performed.
    clusters: one independent cluster label per row, or None for independent rows.

    The p-value needs a correctly specified additive moment model, a regular
    null-nuisance fit and a valid moment CLT. It is not a misspecification-robust
    confidence set for a projected-IV pseudo-true value.
    """
    def __init__(self, y, endog, instruments, *, exog=None, absorb=None, offset=None,
                 weights=None, weight_kind="none", clusters=None, engine="optimized",
                 tol=1e-9, inner_tol=1e-11, max_iter=100):
        self.y = np.asarray(y, dtype=float)
        if self.y.ndim != 1 or not np.all(np.isfinite(self.y)) or np.any(self.y < 0):
            raise ValueError("y must be a finite nonnegative vector")
        n = len(self.y)
        if n < 3 or not np.any(self.y > 0): raise ValueError("insufficient positive-outcome support")
        self.E = _matrix(endog, n, "endog"); self.Z = _matrix(instruments, n, "instruments")
        self.C = np.empty((n,0)) if exog is None else _matrix(exog, n, "exog")
        if self.E.shape[1] < 1 or self.Z.shape[1] < 1:
            raise ValueError("need a target slope and at least one instrument")
        self.offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
        self.w0 = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        if self.offset.shape != (n,) or not np.all(np.isfinite(self.offset)):
            raise ValueError("invalid offset")
        if self.w0.shape != (n,) or not np.all(np.isfinite(self.w0)) or np.any(self.w0 <= 0):
            raise ValueError("base weights must be finite and positive")
        if weight_kind not in {"none","fweight","pweight"}:
            raise ValueError("weight_kind must be none/fweight/pweight")
        if weight_kind == "fweight" and not np.all(self.w0 == np.floor(self.w0)):
            raise ValueError("frequency weights must be integers")
        if weight_kind == "none" and not np.all(self.w0 == 1):
            raise ValueError("nontrivial base weights require an explicit weight_kind")
        self.weight_kind = weight_kind
        groups = [] if absorb is None else list(absorb)
        for g in groups:
            if np.asarray(g).shape != (n,): raise ValueError("invalid FE label array")
        self.plan = FEPlan.from_arrays(groups or [np.zeros(n,dtype=int)])
        self.plan, _ = self.plan.for_engine(engine)
        self.projector = self.plan.projector(engine=engine, method="lsmr")
        self.clusters = None
        self.G = int(round(self.w0.sum())) if weight_kind == "fweight" else n
        if clusters is not None:
            cl = np.asarray(clusters)
            if cl.ndim != 1 or len(cl) != n: raise ValueError("one-way cluster vector required")
            _, self.clusters = np.unique(cl,return_inverse=True)
            self.G = int(self.clusters.max())+1
            if self.G < 2: raise ValueError("at least two independent clusters required")
        if not (0 < inner_tol <= tol < 1) or max_iter < 1:
            raise ValueError("invalid tolerances/iteration limit")
        self.tol, self.inner_tol, self.max_iter = float(tol),float(inner_tol),int(max_iter)
        for g,L in zip(self.plan.groups,self.plan.levels):
            if np.any(np.bincount(g,weights=self.w0*self.y,minlength=L)<=0):
                raise ValueError("all-zero FE level: finite null profiling unavailable; sample is not silently trimmed")
        # Unit-normalize controls, without changing their nuisance span.
        self.cscale = np.max(np.abs(self.C),axis=0) if self.C.shape[1] else np.empty(0)
        if np.any(self.cscale == 0): raise ValueError("zero nuisance control")
        self.Cs = self.C / self.cscale
        reference = self.w0 * (self.y + float(np.average(self.y, weights=self.w0)))
        self.fe_scales = [np.bincount(g, weights=reference, minlength=L)
                          for g,L in zip(self.plan.groups,self.plan.levels)]
        self.control_scale = np.abs(self.Cs).T @ reference
        self.last_diagnostics = None

    @staticmethod
    def _exp(eta):
        if not np.all(np.isfinite(eta)) or np.max(eta)>700 or np.min(eta)<-700:
            raise NumericalError("null mean outside supported floating-point range; no clipping")
        m=np.exp(eta)
        if not np.all(np.isfinite(m)) or np.any(m<=0):
            raise NumericalError("nonfinite/nonpositive null mean")
        return m

    def _score_vector(self, weighted_residual):
        parts = [np.bincount(g,weights=weighted_residual,minlength=L)/scale
                 for g,L,scale in zip(self.plan.groups,self.plan.levels,self.fe_scales)]
        if self.Cs.shape[1]:
            parts.append((self.Cs.T@weighted_residual)/self.control_scale)
        return np.concatenate(parts)

    def _score_error(self, mu):
        return float(np.max(np.abs(self._score_vector(self.w0*(self.y-mu)))))

    def _nuisance_fit(self,beta):
        b=np.asarray(beta,dtype=float).reshape(-1)
        if b.shape!=(self.E.shape[1],) or not np.all(np.isfinite(b)):
            raise ValueError("beta must fix every target endogenous coefficient")
        base=self.offset+self.E@b
        if not np.all(np.isfinite(base)): raise ValueError("nonfinite null predictor")
        eta=base + math.log(float(self.w0@self.y))-float(logsumexp(base+np.log(self.w0)))
        mu=self._exp(eta); projections=0;backtracks=0
        # A one-way, FE-only nuisance problem has a closed form.
        if len(self.plan.groups)==1 and self.Cs.shape[1]==0:
            g=self.plan.groups[0];L=self.plan.levels[0]
            sy=np.bincount(g,weights=self.w0*self.y,minlength=L)
            sm=np.bincount(g,weights=self.w0*mu,minlength=L)
            eta+=(np.log(sy)-np.log(sm))[g];mu=self._exp(eta)
            err=self._score_error(mu)
            if err>max(self.tol,1e-12): raise ConvergenceError("closed-form null nuisance score check failed")
            return eta,mu,dict(iterations=1,projections=0,backtracks=0,nuisance_score_error=err)
        for iteration in range(1,self.max_iter+1):
            err=self._score_error(mu)
            if err<=self.tol:
                return eta,mu,dict(iterations=iteration,projections=projections,
                                   backtracks=backtracks,nuisance_score_error=err)
            w=self.w0*mu; work=self.y/mu-1.
            block=np.column_stack([work,self.Cs])
            block,info=residualize_scaled(self.projector,block,w,tol=max(1e-14,min(self.inner_tol,.01*err)),copy=False)
            projections+=1
            rt,Ct=block[:,0],block[:,1:]
            if Ct.shape[1]:
                small=Ct*np.sqrt(w)[:,None]
                ss=np.linalg.svd(small,compute_uv=False)
                if ss[-1]<=1e-11*ss[0]:
                    raise ValueError("nuisance controls are rank deficient after FE projection")
            dc,resid=weighted_lstsq(Ct,rt,w,fast=False)
            direction=work-resid
            f=self._score_vector(self.w0*(self.y-mu))
            Jd=-self._score_vector(w*direction)
            derivative=float(f@Jd)
            if not np.isfinite(derivative) or derivative>=0:
                raise ConvergenceError("null nuisance direction failed descent check")
            phi=.5*float(f@f)
            step=1.; accepted=False
            for _ in range(45):
                try:
                    candidate=eta+step*direction; mc=self._exp(candidate)
                    fc=self._score_vector(self.w0*(self.y-mc))
                    objective=.5*float(fc@fc)
                except (NumericalError,FloatingPointError,OverflowError):
                    objective=math.inf
                if np.isfinite(objective) and objective<=phi+1e-4*step*derivative:
                    eta,mu=candidate,mc;accepted=True;break
                step*=.5;backtracks+=1
            if not accepted: raise ConvergenceError("null nuisance line search failed")
        raise ConvergenceError("null nuisance fit did not converge")

    def null_scores(self,beta):
        """Return observation scores for auditing; score totals are IV moments."""
        eta,mu,diag=self._nuisance_fit(beta)
        w=self.w0*mu
        block=np.column_stack([self.Cs,self.Z])
        block,info=residualize_scaled(self.projector,block,w,tol=self.inner_tol,copy=False)
        kc=self.Cs.shape[1]; Ct,Zt=block[:,:kc],block[:,kc:]
        if kc:
            sw=np.sqrt(w);cw=Ct*sw[:,None]
            coef,_,rank,_=np.linalg.lstsq(cw,Zt*sw[:,None],rcond=1e-11)
            if rank!=kc: raise ValueError("null nuisance control space is rank deficient")
            Zt=Zt-Ct@coef
        wr=self.w0*(self.y-mu)
        scores=Zt*wr[:,None]
        diag.update(instrument_nuisance_orthogonality=float(np.max(np.abs(self.Cs.T@(w[:,None]*Zt)),initial=0.)),
                    projection_method="lsmr",fixed_sample=True,rows=len(mu),nuisance_controls=kc,
                    fe_levels=list(self.plan.levels),incidental_bias_corrected=False,
                    scope="regular_null_nuisance_fixed_dimension_or_independently_proved_growth_rates",
                    weak_target_identification_allowed=True,many_moments_certified=False,
                    invalid_instruments_robust=False)
        self.last_diagnostics=diag
        return scores,mu,Zt,diag

    def test(self,beta):
        scores,mu,Zt,diag=self.null_scores(beta)
        total=scores.sum(axis=0)
        if self.clusters is not None:
            sums=np.column_stack([np.bincount(self.clusters,weights=scores[:,j],minlength=self.G)
                                  for j in range(scores.shape[1])])
            centered=sums-sums.mean(axis=0)
            meat=centered.T@centered
            diag["variance"]="one_way_independent_cluster"
        elif self.weight_kind=="fweight":
            # Frequency-expanded observations: w, NOT w^2, in the meat.
            per_copy=scores/self.w0[:,None]
            meat=per_copy.T@(self.w0[:,None]*per_copy)-np.outer(total,total)/self.w0.sum()
            diag["variance"]="frequency_expanded_independent_rows"
        else:
            centered=scores-scores.mean(axis=0)
            meat=centered.T@centered
            diag["variance"]="independent_rows"
        out=_quadratic(total,meat,n_units=self.G,diagnostics=diag)
        out.null_value=np.asarray(beta,dtype=float).reshape(-1)
        return out
