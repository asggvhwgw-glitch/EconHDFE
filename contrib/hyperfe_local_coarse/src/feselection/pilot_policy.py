"""Cost-aware continuation of a single useful RHS (heuristics, not error bounds).

The LSMR recurrence is not restarted at checkpoints. Estimates use the same
backward-error quantities as its stopping rule; only the original-target audit
can accept a completed solve. Setup costs are sunk at this stage and excluded
from continuation decisions, but remain in end-to-end benchmark time.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import statistics
import numpy as np
from scipy.sparse.linalg import LinearOperator
from .monitored_lsmr import monitored_lsmr


@dataclass(frozen=True)
class PilotPolicy:
    enabled: bool = True
    checkpoint_steps: int = 8
    forecast_safety: float = 1.5
    maximum_native_fraction: float = .90
    seconds_cap: float = 5.
    exploratory_time_fraction: float = .35
    minimum_contraction: float = 1.02
    history_windows: int = 3

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be boolean')
        for key in ('checkpoint_steps', 'history_windows'):
            val = getattr(self, key)
            if type(val) is not int or val < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('forecast_safety','seconds_cap','minimum_contraction'):
            val = getattr(self,key)
            if isinstance(val,bool) or not math.isfinite(val) or val <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.forecast_safety < 1 or self.minimum_contraction <= 1:
            raise ValueError('forecast_safety >= 1 and minimum_contraction > 1 required')
        if not math.isfinite(self.exploratory_time_fraction) or not 0 <= self.exploratory_time_fraction <= 1:
            raise ValueError('exploratory_time_fraction must be in [0,1]')
        if not math.isfinite(self.maximum_native_fraction) or not 0 < self.maximum_native_fraction <= 1:
            raise ValueError('maximum_native_fraction must be in (0,1]')


class PilotController:
    """Cooperative checkpoint decisions; one expensive operation is not preempted.

    Time and convergence forecasts can be wrong. This controller may reject a
    useful root, but cannot accept a numerically incomplete solve as converged.
    """
    def __init__(self, policy, *, baseline_seconds, remaining_rhs, initial_steps,
                 gain_margin=.20):
        if not math.isfinite(baseline_seconds) or baseline_seconds <= 0:
            raise ValueError('positive finite baseline_seconds required')
        for val in (remaining_rhs, initial_steps):
            if type(val) is not int or val < 1:
                raise ValueError('positive integer RHS/initial steps required')
        if not 0 <= gain_margin < 1:
            raise ValueError('invalid gain margin')
        self.policy=policy;self.baseline=float(baseline_seconds)
        self.remaining=remaining_rhs;self.initial=initial_steps;self.margin=gain_margin
        self.budget=min(policy.seconds_cap,policy.maximum_native_fraction*self.baseline)
        self.records=[]

    def __call__(self, state):
        rec=dict(state)
        normal=state['normal_backward_error'];compatible=state['compatible_relative_residual']
        atol=state['atol'];rtol=state['compatible_tolerance']
        gaps=(normal/max(atol,np.finfo(float).tiny),compatible/max(rtol,np.finfo(float).tiny))
        gap=max(min(gaps),np.finfo(float).tiny)
        rec.update(gap_to_stopping=gap,budget_seconds=self.budget,
                   baseline_rhs_seconds=self.baseline,remaining_rhs=self.remaining)
        self.records.append(rec)
        if state['stop_code']:
            rec.update(continue_iteration=False,reason='numerical_stop',
                       convergence=state['stop_code'] in (0,1,2,4,5))
            return True  # Never overwrite SciPy's genuine stopping status.
        if not all(math.isfinite(v) for v in (gap,state['elapsed'],state['normr'],state['normar'])):
            rec.update(continue_iteration=False,reason='nonfinite_progress')
            return False
        if state['elapsed']>=self.budget:
            rec.update(continue_iteration=False,reason='pilot_time_budget')
            return False
        if state['iteration']<self.initial or len(self.records)<2:
            rec.update(continue_iteration=True,reason='initial_measurement')
            return True
        recent=self.records[-(self.policy.history_windows+1):]
        rates=[]
        for prev,cur in zip(recent[:-1],recent[1:]):
            dk=cur['iteration']-prev['iteration']
            if dk>0:
                rates.append((math.log(prev['gap_to_stopping'])-math.log(cur['gap_to_stopping']))/dk)
        rate=statistics.median(rates) if rates else 0.
        rec['median_log_contraction_per_iteration']=rate
        if rate <= math.log(self.policy.minimum_contraction)/self.policy.checkpoint_steps:
            rec.update(continue_iteration=False,reason='insufficient_progress')
            return False
        predicted=max(0.,math.log(max(1.,gap))/rate)
        step=state['elapsed']/max(1,state['iteration'])
        # Conservative trend extrapolation, not a future error certificate.
        seconds_left=self.policy.forecast_safety*predicted*step
        total_local=state['elapsed']+seconds_left
        future_local=seconds_left+(self.remaining-1)*total_local
        native_alternative=self.remaining*self.baseline
        rec.update(predicted_iterations_left=predicted,
                   predicted_seconds_left=seconds_left,
                   predicted_completed_rhs_seconds=total_local,
                   predicted_remaining_work_seconds=future_local,
                   native_remaining_work_estimate=native_alternative)
        proceed=(total_local <= self.budget and
                 future_local < (1-self.margin)*native_alternative)
        # Krylov residuals can accelerate abruptly when a cluster is resolved.
        # A short-term geometric forecast must not be treated as a theorem.
        # With observed progress, buy ONE more checkpoint only while its
        # estimated cost fits a small, explicit fraction of one native RHS.
        extra_window=self.policy.forecast_safety*step*self.policy.checkpoint_steps
        exploratory_cap=min(self.budget,self.policy.exploratory_time_fraction*self.baseline)
        bridge=(not proceed and state['elapsed']+extra_window <= exploratory_cap)
        rec.update(next_checkpoint_estimated_seconds=extra_window,
                   exploratory_cap_seconds=exploratory_cap,
                   continue_iteration=bool(proceed or bridge),
                   reason='progress_and_cost_support_continuation' if proceed else
                          'bounded_progress_extension_forecast_uncertain' if bridge else
                          'forecast_does_not_pay')
        return bool(proceed or bridge)


def solve_profiled_pilot(system,y,root,*,tol,maxiter,controller):
    """Same rectangular target and adjoint as solve_rect, with one live recurrence."""
    w=system.w;sw=np.sqrt(w);g=system.groups;H=system.H;delta=system.delta
    def mv(x):return sw*system.center(H@root.forward(x))
    def rmv(z):
        v=sw*z
        v-=w*(np.bincount(g,weights=v,minlength=len(delta))/delta)[g]
        return root.adjoint(np.asarray(H.T@v).ravel())
    A=LinearOperator((len(y),H.shape[1]+getattr(root,'k',0)),matvec=mv,rmatvec=rmv,dtype=float)
    out=monitored_lsmr(A,sw*system.center(y),atol=tol,btol=tol,conlim=1e14,
                       maxiter=maxiter,monitor=controller,
                       check_every=controller.policy.checkpoint_steps)
    # Even if interrupted, the fit subtracted here belongs to the ORIGINAL FE
    # space. Finishing native on this residual preserves the projection target.
    r=system.center(y-H@root.forward(out[0]))
    return r,{'status':int(out[1]),'iterations':int(out[2]),
              'normr':float(out[3]),'normar':float(out[4]),
              'controller_checks':controller.records,
              'state_restarts':0}
