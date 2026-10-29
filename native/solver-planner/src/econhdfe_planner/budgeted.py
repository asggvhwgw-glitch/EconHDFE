"""Finite-horizon, bounded optional calibration over a certified incumbent.

No calibration for one-shot/short-horizon fits. The unchanged production auto
adapter computes the first answer. Optional challengers run in disposable fresh
processes; timeouts/capped iterations are missing information, never a fast score.
A pilot uses full rows and a small actual RHS block, not a subsampled FE graph.
A switch requires a certified FULL RHS, cross-route agreement and estimated
payback within the supplied horizon. Calibrating does not reduce statistical
acceptance accuracy. Retained workers are idle between synchronous calls.
"""
from __future__ import annotations
import copy
import math
from time import perf_counter
import numpy as np
from .planner import _Route, _weighted_gap, PlanningError, ProjectionDisagreement, ROUTES
from .structure import compact_codes, positive_weights
from ._bounded_process import TrialWorker, TrialTimeout


class _LegacyAuto:
    name = 'legacy_auto'
    def __init__(self, codes, weights, threads, scratch_bytes):
        from _econhdfe_within import NativeSolver
        self.core = NativeSolver(codes, weights=weights, strategy='auto',
                                 threads=threads, workspace_bytes=scratch_bytes)
    def project(self, y, tol, max_iter):
        out, flags, iters, cert = self.core.solve(y, tol=tol, max_iter=max_iter)
        accepted = (all(flags) and np.isfinite(out).all() and
                    all(np.isfinite(x) and x <= max(tol,128*np.finfo(float).eps) for x in cert))
        if not accepted:
            raise PlanningError('incumbent certificate rejected')
        return out, dict(accepted=True, backend=self.name, route=self.core.route,
                         n_cells=self.core.n_cells, iterations=iters, certificate=cert)
    def update_weights(self, w):
        self.core.update_weights(w)


def payback_decision(incumbent_seconds, challenger_seconds, paid_seconds,
                     remaining_calls, *, margin=1.25):
    """A forecast, not a speed theorem. Use actual promotion/IPC cost, not pilot.

    All exploration paid this call must be earned back by future calls. A finite
    conservative multiplier prevents switching on tiny measured differences.
    """
    saving = max(0.0, incumbent_seconds-challenger_seconds)
    projected = remaining_calls*saving
    minimum = None if saving <= 0 else math.ceil(margin*paid_seconds/saving)
    return dict(accept=bool(saving > 0 and projected >= margin*paid_seconds),
                savings_per_future_call=saving, estimated_future_savings=projected,
                paid_optional_seconds=paid_seconds, payback_margin=margin,
                estimated_remaining_calls_to_pay_back=minimum)


class BudgetedAbsorber:
    """Opt-in v0.3 planner; defaults to no speculative work (expected_calls=1).

    expected_calls counts total nonempty future/current residualize calls, not
    FE iterations. By default, measure a second REAL call before exploring,
    avoiding a speculative warm solve and a cold-setup-biased savings forecast.
    min_calibration_calls defaults to 4. Optional spending is at
    most min(max_probe_seconds, max_overhead_fraction*current_incumbent_call,
    exploration_fraction*remaining_calls*current_incumbent_call) in child work.

    The deadline includes candidate preparation, but cannot interrupt parent
    array/filesystem work or OS cleanup; overruns are recorded. This does not
    cap the necessary incumbent/final solve or later retained-worker calls.
    probe_max_iter applies to each adapter inner projection attempt, not to
    symbolic factorization or a global sum of all Krylov iterations.

    'reweight' adds a same-weight refresh to promotion. This is only a cost
    proxy; arbitrary future IRLS weights/RHS can change which solver is best.
    """
    def __init__(self, codes, *, weights=None, backend='auto', threads=2,
                 scratch_bytes=512 << 20, expected_calls=1, min_calibration_calls=4, calibration_after=2,
                 max_probe_seconds=3.0, max_overhead_fraction=0.5,
                 exploration_fraction=0.1, probe_max_iter=128, probe_rhs=1,
                 max_candidates=2, minimum_probe_seconds=0.15,
                 payback_margin=1.25, workload='fixed', allow_fallback=True,
                 agreement_tolerance=2e-8):
        t = perf_counter()
        if backend not in (*ROUTES, 'auto', 'legacy_auto'):
            raise ValueError('invalid backend')
        for key, val, lower in [('threads',threads,1),('scratch_bytes',scratch_bytes,1024),
                               ('expected_calls',expected_calls,1),('min_calibration_calls',min_calibration_calls,2),('calibration_after',calibration_after,1),
                               ('probe_max_iter',probe_max_iter,1),('probe_rhs',probe_rhs,1),
                               ('max_candidates',max_candidates,0)]:
            if isinstance(val,bool) or not isinstance(val,(int,np.integer)) or val<lower:
                raise ValueError(f'{key} must be an integer >= {lower}')
        for key,val,lower in [('max_probe_seconds',max_probe_seconds,0),
                              ('max_overhead_fraction',max_overhead_fraction,0),
                              ('exploration_fraction',exploration_fraction,0),
                              ('minimum_probe_seconds',minimum_probe_seconds,0),
                              ('payback_margin',payback_margin,1),
                              ('agreement_tolerance',agreement_tolerance,0)]:
            if isinstance(val,bool) or not np.isfinite(val) or val<lower:
                raise ValueError(f'invalid {key}')
        if agreement_tolerance == 0 or workload not in ('fixed','reweight'):
            raise ValueError('invalid agreement_tolerance or workload')
        self._codes, self._levels = compact_codes(codes)
        self._n = len(self._codes)
        self._w = positive_weights(weights,self._n)
        self._threads,self._scratch = int(threads),int(scratch_bytes)
        self._requested = backend
        self._expected, self._minimum_calls = int(expected_calls), int(min_calibration_calls)
        self._calibration_after = int(calibration_after)
        self._budget, self._overhead, self._explore = float(max_probe_seconds),float(max_overhead_fraction),float(exploration_fraction)
        self._probe_iter,self._probe_rhs,self._max_candidates = int(probe_max_iter),int(probe_rhs),int(max_candidates)
        self._minimum_budget,self._margin,self._agreement = float(minimum_probe_seconds),float(payback_margin),float(agreement_tolerance)
        self._workload,self._fallback = workload,bool(allow_fallback)
        self._route = None
        self._selected = None
        self._calls,self._generation = 0,0
        self._pending_update = 0.0
        self._tune_pending = True
        self._plan_id = 0
        self._plan,self._last = {},{}
        self._closed = False
        self._constructor_seconds = perf_counter()-t

    def _local(self,name):
        if name=='legacy_auto':
            return _LegacyAuto(self._codes,self._w,self._threads,self._scratch)
        return _Route(name,self._codes,self._w,self._threads,self._scratch)

    def _worker(self,name,deadline):
        return TrialWorker(name,self._codes,self._w,self._threads,self._scratch,deadline)

    @staticmethod
    def _dispose(route):
        if route is not None and hasattr(route,'close'):
            route.close()

    def _solve_incumbent(self,y,tol,max_iter):
        errors=[]
        name=self._selected or ('legacy_auto' if self._requested=='auto' else self._requested)
        choices=[name]+([x for x in ('fereg','within','compressed_within') if x!=name] if self._fallback else [])
        for choice in choices:
            try:
                if self._route is None:
                    self._route=self._local(choice)
                out,info=self._route.project(y,tol,max_iter)
                if not info.get('accepted'):
                    raise PlanningError('uncertified incumbent')
                self._selected=choice
                return out,info,errors
            except (RuntimeError,ValueError,MemoryError,OSError) as exc:
                errors.append(dict(backend=choice,error=f'{type(exc).__name__}: {exc}'))
                self._dispose(self._route)
                self._route=None
        raise PlanningError(f'no certified incumbent: {errors}')

    def _agree(self,a,b,y):
        gap=_weighted_gap(a,b,self._w)
        if gap>self._agreement:
            scale=np.maximum(np.max(np.abs(y),axis=0),np.finfo(float).tiny)
            if np.max(np.abs(a/scale-b/scale))>self._agreement:
                raise ProjectionDisagreement(f'certified routes disagree, relative gap={gap:.3e}')
        return gap

    def _challengers(self, info):
        # Three numerical routes remain available; existing auto already resolves
        # exact tuple compression/SmallSchur. No extra full-stream sketch is paid.
        names=['fereg','within']
        if info.get('n_cells',self._n)/self._n<=0.7 and info.get('route')!='compressed_within':
            names.insert(0,'compressed_within')
        return [x for x in names if x!=self._selected][:self._max_candidates]

    def _explore_routes(self,y,out,info,incumbent_seconds,tol,max_iter):
        start=perf_counter()
        remaining=max(0,self._expected-self._calls-1)
        budget=min(self._budget,self._overhead*incumbent_seconds,
                   self._explore*remaining*incumbent_seconds)
        self._plan_id+=1
        plan=dict(plan_id=self._plan_id,expected_calls=self._expected,
                  remaining_calls_at_plan=remaining,incumbent=self._selected,
                  incumbent_call_seconds=incumbent_seconds,incumbent_cost_basis='actual current call including update; deferred by default',requested_budget_seconds=budget,
                  probe_max_iter_per_attempt=min(max_iter,self._probe_iter),
                  pilot_rhs=min(y.shape[1],self._probe_rhs),trials=[],selected=self._selected,
                  workload=self._workload,objective='finite-horizon paid total cost forecast',
                  deadline_scope='optional child work; parent I/O and cleanup may overrun',
                  geometry_policy='reuse incumbent exact route; no additional sketch',
                  selected_rhs=y.shape[1],selected_weight_generation=self._generation)
        reason=None
        if self._requested!='auto': reason='explicit_backend_no_calibration'
        elif remaining+1<self._minimum_calls: reason='short_horizon_no_calibration'
        elif self._max_candidates==0 or budget<self._minimum_budget or budget==0: reason='budget_too_small_no_calibration'
        if reason:
            plan.update(reason=reason,optional_seconds=0.0,budget_overrun_seconds=0.0)
            self._plan=plan
            return out,info
        deadline=start+budget
        width=min(y.shape[1],self._probe_rhs)
        # Spread pilot columns across the actual block; never subsample rows.
        indices=np.linspace(0,y.shape[1]-1,width,dtype=int)
        pilot=y[:,indices]
        best_cost=incumbent_seconds  # default uses second actual call; remains a forecast
        for name in self._challengers(info):
            if deadline-perf_counter()<self._minimum_budget:
                plan['trials'].append(dict(backend=name,status='untested_budget'))
                break
            trial=dict(backend=name,status='started')
            candidate=None
            kept=False
            trial_start=perf_counter()
            try:
                candidate=self._worker(name,deadline)
                setup_end=perf_counter()
                cap=min(max_iter,self._probe_iter)
                stages=list(dict.fromkeys((min(32,cap),cap)))
                trial['pilot_stages']=[]
                for stage_cap in stages:
                    stage_start=perf_counter()
                    try:
                        candidate_pilot,pilot_info=candidate.project(pilot,tol,stage_cap,deadline=deadline)
                        trial['pilot_stages'].append(dict(cap=stage_cap,status='certified',seconds=perf_counter()-stage_start))
                        break
                    except TrialTimeout:
                        raise
                    except PlanningError as exc:
                        trial['pilot_stages'].append(dict(cap=stage_cap,status='unranked',seconds=perf_counter()-stage_start,error=str(exc)))
                        numerical=('convergence' in str(exc) or 'certificate' in str(exc))
                        if stage_cap==stages[-1] or not numerical:
                            raise
                        # Reuse the same candidate's geometry/preconditioner, but
                        # never its unaccepted residual. The original pilot is
                        # solved afresh under the SAME overall wall deadline.
                pilot_end=perf_counter()
                trial.update(preparation_seconds=setup_end-trial_start,pilot_wall_seconds=pilot_end-setup_end,
                             pilot_details=pilot_info,pilot_gap=self._agree(candidate_pilot,out[:,indices],pilot))
                if not pilot_info.get('accepted'):
                    raise PlanningError('pilot not certified')
                # Pilot extrapolation is screening information only, never enough
                # to switch. Promotion must finish the actual full block below.
                native=pilot_info.get('worker_timing',{})
                estimate=native.get('solve_seconds',trial['pilot_wall_seconds'])*y.shape[1]/width
                trial['full_solve_screen_estimate_seconds']=estimate
                if estimate*1.1>=best_cost:
                    trial['status']='screened_no_predicted_gain'
                    continue
                if estimate>deadline-perf_counter():
                    trial['status']='untested_promotion_budget'
                    continue
                t=perf_counter()
                complete,complete_info=candidate.project(y,tol,min(max_iter,self._probe_iter),
                                                         deadline=deadline,refresh=self._workload=='reweight')
                full_seconds=perf_counter()-t
                trial.update(promotion_wall_seconds=full_seconds,
                             promotion_details=complete_info,full_gap=self._agree(complete,out,y))
                if not complete_info.get('accepted'):
                    raise PlanningError('promotion not certified')
                spent=perf_counter()-start
                decision=payback_decision(best_cost,full_seconds,spent,remaining,margin=self._margin)
                trial['payback']=decision
                if perf_counter()>deadline:
                    trial['status']='late_promotion_discarded'
                elif decision['accept']:
                    old=self._route
                    self._route=candidate
                    self._selected=name
                    kept=True
                    self._dispose(old)
                    out,info=complete,complete_info
                    best_cost=full_seconds
                    trial['status']='promoted'
                    # One accepted challenger is enough; do not spend the rest
                    # of the budget comparing all nearly equivalent routes.
                    break
                else:
                    trial['status']='certified_but_no_horizon_payback'
            except TrialTimeout as exc:
                trial.update(status='timeout_unranked',error=str(exc))
            except ProjectionDisagreement:
                trial['status']='disagreement_fatal'
                raise
            except (RuntimeError,ValueError,MemoryError,OSError) as exc:
                trial.update(status='failed_or_iteration_limited_unranked',error=f'{type(exc).__name__}: {exc}')
            finally:
                if not kept:
                    self._dispose(candidate)
                trial['total_wall_seconds']=perf_counter()-trial_start
                plan['trials'].append(trial)
        elapsed=perf_counter()-start
        plan.update(selected=self._selected,reason='bounded_trial_promoted' if any(x['status']=='promoted' for x in plan['trials']) else 'kept_certified_incumbent',
                    optional_seconds=elapsed,budget_overrun_seconds=max(0.0,elapsed-budget),
                    final_forecast_call_seconds=best_cost)
        self._plan=plan
        return out,info

    def residualize(self,data,*,tol=1e-10,max_iter=6000,return_info=False):
        if self._closed: raise PlanningError('absorber closed')
        begin=perf_counter()
        y=np.asarray(data,dtype=np.float64)
        vector=y.ndim==1
        if vector:y=y[:,None]
        if y.ndim!=2 or len(y)!=self._n or not np.isfinite(y).all():
            raise ValueError('finite RHS with shape (n,) or (n,rhs) required')
        if not np.isfinite(tol) or not 0<tol<=0.1:
            raise ValueError('tol must be in (0,0.1]')
        if isinstance(max_iter,bool) or not isinstance(max_iter,(int,np.integer)) or max_iter<1:
            raise ValueError('max_iter must be a positive integer')
        if y.shape[1]==0:
            out=y.copy()
            self._last=dict(accepted=True,backend='none',rhs=0,successful_calls=self._calls,plan=self._plan)
            return (out,self.diagnostics) if return_info else out
        t=perf_counter()
        out,native,errors=self._solve_incumbent(y,tol,int(max_iter))
        incumbent=perf_counter()-t+self._pending_update
        did_plan=self._tune_pending and (self._calls+1>=self._calibration_after or self._expected<self._minimum_calls or self._requested!='auto')
        if self._tune_pending and not did_plan:
            self._plan=dict(plan_id=self._plan_id,selected=self._selected,reason='deferred_until_actual_reuse',
                            expected_calls=self._expected,calibration_after=self._calibration_after,
                            optional_seconds=0.0,budget_overrun_seconds=0.0,trials=[],selected_rhs=y.shape[1])
        if did_plan:
            try:
                out,native=self._explore_routes(y,out,native,incumbent,tol,int(max_iter))
            except ProjectionDisagreement:
                self._last=dict(accepted=False,phase='optional_trial',error='certified routes disagree')
                raise
            self._tune_pending=False
        self._calls+=1
        self._last=dict(accepted=True,backend=self._selected,successful_calls=self._calls,
                        calibrated_this_call=did_plan,weight_generation=self._generation,
                        call_seconds=perf_counter()-begin,update_seconds=self._pending_update,
                        constructor_seconds=self._constructor_seconds,
                        plan_rhs_changed=self._plan.get('selected_rhs',y.shape[1])!=y.shape[1],
                        fallback_errors=errors,plan=self._plan,projection=native)
        self._pending_update=0.0
        if vector:out=out[:,0]
        return (out,self.diagnostics) if return_info else out

    def update_weights(self,weights):
        if self._closed:raise PlanningError('absorber closed')
        begin=perf_counter()
        new_w=positive_weights(weights,self._n)
        try:
            if self._route is not None:self._route.update_weights(new_w)
        except (RuntimeError,ValueError,MemoryError,OSError):
            self._dispose(self._route); self._route=None
            raise
        self._w=new_w
        self._generation+=1
        self._pending_update+=perf_counter()-begin

    def retune(self,*,expected_calls=None):
        """Optional explicit exploration on NEXT synchronous nonempty solve.

        expected_calls is the revised total horizon including calls already made.
        Retain the certified incumbent rather than throwing away useful state.
        """
        if self._requested!='auto': raise ValueError('retune requires auto')
        if expected_calls is not None:
            if isinstance(expected_calls,bool) or not isinstance(expected_calls,(int,np.integer)) or expected_calls<1:
                raise ValueError('expected_calls must be a positive integer')
            self._expected=int(expected_calls)
        self._tune_pending=True

    @property
    def backend(self):return self._selected or 'uninitialized'
    @property
    def weight_generation(self):return self._generation
    @property
    def diagnostics(self):return copy.deepcopy(self._last)
    def close(self):
        if not self._closed:
            self._dispose(self._route);self._route=None;self._closed=True
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
    def __del__(self):
        try:self.close()
        except Exception:pass
