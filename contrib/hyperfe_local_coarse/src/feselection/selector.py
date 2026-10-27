"""Opt-in categorical FE selector, layered above the frozen research kernels.

No sample selection, DoF convention, penalties or default estimator changes.
On partial local LSMR, fallback explicitly restarts native within on the partial
residual. Because the subtracted fit is in col(D), this preserves the projection.
A normal-equation audit is NOT a general forward-error certificate.
"""
from __future__ import annotations
from dataclasses import dataclass,field,asdict
from time import perf_counter
from typing import Literal
import hashlib
import numpy as np
import scipy.linalg as la
from scalable_hyperfe import ResourceGuard,memory_bytes
from budget_preconditioner import ComponentRight,solve_rect
from .allocation import (ResourceBudget,AllocationLedger,CandidateRejected,inspect,
                         elimination_candidates,prepare_system,guarded_star)
from .partitioning import candidates
from .coarse import GuardedCoarseRight
from .cost_policy import CostPolicy
from .numerical_screen import NumericalScreenPolicy, preview_partition
from .auditing import audit_columns, column_scales
from .early_screen import EarlyScreenPolicy, scout
from .pilot_policy import PilotPolicy, PilotController, solve_profiled_pilot

@dataclass(frozen=True)
class SelectionConfig:
    mode: Literal['native','observe','auto','candidate'] = 'observe'
    tol: float = 1e-13
    audit_tol: float = 1e-8
    maxiter: int = 2000
    pilot_iterations: int = 24
    eig_cut: float = .005
    max_candidates: int = 3
    native_mode: Literal['adaptive','diagonal','additive'] = 'adaptive'
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    cost: CostPolicy = field(default_factory=CostPolicy)
    screen: NumericalScreenPolicy = field(default_factory=NumericalScreenPolicy)
    early: EarlyScreenPolicy = field(default_factory=EarlyScreenPolicy)
    pilot: PilotPolicy = field(default_factory=PilotPolicy)
    def __post_init__(self):
        if self.mode not in ('native','observe','auto','candidate'):raise ValueError('unknown selection mode')
        if self.native_mode not in ('adaptive','diagonal','additive'):raise ValueError('unknown native mode')
        for key in ('tol','audit_tol'):
            v=getattr(self,key)
            if not np.isfinite(v) or not 0<v<1:raise ValueError(f'{key} must be in (0,1)')
        for key in ('maxiter','pilot_iterations','max_candidates'):
            v=getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,int) or v<1:raise ValueError(f'{key} must be a positive integer')
        if not np.isfinite(self.eig_cut) or not 0<=self.eig_cut<1:raise ValueError('invalid eig_cut')
        if not 0<self.cost.exploration_fraction<=1 or not 0<=self.cost.gain_margin<1:
            raise ValueError('invalid cost policy fractions')

@dataclass
class SelectionResult:
    residuals: np.ndarray
    diagnostics: dict


def _normal(codes,w,y,r):
    if not np.all(np.isfinite(r)):raise FloatingPointError('nonfinite projection residual')
    scale=float(np.max(np.abs(y),initial=0.)) or 1.
    values=[]
    for q in range(codes.shape[1]):
        g=codes[:,q];delta=np.bincount(g,weights=w)
        v=np.bincount(g,weights=w*(r/scale))/np.sqrt(delta)
        values.append(float(la.norm(v)))
    den=float(la.norm(np.sqrt(w)*(y/scale)))
    error=float(la.norm(values))/max(den,np.finfo(float).tiny)
    if not np.isfinite(error):raise FloatingPointError('nonfinite normal-equation audit')
    return error

class FESelector:
    def __init__(self,config:SelectionConfig|None=None):self.config=config or SelectionConfig()
    def prepare(self,codes,weights=None,*,expected_rhs:int|None=None):
        return PreparedFE(codes,weights,self.config,expected_rhs=expected_rhs)

class PreparedFE:
    def __init__(self,codes,weights,config,expected_rhs=None):
        t=perf_counter();self.config=config;self.ledger=AllocationLedger(config.budget)
        c=np.asarray(codes)
        if c.ndim!=2 or not np.issubdtype(c.dtype,np.integer) or np.any(c<0):
            raise ValueError('two-dimensional nonnegative integer codes required')
        n,q=c.shape
        if not n:raise ValueError('empty sample is not supported')
        if q and (int(c.max())>=n or int(c.max())>=np.iinfo(np.int32).max):
            raise ValueError('dense observed codes required; labels cannot exceed sample size')
        if n*max(1,q)>=np.iinfo(np.int32).max:raise ValueError('32-bit sparse-index range exceeded')
        if weights is not None and np.iscomplexobj(weights):
            raise ValueError('weights must be real')
        w=np.ones(n) if weights is None else np.asarray(weights,dtype=float)
        if w.ndim!=1 or w.shape!=(n,) or np.any(~np.isfinite(w)) or np.any(w<=0):
            raise ValueError('finite strictly positive one-dimensional weights required')
        if expected_rhs is not None and (isinstance(expected_rhs,bool) or not isinstance(expected_rhs,int) or expected_rhs<1):
            raise ValueError('expected_rhs must be a positive actual batch width')
        # Own immutable snapshots. Caller mutation cannot make the cache stale.
        self.ledger.admit('owned_input_snapshots',n*(q*4+8)+16*n)
        self.codes=np.array(c,dtype=np.int32,order='F',copy=True)
        self.w=np.array(w/w.max(),dtype=float,copy=True)
        if np.any(self.w<=0):raise ValueError('weight normalization underflow; unsupported numeric dynamic range')
        self.codes.flags.writeable=False;self.w.flags.writeable=False
        self.profile,self.offsets,self.masses=inspect(self.codes,self.w,with_masses=True)
        self.masses.flags.writeable=False;self.offsets.flags.writeable=False
        self.inspection_seconds=perf_counter()-t
        h=hashlib.sha256()
        h.update(str(self.codes.shape).encode());h.update(self.codes.ravel(order='K').data);h.update(self.w.data)
        self.fingerprint=h.hexdigest();self.expected_rhs=expected_rhs
        self.preparation_seconds=perf_counter()-t
        self.native=None;self.root=None;self.system=None;self.cached_route=None
        self.cached_width=None;self.cache_builds=0
    def with_weights(self,weights):
        """Explicit rebuild; neither numeric factors nor decision state are reused."""
        return PreparedFE(self.codes,weights,self.config,expected_rhs=self.expected_rhs)
    def _native(self):
        if self.native is None:
            import within
            cfg={'adaptive':within.PreconditionerConfig.Adaptive,'diagonal':within.PreconditionerConfig.Diagonal,
                 'additive':within.PreconditionerConfig.Additive}[self.config.native_mode]()
            # Validated nonnegative int32 IDs have identical uint32 bits. The
            # immutable F-order snapshot is already in within's preferred layout.
            cat=self.codes.view(np.uint32)
            self.native=within.Solver(within.Design(cat),weights=None if self.profile['equal_weights'] else self.w,
                                      preconditioner=cfg)
        return self.native
    def _native_batch(self,Y,events,*,partial=False):
        """Batch all remaining native work and audit each output once.

        Chunk known normalization/result/coefficient buffers to ~128 MiB;
        within's internal allocations are outside this working-buffer bound.
        """
        import within
        solver=self._native();n,k=Y.shape
        width=max(1,min(k,int(128*2**20//max(1,16*n+8*len(self.masses)))))
        R=np.empty(Y.shape,dtype=float,order='F');normals=[];feasible=[];iters=[]
        solve_seconds=0.;audit_seconds=0.;build_in_call=0.;calls=0
        for start in range(0,k,width):
            end=min(k,start+width);y=Y[:,start:end]
            scales=column_scales(y);yy=np.asfortranarray(y/scales)
            pre=solver.preconditioner
            old_build=0. if pre is None else pre.build_duration_seconds
            t=perf_counter()
            result=solver.solve_batch(yy,options=within.LsmrOptions(tol=self.config.tol,maxiter=self.config.maxiter))
            elapsed=perf_counter()-t;solve_seconds+=elapsed;calls+=1
            if not all(result.converged):raise RuntimeError('native within did not converge')
            pre=solver.preconditioner
            new_build=0. if pre is None else pre.build_duration_seconds
            build_in_call+=max(0.,new_build-old_build)
            R[:,start:end]=np.asarray(result.demeaned)*scales
            ta=perf_counter()
            normal,feas=audit_columns(self.codes,self.w,y,R[:,start:end],self.offsets,self.masses,
                                      coef=np.asarray(result.x),scales=scales)
            audit_seconds+=perf_counter()-ta
            if np.any(normal>self.config.audit_tol) or np.any(feas>self.config.audit_tol):
                raise RuntimeError('native projection audit failed')
            normals.extend(normal.tolist());feasible.extend(feas.tolist());iters.extend(map(int,result.iterations))
        events.append(dict(stage='native_finish_partial_batch' if partial else 'native_batch',seconds=solve_seconds,
                           columns=k,calls=calls,iterations=iters,normal=normals,feasibility=feasible,
                           audit_seconds=audit_seconds,preconditioner_build_in_call_seconds=build_in_call,
                           estimated_warm_seconds=max(0.,solve_seconds-build_in_call),
                           escalated=bool(solver.has_escalated)))
        self._last_normals=np.asarray(normals)
        self._last_warm_seconds=max(0.,solve_seconds-build_in_call)
        return R,solve_seconds,iters
    def _native_solve(self,y,events,*,partial=False):
        r,elapsed,it=self._native_batch(y[:,None],events,partial=partial)
        event=events[-1];event['stage']='native_finish_partial' if partial else 'native_solve'
        event['iterations']=it[0];event['normal']=event['normal'][0];event['feasibility']=event['feasibility'][0]
        return r[:,0],elapsed,it[0]
    def _native_tail(self,Y,R,normals,start,events):
        if start<Y.shape[1]:
            rr,_,_=self._native_batch(Y[:,start:],events)
            R[:,start:]=rr;normals[start:]=self._last_normals
    def _early_rejection(self,Y):
        """No numeric profile, Gram or factors are formed by this check."""
        shortlist=elimination_candidates(self.profile,self.config.budget)
        selected=next((x for x in shortlist if x['admitted']),None)
        if selected is None:return 'no_admitted_elimination'
        n,q=self.codes.shape;m=selected['reduced_upper']+q-1
        retained=self.codes.nbytes+self.w.nbytes+2*Y.nbytes+64*n
        upper=retained+n*(128+96*(q-1))+m*96
        if upper>self.config.budget.limit:return 'profile_workspace_preflight'
        return None
    def _build_candidate(self,Y,budget_seconds,events):
        t=perf_counter();retained=self.codes.nbytes+self.w.nbytes+Y.nbytes+Y.nbytes+64*len(Y)
        shortlist=elimination_candidates(self.profile,self.config.budget)
        events.append(dict(stage='elimination_shortlist',candidates=shortlist,built_limit=1))
        selected=next((x for x in shortlist if x['admitted']),None)
        if selected is None:raise CandidateRejected('no_admitted_elimination',shortlist=shortlist)
        def deadline(stage):
            dt=perf_counter()-t
            if dt>budget_seconds:raise CandidateRejected('soft_construction_deadline',stage=stage,elapsed=dt,budget=budget_seconds)
        system,order=prepare_system(self.codes,self.w,selected['eliminate'],self.ledger,retained)
        deadline('profile');T,scale,stats=guarded_star(system,self.ledger,retained)
        deadline('star');pool=candidates(T,self.ledger,self.config.max_candidates,retained+memory_bytes(system.H))
        events.append(dict(stage='partition_candidates',star=stats,candidates=[{k:v for k,v in p.items() if k!='labels'} for p in pool]))
        valid=[p for p in pool if p['admitted'] and p['singleton_dof_fraction']<.9]
        if not valid:raise CandidateRejected('no_useful_admitted_partition')
        # Construct exactly one candidate. No repeated expensive factorization.
        p=valid[0];deadline('partition')
        # Do not factor every local block before checking whether this partition
        # supplies plausible numerical value. Explicit candidate mode is an
        # experimental bypass, not an endorsement of this routing heuristic.
        if self.config.mode != 'candidate':
            evidence=preview_partition(T,p,self.ledger,eig_cut=self.config.eig_cut,
                policy=self.config.screen,retained_bytes=retained+memory_bytes(system.H))
            events.append(evidence)
            if not evidence['accepted']:
                raise CandidateRejected(evidence['reason'],evidence=evidence)
            deadline('numerical_preview')
        local_mem=(retained+memory_bytes(system.H)+memory_bytes(T)
            +32*p['sum_block_squares']+64*self.config.budget.max_block**2
            +192*min(T.shape[0],self.config.budget.max_block)*min(T.shape[0],self.config.budget.max_coarse))
        self.ledger.admit('local_factors_before_allocation',local_mem,selected_tau=p['tau'])
        base=ComponentRight(T,scale,tau=p['tau'],max_block=self.config.budget.max_block,
                            eig_cut=self.config.eig_cut,max_coarse=self.config.budget.max_coarse)
        root=base;coarse_stats=None
        # Once a valid root exists, its build cost is sunk. A missed soft
        # checkpoint must stop FURTHER construction, not discard usable work.
        # Correctness and per-allocation memory gates remain mandatory.
        if perf_counter()-t>budget_seconds:
            events.append(dict(stage='ready_candidate_over_soft_budget',available='local_only',
                               elapsed=perf_counter()-t,budget=budget_seconds,
                               action='skip_coarse_and_measure_useful_pilot'))
        elif base.Z.shape[1]:
            root=GuardedCoarseRight(system,base,self.ledger,retained+memory_bytes(T))
            coarse_stats=root.stats
            if perf_counter()-t>budget_seconds:
                events.append(dict(stage='ready_candidate_over_soft_budget',available='local_coarse',
                                   elapsed=perf_counter()-t,budget=budget_seconds,
                                   action='retain_completed_root_and_measure_useful_pilot'))
        self.system=system;self.root=root;self.cache_builds+=1
        events.append(dict(stage='candidate_built',seconds=perf_counter()-t,elimination=order[0],
                           tau=p['tau'],blocks=base.stats,coarse=coarse_stats,
                           kind='local_coarse' if coarse_stats else 'local_only'))
    def _local_column(self,y,events,maxiter):
        t=perf_counter()
        yscale=float(np.max(np.abs(y),initial=0.)) or 1.
        controller=getattr(self,'_pilot_controller',None)
        if controller is None:
            r,info=solve_rect(self.system,(y/yscale)[:,None],self.root,tol=self.config.tol,maxiter=maxiter)
            r=r[:,0]*yscale
        else:
            r,meta=solve_profiled_pilot(self.system,y/yscale,self.root,
                tol=self.config.tol,maxiter=maxiter,controller=controller)
            r=r*yscale;info={'rhs':[meta]}
            events.append(dict(stage='pilot_continuation',**meta))
        elapsed=perf_counter()-t;status=info['rhs'][0]['status']
        ta=perf_counter()
        normal,_=audit_columns(self.codes,self.w,y[:,None],r[:,None],self.offsets,self.masses,
                                scales=np.array([yscale]))
        normal=float(normal[0]);audit_seconds=perf_counter()-ta
        ok=status in (0,1,2,4,5) and normal<=self.config.audit_tol
        events.append(dict(stage='local_solve',seconds=elapsed,iterations=info['rhs'][0]['iterations'],
                           status=status,normal=normal,audit_seconds=audit_seconds,accepted=bool(ok),
                           feasibility='original profiled reconstruction; independently tested against dummy LS'))
        if not ok:
            # Native finishes a feasible partial residual; do not add it to r.
            r,extra,_=self._native_solve(r,events,partial=True)
            elapsed+=extra
            # Partial-native audit uses the partial RHS; report the ORIGINAL RHS
            # normalization for the returned column, without skipping any check.
            normal=float(audit_columns(self.codes,self.w,y[:,None],r[:,None],self.offsets,self.masses)[0][0])
        self._last_normals=np.array([normal])
        return r,elapsed,ok
    def residualize(self,Y):
        t=perf_counter();Y=np.asarray(Y);was_1d=Y.ndim==1
        if was_1d:Y=Y[:,None]
        if Y.ndim!=2 or len(Y)!=len(self.codes) or Y.shape[1]<1 or not np.issubdtype(Y.dtype,np.number) or np.iscomplexobj(Y):
            raise ValueError('finite RHS with matching rows and at least one column required')
        k=Y.shape[1]
        if self.expected_rhs is not None and k!=self.expected_rhs:raise ValueError('RHS width differs from declared workload')
        self.ledger.admit('RHS_and_output',self.codes.nbytes+self.w.nbytes+Y.size*8*3+len(Y)*64)
        Y=np.asarray(Y,dtype=float)
        if not np.all(np.isfinite(Y)):raise ValueError('finite RHS required')
        events=[];route='native';R=np.empty_like(Y);normals=np.full(k,np.nan)
        q=self.codes.shape[1]
        if q<2:
            if not q:R[:]=Y;route='identity'
            else:
                g=self.codes[:,0];d=self.masses
                for j in range(k):
                    # Normalize first: avoid RHS-unit-dependent sum overflow.
                    scale=float(np.max(np.abs(Y[:,j]),initial=0.)) or 1.
                    z=Y[:,j]/scale
                    R[:,j]=(z-(np.bincount(g,weights=self.w*z)/d)[g])*scale
                route='single_fe_exact'
            normals[:]=audit_columns(self.codes,self.w,Y,R,self.offsets,self.masses)[0]
            events.append(dict(stage='exact_fastpath',route=route))
        elif self.cached_route and self.cached_width==k:
            route=self.cached_route
            events.append(dict(stage='cache_hit',route=route,design=self.fingerprint))
            j=0
            if route=='local':
                while j<k:
                    R[:,j],_,ok=self._local_column(Y[:,j],events,self.config.maxiter)
                    normals[j]=self._last_normals[0];j+=1
                    if not ok:
                        route='native';self.cached_route='native';self.root=None;self.system=None
                        break
            self._native_tail(Y,R,normals,j,events)
        else:
            self.root=None;self.system=None
            reason=('native_requested' if self.config.mode=='native' else
                    'insufficient_known_rhs' if k<3 else self._early_rejection(Y))
            if reason:
                events.append(dict(stage='bypass',reason=reason,before_first_solve=True))
                self._native_tail(Y,R,normals,0,events)
            else:
                # Useful pilot; never discard first-column work. Setup incurred
                # during adaptive escalation is not future per-column solve cost.
                R[:,0],per,it=self._native_solve(Y[:,0],events)
                normals[0]=self._last_normals[0]
                warm_per=self._last_warm_seconds
                first_it=it
                start=1
                # Scout only after a useful native solve warrants exploration.
                # Its sampling never supplies the target or its nullspace.
                early_verdict='unknown';scout_seconds=0.
                prelim_remaining=k-start
                prelim_budget=self.config.cost.construction_budget(warm_per,prelim_remaining)
                if (self.config.mode != 'candidate' and it >= self.config.cost.minimum_iterations
                        and self.config.cost.exploration_admitted(it,prelim_remaining)
                        and 8*self.inspection_seconds <= prelim_budget):
                    evidence=scout(self.codes,self.w,self.profile,self.ledger,
                        policy=self.config.early,screen_policy=self.config.screen,
                        eig_cut=self.config.eig_cut,max_candidates=self.config.max_candidates,
                        retained_bytes=self.codes.nbytes+self.w.nbytes+2*Y.nbytes+64*len(Y))
                    events.append(evidence)
                    # The 0.4 counterexample establishes that negative proxy
                    # evidence cannot safely veto a full-data candidate.
                    early_verdict=evidence['verdict'];scout_seconds=evidence['seconds']
                    if early_verdict == 'reject':
                        early_verdict='unknown'
                        events.append(dict(stage='sample_evidence_downgraded',
                            raw_verdict='reject',effective_verdict='unknown',
                            reason='sample_negative_is_not_full_system_evidence',
                            action='retain_full_data_budget_and_numerical_checks'))
                # A second actual native column has opportunity cost. Request it
                # for marginal iteration counts unless the route was rejected.
                # Sampled POSITIVE evidence can be spurious and must not cancel
                # a valuable warm confirmation in this ambiguous regime.
                native_event=next((e for e in reversed(events) if e['stage']=='native_solve'),{})
                old_warm_eligible=(self.config.mode != 'candidate' and
                    k >= self.config.cost.warm_confirmation_min_rhs and
                    native_event.get('escalated',False) and
                    it >= self.config.cost.minimum_iterations and
                    self.config.cost.exploration_admitted(it,k-1))
                ambiguous=(it <= self.config.early.warm_confirmation_max_iteration_ratio
                               * max(1,self.config.cost.minimum_iterations))
                confirm=old_warm_eligible and early_verdict!='reject' and ambiguous
                if old_warm_eligible:
                    events.append(dict(stage='warm_confirmation_decision',confirm=bool(confirm),
                        scout_verdict=early_verdict,first_iterations=it,
                        maximum_ambiguous_iterations=self.config.early.warm_confirmation_max_iteration_ratio
                            * max(1,self.config.cost.minimum_iterations),
                        reason='ambiguous_native_startup' if confirm else
                               'proxy_rejected' if early_verdict=='reject' else
                               'first_rhs_far_above_iteration_gate',
                        guarantee='cost heuristic, not a calibrated value-of-information model'))
                if confirm:
                    cold_it=it;first_warm_estimate=warm_per
                    R[:,1],_,it=self._native_solve(Y[:,1],events)
                    normals[1]=self._last_normals[0];warm_per=self._last_warm_seconds
                    start=2
                    events.append(dict(stage='useful_warm_confirmation',columns_retained=2,
                        first_iterations=cold_it,warm_iterations=it,
                        first_warm_seconds_estimate=first_warm_estimate,
                        measured_warm_seconds=warm_per,
                        reason='ambiguous_native_startup_not_resolved_by_proxy'))
                remaining=k-start
                budget=max(0.,self.config.cost.construction_budget(warm_per,remaining)-scout_seconds)
                events.append(dict(stage='cost_observation',first_rhs_seconds=per,first_rhs_iterations=first_it,
                    decision_rhs_index=start-1,decision_iterations=it,
                    estimated_warm_rhs_seconds=warm_per,remaining_rhs=remaining,
                    remaining_native_seconds_estimate=warm_per*remaining,construction_budget_seconds=budget,
                    scout_seconds_charged=scout_seconds,
                    setup_iteration_equivalents=self.config.cost.setup_iteration_equivalents,
                    uncertainty='single useful RHS, excludes observed one-time build; heuristic, not a lower bound'))
                reason=None
                if self.config.mode!='candidate':
                    if early_verdict=='reject':reason='early_scout_rejected_before_full_profile'
                    elif budget<=0:reason='scout_spent_construction_budget'
                    elif it<self.config.cost.minimum_iterations:reason='native_already_fast'
                    elif not self.config.cost.exploration_admitted(it,remaining):reason='insufficient_remaining_iteration_work'
                    elif 8*self.inspection_seconds>budget:reason='inspection_cost_not_amortized'
                if reason:events.append(dict(stage='bypass',reason=reason))
                else:
                    try:
                        self._build_candidate(Y,float('inf') if self.config.mode=='candidate' else budget,events)
                        if self.config.mode=='observe':
                            events.append(dict(stage='observe_only',recommendation='candidate_ready_execution_unchanged'))
                        else:route='local'
                    except (CandidateRejected,ResourceGuard,la.LinAlgError) as exc:
                        events.append(dict(stage='candidate_rejected',reason=getattr(exc,'reason',str(exc)),
                                           details=getattr(exc,'details',{})))
                        self.system=None;self.root=None
                if route=='local':
                    # One live LSMR call: never rerun the first 24 iterations.
                    self._pilot_controller=(PilotController(self.config.pilot,
                        baseline_seconds=max(warm_per,np.finfo(float).eps),
                        remaining_rhs=k-start,initial_steps=self.config.pilot_iterations,
                        gain_margin=self.config.cost.gain_margin)
                        if self.config.pilot.enabled and self.config.mode!='candidate' else None)
                    pilot_limit=self.config.maxiter if self._pilot_controller is not None else self.config.pilot_iterations
                    try:
                        R[:,start],local_s,ok=self._local_column(Y[:,start],events,pilot_limit)
                    finally:
                        self._pilot_controller=None
                    normals[start]=self._last_normals[0]
                    retain=bool(ok and (self.config.mode=='candidate' or self.config.cost.continue_local(local_s,warm_per)))
                    events.append(dict(stage='useful_column_pilot',seconds=local_s,passed=ok,
                                       baseline_rhs_seconds=warm_per,continue_local=retain))
                    if not retain:route='native';self.system=None;self.root=None
                    start+=1
                j=start
                if route=='local':
                    while j<k:
                        R[:,j],_,ok=self._local_column(Y[:,j],events,self.config.maxiter)
                        normals[j]=self._last_normals[0];j+=1
                        if not ok:
                            route='native';self.system=None;self.root=None
                            break
                self._native_tail(Y,R,normals,j,events)
            self.cached_route=route;self.cached_width=k
        # Every accepted column has one original-target audit. Previously native
        # and local solves audited here AGAIN, redoing O(NQk) work and masses.
        if not np.all(np.isfinite(normals)) or np.any(normals>self.config.audit_tol):
            raise RuntimeError('original-target final normal audit failed')
        diag=dict(policy_version='fe-selector-0.5',config=asdict(self.config),profile=self.profile,
                  fingerprint=self.fingerprint,preparation_seconds=self.preparation_seconds,
                  call_seconds=perf_counter()-t,final_route=route,events=events,
                  allocation_events=list(self.ledger.events),normal_residuals=normals.tolist(),
                  cache_builds=self.cache_builds,
                  guarantee='original target and normal/feasibility audit; not universal forward-error or total-RSS certificate')
        return SelectionResult(R[:,0] if was_1d else R,diag)
