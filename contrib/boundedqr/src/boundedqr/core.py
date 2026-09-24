"""User-facing bounded-memory quantile bootstrap orchestration."""
from dataclasses import dataclass
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .base import fn_fit
from .preprocess import CFMPreprocessor
from .certification import coefficient_radius
from .scores import encode_clusters,cluster_scores,mammen_block
from .inference import summarize_process
from .scaling import ColumnCoordinates

class NumericalFailure(RuntimeError):
    """A fit did not pass its declared original-data numerical checks."""

class _CheckedPreprocessor(CFMPreprocessor):
    def __init__(self,*args,verify_radius=True,**kwargs):
        self.verify_radius=verify_radius
        super().__init__(*args,**kwargs)

    def _diagnostics(self,w,beta,a,a0,residual):
        out=super()._diagnostics(w,beta,a,a0,residual)
        if self.verify_radius:
            out['radius_certificate']=coefficient_radius(self.x,self.y,self.tau,w,beta,np.r_[a,a0]).as_dict()
        return out

def solve_perturbations(X,y,tau,W,beta0,*,backend='cpu',batch_size=16,
                        verify_radius=True,cpu_factor=1.,threads=8,
                        audit_backend='cpu',repair='batched',positive_pseudo=False):
    """Solve the finite pseudo-observation QR problems used by quantreg.

    X(n,p) includes an intercept if desired; W(p,B) is quantreg's gradient,
    with its original sign. Each augmented row is (W_b/tau,n*max(abs(y))).
    beta0 is a proposal only. The target is the nonsmoothed finite problem.
    Returned draws pass the stated numerical checks; all_checked is not an
    exact optimality proof or a coefficient-error bound. A finite coefficient
    radius supplies a separate bound under its stated FP64 arithmetic model,
    and its size still matters. Failed checks raise; draws are never silently
    removed. Uninformative radii remain infinity with a reason, even when
    ordinary checks pass. verify_radius=False supplies no coefficient bound.
    Internally, lossless power-of-two column coordinates prevent absolute LP
    thresholds from treating small-unit regressors as zero. Coefficients and
    radii are returned in original units; stationarity diagnostics explicitly
    refer to normalized coordinates. Unrepresentable dynamic ranges raise.
    Column normalization does not remove near collinearity, weak sharpness,
    ties, or nonunique minimizers.
    audit_backend='gpu_fused' explicitly selects GPU FP64 row envelopes for
    GPU batched repair; 'cpu' (default) and 'gpu' retain their original paths.
    CPU solves and scalar repair always use the scalar CPU auditor. Diagnostics
    distinguish the requested and effective choice. CUDA errors propagate.
    positive_pseudo=True optionally tries the a0=tau dual face, requiring an
    outward strictly positive finite-pseudo residual and the original checks.
    Failed face trials restart the unchanged free-pseudo strategy; infinity
    remains an uninformative radius. The default free strategy is unchanged.
    """
    if backend not in ('cpu','gpu'):
        raise ValueError('backend must be cpu or gpu')
    if audit_backend not in ('cpu','gpu','gpu_fused') or repair not in ('batched','scalar'):
        raise ValueError('Invalid audit backend or repair strategy')
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v<1
           for v in (batch_size,threads)):
        raise ValueError('batch_size and threads must be positive integers')
    w=np.asarray(W,dtype=float)
    if w.ndim!=2 or w.shape[1]<1:
        raise ValueError('W must be a p-by-B matrix with at least one draw')
    wall=time.perf_counter()
    with threadpool_limits(threads):
        coordinates=ColumnCoordinates.from_design(X)
        if not np.isfinite(tau) or not 0<float(tau)<1:
            raise ValueError('tau must lie strictly between zero and one')
        w=coordinates.perturbations_in(w,float(tau))
        supplied_initial=np.asarray(beta0,dtype=float).reshape(-1)
        if supplied_initial.size!=coordinates.design.shape[1]:
            raise ValueError('beta0 must have one entry per design column')
        initial=coordinates.coefficients_in(supplied_initial)
        solver=_CheckedPreprocessor(coordinates.design,y,tau,initial,verify_radius=verify_radius,
                                    factor=cpu_factor if backend=='cpu' else .01,
                                    positive_pseudo=positive_pseudo)
        if backend=='cpu':
            beta,diagnostics=solver.fit_batch(w)
        else:
            from .gpu_newton import solve_batch
            from .batched_repair import iter_repair_batches
            from .batched_radius import RadiusAuditor
            if audit_backend=='gpu_fused':
                from .fused_radius import FusedRadiusAuditor
            beta=np.empty((w.shape[1],solver.p))
            fits=[]
            proposals=[]
            proposed_all=np.empty_like(beta)
            for first in range(0,w.shape[1],batch_size):
                last=min(first+batch_size,w.shape[1])
                block=w[:,first:last]
                initial=(solver.beta0[:,None]+solver.jacobian_inverse@block).T
                proposed,info=solve_batch(solver.x,solver.y,tau,block,initial,
                                         batch_size=batch_size,mu_factors=(1.,.4,.15),max_seconds=30.)
                proposals.append(info)
                proposed_all[first:last]=proposed
            if repair=='scalar':
                for j in range(w.shape[1]):
                    b,record=solver._fit_one(w[:,j],proposed_all[j]-solver.beta0,solver.leverage)
                    record['replicate']=j
                    beta[j]=b
                    fits.append(record)
                blocks=[]
            else:
                auditor_class=FusedRadiusAuditor if audit_backend=='gpu_fused' else RadiusAuditor
                product_backend='gpu' if audit_backend=='gpu_fused' else audit_backend
                auditor=(auditor_class(solver.x,solver.y,tau,backend=product_backend,max_batch=batch_size)
                         if verify_radius else None)
                blocks=[]
                for values,duals,block_diag in iter_repair_batches(solver,w,proposed_all,
                                                                  backend=backend,batch_size=min(64,batch_size),
                                                                  positive_pseudo=positive_pseudo):
                    indices=block_diag['replicate_indices']
                    if not block_diag['all_checked']:
                        raise NumericalFailure(f'Batched original-data checks failed: {block_diag}')
                    if auditor is not None:
                        certs=auditor.audit(w[:,indices],values,duals)
                        for f,c in zip(block_diag['fits'],certs):
                            f['radius_certificate']=c.as_dict()
                    beta[indices]=values
                    fits.extend(block_diag['fits'])
                    blocks.append(block_diag)
            diagnostics=dict(fits=fits,proposals=proposals,setup_seconds=solver.setup_seconds,
                             repair=repair,repair_blocks=blocks,
                             audit_backend=audit_backend,
                             auditor_diagnostics=auditor.diagnostics if repair=='batched' and auditor is not None else None,
                             fallback_count=sum(f['fallback'] for f in fits),
                             all_checked=all(f['numerically_checked'] for f in fits))
    diagnostics.update(backend=backend,batch_size=batch_size,threads=threads,
                       total_seconds=time.perf_counter()-wall,verify_radius=verify_radius,
                       audit_backend_requested=audit_backend,
                       positive_pseudo=bool(positive_pseudo),
                       audit_backend_effective=(None if not verify_radius else
                                                'cpu' if backend=='cpu' or repair=='scalar' else audit_backend))
    if not diagnostics['all_checked']:
        bad=[i for i,f in enumerate(diagnostics['fits']) if not f['numerically_checked']]
        raise NumericalFailure(f'Original-data checks failed for draws {bad}; no draws discarded')
    try:
        beta=coordinates.coefficients_out(beta)
    except ValueError as error:
        raise NumericalFailure(str(error)) from error
    # Loss, residuals and dual bounds are invariant under this exact coordinate
    # map. The equality residual and sharpness linear algebra use normalized
    # coefficients; only the reported coefficient radius is mapped to original
    # Euclidean units. An infinite radius is never the acceptance criterion.
    for fit in diagnostics['fits']:
        fit['stationarity_coordinates']='column-normalized'
        if 'radius_certificate' in fit:
            fit['radius_certificate']=coordinates.map_certificate(fit['radius_certificate'])
    diagnostics['column_normalization']=coordinates.diagnostics()
    return beta,diagnostics

@dataclass
class BootstrapResult:
    quantiles: np.ndarray
    coefficients: np.ndarray
    bootstrap_coefficients: np.ndarray
    standard_errors: np.ndarray
    covariance: np.ndarray
    basic_intervals: np.ndarray
    uniform_band: np.ndarray
    coefficient_radii: np.ndarray
    inference: dict
    diagnostics: dict

def bootstrap(X,y,cluster,*,quantiles=(.5,),reps=199,seed=20260915,
              backend='cpu',batch_size=16,threads=8,level=.95,
              base_coefficients=None,reference_psi=None,multipliers=None,
              verify_radius=True,cpu_factor=1.,base_tolerance=1e-8,audit_backend='cpu',
              positive_pseudo=False):
    """Clustered wild-gradient bootstrap for an unweighted linear QR process.

    The same indexed Mammen draws are used across quantiles. Default NumPy RNG
    differs from R: for replication supply both reference_psi and multipliers,
    with rows ordered by the sorted unique cluster labels. Reference baseline
    coefficients should accompany reference_psi. The score is (residual<0)-tau;
    a recovered KKT dual is never used as a replacement.

    This dense first release neither absorbs fixed effects nor supports survey
    weights, missing observations, or automatic collinearity deletion. Explicit
    dummy columns are accepted when full rank. A numerical certificate does not
    justify the econometric assumptions, few-cluster inference, or survey design.
    audit_backend is passed to solve_perturbations; the default remains 'cpu'.
    'gpu_fused' is an explicit optional GPU envelope path, with no silent CPU
    fallback. CPU solves still use the scalar CPU auditor.
    """
    wall=time.perf_counter()
    x=np.asarray(X,dtype=float)
    y=np.asarray(y,dtype=float)
    taus=np.atleast_1d(np.asarray(quantiles,dtype=float))
    if x.ndim!=2 or min(x.shape)<1 or y.shape!=(len(x),):
        raise ValueError('Require nonempty X(n,p) and y(n)')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Missing or infinite input must be handled explicitly before fitting')
    if taus.ndim!=1 or len(taus)<1 or not np.isfinite(taus).all() or np.any((taus<=0)|(taus>=1)):
        raise ValueError('quantiles must be a nonempty vector strictly inside (0,1)')
    if (any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v<1
            for v in (reps,batch_size,threads)) or reps<2 or not 0<level<1):
        raise ValueError('Require reps>=2, batch_size>=1, and 0<level<1')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError('seed must be a nonnegative integer')
    labels,codes=encode_clusters(cluster,len(x))
    q,p=len(taus),x.shape[1]
    supplied=None if base_coefficients is None else np.asarray(base_coefficients,dtype=float).reshape(q,p)
    psi_input=None if reference_psi is None else np.asarray(reference_psi,dtype=float).reshape(q,len(x))
    if psi_input is not None and supplied is None:
        raise ValueError('reference_psi must be accompanied by its base_coefficients')
    v=None if multipliers is None else np.asarray(multipliers,dtype=float)
    if v is not None and (v.shape!=(len(labels),reps) or not np.isfinite(v).all()):
        raise ValueError('multipliers must be finite G-by-reps in sorted-label order')
    coefficients=np.empty((q,p))
    draws=np.empty((reps,q,p))
    radii=np.full((reps,q),np.inf)
    records=[]
    # Keep baseline FN arithmetic away from extreme column units as well.
    # Supplied reference residual indicators remain untouched.
    base_coordinates=ColumnCoordinates.from_design(x) if supplied is None else None
    with threadpool_limits(threads):
        for j,tau in enumerate(taus):
            start=time.perf_counter()
            if supplied is None:
                base=fn_fit(base_coordinates.design,y,tau,tol=base_tolerance)
                if not base.converged:
                    raise NumericalFailure(f'PyFixest FN baseline failed at tau={tau}')
                b=base_coordinates.coefficients_out(base.beta)
                residual=base.residual
                source='pyfixest_fn_0.60.0'
            else:
                b=supplied[j]
                if not np.isfinite(b).all(): raise ValueError('Nonfinite supplied coefficients')
                residual=y-x@b
                source='supplied_reference'
            coefficients[j]=b
            psi=(residual<0).astype(float)-tau if psi_input is None else psi_input[j]
            if not np.isfinite(psi).all(): raise ValueError('Nonfinite reference scores')
            scores=cluster_scores(x,psi,codes,len(labels))
            W=np.empty((p,reps))
            for first in range(0,reps,batch_size):
                last=min(first+batch_size,reps)
                block=(mammen_block(len(labels),range(first,last),seed) if v is None else v[:,first:last])
                W[:,first:last]=scores.T@block
            del scores
            preparation=time.perf_counter()-start
            draws[:,j],diag=solve_perturbations(x,y,float(tau),W,b,backend=backend,
                                               batch_size=batch_size,threads=threads,
                                               verify_radius=verify_radius,cpu_factor=cpu_factor,
                                               audit_backend=audit_backend,
                                               positive_pseudo=positive_pseudo)
            if verify_radius:
                radii[:,j]=[f['radius_certificate']['radius'] for f in diag['fits']]
            records.append(dict(tau=float(tau),base_source=source,
                                base_column_normalization=(base_coordinates.diagnostics()
                                                           if base_coordinates is not None else None),
                                score_source='supplied_reference' if psi_input is not None else 'residual_indicator',
                                preparation_seconds=preparation,solver=diag))
    inference=summarize_process(coefficients,draws,level=level,radii=radii if verify_radius else None)
    return BootstrapResult(taus,coefficients,draws,inference['standard_errors'],
                           inference['covariance'],inference['basic_intervals'],
                           inference['uniform_band'],radii,inference,
                           dict(quantiles=records,total_seconds=time.perf_counter()-wall,
                                observations=len(x),parameters=p,clusters=len(labels),reps=reps,
                                seed=int(seed),backend=backend,batch_size=batch_size,
                                audit_backend_requested=audit_backend,
                                positive_pseudo=bool(positive_pseudo),
                                multiplier_source='supplied' if v is not None else 'indexed_mammen',
                                score_tie_convention='(residual < 0) - tau',
                                error_scope='Fixed FP64 input and constructed W; conditional on baseline scores'))
