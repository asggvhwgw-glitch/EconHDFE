use crate::reductions::{self, Geometry, SmallSchur, TupleIndex};
use ndarray::{Array2, ShapeBuilder};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::sync::Arc;
use within::{Design, LsmrOptions, PreconditionerConfig, Solver};

const REV: &str = "fc44282cea6124f5f18bb142908605a656d35a7c";
fn value(e: impl ToString) -> PyErr {
    PyValueError::new_err(e.to_string())
}
fn runtime(e: impl ToString) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn upstream(
    groups: &[Vec<u32>],
    weights: &[f64],
    preconditioner: Option<&PreconditionerConfig>,
) -> reductions::Result<Solver<'static>> {
    let n = weights.len();
    let flat = groups.iter().flat_map(|g| g.iter().copied()).collect();
    let a = Array2::from_shape_vec((n, groups.len()).f(), flat).map_err(|e| e.to_string())?;
    let design = Design::from_categories(a.view())
        .map_err(|e| e.to_string())?
        .into_owned();
    Solver::new(design, Some(weights), preconditioner).map_err(|e| e.to_string())
}
enum Engine {
    Schur(SmallSchur),
    Iterative {
        solver: Solver<'static>,
        mass: Vec<f64>,
        compressed: bool,
    },
}
#[pyclass(module = "_econhdfe_within")]
pub struct NativeSolver {
    geometry: Geometry,
    index: Option<Arc<TupleIndex>>,
    weights: Vec<f64>,
    strategy: String,
    budget: usize,
    engine: Engine,
    pool: Arc<rayon::ThreadPool>,
    generation: usize,
    fallbacks: usize,
    schur_fallbacks: usize,
}
impl NativeSolver {
    fn build_engine(
        g: &Geometry,
        ix: Option<&TupleIndex>,
        w: &[f64],
        strategy: &str,
        budget: usize,
    ) -> reductions::Result<Engine> {
        if matches!(strategy, "auto" | "schur") {
            if let Some(s) = SmallSchur::build(g, w, budget)? {
                return Ok(Engine::Schur(s));
            }
            if strategy == "schur" {
                return Err("Schur rejected: size, resource, rank or conditioning gate".into());
            }
        }
        let compressed = ix.is_some_and(|ix| {
            strategy == "compressed"
                || (strategy == "auto" && ix.cells() as f64 / g.n as f64 <= 0.7)
        });
        if compressed {
            let ix = ix.unwrap();
            let mass = ix.masses(w)?;
            let groups = ix.groups(g);
            let solver = upstream(&groups, &mass, None)?;
            Ok(Engine::Iterative {
                solver,
                mass,
                compressed: true,
            })
        } else {
            Ok(Engine::Iterative {
                solver: upstream(&g.groups, w, None)?,
                mass: vec![],
                compressed: false,
            })
        }
    }
    fn label(&self) -> &str {
        match &self.engine {
            Engine::Schur(_) => "schur",
            Engine::Iterative {
                compressed: true, ..
            } => "compressed_within",
            _ => "within",
        }
    }
    fn project(
        &self,
        y: &[f64],
        options: &LsmrOptions,
    ) -> reductions::Result<(Vec<f64>, bool, usize)> {
        match &self.engine {
            Engine::Schur(s) => Ok((s.residualize(&self.geometry, &self.weights, y)?, true, 1)),
            Engine::Iterative {
                solver,
                mass,
                compressed,
            } => {
                if *compressed {
                    let ix = self.index.as_ref().unwrap();
                    let means = ix.means(y, &self.weights, mass)?;
                    let r = solver.solve(&means, options).map_err(|e| e.to_string())?;
                    let mut out = y.to_vec();
                    ix.restore(&mut out, &means, &r.demeaned)?;
                    Ok((out, r.converged, r.iterations))
                } else {
                    let r = solver.solve(y, options).map_err(|e| e.to_string())?;
                    Ok((r.demeaned, r.converged, r.iterations))
                }
            }
        }
    }
    fn certify(
        &self,
        y: &[f64],
        tol: f64,
        max_iter: usize,
        mut project: impl FnMut(&[f64], &LsmrOptions) -> reductions::Result<(Vec<f64>, bool, usize)>,
    ) -> reductions::Result<(Vec<f64>, bool, usize, f64)> {
        // The inner global backward error and the original-row maximum mean
        // are different metrics. Only the latter decides our acceptance gate.
        let mut options = LsmrOptions {
            tol: (tol * 0.01).max(f64::MIN_POSITIVE),
            maxiter: max_iter,
            local_size: None,
        };
        let gate = tol.max(64.0 * f64::EPSILON);
        let (mut result, mut ok, mut iterations) = project(y, &options)?;
        let mut error = reductions::stationarity(&self.geometry, &self.weights, &result);
        let mut tightened = false;
        while ok && error.is_finite() && error > gate && iterations < max_iter {
            options.maxiter = max_iter - iterations;
            let (corrected, success, spent) = project(&result, &options)?;
            iterations += spent;
            let next_error = reductions::stationarity(&self.geometry, &self.weights, &corrected);
            let contracted = next_error <= 0.5 * error;
            // Keep the best certified metric if a roundoff-limited replacement
            // regresses. Both candidates still have y-r in the original FE span.
            if next_error < error {
                result = corrected;
                error = next_error;
            }
            ok = success;
            if !ok || !next_error.is_finite() || error <= gate || spent == 0 {
                break;
            }
            if contracted {
                tightened = false;
            } else if tightened {
                // A stricter inner solve also failed to halve the certificate:
                // do not spend the entire iteration budget on a roundoff floor.
                break;
            } else {
                // Retry the stalled residual with a measured, stricter target.
                // Limit the change to one decade to avoid a huge overcorrection.
                options.tol = (options.tol * (gate / error).clamp(0.1, 0.5)).max(f64::MIN_POSITIVE);
                tightened = true;
            }
        }
        Ok((
            result,
            ok && error.is_finite() && error <= gate,
            iterations,
            error,
        ))
    }
    fn one(
        &mut self,
        y: &mut [f64],
        tol: f64,
        max_iter: usize,
        other_rhs_bytes: usize,
    ) -> reductions::Result<(bool, usize, f64)> {
        if y.iter().any(|v| !v.is_finite()) {
            return Err("RHS must be finite".into());
        }
        let scale = y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if scale == 0.0 {
            return Ok((true, 0, 0.0));
        }
        for v in y.iter_mut() {
            *v /= scale;
        }
        // Residual replacement is another projection of r, so it preserves
        // y-r in col(D); all passes share the caller's total iteration budget.
        let (mut result, mut ok, mut iterations, mut error) =
            self.certify(y, tol, max_iter, |rhs, options| self.project(rhs, options))?;
        if !ok
            && iterations < max_iter
            && reductions::fallback_workspace(&self.geometry)
                .and_then(|n| n.checked_add(self.extra_resident_bytes()))
                .and_then(|n| n.checked_add(other_rhs_bytes))
                .is_some_and(|n| n <= self.budget)
        {
            // Use original rows and a fresh diagonal map: a stalled adaptive
            // preconditioner can keep reproducing its own roundoff error.
            // This changes neither the FE operator nor its numerical rank.
            let solver = upstream(
                &self.geometry.groups,
                &self.weights,
                Some(&PreconditionerConfig::Diagonal),
            )?;
            self.fallbacks += 1;
            self.schur_fallbacks += usize::from(matches!(&self.engine, Engine::Schur(_)));
            let (r, success, spent, metric) =
                self.certify(y, tol, max_iter - iterations, |rhs, options| {
                    let r = solver.solve(rhs, options).map_err(|e| e.to_string())?;
                    Ok((r.demeaned, r.converged, r.iterations))
                })?;
            if success || metric < error {
                result = r;
                error = metric;
            }
            ok = success;
            iterations += spent;
        }
        for (dst, r) in y.iter_mut().zip(result) {
            *dst = r * scale;
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err("residual overflow while restoring original units".into());
        }
        // Additional backward check, NOT a condition-independent forward bound.
        Ok((ok, iterations, error))
    }
}
#[pymethods]
impl NativeSolver {
    #[new]
    #[pyo3(signature=(codes, weights=None, strategy="auto", threads=1, workspace_bytes=536870912))]
    fn new(
        py: Python<'_>,
        codes: PyReadonlyArray2<'_, u32>,
        weights: Option<PyReadonlyArray1<'_, f64>>,
        strategy: &str,
        threads: usize,
        workspace_bytes: usize,
    ) -> PyResult<Self> {
        if !matches!(strategy, "auto" | "plain" | "compressed" | "schur") {
            return Err(value("invalid strategy"));
        }
        if threads == 0 || workspace_bytes == 0 {
            return Err(value("threads and workspace_bytes must be positive"));
        }
        let a = codes.as_array();
        let groups = a
            .columns()
            .into_iter()
            .map(|c| c.iter().copied().collect())
            .collect();
        let raw_w = weights.map(|w| w.as_array().iter().copied().collect::<Vec<_>>());
        let strategy = strategy.to_owned();
        py.detach(move || {
            let g = Geometry::new(groups)?;
            let w = reductions::weights(raw_w.as_deref(), g.n)?;
            let pool = Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|e| e.to_string())?,
            );
            let direct = if matches!(strategy.as_str(), "auto" | "schur") {
                SmallSchur::build(&g, &w, workspace_bytes)?
            } else {
                None
            };
            let index = if direct.is_none() && matches!(strategy.as_str(), "auto" | "compressed") {
                match TupleIndex::build(&g, workspace_bytes) {
                    Ok(ix) => Some(Arc::new(ix)),
                    Err(e) if strategy == "compressed" => return Err(e),
                    Err(_) => None,
                }
            } else {
                None
            };
            let engine = match direct {
                Some(s) => Engine::Schur(s),
                None => pool.install(|| {
                    Self::build_engine(&g, index.as_deref(), &w, &strategy, workspace_bytes)
                })?,
            };
            Ok(Self {
                geometry: g,
                index,
                weights: w,
                strategy,
                budget: workspace_bytes,
                engine,
                pool,
                generation: 0,
                fallbacks: 0,
                schur_fallbacks: 0,
            })
        })
        .map_err(value)
    }
    #[pyo3(signature=(data, tol=1e-10, max_iter=16000))]
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
        tol: f64,
        max_iter: usize,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<bool>, Vec<usize>, Vec<f64>)> {
        if !tol.is_finite() || tol <= 0.0 || tol > 1.0 || max_iter == 0 {
            return Err(value("invalid tolerance or iteration limit"));
        }
        let a = data.as_array();
        let (n, k) = (a.nrows(), a.ncols());
        if n != self.geometry.n {
            return Err(value("RHS row count mismatch"));
        }
        let other_rhs_bytes = n.saturating_mul(k.saturating_sub(1)).saturating_mul(8);
        let mut out: Vec<f64> = a.columns().into_iter().flat_map(|c| c.to_vec()).collect();
        let pool = Arc::clone(&self.pool);
        let (out, ok, iterations, errors) = py
            .detach(move || {
                pool.install(|| {
                    let mut ok = Vec::with_capacity(k);
                    let mut iterations = Vec::with_capacity(k);
                    let mut errors = Vec::with_capacity(k);
                    for col in out.chunks_mut(n) {
                        let (o, i, e) = self.one(col, tol, max_iter, other_rhs_bytes)?;
                        ok.push(o);
                        iterations.push(i);
                        errors.push(e);
                    }
                    Ok::<_, String>((out, ok, iterations, errors))
                })
            })
            .map_err(runtime)?;
        let array = Array2::from_shape_vec((n, k).f(), out).map_err(value)?;
        Ok((array.into_pyarray(py), ok, iterations, errors))
    }
    fn update_weights(
        &mut self,
        py: Python<'_>,
        weights: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let raw: Vec<f64> = weights.as_array().iter().copied().collect();
        let pool = Arc::clone(&self.pool);
        let (w, engine) = py
            .detach(|| {
                pool.install(|| {
                    let w = reductions::weights(Some(&raw), self.geometry.n)?;
                    let e = Self::build_engine(
                        &self.geometry,
                        self.index.as_deref(),
                        &w,
                        &self.strategy,
                        self.budget,
                    )?;
                    Ok::<_, String>((w, e))
                })
            })
            .map_err(value)?;
        self.weights = w;
        self.engine = engine;
        self.generation += 1;
        Ok(())
    }
    fn set_threads(&mut self, threads: usize) -> PyResult<()> {
        if threads == 0 {
            return Err(value("threads must be positive"));
        }
        if self.pool.current_num_threads() != threads {
            self.pool = Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(runtime)?,
            );
        }
        Ok(())
    }
    #[getter]
    fn route(&self) -> String {
        self.label().to_owned()
    }
    #[getter]
    fn n_obs(&self) -> usize {
        self.geometry.n
    }
    #[getter]
    fn n_cells(&self) -> usize {
        self.index.as_ref().map_or(self.geometry.n, |i| i.cells())
    }
    #[getter]
    fn weight_generation(&self) -> usize {
        self.generation
    }
    #[getter]
    fn fallback_count(&self) -> usize {
        self.fallbacks
    }
    #[getter]
    fn schur_fallback_count(&self) -> usize {
        self.schur_fallbacks
    }
    #[getter]
    fn extra_resident_bytes(&self) -> usize {
        match &self.engine {
            Engine::Schur(s) => s.resident_bytes(),
            _ => 0,
        }
    }
    #[getter]
    fn schur_condition(&self) -> Option<f64> {
        match &self.engine {
            Engine::Schur(s) => Some(s.condition),
            _ => None,
        }
    }
}
#[pymodule]
fn _econhdfe_within(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeSolver>()?;
    m.add("UPSTREAM_REV", REV)?;
    m.add("ABI_VERSION", 1)?;
    Ok(())
}
