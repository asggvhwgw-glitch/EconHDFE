#![forbid(unsafe_code)]
//! Experimental, opt-in categorical-intercept absorber. No statistical policy.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use fereg_core::{absorber::{Absorber, SolveOpts}, api::Ctx, par::with_pool,
                 session::Term, solve::{FeSolver, SolverConfig}};
use ndarray::{Array2, ShapeBuilder};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{prelude::*, exceptions::{PyRuntimeError, PyValueError}};
use serde_json::{json, Value};
use within::{Design, LsmrOptions, PreconditionerConfig, Solver};

const WITHIN_REV: &str = "fc44282cea6124f5f18bb142908605a656d35a7c";
const FEREG_REV: &str = "0fa59e3b539addbddaee8bd71acd031c497166a0";
type Res<T> = Result<T, String>;
fn value(e: impl ToString) -> PyErr { PyValueError::new_err(e.to_string()) }
fn runtime(e: impl ToString) -> PyErr { PyRuntimeError::new_err(e.to_string()) }

fn weights(raw: Option<&[f64]>, n: usize) -> Res<Vec<f64>> {
    let Some(w) = raw else { return Ok(vec![1.0; n]); };
    if w.len() != n || w.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("weights must have n rows and be finite and strictly positive".into());
    }
    let scale = w.iter().copied().fold(0.0, f64::max);
    let out: Vec<f64> = w.iter().map(|v| v / scale).collect();
    if out.iter().any(|v| *v == 0.0) {
        return Err("weight dynamic range underflows float64; no weights were changed".into());
    }
    Ok(out)
}
fn compact(col: &mut [u32]) -> usize {
    // Linear-time dense case; avoid allocating according to arbitrary caller labels.
    let max = *col.iter().max().unwrap();
    if (max as usize) < col.len().saturating_mul(2) {
        let mut map = vec![u32::MAX; max as usize + 1];
        for &v in col.iter() { map[v as usize] = 0; }
        let mut g = 0;
        for slot in &mut map { if *slot == 0 { *slot = g; g += 1; } }
        for v in col { *v = map[*v as usize]; }
        g as usize
    } else {
        let mut map = HashMap::new();
        for v in col {
            let next = map.len() as u32;
            *v = *map.entry(*v).or_insert(next);
        }
        map.len()
    }
}
fn masses(groups: &[Vec<u32>], levels: &[usize], w: &[f64]) -> Vec<Vec<f64>> {
    groups.iter().zip(levels).map(|(g, &l)| {
        let mut out = vec![0.0; l];
        for (&i, &wi) in g.iter().zip(w) { out[i as usize] += wi; }
        out
    }).collect()
}
fn sampled_duplicates(groups: &[Vec<u32>], n: usize) -> f64 {
    // A hint only. This is NOT an estimate of the population distinct-tuple ratio.
    let s = n.min(8192);
    let mut seen = HashSet::with_capacity(s);
    for t in 0..s {
        let i = t * n / s;
        seen.insert(groups.iter().map(|g| g[i]).collect::<Vec<_>>());
    }
    1.0 - seen.len() as f64 / s as f64
}
fn choose(levels: &[usize], rhs: usize, duplicates: f64) -> (&'static str, &'static str) {
    let mut l = levels.to_vec(); l.sort_unstable_by(|a,b| b.cmp(a));
    if l.len() == 1 { return ("fereg", "one_factor"); }
    if l.iter().skip(1).sum::<usize>() <= 4096 {
        return ("fereg", "small_reduced_dimension");
    }
    // Protect the regime where earlier direct-core tests favored within.
    if l.len() >= 3 && l[2] >= 4096 && l[0] <= 4 * l[2] {
        return ("within", "three_comparable_large_factors");
    }
    if duplicates >= 0.01 { return ("fereg", "sampled_repeated_tuples"); }
    if l.len() == 2 { return ("fereg", "two_way_elimination_candidate"); }
    if l[0] >= 8 * l[1] { return ("fereg", "strong_size_asymmetry"); }
    if rhs >= 4 { return ("fereg", "multi_rhs_candidate"); }
    ("within", "conservative_default")
}

enum Kernel { Within(Solver<'static>), Fereg(Box<FeSolver>) }
struct Projected { residual: Vec<f64>, iterations: Vec<usize>, converged: Vec<bool> }

#[pyclass(module = "econhdfe_dual._native")]
pub struct NativeDualSolver {
    groups: Vec<Vec<u32>>, levels: Vec<usize>, n: usize,
    w: Vec<f64>, mass: Vec<Vec<f64>>, duplicate_hint: f64,
    requested: String, selected: String, reason: String,
    threads: usize, pool: Arc<rayon::ThreadPool>,
    design: Option<Design<'static>>, kernel: Option<Kernel>,
    scratch_bytes: usize, allow_fallback: bool,
    weight_generation: usize, kernel_builds: usize, fallback_count: usize,
    last: Value,
}
impl NativeDualSolver {
    fn activate(&mut self, name: &str) -> Res<()> {
        // Never retain a failed or obsolete numerical kernel.
        self.kernel = None;
        let core = if name == "within" {
            if self.design.is_none() {
                let flat = self.groups.iter().flat_map(|v| v.iter().copied()).collect();
                let a = Array2::from_shape_vec((self.n, self.groups.len()).f(), flat)
                    .map_err(|e| e.to_string())?;
                self.design = Some(Design::from_categories(a.view()).map_err(|e| e.to_string())?.into_owned());
            }
            let design = self.design.as_ref().unwrap();
            let solver = self.pool.install(|| Solver::new(design, Some(&self.w),
                None::<&PreconditionerConfig>)).map_err(|e| e.to_string())?;
            Kernel::Within(solver)
        } else {
            // Do not hold an unused within design while using fereg.
            self.design = None;
            let terms: Vec<Term> = self.groups.iter().enumerate().map(|(j, g)| Term {
                factors: vec![j], codes: g.clone(), g: self.levels[j], intercept: true,
                slopes: vec![], slope_vars: vec![], save: false,
            }).collect();
            let ctx = Ctx::quiet(self.threads);
            let cfg = SolverConfig::default();
            let solver = with_pool(self.threads, |p|
                FeSolver::new(&terms, Some(&self.w), self.n, cfg, p, &ctx))
                .map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
            Kernel::Fereg(Box::new(solver))
        };
        self.kernel = Some(core);
        self.selected = name.into();
        self.kernel_builds += 1;
        Ok(())
    }
    fn project(&mut self, z: &[f64], r: usize, tol: f64, max_iter: usize) -> Res<Projected> {
        let n = self.n;
        match self.kernel.as_mut().ok_or("kernel is not initialized")? {
            Kernel::Within(solver) => {
                let columns: Vec<Vec<f64>> = (0..r).map(|j|
                    (0..n).map(|i| z[i*r+j]).collect()).collect();
                let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
                let opt = LsmrOptions { tol, maxiter: max_iter, local_size: None };
                let out = self.pool.install(|| solver.solve_batch(&refs, &opt))
                    .map_err(|e| e.to_string())?;
                let mut residual = vec![0.0; n*r];
                for j in 0..r { for i in 0..n { residual[i*r+j] = out.demeaned[j*n+i]; } }
                Ok(Projected { residual, iterations: out.iterations, converged: out.converged })
            }
            Kernel::Fereg(solver) => {
                let mut data = vec![0.0; n*r];
                for (i, &orig) in solver.row_order().iter().enumerate() {
                    data[i*r..(i+1)*r].copy_from_slice(&z[orig as usize*r..(orig as usize+1)*r]);
                }
                let opt = SolveOpts { tol, maxiter: max_iter, want_coefs: false };
                let ctx = Ctx::quiet(self.threads);
                let info = with_pool(self.threads, |p| solver.residualize(&mut data, r, &opt, None, p, &ctx))
                    .map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
                let mut residual = vec![0.0; n*r];
                for (i, &orig) in solver.row_order().iter().enumerate() {
                    residual[orig as usize*r..(orig as usize+1)*r].copy_from_slice(&data[i*r..(i+1)*r]);
                }
                Ok(Projected { residual, iterations: vec![info.iterations; r], converged: vec![true; r] })
            }
        }
    }
    fn certificate(&self, z: &[f64], r: usize) -> Vec<f64> {
        if z.iter().any(|v| !v.is_finite()) { return vec![f64::INFINITY; r]; }
        let mut err = vec![0.0_f64; r];
        for ((g, &l), mass) in self.groups.iter().zip(&self.levels).zip(&self.mass) {
            let mut sums = vec![0.0; l*r];
            for i in 0..self.n {
                let start = g[i] as usize*r;
                for j in 0..r { sums[start+j] += self.w[i]*z[i*r+j]; }
            }
            for level in 0..l { for j in 0..r {
                err[j] = err[j].max((sums[level*r+j] / mass[level]).abs());
            } }
        }
        err
    }
    fn certified(&mut self, original: &[f64], r: usize, tol: f64, max_iter: usize,
                 attempts: &mut Vec<Value>) -> Res<Projected> {
        let gate = tol.max(128.0*f64::EPSILON);
        let mut source = original.to_vec();
        let mut total = vec![0usize; r];
        // At most one residual-replacement correction per kernel attempt.
        for correction in 0..2 {
            let t = Instant::now();
            let out = match self.project(&source, r, (tol*0.1).max(1e-15), max_iter) {
                Ok(v) => v,
                Err(e) => {
                    attempts.push(json!({"backend":self.selected, "correction":correction,
                        "seconds":t.elapsed().as_secs_f64(), "error":e}));
                    return Err(e);
                }
            };
            let seconds = t.elapsed().as_secs_f64();
            for (sum, &it) in total.iter_mut().zip(&out.iterations) { *sum += it; }
            let check = Instant::now();
            let cert = self.certificate(&out.residual, r);
            let valid = out.converged.iter().all(|v| *v) && cert.iter().all(|v| v.is_finite() && *v <= gate);
            attempts.push(json!({"backend":self.selected,"correction":correction,
                "seconds":seconds,"certificate_seconds":check.elapsed().as_secs_f64(),
                "iterations":out.iterations,"converged":out.converged,"certificate":cert,"accepted":valid}));
            if valid { return Ok(Projected { residual:out.residual, iterations:total, converged:out.converged }); }
            if !out.converged.iter().all(|v| *v) || out.residual.iter().any(|v| !v.is_finite()) { break; }
            source = out.residual;
        }
        Err("independent weighted group-mean certificate did not pass".into())
    }
    fn run(&mut self, input: Vec<f64>, k: usize, tol: f64, max_iter: usize) -> Res<(Vec<f64>, String)> {
        let clock = Instant::now();
        if input.iter().any(|v| !v.is_finite()) { return Err("RHS must be finite".into()); }
        if k == 0 {
            return Ok((input, json!({"accepted":true,"rhs":0,"backend":"none","attempts":[]}).to_string()));
        }
        let mut scales = vec![0.0_f64; k];
        for row in input.chunks_exact(k) { for j in 0..k { scales[j] = scales[j].max(row[j].abs()); } }
        let (chosen, reason) = if self.requested == "auto" {
            choose(&self.levels, k, self.duplicate_hint)
        } else { (self.requested.as_str(), "explicit_backend") };
        let chosen = chosen.to_string();
        let reason = reason.to_string();
        let mut setup_s = 0.0;
        let mut attempts = vec![];
        if self.kernel.is_none() {
            self.reason = reason;
            let t = Instant::now();
            if let Err(e) = self.activate(&chosen) {
                attempts.push(json!({"backend":chosen,"phase":"build","error":e}));
                if !self.allow_fallback { return Err(e); }
                let alt = if chosen == "within" { "fereg" } else { "within" };
                self.activate(alt)?;
                self.fallback_count += 1;
            }
            setup_s += t.elapsed().as_secs_f64();
        }
        let initial_backend = self.selected.clone();
        // A per-block scratch admission estimate, not a hard process-RSS cap.
        let per_rhs = self.n.saturating_mul(8).saturating_mul(8).max(1);
        let width = (self.scratch_bytes / per_rhs).max(1).min(k);
        let mut result = vec![0.0; input.len()];
        let mut all_iterations = vec![0usize; k];
        let mut all_cert = vec![0.0; k];
        for j0 in (0..k).step_by(width) {
            let r = width.min(k-j0);
            let mut source = vec![0.0; self.n*r];
            for i in 0..self.n { for j in 0..r {
                let scale = scales[j0+j];
                source[i*r+j] = if scale == 0.0 { 0.0 } else { input[i*k+j0+j]/scale };
            } }
            let attempt_start = attempts.len();
            let projected = match self.certified(&source, r, tol, max_iter, &mut attempts) {
                Ok(p) => p,
                Err(e) => {
                    if !self.allow_fallback { self.last = json!({"accepted":false,"attempts":attempts,"error":e}); return Err(e); }
                    let alt = if self.selected == "within" { "fereg" } else { "within" };
                    let t = Instant::now();
                    self.activate(alt)?;
                    setup_s += t.elapsed().as_secs_f64();
                    self.fallback_count += 1;
                    self.certified(&source, r, tol, max_iter, &mut attempts).map_err(|last| {
                        self.last = json!({"accepted":false,"attempts":attempts,"error":last}); last
                    })?
                }
            };
            let cert = self.certificate(&projected.residual, r);
            for a in &mut attempts[attempt_start..] { a["rhs_start"] = json!(j0); a["rhs_width"] = json!(r); }
            for j in 0..r { all_iterations[j0+j] = projected.iterations[j]; all_cert[j0+j] = cert[j]; }
            for i in 0..self.n { for j in 0..r {
                result[i*k+j0+j] = projected.residual[i*r+j]*scales[j0+j];
            } }
        }
        if result.iter().any(|v| !v.is_finite()) { return Err("residual overflow on return to original units".into()); }
        self.last = json!({"accepted":true,"initial_backend":initial_backend,"backend":self.selected,
            "decision_reason":self.reason,"requested_backend":self.requested,"rhs":k,"rhs_block_width":width,
            "n":self.n,"levels":self.levels,"sample_duplicate_fraction":self.duplicate_hint,
            "kernel_builds":self.kernel_builds,"weight_generation":self.weight_generation,
            "fallback_count":self.fallback_count,"setup_seconds":setup_s,"total_seconds":clock.elapsed().as_secs_f64(),
            "iterations":all_iterations,"certificate":all_cert,"certificate_gate":tol.max(128.0*f64::EPSILON),
            "certificate_definition":"max absolute weighted group mean / original column maxabs",
            "max_iter_scope":"per projection attempt; at most two attempts per kernel per RHS block",
            "attempts":attempts,"within_revision":WITHIN_REV,"fereg_revision":FEREG_REV});
        Ok((result, self.last.to_string()))
    }
}
#[pymethods]
impl NativeDualSolver {
    #[new]
    #[pyo3(signature=(codes, weights=None, backend="auto", threads=2, scratch_bytes=536870912, allow_fallback=true))]
    fn new(py: Python<'_>, codes: PyReadonlyArray2<'_, u32>, weights: Option<PyReadonlyArray1<'_, f64>>,
           backend: &str, threads: usize, scratch_bytes: usize, allow_fallback: bool) -> PyResult<Self> {
        if !matches!(backend, "auto" | "within" | "fereg") || threads == 0 || scratch_bytes < 1024 {
            return Err(value("invalid backend, threads or scratch_bytes"));
        }
        let a = codes.as_array();
        let n = a.nrows();
        if n == 0 || n > u32::MAX as usize || a.ncols() == 0 || a.ncols() > 32 {
            return Err(value("require 1..u32::MAX observations and 1..32 categorical FE factors"));
        }
        let mut groups: Vec<Vec<u32>> = a.columns().into_iter().map(|c| c.iter().copied().collect()).collect();
        let raw_w = weights.map(|w| w.as_array().iter().copied().collect::<Vec<_>>());
        let requested = backend.to_string();
        py.detach(move || -> Res<Self> {
            let levels: Vec<usize> = groups.iter_mut().map(|c| compact(c)).collect();
            let w = crate::weights(raw_w.as_deref(), n)?;
            let mass = masses(&groups, &levels, &w);
            let duplicate_hint = sampled_duplicates(&groups, n);
            let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| e.to_string())?);
            Ok(Self { groups, levels, n, w, mass, duplicate_hint, requested,
                selected:"uninitialized".into(), reason:"pending_first_rhs".into(), threads, pool,
                design:None, kernel:None, scratch_bytes, allow_fallback, weight_generation:0,
                kernel_builds:0, fallback_count:0, last:json!({}) })
        }).map_err(value)
    }
    #[pyo3(signature=(data, tol=1e-10, max_iter=6000))]
    fn solve_block<'py>(&mut self, py: Python<'py>, data: PyReadonlyArray2<'py, f64>, tol: f64,
                       max_iter: usize) -> PyResult<(Bound<'py, PyArray2<f64>>, String)> {
        if !tol.is_finite() || tol <= 0.0 || tol > 0.1 || max_iter == 0 {
            return Err(value("tol must be in (0,0.1], max_iter must be positive"));
        }
        let a = data.as_array();
        let (n,k) = (a.nrows(),a.ncols());
        if n != self.n { return Err(value("RHS row count mismatch")); }
        n.checked_mul(k).ok_or_else(|| value("RHS dimensions overflow"))?;
        let input = a.iter().copied().collect::<Vec<_>>();
        let (output, info) = py.detach(|| self.run(input,k,tol,max_iter)).map_err(runtime)?;
        let arr = Array2::from_shape_vec((n,k),output).map_err(value)?;
        Ok((arr.into_pyarray(py),info))
    }
    fn update_weights(&mut self, py: Python<'_>, weights: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let raw: Vec<f64> = weights.as_array().iter().copied().collect();
        py.detach(|| -> Res<()> {
            // Validate before mutation. Invalid input leaves numerical state untouched.
            let new_w = crate::weights(Some(&raw),self.n)?;
            let new_mass = masses(&self.groups,&self.levels,&new_w);
            if let Some(old) = self.kernel.take() {
                let changed: Res<Kernel> = match old {
                    Kernel::Within(_) => {
                        let design = self.design.as_ref().ok_or("missing cached design")?;
                        self.pool.install(|| Solver::new(design,Some(&new_w),None::<&PreconditionerConfig>))
                            .map(Kernel::Within).map_err(|e| e.to_string())
                    }
                    Kernel::Fereg(mut solver) => {
                        let sorted: Vec<f64> = solver.row_order().iter().map(|&i|new_w[i as usize]).collect();
                        let ctx = Ctx::quiet(self.threads);
                        match with_pool(self.threads,|p|solver.set_weights(Some(&sorted),p,&ctx)) {
                            Ok(Ok(())) => Ok(Kernel::Fereg(solver)),
                            Ok(Err(e)) | Err(e) => Err(e.to_string()),
                        }
                    }
                };
                // On an internal update error, keep old public weights and invalidate cache;
                // the next solve reconstructs a kernel for those old weights.
                self.kernel = Some(changed?);
            }
            self.w = new_w; self.mass = new_mass; self.weight_generation += 1;
            Ok(())
        }).map_err(value)
    }
    #[getter]
    fn diagnostics_json(&self) -> String { self.last.to_string() }
    #[getter]
    fn selected_backend(&self) -> String { self.selected.clone() }
    #[getter]
    fn weight_generation(&self) -> usize { self.weight_generation }
}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeDualSolver>()?;
    m.add("WITHIN_REV",WITHIN_REV)?; m.add("FEREG_REV",FEREG_REV)?;
    Ok(())
}
