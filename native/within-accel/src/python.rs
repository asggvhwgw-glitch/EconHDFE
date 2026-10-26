use std::sync::Arc;
use ndarray::{Array2, ShapeBuilder};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::{PyRuntimeError, PyValueError}, prelude::*};
use within::{Design, LsmrOptions, PreconditionerConfig, Solver};
use crate::reductions::{self, Geometry, SmallSchur, TupleIndex};

const REV: &str = "fc44282cea6124f5f18bb142908605a656d35a7c";
fn value(e: impl ToString) -> PyErr { PyValueError::new_err(e.to_string()) }
fn runtime(e: impl ToString) -> PyErr { PyRuntimeError::new_err(e.to_string()) }

fn upstream(groups: &[Vec<u32>], weights: &[f64]) -> reductions::Result<Solver<'static>> {
    let n=weights.len();
    let flat=groups.iter().flat_map(|g|g.iter().copied()).collect();
    let a=Array2::from_shape_vec((n,groups.len()).f(),flat).map_err(|e|e.to_string())?;
    let design=Design::from_categories(a.view()).map_err(|e|e.to_string())?.into_owned();
    Solver::new(design,Some(weights),None::<&PreconditionerConfig>).map_err(|e|e.to_string())
}
enum Engine {
    Schur(SmallSchur),
    Iterative { solver: Solver<'static>, mass: Vec<f64>, compressed: bool },
}
#[pyclass(module="_econhdfe_within")]
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
}
impl NativeSolver {
    fn build_engine(g:&Geometry, ix:Option<&TupleIndex>, w:&[f64], strategy:&str, budget:usize)->reductions::Result<Engine> {
        if matches!(strategy,"auto"|"schur") {
            if let Some(s)=SmallSchur::build(g,w,budget)? {return Ok(Engine::Schur(s));}
            if strategy=="schur" {return Err("Schur rejected: size, resource, rank or conditioning gate".into());}
        }
        let compressed=ix.is_some_and(|ix|strategy=="compressed" || (strategy=="auto" && ix.cells() as f64/g.n as f64<=0.7));
        if compressed {
            let ix=ix.unwrap();let mass=ix.masses(w)?;let groups=ix.groups(g);
            let solver=upstream(&groups,&mass)?;
            Ok(Engine::Iterative{solver,mass,compressed:true})
        } else {Ok(Engine::Iterative{solver:upstream(&g.groups,w)?,mass:vec![],compressed:false})}
    }
    fn label(&self)->&str {match &self.engine {Engine::Schur(_)=>"schur",Engine::Iterative{compressed:true,..}=>"compressed_within",_=>"within"}}
    fn one(&mut self, y:&mut [f64], tol:f64, max_iter:usize)->reductions::Result<(bool,usize,f64)> {
        if y.iter().any(|v|!v.is_finite()) {return Err("RHS must be finite".into());}
        let scale=y.iter().map(|v|v.abs()).fold(0.0_f64,f64::max);
        if scale==0.0 {return Ok((true,0,0.0));}
        for v in y.iter_mut() {*v/=scale;}
        let options=LsmrOptions{tol,maxiter:max_iter,local_size:None};
        let (mut result, mut ok, mut iterations)=match &self.engine {
            Engine::Schur(s)=>{
                let mut r=s.residualize(&self.geometry,&self.weights,y)?;
                // Iterative refinement uses residuals, never a ridge or a truncated eigenspace.
                for _ in 0..2 {
                    if reductions::stationarity(&self.geometry,&self.weights,&r)<=tol {break;}
                    r=s.residualize(&self.geometry,&self.weights,&r)?;
                }
                (r,true,1)
            }
            Engine::Iterative{solver,mass,compressed}=>{
                if *compressed {
                    let ix=self.index.as_ref().unwrap();let means=ix.means(y,&self.weights,mass)?;
                    let r=solver.solve(&means,&options).map_err(|e|e.to_string())?;
                    let mut out=y.to_vec();ix.restore(&mut out,&means,&r.demeaned)?;
                    (out,r.converged,r.iterations)
                } else {
                    let r=solver.solve(y,&options).map_err(|e|e.to_string())?;
                    (r.demeaned,r.converged,r.iterations)
                }
            }
        };
        let mut error=reductions::stationarity(&self.geometry,&self.weights,&result);
        if matches!(&self.engine,Engine::Schur(_)) && (!error.is_finite()||error>tol) {
            let solver=upstream(&self.geometry.groups,&self.weights)?;
            let r=solver.solve(y,&options).map_err(|e|e.to_string())?;
            result=r.demeaned;ok=r.converged;iterations+=r.iterations;self.fallbacks+=1;
            error=reductions::stationarity(&self.geometry,&self.weights,&result);
        }
        for (dst,r) in y.iter_mut().zip(result) {*dst=r*scale;}
        if y.iter().any(|v|!v.is_finite()) {return Err("residual overflow while restoring original units".into());}
        // Stationarity is an additional backward-error gate, not a forward-error claim.
        Ok((ok && error<=tol.max(64.0*f64::EPSILON),iterations,error))
    }
}
#[pymethods]
impl NativeSolver {
    #[new]
    #[pyo3(signature=(codes, weights=None, strategy="auto", threads=1, workspace_bytes=536870912))]
    fn new(py:Python<'_>,codes:PyReadonlyArray2<'_,u32>,weights:Option<PyReadonlyArray1<'_,f64>>,strategy:&str,threads:usize,workspace_bytes:usize)->PyResult<Self> {
        if !matches!(strategy,"auto"|"plain"|"compressed"|"schur") {return Err(value("invalid strategy"));}
        if threads==0 || workspace_bytes==0 {return Err(value("threads and workspace_bytes must be positive"));}
        let a=codes.as_array();
        let groups=a.columns().into_iter().map(|c|c.iter().copied().collect()).collect();
        let raw_w=weights.map(|w|w.as_array().iter().copied().collect::<Vec<_>>());
        let strategy=strategy.to_owned();
        py.detach(move || {
            let g=Geometry::new(groups)?;let w=reductions::weights(raw_w.as_deref(),g.n)?;
            let pool=Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e|e.to_string())?);
            // Try the cheap Schur gate first; do not pay a tuple sort when Schur already applies.
            let direct=if matches!(strategy.as_str(),"auto"|"schur") {SmallSchur::build(&g,&w,workspace_bytes)?} else {None};
            let index=if direct.is_none() && matches!(strategy.as_str(),"auto"|"compressed") {
                match TupleIndex::build(&g,workspace_bytes) {
                    Ok(ix)=>Some(Arc::new(ix)),
                    Err(e) if strategy=="compressed"=>return Err(e),
                    Err(_)=>None,
                }
            } else {None};
            let engine=match direct {Some(s)=>Engine::Schur(s),None=>pool.install(||Self::build_engine(&g,index.as_deref(),&w,&strategy,workspace_bytes))?};
            Ok(Self{geometry:g,index,weights:w,strategy,budget:workspace_bytes,engine,pool,generation:0,fallbacks:0})
        }).map_err(value)
    }
    #[pyo3(signature=(data, tol=1e-10, max_iter=16000))]
    fn solve<'py>(&mut self,py:Python<'py>,data:PyReadonlyArray2<'py,f64>,tol:f64,max_iter:usize)->PyResult<(Bound<'py,PyArray2<f64>>,Vec<bool>,Vec<usize>,Vec<f64>)> {
        if !tol.is_finite() || tol<=0.0 || tol>1.0 || max_iter==0 {return Err(value("invalid tolerance or iteration limit"));}
        let a=data.as_array();let (n,k)=(a.nrows(),a.ncols());
        if n!=self.geometry.n {return Err(value("RHS row count mismatch"));}
        // Single owned F-order result buffer. No per-RHS result collection / concatenation.
        let mut out:Vec<f64>=a.columns().into_iter().flat_map(|c|c.to_vec()).collect();
        let pool=Arc::clone(&self.pool);
        let (out,ok,iterations,errors)=py.detach(move ||pool.install(|| {
            let mut ok=Vec::with_capacity(k);let mut iterations=Vec::with_capacity(k);let mut errors=Vec::with_capacity(k);
            for col in out.chunks_mut(n) {let (o,i,e)=self.one(col,tol,max_iter)?;ok.push(o);iterations.push(i);errors.push(e);}
            Ok::<_,String>((out,ok,iterations,errors))
        })).map_err(runtime)?;
        let array=Array2::from_shape_vec((n,k).f(),out).map_err(value)?;
        Ok((array.into_pyarray(py),ok,iterations,errors))
    }
    fn update_weights(&mut self,py:Python<'_>,weights:PyReadonlyArray1<'_,f64>)->PyResult<()> {
        let raw:Vec<f64>=weights.as_array().iter().copied().collect();let pool=Arc::clone(&self.pool);
        // Build-then-swap: a rejected update leaves the previous working state intact.
        let (w,engine)=py.detach(||pool.install(|| {
            let w=reductions::weights(Some(&raw),self.geometry.n)?;
            let e=Self::build_engine(&self.geometry,self.index.as_deref(),&w,&self.strategy,self.budget)?;
            Ok::<_,String>((w,e))
        })).map_err(value)?;
        self.weights=w;self.engine=engine;self.generation+=1;Ok(())
    }
    fn set_threads(&mut self, threads:usize)->PyResult<()> {
        if threads==0 {return Err(value("threads must be positive"));}
        if self.pool.current_num_threads()!=threads {
            self.pool=Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(runtime)?);
        }
        Ok(())
    }
    #[getter] fn route(&self)->String {self.label().to_owned()}
    #[getter] fn n_obs(&self)->usize {self.geometry.n}
    #[getter] fn n_cells(&self)->usize {self.index.as_ref().map_or(self.geometry.n,|i|i.cells())}
    #[getter] fn weight_generation(&self)->usize {self.generation}
    #[getter] fn fallback_count(&self)->usize {self.fallbacks}
    #[getter] fn schur_condition(&self)->Option<f64> {match &self.engine {Engine::Schur(s)=>Some(s.condition),_=>None}}
}
#[pymodule]
fn _econhdfe_within(m:&Bound<'_,PyModule>)->PyResult<()> {
    m.add_class::<NativeSolver>()?;m.add("UPSTREAM_REV",REV)?;m.add("ABI_VERSION",1)?;Ok(())
}
