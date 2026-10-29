//! Low-level, identical-data fereg-core vs pinned within comparison.
//! Timings include FE setup (but exclude synthetic DGP, source copies and verification).
//! fereg-core accepts row-major R RHS and reorders observations; within accepts R slices.
//! The source marshaling time is reported separately to avoid hiding an API difference.
use std::time::Instant;
use fereg_core::absorber::{Absorber, SolveOpts};
use fereg_core::api::Ctx;
use fereg_core::par::with_pool;
use fereg_core::session::Term;
use fereg_core::solve::{FeSolver, SolverConfig};
use ndarray::{Array2, ShapeBuilder};
use serde_json::{json, Value};

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn randint(i: usize, salt: u64, k: usize) -> u32 {
    (mix((i as u64) ^ salt) % (k.max(2) as u64)) as u32
}
fn unit(i: usize, salt: u64) -> f64 {
    ((mix((i as u64) ^ salt) >> 11) as f64) / ((1u64 << 53) as f64)
}
fn dense_codes(col: &mut [u32]) -> usize {
    let mut lut = vec![u32::MAX; 1 + (*col.iter().max().unwrap_or(&0) as usize)];
    for &v in col.iter() { lut[v as usize] = 0; }
    let mut g = 0;
    for v in lut.iter_mut() {
        if *v == 0 { *v = g; g += 1; }
    }
    for v in col.iter_mut() { *v = lut[*v as usize]; }
    g as usize
}
struct Dataset {
    codes: Vec<Vec<u32>>, levels: Vec<usize>, y: Vec<f64>, w0: Vec<f64>, n: usize, rhs: usize,
}
fn data(case: &str, n: usize, rhs: usize) -> Dataset {
    let mut columns = if case == "weak_links" { vec![vec![0u32; n]; 2] } else { vec![vec![0u32; n]; 3] };
    let communities=32;
    let a_per=(n/12/communities).max(2);
    let b_per=(n/16/communities).max(2);
    let u=(n/8).max(1);
    for i in 0..n {
        let r0=randint(i,0xA38F,1000) as usize;
        let r1=randint(i,0x91F9,1000) as usize;
        let _=(r0,r1);
        match case {
            "one_big" => {
                columns[0][i]=(i%(n/4).max(2)) as u32;
                columns[1][i]=randint(i,0x3ea3,48);
                columns[2][i]=randint(i,0xba16,16);
            }
            "two_big" | "reweight" => {
                columns[0][i]=(i%(n/8).max(2)) as u32;
                columns[1][i]=randint(i,0xaaa4,(n/12).max(2));
                columns[2][i]=randint(i,0xbbb7,16);
            }
            "three_big" => {
                columns[0][i]=(i%(n/10).max(2)) as u32;
                columns[1][i]=randint(i,0xcccc,(n/12).max(2));
                columns[2][i]=randint(i,0xdddd,(n/16).max(2));
            }
            "weak_links" => {
                let community=i%communities;
                columns[0][i]=(community*a_per+(i/communities)%a_per) as u32;
                let bridge=unit(i,0xFA33)<0.0003;
                columns[1][i]=if bridge {
                    randint(i,0x5c2,b_per*communities)
                } else {
                    (community*b_per) as u32+randint(i,0x6a7,b_per)
                };
            }
            "hub" => {
                let b=(n/24).max(2);
                columns[0][i]=(i%(n/10).max(2)) as u32;
                columns[1][i]=if i/b >= 5 {
                    randint(i,0xdead,(b/100).max(2))
                } else {(i%b) as u32};
                columns[2][i]=randint(i,0xbeef,16);
            }
            "repeated_cells" => {
                let k=i/8;
                columns[0][i]=(k%(u/3).max(2)) as u32;
                columns[1][i]=randint(k,0xeba4,(u/6).max(2));
                columns[2][i]=randint(k,0xbeb9,32);
            }
            _ => panic!("unknown case {case}")
        }
    }
    let levels=columns.iter_mut().map(|c|dense_codes(c)).collect::<Vec<_>>();
    let mut y=vec![0.0f64;n*rhs];
    for i in 0..n {
        for r in 0..rhs {
            let x=unit(i,0xF17Eu64 + (r as u64)*977)-0.5;
            let noise=unit(i,0x9DD9 + (r as u64)*71)-0.5;
            let fe:f64=columns.iter().enumerate().map(|(j,c)|{
                (f64::from(c[i])*(0.00117+0.00064*(j as f64)) + (r as f64)*0.13).sin()*0.4/(1+j) as f64
            }).sum();
            y[i*rhs+r]=0.65*x+0.7*noise+fe;
        }
    }
    let w0:Vec<f64>=(0..n).map(|i| {
        if case=="reweight" { (0.65*(unit(i,0xABC4)-0.5)).exp() }else{1.0}
    }).collect();
    Dataset{codes:columns,levels,y,w0,n,rhs}
}
fn weights(d:&Dataset, round:usize)->Vec<f64>{
    if round==0{return d.w0.clone();}
    (0..d.n).map(|i| d.w0[i]*(0.18*((i as f64)*(0.011+0.002*round as f64)).sin()).exp()).collect()
}
fn moments(d:&Dataset, residual:&[f64], w:&[f64])->(f64,f64){
    let (n,r)=(d.n,d.rhs);
    let sum_w=w.iter().sum::<f64>();
    let mut worst=0.0_f64;
    let mut ratio=0.0_f64;
    for c in 0..r{
        let norm=((0..n).map(|i|w[i]*d.y[i*r+c].powi(2)).sum::<f64>()/sum_w).sqrt().max(1e-30);
        let resnorm=((0..n).map(|i|w[i]*residual[i*r+c].powi(2)).sum::<f64>()/sum_w).sqrt()/norm;
        ratio=ratio.max(resnorm);
        for (j,codes) in d.codes.iter().enumerate(){
            let mut mass=vec![0.0;d.levels[j]];
            let mut num=vec![0.0;d.levels[j]];
            for i in 0..n{
                let k=codes[i] as usize;
                mass[k]+=w[i];
                num[k]+=w[i]*residual[i*r+c];
            }
            let val=mass.iter().zip(num.iter()).map(|(m,s)|if *m>0.0{s*s/m}else{0.0}).sum::<f64>();
            worst=worst.max((val/sum_w).sqrt()/norm);
        }
    }
    (worst,ratio)
}
fn mem_hwm() -> f64{
    let Ok(s)=std::fs::read_to_string("/proc/self/status") else{return 0.0};
    s.lines().find(|l|l.starts_with("VmHWM:")).and_then(|l|l.split_whitespace().nth(1))
       .and_then(|x|x.parse::<f64>().ok()).unwrap_or(0.0)/1024.0
}
fn run_fereg(d:&Dataset,repeat:usize,weighted:bool,threads:usize)->Result<Value,Box<dyn std::error::Error>> {
    let ctx=Ctx::quiet(threads);
    let terms=d.codes.iter().enumerate().map(|(i,c)|Term{
        factors:vec![i],codes:c.clone(),g:d.levels[i],intercept:true,
        slopes:Vec::new(),slope_vars:Vec::new(),save:false
    }).collect::<Vec<_>>();
    let initial=if weighted { Some(d.w0.as_slice()) }else{None};
    let t_pool=Instant::now();
    let result=with_pool(threads,|pool|->Result<Value,fereg_core::error::RfError>{
        let pool_seconds=t_pool.elapsed().as_secs_f64();
        let t0=Instant::now();
        let mut solver=FeSolver::new(&terms,initial,d.n,SolverConfig::default(),pool,&ctx)?;
        let setup=t0.elapsed().as_secs_f64();
        let t0=Instant::now();
        let order=solver.row_order().to_vec();
        let mut source=vec![0.0f64;d.n*d.rhs];
        for (row,&original) in order.iter().enumerate(){
            let i=original as usize;
            source[row*d.rhs..(row+1)*d.rhs].copy_from_slice(&d.y[i*d.rhs..(i+1)*d.rhs]);
        }
        let marshaling=t0.elapsed().as_secs_f64();
        let opt=SolveOpts {tol:1e-10,maxiter:6000,want_coefs:false};
        let mut rounds=Vec::new();
        for round in 0..repeat{
            let w=if weighted {weights(d,round)}else{vec![1.0f64;d.n]};
            let t0=Instant::now();
            if round>0 && weighted {
                let wr=order.iter().map(|&i|w[i as usize]).collect::<Vec<_>>();
                solver.set_weights(Some(&wr),pool,&ctx)?;
            }
            let update=t0.elapsed().as_secs_f64();
            let mut z=source.clone();
            let t1=Instant::now();
            let info=solver.residualize(&mut z,d.rhs,&opt,None,pool,&ctx)?;
            let elapsed=t1.elapsed().as_secs_f64();
            let mut residual=vec![0.0f64;d.n*d.rhs];
            for (row,&orig) in order.iter().enumerate() {
                let orig=orig as usize;
                residual[orig*d.rhs..(orig+1)*d.rhs].copy_from_slice(&z[row*d.rhs..(row+1)*d.rhs]);
            }
            let (orthog,resid_norm)=moments(d,&residual,&w);
            rounds.push(json!({
                "round":round,"weight_update_s":update,"solve_s":elapsed,
                "iter":info.iterations,"exact":info.exact,"escalated":info.escalated,
                "moment":orthog,"resid_norm":resid_norm,"valid":orthog<=5e-8,
                "precond":solver.precond_kinds()
            }));
        }
        Ok(json!({"setup_s":setup,"marshal_s":marshaling,"pool_s":pool_seconds,
            "rounds":rounds,"peak_rss_mb":mem_hwm()}))
    })??;
    Ok(result)
}
fn run_within(d:&Dataset,repeat:usize,weighted:bool,threads:usize)->Result<Value,Box<dyn std::error::Error>>{
    let mut flat=Vec::with_capacity(d.n*d.codes.len());
    for c in &d.codes {flat.extend_from_slice(c);}
    let a=Array2::from_shape_vec((d.n,d.codes.len()).f(),flat)?;
    let ys=(0..d.rhs).map(|c|(0..d.n).map(|i|d.y[i*d.rhs+c]).collect::<Vec<_>>()).collect::<Vec<_>>();
    let yrefs=ys.iter().map(|v|v.as_slice()).collect::<Vec<_>>();
    let opts=within::LsmrOptions {tol:1e-10,maxiter:6000,..Default::default()};
    let pool=rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
    let result=pool.install(||->Result<Value,String>{
        let t0=Instant::now();
        let design=within::Design::from_categories(a.view()).map_err(|e|e.to_string())?;
        let setup=t0.elapsed().as_secs_f64();
        let mut rounds=Vec::new();
        let mut resident:Option<within::Solver<'_>>=None;
        for round in 0..repeat{
            let w=if weighted {weights(d,round)}else{vec![1.0;d.n]};
            let t0=Instant::now();
            if round==0 || weighted {
                resident=Some(within::Solver::new(&design,if weighted{Some(w.as_slice())}else{None},None::<&within::PreconditionerConfig>).map_err(|e|e.to_string())?);
            }
            let update=t0.elapsed().as_secs_f64();
            let t1=Instant::now();
            let out=resident.as_ref().unwrap().solve_batch(&yrefs,Some(&opts)).map_err(|e|e.to_string())?;
            let elapsed=t1.elapsed().as_secs_f64();
            let mut resid=vec![0.0;d.n*d.rhs];
            for c in 0..d.rhs{
                for i in 0..d.n{resid[i*d.rhs+c]=out.demeaned[c*d.n+i];}
            }
            let (orthog,resid_norm)=moments(d,&resid,&w);
            rounds.push(json!({
                "round":round,"weight_update_s":update,"solve_s":elapsed,
                "iters":out.iterations,"converged":out.converged,
                "moment":orthog,"resid_norm":resid_norm,
                "valid":orthog<=5e-8 && out.converged.iter().all(|v|*v),
                "time_setup":out.time_setup,"time_solve":out.time_solve
            }));
        }
        Ok(json!({"setup_s":setup,"marshal_s":0.0,"pool_s":0.0,
            "rounds":rounds,"peak_rss_mb":mem_hwm()}))
    }).map_err(std::io::Error::other)?;
    Ok(result)
}
fn main()->Result<(),Box<dyn std::error::Error>>{
    let args=std::env::args().collect::<Vec<_>>();
    if args.len()!=7 {
        eprintln!("Usage: kernel-pair <case> <fereg|within> <n> <rhs> <reps> <threads>");
        std::process::exit(1)
    }
    let case=&args[1];
    let method=&args[2];
    let n=args[3].parse::<usize>()?;
    let rhs=args[4].parse::<usize>()?;
    let reps=args[5].parse::<usize>()?;
    let threads=args[6].parse::<usize>()?;
    assert!(n>=1000 && rhs>0 && rhs<=16 && reps>0 && threads>0);
    let t=Instant::now();
    let d=data(case,n,rhs);
    let dgp=t.elapsed().as_secs_f64();
    let weighted=case=="reweight";
    let t=Instant::now();
    let result=if method=="fereg"{
        run_fereg(&d,reps,weighted,threads)?
    }else if method=="within"{
        run_within(&d,reps,weighted,threads)?
    }else{panic!("unknown method {method}")};
    let wall=t.elapsed().as_secs_f64();
    let valid=result["rounds"].as_array().unwrap().iter().all(|v|v["valid"]==true);
    let record=json!({
       "case":case,"method":method,"n":n,"rhs":rhs,"reps":reps,"threads":threads,
       "dimensions":d.levels,"weighted":weighted,"dgp_s":dgp,"bench_wall_s":wall,
       "result":result,"valid":valid,
       "fereg_rev":"0fa59e3b539addbddaee8bd71acd031c497166a0",
       "within_rev":"fc44282cea6124f5f18bb142908605a656d35a7c"
    });
    println!("BENCH_KERNEL {}",record);
    if !valid {std::process::exit(2)}
    Ok(())
}
