use std::sync::Arc;

pub type Result<T> = std::result::Result<T, String>;

#[derive(Debug, Clone)]
pub struct Geometry {
    pub groups: Arc<Vec<Vec<u32>>>,
    pub levels: Vec<usize>,
    pub n: usize,
}
impl Geometry {
    pub fn new(groups: Vec<Vec<u32>>) -> Result<Self> {
        let n = groups.first().map_or(0, Vec::len);
        if n == 0 || n > u32::MAX as usize || groups.iter().any(|g| g.len() != n) {
            return Err("expected nonempty equal-length groups with at most u32::MAX rows".into());
        }
        let mut levels = Vec::new();
        for g in &groups {
            let l = *g.iter().max().unwrap() as usize + 1;
            if l > n {
                return Err("codes must be dense, zero-based u32".into());
            }
            let mut seen = vec![false; l];
            for &v in g {
                seen[v as usize] = true;
            }
            if seen.iter().any(|&v| !v) {
                return Err("codes must be dense, zero-based u32".into());
            }
            levels.push(l);
        }
        Ok(Self {
            groups: Arc::new(groups),
            levels,
            n,
        })
    }
}

pub fn weights(w: Option<&[f64]>, n: usize) -> Result<Vec<f64>> {
    if let Some(w) = w {
        if w.len() != n || w.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err("weights must be finite, strictly positive and match the sample".into());
        }
        let max = w.iter().copied().fold(0.0_f64, f64::max);
        let out: Vec<_> = w.iter().map(|v| v / max).collect();
        if out.contains(&0.0) {
            return Err("weight scaling underflow; no row was dropped".into());
        }
        Ok(out)
    } else {
        Ok(vec![1.0; n])
    }
}
fn add(sum: &mut f64, correction: &mut f64, x: f64) {
    let t = x - *correction;
    let next = *sum + t;
    *correction = (next - *sum) - t;
    *sum = next;
}
fn finite(x: &[f64]) -> Result<()> {
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err("nonfinite numerical result".into())
    }
}

/// Conservative primary + fallback buffers for one RHS. Keep both copies of
/// the adapter's resident + one-RHS estimate: the primary engine stays live.
/// Admission is not a peak-RSS bound; upstream allocations remain opaque.
pub fn fallback_workspace(g: &Geometry) -> Option<usize> {
    let levels = g.levels.iter().try_fold(0usize, |n, &l| n.checked_add(l))?;
    let per_row = g.groups.len().checked_mul(8)?.checked_add(88)?;
    g.n.checked_mul(per_row)?
        .checked_add(levels.checked_mul(128)?)?
        .checked_mul(2)
}

/// Guard the actual fallback allocation, including buffers retained by the
/// primary solver and the bridge. Overflow and an exhausted budget fail closed.
pub fn admitted_fallback<T>(
    g: &Geometry,
    retained_bytes: usize,
    other_rhs_bytes: usize,
    budget: usize,
    remaining_iter: usize,
    fallback: impl FnOnce(usize) -> Result<T>,
) -> Result<Option<T>> {
    let required = fallback_workspace(g)
        .and_then(|n| n.checked_add(retained_bytes))
        .and_then(|n| n.checked_add(other_rhs_bytes));
    if remaining_iter == 0 || required.is_none_or(|n| n > budget) {
        Ok(None)
    } else {
        fallback(remaining_iter).map(Some)
    }
}

#[derive(Debug, Clone)]
pub struct TupleIndex {
    pub map: Vec<u32>,
    pub representatives: Vec<u32>,
}
impl TupleIndex {
    /// Stable LSD radix passes, O(N K log_256(max label)); equality is exact.
    /// No mixed-radix overflow, hashing, or Python tuple allocation.
    pub fn build(g: &Geometry, budget: usize) -> Result<Self> {
        let required = g.n.checked_mul(16).ok_or("compression size overflow")?;
        if required > budget {
            return Err("compression workspace budget exceeded".into());
        }
        let mut order: Vec<u32> = (0..g.n as u32).collect();
        let mut scratch = vec![0u32; g.n];
        for (col, &l) in g.groups.iter().zip(&g.levels).rev() {
            let mut shift = 0;
            while shift < 32 && ((l - 1) as u64 >> shift) != 0 {
                let mut counts = [0usize; 256];
                for &i in &order {
                    counts[((col[i as usize] >> shift) & 255) as usize] += 1;
                }
                let mut prefix = 0;
                for count in &mut counts {
                    let old = *count;
                    *count = prefix;
                    prefix += old;
                }
                for &i in &order {
                    let bin = ((col[i as usize] >> shift) & 255) as usize;
                    scratch[counts[bin]] = i;
                    counts[bin] += 1;
                }
                std::mem::swap(&mut order, &mut scratch);
                shift += 8;
            }
        }
        drop(scratch);
        let mut map = vec![0u32; g.n];
        let mut representatives = Vec::new();
        let mut previous: Option<u32> = None;
        for i in order {
            if previous.is_none_or(|p| g.groups.iter().any(|c| c[p as usize] != c[i as usize])) {
                representatives.push(i);
            }
            map[i as usize] = (representatives.len() - 1) as u32;
            previous = Some(i);
        }
        Ok(Self {
            map,
            representatives,
        })
    }
    pub fn cells(&self) -> usize {
        self.representatives.len()
    }
    pub fn groups(&self, g: &Geometry) -> Vec<Vec<u32>> {
        g.groups
            .iter()
            .map(|c| {
                self.representatives
                    .iter()
                    .map(|&i| c[i as usize])
                    .collect()
            })
            .collect()
    }
    pub fn masses(&self, w: &[f64]) -> Result<Vec<f64>> {
        if self.map.len() != w.len() {
            return Err("weight shape mismatch".into());
        }
        let mut sums = vec![0.0; self.cells()];
        let mut corrections = sums.clone();
        for (i, &v) in w.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                return Err("invalid weight".into());
            }
            let c = self.map[i] as usize;
            add(&mut sums[c], &mut corrections[c], v);
        }
        finite(&sums)?;
        Ok(sums)
    }
    pub fn means(&self, y: &[f64], w: &[f64], mass: &[f64]) -> Result<Vec<f64>> {
        if y.len() != self.map.len() || y.len() != w.len() || mass.len() != self.cells() {
            return Err("aggregation shape mismatch".into());
        }
        finite(y)?;
        let mut sums = vec![0.0; self.cells()];
        let mut corrections = sums.clone();
        for i in 0..y.len() {
            let c = self.map[i] as usize;
            add(&mut sums[c], &mut corrections[c], w[i] * y[i]);
        }
        for (s, &m) in sums.iter_mut().zip(mass) {
            *s /= m;
        }
        finite(&sums)?;
        Ok(sums)
    }
    pub fn restore(&self, y: &mut [f64], means: &[f64], residual: &[f64]) -> Result<()> {
        if y.len() != self.map.len()
            || means.len() != self.cells()
            || residual.len() != self.cells()
        {
            return Err("restoration shape mismatch".into());
        }
        for (i, value) in y.iter_mut().enumerate() {
            let c = self.map[i] as usize;
            *value -= means[c] - residual[c];
        }
        finite(y)
    }
}

/// Dense, scaled, small Schur complement. Refuses extra nullity / bad conditioning.
/// Larger systems always retain the pinned within operator and preconditioner.
pub struct SmallSchur {
    pub big: usize,
    pub q: usize,
    pub condition: f64,
    offsets: Vec<Option<usize>>,
    mass: Vec<f64>,
    cross: Vec<f64>,
    scale: Vec<f64>,
    chol: Vec<f64>,
}
impl SmallSchur {
    pub fn resident_bytes(&self) -> usize {
        let entries = [
            self.mass.capacity(),
            self.cross.capacity(),
            self.scale.capacity(),
            self.chol.capacity(),
        ]
        .into_iter()
        .fold(0usize, usize::saturating_add);
        std::mem::size_of::<Self>()
            .saturating_add(
                self.offsets
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Option<usize>>()),
            )
            .saturating_add(entries.saturating_mul(std::mem::size_of::<f64>()))
    }
    pub fn build(g: &Geometry, w: &[f64], budget: usize) -> Result<Option<Self>> {
        let big = g
            .levels
            .iter()
            .enumerate()
            .max_by_key(|(_, l)| *l)
            .unwrap()
            .0;
        let l = g.levels[big];
        let mut offsets = vec![None; g.groups.len()];
        let mut q = 0usize;
        for (t, &levels) in g.levels.iter().enumerate() {
            if t != big {
                offsets[t] = Some(q);
                q = q.checked_add(levels - 1).ok_or("Schur width overflow")?;
            }
        }
        if q > 128 {
            return Ok(None);
        }
        let entries = l
            .checked_mul(q)
            .and_then(|v| v.checked_add(6 * q * q + 4 * l + 8 * q))
            .and_then(|v| v.checked_mul(8))
            .ok_or("Schur size overflow")?;
        if entries > budget {
            return Ok(None);
        }
        let mut mass = vec![0.0; l];
        let mut mc = vec![0.0; l];
        let mut cross = vec![0.0; l * q];
        let mut b = vec![0.0; q * q];
        let mut row = Vec::with_capacity(g.groups.len());
        for i in 0..g.n {
            let a = g.groups[big][i] as usize;
            add(&mut mass[a], &mut mc[a], w[i]);
            row.clear();
            for (t, off) in offsets.iter().enumerate() {
                if let Some(off) = off {
                    let c = g.groups[t][i] as usize;
                    if c > 0 {
                        row.push(off + c - 1);
                    }
                }
            }
            for &j in &row {
                cross[a * q + j] += w[i];
                for &k in &row {
                    b[j * q + k] += w[i];
                }
            }
        }
        if mass.iter().any(|v| *v <= 0.0 || !v.is_finite()) {
            return Ok(None);
        }
        let raw_diag: Vec<f64> = (0..q).map(|j| b[j * q + j]).collect();
        let mut error = vec![0.0; q * q];
        let mut nz = Vec::with_capacity(q);
        for a in 0..l {
            nz.clear();
            nz.extend((0..q).filter(|&j| cross[a * q + j] != 0.0));
            for (pos, &j) in nz.iter().enumerate() {
                for &k in &nz[..=pos] {
                    add(
                        &mut b[j * q + k],
                        &mut error[j * q + k],
                        -cross[a * q + j] * cross[a * q + k] / mass[a],
                    );
                }
            }
        }
        let mut scale = vec![0.0; q];
        for j in 0..q {
            if b[j * q + j] <= 1e-10 * raw_diag[j] || !b[j * q + j].is_finite() {
                return Ok(None);
            }
            scale[j] = b[j * q + j].sqrt();
        }
        for j in 0..q {
            for k in 0..=j {
                let value = b[j * q + k] / scale[j] / scale[k];
                b[j * q + k] = value;
                b[k * q + j] = value;
            }
        }
        let matrix_norm = (0..q)
            .map(|j| b[j * q..(j + 1) * q].iter().map(|v| v.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let mut chol = b;
        for j in 0..q {
            for k in 0..=j {
                let mut v = chol[j * q + k];
                for h in 0..k {
                    v -= chol[j * q + h] * chol[k * q + h];
                }
                if j == k {
                    if !v.is_finite() || v <= 1e-12 {
                        return Ok(None);
                    }
                    chol[j * q + k] = v.sqrt();
                } else {
                    chol[j * q + k] = v / chol[k * q + k];
                }
            }
        }
        let mut inverse_rows = vec![0.0; q];
        for j in 0..q {
            let mut v = vec![0.0; q];
            v[j] = 1.0;
            cholesky_solve(&chol, &mut v);
            for k in 0..q {
                inverse_rows[k] += v[k].abs();
            }
        }
        let condition = if q == 0 {
            1.0
        } else {
            matrix_norm * inverse_rows.into_iter().fold(0.0_f64, f64::max)
        };
        if !condition.is_finite() || condition > 1e8 {
            return Ok(None);
        }
        Ok(Some(Self {
            big,
            q,
            condition,
            offsets,
            mass,
            cross,
            scale,
            chol,
        }))
    }
    pub fn residualize(&self, g: &Geometry, w: &[f64], y: &[f64]) -> Result<Vec<f64>> {
        if y.len() != g.n || w.len() != g.n {
            return Err("Schur shape mismatch".into());
        }
        let q = self.q;
        let l = self.mass.len();
        let mut ba = vec![0.0; l];
        let mut bc = vec![0.0; l];
        let mut rhs = vec![0.0; q];
        let mut rc = vec![0.0; q];
        for i in 0..g.n {
            let a = g.groups[self.big][i] as usize;
            let wy = w[i] * y[i];
            add(&mut ba[a], &mut bc[a], wy);
            for (t, off) in self.offsets.iter().enumerate() {
                if let Some(off) = off {
                    let c = g.groups[t][i] as usize;
                    if c > 0 {
                        let j = off + c - 1;
                        add(&mut rhs[j], &mut rc[j], wy);
                    }
                }
            }
        }
        for a in 0..l {
            for j in 0..q {
                add(
                    &mut rhs[j],
                    &mut rc[j],
                    -self.cross[a * q + j] * (ba[a] / self.mass[a]),
                );
            }
        }
        for j in 0..q {
            rhs[j] /= self.scale[j];
        }
        cholesky_solve(&self.chol, &mut rhs);
        for j in 0..q {
            rhs[j] /= self.scale[j];
        }
        for a in 0..l {
            let v = (0..q).map(|j| self.cross[a * q + j] * rhs[j]).sum::<f64>();
            ba[a] = (ba[a] - v) / self.mass[a];
        }
        let mut out = y.to_vec();
        for i in 0..g.n {
            let mut fitted = ba[g.groups[self.big][i] as usize];
            for (t, off) in self.offsets.iter().enumerate() {
                if let Some(off) = off {
                    let c = g.groups[t][i] as usize;
                    if c > 0 {
                        fitted += rhs[off + c - 1];
                    }
                }
            }
            out[i] -= fitted;
        }
        finite(&out)?;
        Ok(out)
    }
}
fn cholesky_solve(chol: &[f64], x: &mut [f64]) {
    let q = x.len();
    for j in 0..q {
        for k in 0..j {
            x[j] -= chol[j * q + k] * x[k];
        }
        x[j] /= chol[j * q + j];
    }
    for j in (0..q).rev() {
        for k in j + 1..q {
            x[j] -= chol[k * q + j] * x[k];
        }
        x[j] /= chol[j * q + j];
    }
}

/// One original-row weighted cyclic projection, with compensated group sums.
/// Each subtraction lies in the original FE span. This is a bounded polishing
/// step, not a convergence claim; the caller must recertify the result.
/// Scratch is 8*N + 32*max(levels), within the existing one-RHS P estimate.
pub fn cyclic_polish(g: &Geometry, w: &[f64], r: &[f64]) -> Result<Vec<f64>> {
    if w.len() != g.n || r.len() != g.n {
        return Err("polishing shape mismatch".into());
    }
    let mut out = r.to_vec();
    for (col, &l) in g.groups.iter().zip(&g.levels) {
        let mut sum = vec![0.0; l];
        let mut sc = vec![0.0; l];
        let mut mass = vec![0.0; l];
        let mut mc = vec![0.0; l];
        for i in 0..g.n {
            let a = col[i] as usize;
            add(&mut sum[a], &mut sc[a], w[i] * out[i]);
            add(&mut mass[a], &mut mc[a], w[i]);
        }
        for a in 0..l {
            sum[a] /= mass[a];
        }
        finite(&sum)?;
        for i in 0..g.n {
            out[i] -= sum[col[i] as usize];
        }
    }
    finite(&out)?;
    Ok(out)
}

/// Dimensionless group-mean stationarity check on already scaled RHS.
/// This is not a condition-independent forward-error certificate.
pub fn stationarity(g: &Geometry, w: &[f64], r: &[f64]) -> f64 {
    let mut worst = 0.0_f64;
    for (col, &l) in g.groups.iter().zip(&g.levels) {
        let mut sum = vec![0.0; l];
        let mut mass = vec![0.0; l];
        for i in 0..g.n {
            let a = col[i] as usize;
            sum[a] += w[i] * r[i];
            mass[a] += w[i];
        }
        for a in 0..l {
            let mean = (sum[a] / mass[a]).abs();
            // f64::max ignores NaN; an invalid certificate must fail closed.
            if !mean.is_finite() {
                return f64::INFINITY;
            }
            worst = worst.max(mean);
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn radix_repeated_and_nonadjacent() {
        let g = Geometry::new(vec![vec![1, 0, 1, 0, 1, 0], vec![0, 1, 0, 1, 2, 1]]).unwrap();
        let ix = TupleIndex::build(&g, 1024).unwrap();
        assert_eq!(ix.cells(), 3);
        assert_eq!(ix.map[0], ix.map[2]);
        assert_eq!(ix.map[1], ix.map[5]);
        let w = weights(Some(&[1., 2., 3., 4., 5., 6.]), 6).unwrap();
        let y = [2., 5., 8., 7., 11., 9.];
        let mass = ix.masses(&w).unwrap();
        let means = ix.means(&y, &w, &mass).unwrap();
        let fit = [1.5, -0.25, 2.25];
        let residual: Vec<_> = means.iter().zip(fit).map(|(m, f)| m - f).collect();
        let mut out = y.to_vec();
        ix.restore(&mut out, &means, &residual).unwrap();
        for i in 0..6 {
            assert!((out[i] - y[i] + fit[ix.map[i] as usize]).abs() < 1e-12);
        }
    }
    #[test]
    fn radix_multiple_bytes() {
        let a: Vec<_> = (0..65537u32).rev().collect();
        let g = Geometry::new(vec![a]).unwrap();
        let ix = TupleIndex::build(&g, 2_000_000).unwrap();
        assert_eq!(ix.cells(), 65537);
        assert_eq!(ix.map[0], 65536);
        assert_eq!(ix.map[65536], 0);
    }
    #[test]
    fn invalid_and_budget() {
        assert!(Geometry::new(vec![vec![0, 2]]).is_err());
        assert!(weights(Some(&[1., 0.]), 2).is_err());
        let g = Geometry::new(vec![vec![0, 1]]).unwrap();
        assert!(TupleIndex::build(&g, 1).is_err());
    }
    #[test]
    fn nonfinite_certificate_is_rejected() {
        let g = Geometry::new(vec![vec![0, 0, 1, 1], vec![0, 1, 0, 1]]).unwrap();
        let w = [1.; 4];
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(!stationarity(&g, &w, &[bad, 1., -1., 0.]).is_finite());
        }
    }
    #[test]
    fn fallback_admission_enforces_retained_and_bridge_buffers() {
        let g = Geometry::new(
            [200, 70, 25]
                .into_iter()
                .map(|l| (0..4000).map(|i| (i % l) as u32).collect())
                .collect(),
        )
        .unwrap();
        assert_eq!(fallback_workspace(&g), Some(971_520));
        // Exact known-buffer boundary admits one construction and carries the
        // remaining iteration allowance to that construction/solve pair.
        let mut calls = 0;
        let result = admitted_fallback(&g, 0, 0, 971_520, 17, |remaining| {
            calls += 1;
            Ok(remaining)
        })
        .unwrap();
        assert_eq!(result, Some(17));
        assert_eq!(calls, 1);
        for (retained, other, budget) in [
            (0, 0, 971_519),
            (0, 32_000, 980_000),
            (32_000, 0, 980_000),
            (usize::MAX, 0, usize::MAX),
            (0, usize::MAX, usize::MAX),
        ] {
            let refused: Option<()> = admitted_fallback(&g, retained, other, budget, 17, |_| {
                panic!("fallback allocated beyond admitted known buffers")
            })
            .unwrap();
            assert!(refused.is_none());
        }
    }
    #[test]
    fn fallback_admission_respects_iterations_and_propagates_errors() {
        let g = Geometry::new(vec![vec![0, 1]]).unwrap();
        let refused: Option<()> = admitted_fallback(&g, 0, 0, usize::MAX, 0, |_| {
            panic!("fallback allocated with no remaining iterations")
        })
        .unwrap();
        assert!(refused.is_none());
        let failed: Result<Option<()>> = admitted_fallback(&g, 0, 0, usize::MAX, 1, |_| {
            Err("construction failed".into())
        });
        assert_eq!(failed, Err("construction failed".into()));
    }
    #[test]
    fn schur_known_residual_weighted() {
        let mut gs = vec![vec![], vec![], vec![]];
        let mut w = vec![];
        let mut y = vec![];
        let mut truth = vec![];
        for a in 0..8 {
            for b in 0..3 {
                for c in 0..2 {
                    for rep in 0..2 {
                        gs[0].push(a);
                        gs[1].push(b);
                        gs[2].push(c);
                        let wi = (1 + a + b + c) as f64;
                        let r = if rep == 0 { 1.0 / wi } else { -1.0 / wi };
                        w.push(wi);
                        y.push(a as f64 + 0.2 * b as f64 - 0.7 * c as f64 + r);
                        truth.push(r);
                    }
                }
            }
        }
        let g = Geometry::new(gs).unwrap();
        let w = weights(Some(&w), g.n).unwrap();
        let s = SmallSchur::build(&g, &w, 100000).unwrap().unwrap();
        let r = s.residualize(&g, &w, &y).unwrap();
        assert!(r.iter().zip(&truth).all(|(a, b)| (a - b).abs() < 1e-11));
        assert!(stationarity(&g, &w, &r) < 1e-11);
    }
    #[test]
    fn schur_refuses_additional_nullity() {
        let g = Geometry::new(vec![vec![0, 0, 1, 1], vec![0, 0, 1, 1]]).unwrap();
        assert!(SmallSchur::build(&g, &vec![1.; 4], 10000)
            .unwrap()
            .is_none());
    }
}
