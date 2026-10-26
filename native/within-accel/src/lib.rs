#![forbid(unsafe_code)]
//! Exact numerical reductions; no sample, rank/DoF, or inference policy lives here.
mod reductions;
#[cfg(feature = "python")]
mod python;
