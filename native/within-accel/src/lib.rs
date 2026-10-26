#![forbid(unsafe_code)]
//! Exact numerical reductions; no sample, rank/DoF, or inference policy lives here.
#[cfg(feature = "python")]
mod python;
mod reductions;
