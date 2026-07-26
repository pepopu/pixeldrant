//! Core primitives shared by every index and benchmark: dataset loading,
//! distance kernels, exact search, and evaluation metrics.

pub mod brute;
pub mod dataset;
pub mod distance;
pub mod eval;
pub mod scorer;
pub mod topk;
