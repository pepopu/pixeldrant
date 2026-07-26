//! Routing-distance abstraction.
//!
//! The beam search only needs *approximately* correct distances to decide
//! where to walk next; exact distances for the returned top-k come from the
//! full-precision vectors inside every disk block the search already reads
//! (rerank). Exact routing over `FloatVectors` is the M2/M3 baseline;
//! quantized codebooks (`nvvec-index::quant`) shrink routing RAM.

use crate::dataset::FloatVectors;
use crate::distance::l2_sq;

pub trait RouteScorer: Sync {
    /// (Possibly approximate) squared L2 between `query` and point `id`.
    fn score(&self, query: &[f32], id: u32) -> f32;

    /// Resident memory this scorer keeps, in bytes (for the RAM bookkeeping
    /// that this whole project is about).
    fn memory_bytes(&self) -> usize;
}

/// Exact routing: full-precision vectors in RAM.
impl RouteScorer for FloatVectors {
    #[inline]
    fn score(&self, query: &[f32], id: u32) -> f32 {
        l2_sq(query, self.get(id as usize))
    }

    fn memory_bytes(&self) -> usize {
        self.count * self.dim * size_of::<f32>()
    }
}
