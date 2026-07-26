//! Exact k-NN by parallel linear scan — the recall reference every index is
//! judged against, and the honest baseline any index must actually beat.

use rayon::prelude::*;

use crate::dataset::FloatVectors;
use crate::distance::l2_sq;
use crate::topk::TopK;

/// Returns, per query, the exact k nearest (distance, id) in ascending order.
pub fn search(base: &FloatVectors, queries: &FloatVectors, k: usize) -> Vec<Vec<(f32, u32)>> {
    assert_eq!(base.dim, queries.dim);
    (0..queries.count)
        .into_par_iter()
        .map(|qi| {
            let q = queries.get(qi);
            let mut topk = TopK::new(k);
            for (i, v) in base.iter().enumerate() {
                topk.push(l2_sq(q, v), i as u32);
            }
            topk.into_sorted()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::FloatVectors;

    #[test]
    fn finds_exact_neighbors() {
        // 4 points on a line; query at 0.9 → nearest are 1.0, 0.0
        let base = FloatVectors::from_raw(1, vec![0.0, 1.0, 2.0, 3.0]);
        let queries = FloatVectors::from_raw(1, vec![0.9]);
        let results = search(&base, &queries, 2);
        let ids: Vec<u32> = results[0].iter().map(|&(_, id)| id).collect();
        assert_eq!(ids, vec![1, 0]);
    }
}
