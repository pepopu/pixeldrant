//! Recall and latency reporting shared by all benchmarks.

use crate::dataset::IdLists;

/// Mean recall@k: |top-k results ∩ top-k ground truth| / k over queries.
/// `results` may cover only the first N queries of the ground truth.
///
/// Note: with L2 ties at the k-th position even an exact scan can score
/// marginally below 1.0 against a fixed ground-truth file; ~0.999+ passes.
pub fn recall_at_k(results: &[Vec<u32>], gt: &IdLists, k: usize) -> f64 {
    assert!(k <= gt.dim, "ground truth only has {} neighbors per query", gt.dim);
    assert!(results.len() <= gt.count);
    assert!(!results.is_empty());
    let mut hits = 0usize;
    for (qi, res) in results.iter().enumerate() {
        let truth = &gt.get(qi)[..k];
        hits += res.iter().take(k).filter(|id| truth.contains(id)).count();
    }
    hits as f64 / (results.len() * k) as f64
}

pub struct LatencyStats {
    pub mean_us: f64,
    pub p50_us: f64,
    pub p90_us: f64,
    pub p99_us: f64,
}

pub fn latency_stats(mut samples_us: Vec<f64>) -> LatencyStats {
    assert!(!samples_us.is_empty());
    samples_us.sort_by(f64::total_cmp);
    let pct = |p: f64| {
        let rank = (samples_us.len() as f64 * p / 100.0).ceil() as usize;
        samples_us[rank.saturating_sub(1).min(samples_us.len() - 1)]
    };
    LatencyStats {
        mean_us: samples_us.iter().sum::<f64>() / samples_us.len() as f64,
        p50_us: pct(50.0),
        p90_us: pct(90.0),
        p99_us: pct(99.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles() {
        let samples: Vec<f64> = (1..=100).map(f64::from).collect();
        let stats = latency_stats(samples);
        assert_eq!(stats.p50_us, 50.0);
        assert_eq!(stats.p99_us, 99.0);
        assert_eq!(stats.mean_us, 50.5);
    }
}
