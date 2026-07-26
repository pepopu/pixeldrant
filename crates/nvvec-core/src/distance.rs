//! Distance kernels.
//!
//! Written to auto-vectorize under `-C target-cpu=native` (set in
//! `.cargo/config.toml`). Hand-written SIMD only comes later if profiling
//! shows LLVM missing the mark — measure first.

/// Squared Euclidean distance. No sqrt: monotonic, so rankings are unchanged.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    const LANES: usize = 8;
    let split = a.len() - a.len() % LANES;
    // Independent accumulators per lane break the loop-carried dependency,
    // which is what lets LLVM emit packed FMA instructions here.
    let mut acc = [0.0f32; LANES];
    for (ca, cb) in a[..split].chunks_exact(LANES).zip(b[..split].chunks_exact(LANES)) {
        for l in 0..LANES {
            let d = ca[l] - cb[l];
            acc[l] += d * d;
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in a[split..].iter().zip(&b[split..]) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_value() {
        assert_eq!(l2_sq(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
    }

    #[test]
    fn handles_non_multiple_of_lane_width() {
        // dim 11 exercises both the vectorized body and the scalar tail
        let a: Vec<f32> = (0..11).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..11).map(|i| (i + 1) as f32).collect();
        assert_eq!(l2_sq(&a, &b), 11.0);
    }

    #[test]
    fn zero_for_identical() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.5).collect();
        assert_eq!(l2_sq(&a, &a), 0.0);
    }
}
