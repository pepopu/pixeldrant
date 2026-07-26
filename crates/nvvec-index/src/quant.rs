//! Scalar quantization (SQ8): one byte per dimension for routing.
//!
//! Per-dimension affine codes: `value ≈ min[d] + code * scale[d]`. Distance
//! is asymmetric (f32 query vs decoded code) — no query quantization error.
//! 4x smaller than f32; with block-side exact rerank the quantization error
//! only affects which nodes get explored, never the returned scores.
//! (RaBitQ-style 1–2 bit codes are the follow-up once SQ8 is proven.)

use nvvec_core::dataset::FloatVectors;
use nvvec_core::scorer::RouteScorer;

pub struct Sq8Codebook {
    pub dim: usize,
    pub count: usize,
    min: Vec<f32>,
    /// (max-min)/255 per dimension; 1.0 for constant dimensions.
    scale: Vec<f32>,
    codes: Vec<u8>,
}

impl Sq8Codebook {
    pub fn train(base: &FloatVectors) -> Self {
        let dim = base.dim;
        let mut min = vec![f32::INFINITY; dim];
        let mut max = vec![f32::NEG_INFINITY; dim];
        for v in base.iter() {
            for d in 0..dim {
                min[d] = min[d].min(v[d]);
                max[d] = max[d].max(v[d]);
            }
        }
        let scale: Vec<f32> = min
            .iter()
            .zip(&max)
            .map(|(lo, hi)| if hi > lo { (hi - lo) / 255.0 } else { 1.0 })
            .collect();
        let mut codes = Vec::with_capacity(base.count * dim);
        for v in base.iter() {
            for d in 0..dim {
                codes.push(((v[d] - min[d]) / scale[d]).round().clamp(0.0, 255.0) as u8);
            }
        }
        Self { dim, count: base.count, min, scale, codes }
    }

    #[inline]
    pub fn code(&self, id: u32) -> &[u8] {
        let i = id as usize;
        &self.codes[i * self.dim..(i + 1) * self.dim]
    }

    /// Asymmetric squared L2: f32 query against the decoded code.
    #[inline]
    pub fn dist(&self, q: &[f32], id: u32) -> f32 {
        let code = self.code(id);
        debug_assert_eq!(q.len(), self.dim);
        let mut sum = 0.0f32;
        for d in 0..self.dim {
            let rec = self.min[d] + self.scale[d] * code[d] as f32;
            let diff = q[d] - rec;
            sum += diff * diff;
        }
        sum
    }
}

impl RouteScorer for Sq8Codebook {
    #[inline]
    fn score(&self, query: &[f32], id: u32) -> f32 {
        self.dist(query, id)
    }

    fn memory_bytes(&self) -> usize {
        self.codes.len() + (self.min.len() + self.scale.len()) * size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nvvec_core::distance::l2_sq;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn random_vectors(n: usize, dim: usize, seed: u64) -> FloatVectors {
        let mut state = seed;
        let data: Vec<f32> = (0..n * dim)
            .map(|_| (splitmix64(&mut state) >> 40) as f32 / (1u64 << 24) as f32)
            .collect();
        FloatVectors::from_raw(dim, data)
    }

    #[test]
    fn distance_approximation_is_tight() {
        let base = random_vectors(500, 32, 7);
        let queries = random_vectors(20, 32, 8);
        let cb = Sq8Codebook::train(&base);
        for qi in 0..queries.count {
            let q = queries.get(qi);
            for id in 0..base.count as u32 {
                let exact = l2_sq(q, base.get(id as usize));
                let approx = cb.dist(q, id);
                // uniform [0,1) data, 32 dims: quantization step is 1/255
                // per dim, so absolute error stays far below typical
                // inter-point distances (~O(1))
                assert!(
                    (exact - approx).abs() < 0.05,
                    "q{qi} id{id}: exact {exact} vs approx {approx}"
                );
            }
        }
    }

    #[test]
    fn preserves_nearest_neighbor_ordering_mostly() {
        let base = random_vectors(300, 24, 9);
        let queries = random_vectors(30, 24, 10);
        let cb = Sq8Codebook::train(&base);
        let mut agree = 0;
        for qi in 0..queries.count {
            let q = queries.get(qi);
            let exact_nn = (0..base.count as u32)
                .min_by(|&a, &b| {
                    l2_sq(q, base.get(a as usize)).total_cmp(&l2_sq(q, base.get(b as usize)))
                })
                .unwrap();
            let approx_nn = (0..base.count as u32)
                .min_by(|&a, &b| cb.dist(q, a).total_cmp(&cb.dist(q, b)))
                .unwrap();
            agree += (exact_nn == approx_nn) as usize;
        }
        assert!(agree >= 27, "only {agree}/30 nearest neighbors preserved");
    }

    #[test]
    fn memory_is_quarter_of_f32() {
        let base = random_vectors(1000, 64, 11);
        let cb = Sq8Codebook::train(&base);
        let raw = 1000 * 64 * 4;
        assert!(cb.memory_bytes() < raw / 3, "{} vs raw {raw}", cb.memory_bytes());
    }

    #[test]
    fn constant_dimension_roundtrips() {
        let base = FloatVectors::from_raw(2, vec![5.0, 1.0, 5.0, 2.0, 5.0, 3.0]);
        let cb = Sq8Codebook::train(&base);
        assert_eq!(cb.dist(&[5.0, 1.0], 0), 0.0);
    }
}
