//! Vamana construction (DiskANN §2): two passes of insert-and-prune with
//! fine-grained per-node locks; deadlock-free because at most one adjacency
//! lock is held at a time.

use nvvec_core::dataset::FloatVectors;
use nvvec_core::distance::l2_sq;
use parking_lot::RwLock;
use rayon::prelude::*;

use crate::graph::VamanaGraph;
use crate::search::{SearchScratch, beam_search};

pub struct BuildParams {
    /// Max out-degree.
    pub r: usize,
    /// Candidate list size during construction (the paper's L).
    pub l_build: usize,
    /// Pruning slack; > 1.0 keeps longer "highway" edges for navigability.
    pub alpha: f32,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self { r: 32, l_build: 100, alpha: 1.2 }
    }
}

pub fn build(base: &FloatVectors, params: &BuildParams) -> VamanaGraph {
    let n = base.count;
    assert!(n > 0 && n < u32::MAX as usize);
    assert!(params.r >= 2 && params.r <= u8::MAX as usize);
    assert!(params.l_build >= params.r);

    let medoid = find_medoid(base);
    let adj: Vec<RwLock<Vec<u32>>> =
        (0..n).map(|_| RwLock::new(Vec::with_capacity(params.r + 1))).collect();

    // Pass 1 with alpha=1.0 builds a navigable approximation cheaply;
    // pass 2 re-prunes every node with the target alpha (paper's schedule).
    for (pass, alpha) in [(0u64, 1.0f32), (1, params.alpha)] {
        let order = permutation(n, 42 + pass);
        order.par_chunks(4096).for_each_init(
            || SearchScratch::new(n),
            |scratch, chunk| {
                for &p in chunk {
                    insert_point(base, &adj, medoid, p, alpha, params, scratch);
                }
            },
        );
    }

    let adj: Vec<Vec<u32>> = adj.into_iter().map(|l| l.into_inner()).collect();
    VamanaGraph::from_adjacency(params.r, medoid, adj)
}

fn insert_point(
    base: &FloatVectors,
    adj: &[RwLock<Vec<u32>>],
    medoid: u32,
    p: u32,
    alpha: f32,
    params: &BuildParams,
    scratch: &mut SearchScratch,
) {
    let q = base.get(p as usize);
    beam_search(
        base,
        q,
        medoid,
        params.l_build,
        |id, buf| {
            buf.clear();
            buf.extend_from_slice(&adj[id as usize].read());
        },
        scratch,
    );

    // Prune over everything the search expanded plus p's current neighbors.
    let mut cand: Vec<(f32, u32)> = scratch.expanded.clone();
    for &nb in adj[p as usize].read().iter() {
        cand.push((l2_sq(q, base.get(nb as usize)), nb));
    }
    let new_nb = robust_prune(base, p, cand, alpha, params.r);
    *adj[p as usize].write() = new_nb.clone();

    // Reverse edges, pruning receivers that overflow R.
    for &nb in &new_nb {
        let mut a = adj[nb as usize].write();
        if a.contains(&p) {
            continue;
        }
        if a.len() < params.r {
            a.push(p);
        } else {
            let v_nb = base.get(nb as usize);
            let mut cand: Vec<(f32, u32)> =
                a.iter().map(|&x| (l2_sq(v_nb, base.get(x as usize)), x)).collect();
            cand.push((l2_sq(v_nb, base.get(p as usize)), p));
            *a = robust_prune(base, nb, cand, alpha, params.r);
        }
    }
}

/// RobustPrune (DiskANN Algorithm 2): greedily keep the closest candidate,
/// then drop every candidate it "occludes" (alpha * d(kept, c) <= d(p, c)).
///
/// Distances here are *squared* L2, while alpha is defined on the metric —
/// so the comparison uses alpha^2.
pub(crate) fn robust_prune(
    base: &FloatVectors,
    p: u32,
    mut cand: Vec<(f32, u32)>,
    alpha: f32,
    r: usize,
) -> Vec<u32> {
    cand.sort_unstable_by_key(|c| c.1);
    cand.dedup_by_key(|c| c.1);
    cand.retain(|c| c.1 != p);
    cand.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    let alpha_sq = alpha * alpha;
    let mut result = Vec::with_capacity(r);
    let mut alive = vec![true; cand.len()];
    for i in 0..cand.len() {
        if !alive[i] {
            continue;
        }
        let (_, pi) = cand[i];
        result.push(pi);
        if result.len() == r {
            break;
        }
        let v_i = base.get(pi as usize);
        for j in (i + 1)..cand.len() {
            if !alive[j] {
                continue;
            }
            let (d_pj, pj) = cand[j];
            if alpha_sq * l2_sq(v_i, base.get(pj as usize)) <= d_pj {
                alive[j] = false;
            }
        }
    }
    result
}

/// Entry point: the base point closest to the dataset mean.
fn find_medoid(base: &FloatVectors) -> u32 {
    let dim = base.dim;
    let sums = (0..base.count)
        .into_par_iter()
        .fold(
            || vec![0f64; dim],
            |mut acc, i| {
                for (a, x) in acc.iter_mut().zip(base.get(i)) {
                    *a += *x as f64;
                }
                acc
            },
        )
        .reduce(
            || vec![0f64; dim],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += *y;
                }
                a
            },
        );
    let mean: Vec<f32> = sums.iter().map(|&s| (s / base.count as f64) as f32).collect();
    (0..base.count)
        .into_par_iter()
        .map(|i| (l2_sq(&mean, base.get(i)), i))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap()
        .1 as u32
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic Fisher–Yates shuffle of 0..n.
fn permutation(n: usize, seed: u64) -> Vec<u32> {
    let mut v: Vec<u32> = (0..n as u32).collect();
    let mut state = seed;
    for i in (1..n).rev() {
        let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use nvvec_core::dataset::FloatVectors;

    #[test]
    fn robust_prune_drops_occluded() {
        // Points on a line at 0, 1, 2, 3. From p=0, candidate 1 occludes
        // both 2 and 3 (they are closer to 1 than to 0 by margin > alpha).
        let base = FloatVectors::from_raw(1, vec![0.0, 1.0, 2.0, 3.0]);
        let cand = vec![(1.0, 1), (4.0, 2), (9.0, 3)];
        let kept = robust_prune(&base, 0, cand, 1.1, 4);
        assert_eq!(kept, vec![1]);
    }

    #[test]
    fn robust_prune_respects_r_and_dedups() {
        let base = FloatVectors::from_raw(1, vec![0.0, 1.0, -1.0, 2.0]);
        // duplicate entries and p itself must be ignored
        let cand = vec![(1.0, 1), (1.0, 1), (0.0, 0), (1.0, 2), (4.0, 3)];
        let kept = robust_prune(&base, 0, cand, 1.0, 2);
        assert!(kept.len() <= 2);
        assert!(!kept.contains(&0));
    }

    #[test]
    fn permutation_is_deterministic_and_complete() {
        let a = permutation(1000, 7);
        let b = permutation(1000, 7);
        assert_eq!(a, b);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..1000).collect::<Vec<u32>>());
    }
}
