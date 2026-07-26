//! Best-first beam search with a bounded candidate pool.
//!
//! The adjacency is supplied as a closure so the same routine serves the
//! in-memory build graph (RwLock'd, still mutating), the frozen
//! `VamanaGraph`, and — in M2 — neighbor lists parsed out of disk blocks.

use nvvec_core::dataset::FloatVectors;
use nvvec_core::distance::l2_sq;
use nvvec_core::scorer::RouteScorer;

#[derive(Clone, Copy)]
struct Candidate {
    dist: f32,
    id: u32,
    expanded: bool,
}

/// Reusable per-thread search state. Allocating the visited-marker array is
/// O(n), so create one scratch per thread and reuse it across queries.
pub struct SearchScratch {
    visit_epoch: Vec<u32>,
    epoch: u32,
    pool: Vec<Candidate>,
    /// Nodes expanded by the last search, with their distances. The build
    /// uses these as pruning candidates; queries can ignore them.
    pub expanded: Vec<(f32, u32)>,
    nb_buf: Vec<u32>,
}

impl SearchScratch {
    pub fn new(num_nodes: usize) -> Self {
        Self {
            visit_epoch: vec![0; num_nodes],
            epoch: 0,
            pool: Vec::new(),
            expanded: Vec::new(),
            nb_buf: Vec::new(),
        }
    }

    fn begin(&mut self) {
        if self.epoch == u32::MAX {
            self.visit_epoch.fill(0);
            self.epoch = 0;
        }
        self.epoch += 1;
        self.pool.clear();
        self.expanded.clear();
    }

    /// True the first time `id` is seen in the current search.
    #[inline]
    fn first_visit(&mut self, id: u32) -> bool {
        let slot = &mut self.visit_epoch[id as usize];
        if *slot == self.epoch {
            false
        } else {
            *slot = self.epoch;
            true
        }
    }

    /// Top-k (dist, id) of the last search, ascending by distance.
    pub fn top_k(&self, k: usize) -> impl Iterator<Item = (f32, u32)> + '_ {
        self.pool.iter().take(k).map(|c| (c.dist, c.id))
    }
}

/// Round-based beam search with width `w`: each round expands the up-to-`w`
/// closest unexpanded candidates at once. `expand` receives the whole
/// frontier and must APPEND all their neighbors to the buffer — for the disk
/// index that is one batched read of `w` blocks, which is exactly the shape
/// the io_uring backend (M3) wants submitted together.
///
/// Routing distances come from `scorer` (exact `FloatVectors` or a
/// quantized codebook). With `w == 1` and an exact scorer this is
/// step-for-step identical to [`beam_search`].
/// Returns the number of expanded nodes (= block reads for the disk index).
pub fn beam_search_batched<S: RouteScorer>(
    scorer: &S,
    query: &[f32],
    entry: u32,
    ef: usize,
    w: usize,
    mut expand: impl FnMut(&[u32], &mut Vec<u32>),
    scratch: &mut SearchScratch,
) -> usize {
    assert!(ef > 0 && w > 0);
    scratch.begin();
    scratch.first_visit(entry);
    let d0 = scorer.score(query, entry);
    scratch.pool.push(Candidate { dist: d0, id: entry, expanded: false });

    let mut expansions = 0usize;
    let mut frontier: Vec<u32> = Vec::with_capacity(w);
    loop {
        frontier.clear();
        let lim = scratch.pool.len().min(ef);
        for i in 0..lim {
            if frontier.len() == w {
                break;
            }
            if !scratch.pool[i].expanded {
                scratch.pool[i].expanded = true;
                scratch.expanded.push((scratch.pool[i].dist, scratch.pool[i].id));
                frontier.push(scratch.pool[i].id);
            }
        }
        if frontier.is_empty() {
            break;
        }
        expansions += frontier.len();

        let mut nb_buf = std::mem::take(&mut scratch.nb_buf);
        nb_buf.clear();
        expand(&frontier, &mut nb_buf);
        for i in 0..nb_buf.len() {
            let nb = nb_buf[i];
            if !scratch.first_visit(nb) {
                continue;
            }
            let d = scorer.score(query, nb);
            if scratch.pool.len() < ef || d < scratch.pool.last().unwrap().dist {
                let pos = scratch.pool.partition_point(|c| c.dist <= d);
                scratch.pool.insert(pos, Candidate { dist: d, id: nb, expanded: false });
                if scratch.pool.len() > ef {
                    scratch.pool.pop();
                }
            }
        }
        scratch.nb_buf = nb_buf;
    }
    expansions
}

/// Greedy best-first search keeping the `ef` closest candidates seen.
/// `neighbors` must fill the provided buffer with the adjacency of a node.
/// Results are left in the scratch (see [`SearchScratch::top_k`]);
/// returns the number of expanded nodes (= disk reads per query in M2).
pub fn beam_search(
    base: &FloatVectors,
    query: &[f32],
    entry: u32,
    ef: usize,
    mut neighbors: impl FnMut(u32, &mut Vec<u32>),
    scratch: &mut SearchScratch,
) -> usize {
    assert!(ef > 0);
    scratch.begin();
    scratch.first_visit(entry);
    let d0 = l2_sq(query, base.get(entry as usize));
    scratch.pool.push(Candidate { dist: d0, id: entry, expanded: false });

    let mut hops = 0usize;
    loop {
        // Closest unexpanded candidate within the beam.
        let lim = scratch.pool.len().min(ef);
        let Some(idx) = scratch.pool[..lim].iter().position(|c| !c.expanded) else {
            break;
        };
        let cur = scratch.pool[idx];
        scratch.pool[idx].expanded = true;
        scratch.expanded.push((cur.dist, cur.id));
        hops += 1;

        // Take the buffer out to avoid a double mutable borrow of scratch.
        let mut nb_buf = std::mem::take(&mut scratch.nb_buf);
        neighbors(cur.id, &mut nb_buf);
        for i in 0..nb_buf.len() {
            let nb = nb_buf[i];
            if !scratch.first_visit(nb) {
                continue;
            }
            let d = l2_sq(query, base.get(nb as usize));
            if scratch.pool.len() < ef || d < scratch.pool.last().unwrap().dist {
                let pos = scratch.pool.partition_point(|c| c.dist <= d);
                scratch.pool.insert(pos, Candidate { dist: d, id: nb, expanded: false });
                if scratch.pool.len() > ef {
                    scratch.pool.pop();
                }
            }
        }
        scratch.nb_buf = nb_buf;
    }
    hops
}
