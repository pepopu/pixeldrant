//! M3b: cross-query pipelined search (Linux only).
//!
//! One scheduler thread drives up to `concurrency` query state machines
//! over a single deep io_uring. Each query keeps the exact round semantics
//! of `beam_search_batched` (expand its w best unexpanded candidates, wait
//! for its own reads, integrate, repeat) — but *different* queries overlap
//! freely, so the device queue stays ~`concurrency * w` deep instead of
//! draining to zero between every round. That queue depth is what turns a
//! per-query latency game into an IOPS-saturation game.
//!
//! A finished query immediately hands its state slot to the next pending
//! query, so the pipeline never breathes out.

use std::collections::HashSet;
use std::io;

use nvvec_core::dataset::FloatVectors;
use nvvec_core::distance::l2_sq;
use nvvec_core::scorer::RouteScorer;
use nvvec_core::topk::TopK;
use nvvec_io::UringExecutor;

use crate::disk::{DiskIndexMeta, decode_vector, parse_neighbors};

pub struct PipelineParams {
    pub k: usize,
    pub ef: usize,
    /// Beam width per query per round.
    pub w: usize,
    /// Query state machines in flight at once.
    pub concurrency: usize,
}

#[derive(Clone, Copy)]
struct Cand {
    dist: f32,
    id: u32,
    expanded: bool,
}

struct QState {
    qi: usize,
    /// Routing pool, ordered by (possibly approximate) scorer distances.
    pool: Vec<Cand>,
    /// Exact-scored expanded nodes (rerank from block vectors) — the
    /// returned top-k comes from here, never from approximate scores.
    exact: TopK,
    visited: HashSet<u32>,
    /// Neighbor ids parsed out of this round's completed block reads.
    pending_nb: Vec<u32>,
    in_flight: u32,
    reads: usize,
    active: bool,
}

impl QState {
    fn new(k: usize) -> Self {
        Self {
            qi: 0,
            pool: Vec::new(),
            exact: TopK::new(k),
            visited: HashSet::with_capacity(8192),
            pending_nb: Vec::new(),
            in_flight: 0,
            reads: 0,
            active: false,
        }
    }

    fn seed(
        &mut self,
        qi: usize,
        k: usize,
        meta: &DiskIndexMeta,
        scorer: &impl RouteScorer,
        queries: &FloatVectors,
    ) {
        self.qi = qi;
        self.pool.clear();
        self.exact = TopK::new(k);
        self.visited.clear();
        self.pending_nb.clear();
        self.in_flight = 0;
        self.reads = 0;
        self.active = true;
        let entry = meta.medoid;
        self.visited.insert(entry);
        let dist = scorer.score(queries.get(qi), entry);
        self.pool.push(Cand { dist, id: entry, expanded: false });
    }
}

fn pool_insert(pool: &mut Vec<Cand>, ef: usize, dist: f32, id: u32) {
    if pool.len() >= ef && dist >= pool.last().unwrap().dist {
        return;
    }
    let pos = pool.partition_point(|c| c.dist <= dist);
    pool.insert(pos, Cand { dist, id, expanded: false });
    if pool.len() > ef {
        pool.pop();
    }
}

/// Run all `queries` through the on-disk graph, `concurrency` at a time.
/// Returns per-query (top-k ascending, block reads).
///
/// Routing uses `scorer` (exact `FloatVectors` or a quantized codebook);
/// every expanded node is *reranked exactly* against the full-precision
/// vector in its block, and the returned top-k comes only from those exact
/// scores — approximate routing can misdirect exploration but can never
/// corrupt a returned distance.
///
/// The executor must have `capacity() >= concurrency * w` and be freshly
/// opened on the same index file that `meta` describes.
pub fn search_pipelined<S: RouteScorer>(
    meta: &DiskIndexMeta,
    exec: &mut UringExecutor,
    scorer: &S,
    queries: &FloatVectors,
    params: &PipelineParams,
) -> io::Result<Vec<(Vec<(f32, u32)>, usize)>> {
    let nq = queries.count;
    let mut results: Vec<(Vec<(f32, u32)>, usize)> = vec![(Vec::new(), 0); nq];
    if nq == 0 {
        return Ok(results);
    }
    let concurrency = params.concurrency.clamp(1, nq.max(1)).min(u16::MAX as usize);
    assert!(params.w >= 1 && params.ef >= params.k);
    assert!(
        exec.capacity() >= concurrency * params.w,
        "executor capacity {} < concurrency {concurrency} * w {}",
        exec.capacity(),
        params.w
    );

    let dim = meta.dim as usize;
    let r = meta.r as usize;
    let node_len = meta.node_len as usize;
    let npb = meta.nodes_per_block as u64;

    let mut states: Vec<QState> = Vec::with_capacity(concurrency);
    let mut next_q = 0usize;
    for _ in 0..concurrency {
        let mut s = QState::new(params.k);
        s.seed(next_q, params.k, meta, scorer, queries);
        next_q += 1;
        states.push(s);
    }

    let mut remaining = nq;
    let mut frontier: Vec<u32> = Vec::with_capacity(params.w);
    let mut decode_buf: Vec<f32> = Vec::with_capacity(dim);

    while remaining > 0 {
        // Phase A: every idle state integrates its round and expands.
        for sid in 0..states.len() {
            // loop so a converged query immediately reseeds and expands
            loop {
                let s = &mut states[sid];
                if !s.active || s.in_flight > 0 {
                    break;
                }
                let q = queries.get(s.qi);
                for i in 0..s.pending_nb.len() {
                    let nb = s.pending_nb[i];
                    if s.visited.insert(nb) {
                        pool_insert(&mut s.pool, params.ef, scorer.score(q, nb), nb);
                    }
                }
                s.pending_nb.clear();

                frontier.clear();
                let lim = s.pool.len().min(params.ef);
                for i in 0..lim {
                    if frontier.len() == params.w {
                        break;
                    }
                    if !s.pool[i].expanded {
                        s.pool[i].expanded = true;
                        frontier.push(s.pool[i].id);
                    }
                }

                if frontier.is_empty() {
                    let exact = std::mem::replace(&mut s.exact, TopK::new(params.k));
                    results[s.qi] = (exact.into_sorted(), s.reads);
                    remaining -= 1;
                    if next_q < nq {
                        s.seed(next_q, params.k, meta, scorer, queries);
                        next_q += 1;
                        continue; // expand the fresh query right away
                    }
                    s.active = false;
                    break;
                }

                for &node in &frontier {
                    let tag = (sid as u64) | ((node as u64) << 16);
                    exec.stage_read(1 + node as u64 / npb, tag)?;
                    s.in_flight += 1;
                    s.reads += 1;
                }
                break;
            }
        }
        if remaining == 0 {
            break;
        }
        exec.flush()?;

        // Phase B: block until at least one read lands, drain everything.
        // Each landed block yields the node's neighbors (routing) AND its
        // full-precision vector (exact rerank, effectively free).
        exec.drain(1, |tag, block_bytes| {
            let sid = (tag & 0xFFFF) as usize;
            let node = (tag >> 16) as u32;
            let off = (node as u64 % npb) as usize * node_len;
            let node_bytes = &block_bytes[off..off + node_len];
            let s = &mut states[sid];
            decode_vector(node_bytes, dim, &mut decode_buf);
            s.exact.push(l2_sq(queries.get(s.qi), &decode_buf), node);
            s.pending_nb.extend(parse_neighbors(node_bytes, dim, r));
            s.in_flight -= 1;
        })?;
    }
    Ok(results)
}
