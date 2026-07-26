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
use nvvec_io::UringExecutor;

use crate::disk::{DiskIndexMeta, parse_neighbors};

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
    pool: Vec<Cand>,
    visited: HashSet<u32>,
    /// Neighbor ids parsed out of this round's completed block reads.
    pending_nb: Vec<u32>,
    in_flight: u32,
    reads: usize,
    active: bool,
}

impl QState {
    fn new() -> Self {
        Self {
            qi: 0,
            pool: Vec::new(),
            visited: HashSet::with_capacity(8192),
            pending_nb: Vec::new(),
            in_flight: 0,
            reads: 0,
            active: false,
        }
    }

    fn seed(&mut self, qi: usize, meta: &DiskIndexMeta, base: &FloatVectors, queries: &FloatVectors) {
        self.qi = qi;
        self.pool.clear();
        self.visited.clear();
        self.pending_nb.clear();
        self.in_flight = 0;
        self.reads = 0;
        self.active = true;
        let entry = meta.medoid;
        self.visited.insert(entry);
        let dist = l2_sq(queries.get(qi), base.get(entry as usize));
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
/// Returns per-query (top-k ascending, block reads). Routing distances come
/// from `base` in RAM (M4 swaps in quantized codes).
///
/// The executor must have `capacity() >= concurrency * w` and be freshly
/// opened on the same index file that `meta` describes.
pub fn search_pipelined(
    meta: &DiskIndexMeta,
    exec: &mut UringExecutor,
    base: &FloatVectors,
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
        let mut s = QState::new();
        s.seed(next_q, meta, base, queries);
        next_q += 1;
        states.push(s);
    }

    let mut remaining = nq;
    let mut frontier: Vec<u32> = Vec::with_capacity(params.w);

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
                        pool_insert(&mut s.pool, params.ef, l2_sq(q, base.get(nb as usize)), nb);
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
                    results[s.qi] = (
                        s.pool.iter().take(params.k).map(|c| (c.dist, c.id)).collect(),
                        s.reads,
                    );
                    remaining -= 1;
                    if next_q < nq {
                        s.seed(next_q, meta, base, queries);
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
        exec.drain(1, |tag, block_bytes| {
            let sid = (tag & 0xFFFF) as usize;
            let node = (tag >> 16) as u32;
            let off = (node as u64 % npb) as usize * node_len;
            let s = &mut states[sid];
            s.pending_nb.extend(parse_neighbors(&block_bytes[off..off + node_len], dim, r));
            s.in_flight -= 1;
        })?;
    }
    Ok(results)
}
