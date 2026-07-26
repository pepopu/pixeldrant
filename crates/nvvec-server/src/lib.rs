//! Minimal HTTP serving layer (M5) over the disk index.
//!
//! Startup loads the index header plus one routing scorer — either exact
//! f32 vectors or an SQ8 codebook (1/4 RAM; the recommended mode: the
//! full-precision vectors then live only on disk). Search runs the same
//! rerank-always beam search as the benchmarks.
//!
//! v1 serving model: one shared `DiskIndex` (positional reads are
//! thread-safe), a scratch pool to avoid per-request O(n) allocations, and
//! `spawn_blocking` so disk I/O never parks the async runtime.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use nvvec_core::dataset::{FloatVectors, read_fvecs};
use nvvec_core::scorer::RouteScorer;
use nvvec_index::{DiskIndex, SearchScratch, Sq8Codebook};
use nvvec_io::{BlockReader, PreadReader};

pub enum Scorer {
    Exact(FloatVectors),
    Sq8(Sq8Codebook),
}

impl RouteScorer for Scorer {
    #[inline]
    fn score(&self, query: &[f32], id: u32) -> f32 {
        match self {
            Scorer::Exact(v) => v.score(query, id),
            Scorer::Sq8(c) => c.score(query, id),
        }
    }

    fn memory_bytes(&self) -> usize {
        match self {
            Scorer::Exact(v) => v.memory_bytes(),
            Scorer::Sq8(c) => c.memory_bytes(),
        }
    }
}

pub struct SearchDefaults {
    pub k: usize,
    pub ef: usize,
    pub w: usize,
}

pub struct AppState {
    pub index: DiskIndex<Box<dyn BlockReader>>,
    pub scorer: Scorer,
    pub defaults: SearchDefaults,
    scratches: Mutex<Vec<(SearchScratch, Vec<u8>)>>,
}

impl AppState {
    pub fn open(index_path: &Path, scorer: Scorer, defaults: SearchDefaults) -> anyhow::Result<Self> {
        let reader: Box<dyn BlockReader> = Box::new(PreadReader::open(index_path)?);
        let index = DiskIndex::with_reader(reader)?;
        anyhow::ensure!(
            match &scorer {
                Scorer::Exact(v) => v.count as u64 == index.meta.n && v.dim as u32 == index.meta.dim,
                Scorer::Sq8(c) => c.count as u64 == index.meta.n && c.dim as u32 == index.meta.dim,
            },
            "scorer shape does not match index ({} x {})",
            index.meta.n,
            index.meta.dim
        );
        Ok(Self { index, scorer, defaults, scratches: Mutex::new(Vec::new()) })
    }

    fn search(&self, query: &[f32], k: usize, ef: usize, w: usize) -> std::io::Result<(Vec<(f32, u32)>, usize)> {
        let (mut scratch, mut io_bufs) = self
            .scratches
            .lock()
            .pop()
            .unwrap_or_else(|| (SearchScratch::new(self.index.meta.n as usize), Vec::new()));
        let result = self.index.search(&self.scorer, query, k, ef, w, &mut scratch, &mut io_bufs);
        self.scratches.lock().push((scratch, io_bufs));
        result
    }
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub k: Option<usize>,
    pub ef: Option<usize>,
    pub w: Option<usize>,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub ids: Vec<u32>,
    pub distances: Vec<f32>,
    pub reads: usize,
    pub took_us: u128,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub vectors: u64,
    pub dim: u32,
    pub routing_memory_bytes: usize,
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        vectors: state.index.meta.n,
        dim: state.index.meta.dim,
        routing_memory_bytes: state.scorer.memory_bytes(),
    })
}

async fn search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    if req.vector.len() != state.index.meta.dim as usize {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("vector dim {} != index dim {}", req.vector.len(), state.index.meta.dim),
        ));
    }
    let k = req.k.unwrap_or(state.defaults.k).max(1);
    let ef = req.ef.unwrap_or(state.defaults.ef).max(k);
    let w = req.w.unwrap_or(state.defaults.w).clamp(1, 64);

    let t = Instant::now();
    let state2 = state.clone();
    let (results, reads) = tokio::task::spawn_blocking(move || {
        state2.search(&req.vector, k, ef, w)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let (distances, ids) = results.into_iter().map(|(d, id)| (d, id)).unzip();
    Ok(Json(SearchResponse { ids, distances, reads, took_us: t.elapsed().as_micros() }))
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/search", post(search))
        .with_state(state)
}

/// Load the scorer per CLI configuration.
pub fn load_scorer(routing: &str, base: Option<&Path>, codes: Option<&Path>) -> anyhow::Result<Scorer> {
    match routing {
        "exact" => {
            let path = base.ok_or_else(|| anyhow::anyhow!("--routing exact requires --base <fvecs>"))?;
            Ok(Scorer::Exact(read_fvecs(path)?))
        }
        "sq8" => {
            let path = codes.ok_or_else(|| anyhow::anyhow!("--routing sq8 requires --codes <sq8 file>"))?;
            Ok(Scorer::Sq8(Sq8Codebook::load(path)?))
        }
        other => anyhow::bail!("unknown routing '{other}' (expected exact or sq8)"),
    }
}
