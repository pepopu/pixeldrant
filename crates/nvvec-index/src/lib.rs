//! Vamana graph index (the DiskANN graph, NeurIPS'19).
//!
//! M1 keeps everything in memory: the graph here, vectors in
//! `FloatVectors`. M2 moves both into 4 KiB blocks behind
//! `nvvec_io::BlockReader` — which is why `beam_search` takes the
//! adjacency as a closure instead of a concrete graph type.

pub mod build;
pub mod disk;
pub mod graph;
#[cfg(target_os = "linux")]
pub mod pipeline;
pub mod search;

pub use build::{BuildParams, build};
pub use disk::{DiskIndex, DiskIndexMeta, write_disk_index};
pub use graph::VamanaGraph;
pub use search::{SearchScratch, beam_search, beam_search_batched};
