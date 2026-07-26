//! End-to-end index quality checks on synthetic data: build a graph, search
//! it, and compare against exact brute-force results.

use nvvec_core::dataset::FloatVectors;
use nvvec_index::{BuildParams, SearchScratch, VamanaGraph, beam_search, build};

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn random_vectors(n: usize, dim: usize, seed: u64) -> FloatVectors {
    let mut state = seed;
    let data: Vec<f32> =
        (0..n * dim).map(|_| (splitmix64(&mut state) >> 40) as f32 / (1u64 << 24) as f32).collect();
    FloatVectors::from_raw(dim, data)
}

fn search_ids(
    graph: &VamanaGraph,
    base: &FloatVectors,
    query: &[f32],
    k: usize,
    ef: usize,
    scratch: &mut SearchScratch,
) -> Vec<u32> {
    beam_search(
        base,
        query,
        graph.medoid,
        ef,
        |id, buf| {
            buf.clear();
            buf.extend_from_slice(graph.neighbors_of(id));
        },
        scratch,
    );
    scratch.top_k(k).map(|(_, id)| id).collect()
}

#[test]
fn recall_beats_090_on_random_data() {
    let (n, dim, k) = (2000, 16, 10);
    let base = random_vectors(n, dim, 1);
    let queries = random_vectors(100, dim, 2);
    let params = BuildParams { r: 16, l_build: 40, alpha: 1.2 };
    let graph = build(&base, &params);

    // structural invariants
    assert_eq!(graph.num_nodes(), n);
    assert!(graph.avg_degree() > 4.0, "suspiciously sparse: {}", graph.avg_degree());
    for id in 0..n as u32 {
        let nbs = graph.neighbors_of(id);
        assert!(nbs.len() <= params.r);
        assert!(!nbs.contains(&id), "self-loop at {id}");
    }

    let exact = nvvec_core::brute::search(&base, &queries, k);
    let mut scratch = SearchScratch::new(n);
    let mut hits = 0;
    for (qi, exact_res) in exact.iter().enumerate() {
        let got = search_ids(&graph, &base, queries.get(qi), k, 64, &mut scratch);
        let truth: Vec<u32> = exact_res.iter().map(|&(_, id)| id).collect();
        hits += got.iter().filter(|id| truth.contains(id)).count();
    }
    let recall = hits as f64 / (exact.len() * k) as f64;
    assert!(recall >= 0.9, "recall {recall} < 0.9");
}

#[test]
fn self_search_returns_self() {
    let base = random_vectors(500, 8, 3);
    let graph = build(&base, &BuildParams { r: 12, l_build: 30, alpha: 1.2 });
    let mut scratch = SearchScratch::new(500);
    let mut ok = 0;
    for qi in (0..500).step_by(10) {
        let got = search_ids(&graph, &base, base.get(qi), 1, 30, &mut scratch);
        ok += (got == vec![qi as u32]) as usize;
    }
    // every point should find itself (dist 0) essentially always
    assert!(ok >= 48, "only {ok}/50 self-searches succeeded");
}

#[test]
fn save_load_roundtrip() {
    let base = random_vectors(300, 8, 4);
    let graph = build(&base, &BuildParams { r: 8, l_build: 20, alpha: 1.2 });
    let path = std::env::temp_dir().join(format!("nvvec-graph-{}", std::process::id()));
    graph.save(&path).unwrap();
    let loaded = VamanaGraph::load(&path).unwrap();
    assert_eq!(loaded.num_nodes(), graph.num_nodes());
    assert_eq!(loaded.medoid, graph.medoid);
    assert_eq!(loaded.r(), graph.r());
    for id in [0u32, 7, 123, 299] {
        assert_eq!(loaded.neighbors_of(id), graph.neighbors_of(id));
    }
    std::fs::remove_file(&path).ok();
}
