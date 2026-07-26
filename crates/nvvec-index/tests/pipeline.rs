//! M3b acceptance: the pipelined executor must match the round-based
//! single-query search exactly — same results, same read counts — while
//! interleaving many queries.
#![cfg(target_os = "linux")]

use nvvec_core::dataset::FloatVectors;
use nvvec_core::scorer::RouteScorer;
use nvvec_index::pipeline::{PipelineParams, search_pipelined};
use nvvec_index::{BuildParams, DiskIndex, SearchScratch, build, write_disk_index};
use nvvec_io::{UringExecutor, UringOpts};

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

#[test]
fn pipelined_matches_batched_search() {
    let (n, dim, k, ef, w) = (1500, 24, 10, 50, 2);
    let base = random_vectors(n, dim, 31);
    let queries = random_vectors(120, dim, 32);
    let graph = build(&base, &BuildParams { r: 16, l_build: 40, alpha: 1.2 });
    let path = std::env::temp_dir().join(format!("nvvec-pipe-{}", std::process::id()));
    write_disk_index(&path, &base, &graph).unwrap();

    // reference: per-query round-based search through the pread index
    let index = DiskIndex::open(&path).unwrap();
    let mut scratch = SearchScratch::new(n);
    let mut io_bufs = Vec::new();
    let mut expected = Vec::new();
    for qi in 0..queries.count {
        expected.push(
            index.search(&base, queries.get(qi), k, ef, w, &mut scratch, &mut io_bufs).unwrap(),
        );
    }

    // pipelined: 8 queries in flight over one ring
    let concurrency = 8;
    let mut exec = UringExecutor::open(
        &path,
        UringOpts { queue_depth: (concurrency * w) as u32, direct: false },
    )
    .unwrap();
    let got = search_pipelined(
        &index.meta,
        &mut exec,
        &base,
        &queries,
        &PipelineParams { k, ef, w, concurrency },
    )
    .unwrap();

    assert_eq!(got.len(), expected.len());
    for (qi, ((got_res, got_reads), (exp_res, exp_reads))) in
        got.iter().zip(expected.iter()).enumerate()
    {
        assert_eq!(got_reads, exp_reads, "query {qi}: read count diverged");
        assert_eq!(got_res, exp_res, "query {qi}: results diverged");
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn sq8_routing_with_exact_rerank_holds_recall() {
    use nvvec_core::distance::l2_sq;
    use nvvec_index::Sq8Codebook;

    let (n, dim, k, ef, w, c) = (1500, 24, 10, 50, 2, 8);
    let base = random_vectors(n, dim, 51);
    let queries = random_vectors(100, dim, 52);
    let graph = build(&base, &BuildParams { r: 16, l_build: 40, alpha: 1.2 });
    let path = std::env::temp_dir().join(format!("nvvec-pipe-sq8-{}", std::process::id()));
    write_disk_index(&path, &base, &graph).unwrap();
    let meta = DiskIndex::open(&path).unwrap().meta;
    let exact_gt = nvvec_core::brute::search(&base, &queries, k);

    let mut recalls = Vec::new();
    let codebook = Sq8Codebook::train(&base);
    assert!(codebook.memory_bytes() * 3 < base.count * base.dim * 4);

    // run once with exact routing, once with sq8 routing
    for use_sq8 in [false, true] {
        let mut exec = UringExecutor::open(
            &path,
            UringOpts { queue_depth: (c * w) as u32, direct: false },
        )
        .unwrap();
        let params = PipelineParams { k, ef, w, concurrency: c };
        let out = if use_sq8 {
            search_pipelined(&meta, &mut exec, &codebook, &queries, &params).unwrap()
        } else {
            search_pipelined(&meta, &mut exec, &base, &queries, &params).unwrap()
        };

        let mut hits = 0;
        for (qi, (res, _)) in out.iter().enumerate() {
            // returned distances must be EXACT regardless of routing scorer
            for &(dist, id) in res {
                let true_dist = l2_sq(queries.get(qi), base.get(id as usize));
                assert_eq!(dist, true_dist, "query {qi} id {id}: rerank not exact");
            }
            let truth: Vec<u32> = exact_gt[qi].iter().map(|&(_, id)| id).collect();
            hits += res.iter().filter(|(_, id)| truth.contains(id)).count();
        }
        recalls.push(hits as f64 / (out.len() * k) as f64);
    }
    let (exact_recall, sq8_recall) = (recalls[0], recalls[1]);
    assert!(
        sq8_recall >= exact_recall - 0.03,
        "sq8 routing lost too much recall: {sq8_recall} vs {exact_recall}"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn pipelined_handles_more_concurrency_than_queries() {
    let (n, dim) = (600, 8);
    let base = random_vectors(n, dim, 41);
    let queries = random_vectors(3, dim, 42);
    let graph = build(&base, &BuildParams { r: 8, l_build: 20, alpha: 1.2 });
    let path = std::env::temp_dir().join(format!("nvvec-pipe-small-{}", std::process::id()));
    write_disk_index(&path, &base, &graph).unwrap();
    let index = DiskIndex::open(&path).unwrap();

    let mut exec =
        UringExecutor::open(&path, UringOpts { queue_depth: 64, direct: false }).unwrap();
    let got = search_pipelined(
        &index.meta,
        &mut exec,
        &base,
        &queries,
        &PipelineParams { k: 5, ef: 20, w: 4, concurrency: 16 },
    )
    .unwrap();
    assert_eq!(got.len(), 3);
    for (res, reads) in &got {
        assert_eq!(res.len(), 5);
        assert!(*reads > 0);
    }
    std::fs::remove_file(&path).ok();
}
