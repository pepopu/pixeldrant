//! Disk index acceptance (M2): layout roundtrip, w=1 equivalence with the
//! in-memory search, and I/O counting.

use nvvec_core::dataset::FloatVectors;
use nvvec_index::{
    BuildParams, DiskIndex, SearchScratch, beam_search, build, write_disk_index,
};

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

fn temp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("nvvec-disk-{}-{name}", std::process::id()))
}

#[test]
fn layout_roundtrip_and_search_parity() {
    let (n, dim) = (1500, 24);
    let base = random_vectors(n, dim, 11);
    let graph = build(&base, &BuildParams { r: 16, l_build: 40, alpha: 1.2 });
    let path = temp("idx");
    let meta = write_disk_index(&path, &base, &graph).unwrap();
    assert_eq!(meta.nodes_per_block, (4096 / (24 * 4 + 4 + 16 * 4)) as u32);

    let index = DiskIndex::open(&path).unwrap();
    assert_eq!(index.meta.n, n as u64);
    assert_eq!(index.meta.medoid, graph.medoid);

    // stored vectors and adjacency must match the sources byte-for-byte
    for node in [0u32, 1, 5, 777, (n - 1) as u32] {
        let (vec_disk, nbs_disk) = index.read_node(node).unwrap();
        assert_eq!(vec_disk.as_slice(), base.get(node as usize), "vector of {node}");
        assert_eq!(nbs_disk.as_slice(), graph.neighbors_of(node), "neighbors of {node}");
    }

    // w=1 disk search must equal the in-memory search step for step
    let queries = random_vectors(40, dim, 12);
    let (k, ef) = (10, 50);
    let mut mem_scratch = SearchScratch::new(n);
    let mut disk_scratch = SearchScratch::new(n);
    let mut io_bufs = Vec::new();
    for qi in 0..queries.count {
        let q = queries.get(qi);
        let mem_hops = beam_search(
            &base,
            q,
            graph.medoid,
            ef,
            |id, buf| {
                buf.clear();
                buf.extend_from_slice(graph.neighbors_of(id));
            },
            &mut mem_scratch,
        );
        let mem_ids: Vec<(f32, u32)> = mem_scratch.top_k(k).collect();

        let (disk_ids, reads) =
            index.search(&base, q, k, ef, 1, &mut disk_scratch, &mut io_bufs).unwrap();
        assert_eq!(disk_ids, mem_ids, "query {qi} diverged");
        assert_eq!(reads, mem_hops, "query {qi} read count != hops");
    }

    // wider beams may explore a superset; recall must not collapse
    let exact = nvvec_core::brute::search(&base, &queries, k);
    for w in [2usize, 4, 8] {
        let mut hits = 0;
        for qi in 0..queries.count {
            let (got, _) =
                index.search(&base, queries.get(qi), k, ef, w, &mut disk_scratch, &mut io_bufs).unwrap();
            let truth: Vec<u32> = exact[qi].iter().map(|&(_, id)| id).collect();
            hits += got.iter().filter(|(_, id)| truth.contains(id)).count();
        }
        let recall = hits as f64 / (queries.count * k) as f64;
        assert!(recall >= 0.85, "w={w} recall {recall} too low");
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_corrupt_header() {
    let base = random_vectors(200, 8, 21);
    let graph = build(&base, &BuildParams { r: 8, l_build: 20, alpha: 1.2 });
    let path = temp("corrupt");
    write_disk_index(&path, &base, &graph).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[0] ^= 0xFF; // break magic
    std::fs::write(&path, &bytes).unwrap();
    assert!(DiskIndex::open(&path).is_err());
    std::fs::remove_file(&path).ok();
}
