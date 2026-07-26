//! On-disk index in the DiskANN layout: header block, then data blocks with
//! nodes packed `BLOCK_SIZE / node_len` per block. A node is
//! `[vector: dim*f32][degree: u32][neighbors: r*u32 zero-padded]`, so one
//! block read yields both a node's adjacency (for routing) and its
//! full-precision vector (for reranking — unused until M4, but the layout
//! is fixed here).
//!
//! M2 routing scores candidates from RAM (`FloatVectors`); M4 will replace
//! that with in-memory quantized codes. The disk's job in both is the same:
//! one 4 KiB read per expanded node, batched `w` at a time.

use std::io::{self, Write};
use std::path::Path;

use nvvec_core::dataset::FloatVectors;
use nvvec_core::distance::l2_sq;
use nvvec_core::scorer::RouteScorer;
use nvvec_core::topk::TopK;
use nvvec_io::{BLOCK_SIZE, BlockRead, BlockReader, PreadReader};

use crate::graph::VamanaGraph;
use crate::search::{SearchScratch, beam_search_batched};

const MAGIC: &[u8; 4] = b"NVVD";
const VERSION: u32 = 1;

#[derive(Clone, Copy, Debug)]
pub struct DiskIndexMeta {
    pub n: u64,
    pub dim: u32,
    pub r: u32,
    pub medoid: u32,
    pub node_len: u32,
    pub nodes_per_block: u32,
}

impl DiskIndexMeta {
    fn data_blocks(&self) -> u64 {
        self.n.div_ceil(self.nodes_per_block as u64)
    }
}

/// Serialize vectors + graph into the block layout.
pub fn write_disk_index(
    path: &Path,
    base: &FloatVectors,
    graph: &VamanaGraph,
) -> io::Result<DiskIndexMeta> {
    assert_eq!(base.count, graph.num_nodes());
    let (dim, r) = (base.dim, graph.r());
    let node_len = dim * 4 + 4 + r * 4;
    if node_len > BLOCK_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("node ({node_len} B) exceeds block size; multi-block nodes not supported yet"),
        ));
    }
    let npb = BLOCK_SIZE / node_len;
    let meta = DiskIndexMeta {
        n: base.count as u64,
        dim: dim as u32,
        r: r as u32,
        medoid: graph.medoid,
        node_len: node_len as u32,
        nodes_per_block: npb as u32,
    };

    let mut w = io::BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
    let mut header = [0u8; BLOCK_SIZE];
    header[..4].copy_from_slice(MAGIC);
    header[4..8].copy_from_slice(&VERSION.to_le_bytes());
    header[8..16].copy_from_slice(&meta.n.to_le_bytes());
    header[16..20].copy_from_slice(&meta.dim.to_le_bytes());
    header[20..24].copy_from_slice(&meta.r.to_le_bytes());
    header[24..28].copy_from_slice(&meta.medoid.to_le_bytes());
    header[28..32].copy_from_slice(&meta.node_len.to_le_bytes());
    header[32..36].copy_from_slice(&meta.nodes_per_block.to_le_bytes());
    w.write_all(&header)?;

    let mut block = vec![0u8; BLOCK_SIZE];
    for first in (0..base.count).step_by(npb) {
        block.fill(0);
        for slot in 0..npb.min(base.count - first) {
            let node = first + slot;
            let buf = &mut block[slot * node_len..(slot + 1) * node_len];
            for (i, &x) in base.get(node).iter().enumerate() {
                buf[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
            }
            let deg_off = dim * 4;
            let nbs = graph.neighbors_of(node as u32);
            buf[deg_off..deg_off + 4].copy_from_slice(&(nbs.len() as u32).to_le_bytes());
            for (i, &nb) in nbs.iter().enumerate() {
                let o = deg_off + 4 + i * 4;
                buf[o..o + 4].copy_from_slice(&nb.to_le_bytes());
            }
        }
        w.write_all(&block)?;
    }
    w.flush()?;
    Ok(meta)
}

pub struct DiskIndex<R: BlockReader> {
    pub meta: DiskIndexMeta,
    reader: R,
}

impl DiskIndex<PreadReader> {
    pub fn open(path: &Path) -> io::Result<Self> {
        Self::with_reader(PreadReader::open(path)?)
    }
}

impl<R: BlockReader> DiskIndex<R> {
    pub fn with_reader(reader: R) -> io::Result<Self> {
        let err = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, msg.to_string());
        let mut header = vec![0u8; BLOCK_SIZE];
        reader.read_block(0, &mut header)?;
        if &header[..4] != MAGIC {
            return Err(err("not a disk index (bad magic)"));
        }
        if u32::from_le_bytes(header[4..8].try_into().unwrap()) != VERSION {
            return Err(err("unsupported disk index version"));
        }
        let meta = DiskIndexMeta {
            n: u64::from_le_bytes(header[8..16].try_into().unwrap()),
            dim: u32::from_le_bytes(header[16..20].try_into().unwrap()),
            r: u32::from_le_bytes(header[20..24].try_into().unwrap()),
            medoid: u32::from_le_bytes(header[24..28].try_into().unwrap()),
            node_len: u32::from_le_bytes(header[28..32].try_into().unwrap()),
            nodes_per_block: u32::from_le_bytes(header[32..36].try_into().unwrap()),
        };
        let expect_node_len = meta.dim * 4 + 4 + meta.r * 4;
        if meta.node_len != expect_node_len
            || meta.nodes_per_block as usize != BLOCK_SIZE / expect_node_len as usize
            || meta.medoid as u64 >= meta.n
            || reader.num_blocks() != 1 + meta.data_blocks()
        {
            return Err(err("corrupt disk index header"));
        }
        Ok(Self { meta, reader })
    }

    #[inline]
    fn locate(&self, node: u32) -> (u64, usize) {
        let npb = self.meta.nodes_per_block as u64;
        (1 + node as u64 / npb, (node as u64 % npb) as usize * self.meta.node_len as usize)
    }

    /// Fetch one node's vector and neighbor list (test/debug convenience;
    /// the search path uses batched reads instead).
    pub fn read_node(&self, node: u32) -> io::Result<(Vec<f32>, Vec<u32>)> {
        let mut buf = vec![0u8; BLOCK_SIZE];
        let (blk, off) = self.locate(node);
        self.reader.read_block(blk, &mut buf)?;
        let bytes = &buf[off..off + self.meta.node_len as usize];
        let dim = self.meta.dim as usize;
        let vector = bytes[..dim * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok((vector, parse_neighbors(bytes, dim, self.meta.r as usize)))
    }

    /// Beam search over the on-disk graph: one block read per expanded node,
    /// `w` reads batched per round. Routing via `scorer` (exact RAM vectors
    /// or quantized codes); every expanded node is reranked exactly against
    /// the full-precision vector in its block, and the returned top-k comes
    /// only from exact scores. Returns (top-k ascending, block reads).
    pub fn search<S: RouteScorer>(
        &self,
        scorer: &S,
        query: &[f32],
        k: usize,
        ef: usize,
        w: usize,
        scratch: &mut SearchScratch,
        io_bufs: &mut Vec<u8>,
    ) -> io::Result<(Vec<(f32, u32)>, usize)> {
        io_bufs.resize(w * BLOCK_SIZE, 0);
        let dim = self.meta.dim as usize;
        let (r, node_len) = (self.meta.r as usize, self.meta.node_len as usize);
        let mut io_err: Option<io::Error> = None;
        let mut exact = TopK::new(k);
        let mut decode_buf: Vec<f32> = Vec::with_capacity(dim);

        let reads = beam_search_batched(
            scorer,
            query,
            self.meta.medoid,
            ef,
            w,
            |frontier, out| {
                if io_err.is_some() {
                    return;
                }
                let mut batch: Vec<BlockRead<'_>> = io_bufs
                    .chunks_exact_mut(BLOCK_SIZE)
                    .zip(frontier)
                    .map(|(buf, &node)| BlockRead { block_id: self.locate(node).0, buf })
                    .collect();
                if let Err(e) = self.reader.read_batch(&mut batch) {
                    io_err = Some(e);
                    return;
                }
                drop(batch);
                for (i, &node) in frontier.iter().enumerate() {
                    let off = self.locate(node).1;
                    let bytes = &io_bufs[i * BLOCK_SIZE + off..i * BLOCK_SIZE + off + node_len];
                    decode_vector(bytes, dim, &mut decode_buf);
                    exact.push(l2_sq(query, &decode_buf), node);
                    out.extend(parse_neighbors(bytes, dim, r));
                }
            },
            scratch,
        );
        if let Some(e) = io_err {
            return Err(e);
        }
        Ok((exact.into_sorted(), reads))
    }
}

/// Decode the little-endian f32 vector at the head of a node's bytes into a
/// reusable buffer. Rerank then calls the ordinary `l2_sq` on it — same
/// kernel, same summation order, bit-identical to RAM-based scoring.
#[inline]
pub(crate) fn decode_vector(node_bytes: &[u8], dim: usize, out: &mut Vec<f32>) {
    out.clear();
    out.extend(
        node_bytes[..dim * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
    );
}

pub(crate) fn parse_neighbors(node_bytes: &[u8], dim: usize, r: usize) -> Vec<u32> {
    let deg_off = dim * 4;
    let deg =
        u32::from_le_bytes(node_bytes[deg_off..deg_off + 4].try_into().unwrap()) as usize;
    node_bytes[deg_off + 4..deg_off + 4 + deg.min(r) * 4]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
