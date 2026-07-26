//! Frozen Vamana graph: fixed-stride adjacency, ready to be memcpy'd into
//! the M2 disk layout (one node's neighbors always fit its 4 KiB block).

use std::io::{self, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"NVVG";
const VERSION: u32 = 1;

pub struct VamanaGraph {
    r: usize,
    pub medoid: u32,
    /// Out-degree per node (<= r).
    degrees: Vec<u8>,
    /// Fixed stride r per node; slots past the degree are zero padding.
    neighbors: Vec<u32>,
}

impl VamanaGraph {
    pub(crate) fn from_adjacency(r: usize, medoid: u32, adj: Vec<Vec<u32>>) -> Self {
        assert!(r <= u8::MAX as usize);
        let n = adj.len();
        let mut degrees = vec![0u8; n];
        let mut neighbors = vec![0u32; n * r];
        for (i, list) in adj.into_iter().enumerate() {
            assert!(list.len() <= r, "node {i} has degree {} > R={r}", list.len());
            degrees[i] = list.len() as u8;
            neighbors[i * r..i * r + list.len()].copy_from_slice(&list);
        }
        Self { r, medoid, degrees, neighbors }
    }

    pub fn num_nodes(&self) -> usize {
        self.degrees.len()
    }

    pub fn r(&self) -> usize {
        self.r
    }

    #[inline]
    pub fn neighbors_of(&self, id: u32) -> &[u32] {
        let i = id as usize;
        &self.neighbors[i * self.r..i * self.r + self.degrees[i] as usize]
    }

    pub fn avg_degree(&self) -> f64 {
        self.degrees.iter().map(|&d| d as u64).sum::<u64>() as f64 / self.num_nodes() as f64
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut w = io::BufWriter::new(std::fs::File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(self.num_nodes() as u64).to_le_bytes())?;
        w.write_all(&(self.r as u32).to_le_bytes())?;
        w.write_all(&self.medoid.to_le_bytes())?;
        w.write_all(&self.degrees)?;
        let mut buf = Vec::with_capacity(self.neighbors.len() * 4);
        for &nb in &self.neighbors {
            buf.extend_from_slice(&nb.to_le_bytes());
        }
        w.write_all(&buf)?;
        w.flush()
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let err = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {msg}", path.display()));
        let mut f = io::BufReader::new(std::fs::File::open(path)?);
        let mut header = [0u8; 24];
        f.read_exact(&mut header)?;
        if &header[..4] != MAGIC {
            return Err(err("not a graph file"));
        }
        if u32::from_le_bytes(header[4..8].try_into().unwrap()) != VERSION {
            return Err(err("unsupported version"));
        }
        let n = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        let r = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
        let medoid = u32::from_le_bytes(header[20..24].try_into().unwrap());
        if r == 0 || r > u8::MAX as usize || medoid as usize >= n {
            return Err(err("corrupt header"));
        }
        let mut degrees = vec![0u8; n];
        f.read_exact(&mut degrees)?;
        let mut raw = vec![0u8; n * r * 4];
        f.read_exact(&mut raw)?;
        let neighbors: Vec<u32> = raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok(Self { r, medoid, degrees, neighbors })
    }
}
