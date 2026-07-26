//! Loaders for the TEXMEX benchmark formats (`.fvecs` / `.ivecs`).
//!
//! Record layout (little-endian): `[dim: u32][dim elements]`, repeated.
//! SIFT1M ships as `sift_base.fvecs` (1M x 128), `sift_query.fvecs`
//! (10k x 128) and `sift_groundtruth.ivecs` (10k x 100, neighbor ids
//! sorted by increasing L2 distance).

use std::io;
use std::path::{Path, PathBuf};

/// Dense row-major f32 vectors.
pub struct FloatVectors {
    pub dim: usize,
    pub count: usize,
    data: Vec<f32>,
}

impl FloatVectors {
    pub fn from_raw(dim: usize, data: Vec<f32>) -> Self {
        assert!(dim > 0 && data.len() % dim == 0);
        Self { dim, count: data.len() / dim, data }
    }

    #[inline]
    pub fn get(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[f32]> {
        self.data.chunks_exact(self.dim)
    }

    /// Copy of the first `n` rows (e.g. to benchmark on a query subset).
    pub fn head(&self, n: usize) -> FloatVectors {
        let n = n.min(self.count);
        FloatVectors::from_raw(self.dim, self.data[..n * self.dim].to_vec())
    }
}

/// Dense row-major u32 id lists (ground-truth neighbors).
pub struct IdLists {
    pub dim: usize,
    pub count: usize,
    data: Vec<u32>,
}

impl IdLists {
    #[inline]
    pub fn get(&self, i: usize) -> &[u32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

fn record_layout(bytes: &[u8], path: &Path) -> io::Result<(usize, usize)> {
    let err =
        |msg: String| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {msg}", path.display()));
    if bytes.len() < 4 {
        return Err(err("file too small".to_string()));
    }
    let dim = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    if dim == 0 || dim > 65_536 {
        return Err(err(format!("implausible dimension {dim}")));
    }
    let rec = 4 + dim * 4;
    if !bytes.len().is_multiple_of(rec) {
        return Err(err(format!("size {} is not a multiple of record size {rec}", bytes.len())));
    }
    Ok((dim, bytes.len() / rec))
}

fn parse_records(
    bytes: &[u8],
    path: &Path,
    mut push: impl FnMut(&[u8]),
) -> io::Result<(usize, usize)> {
    let (dim, count) = record_layout(bytes, path)?;
    let rec = 4 + dim * 4;
    for r in 0..count {
        let off = r * rec;
        let d = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        if d != dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: record {r} has dim {d}, expected {dim}", path.display()),
            ));
        }
        push(&bytes[off + 4..off + rec]);
    }
    Ok((dim, count))
}

pub fn read_fvecs(path: &Path) -> io::Result<FloatVectors> {
    let bytes = std::fs::read(path)?;
    let mut data = Vec::with_capacity(bytes.len() / 4);
    let (dim, count) = parse_records(&bytes, path, |payload| {
        data.extend(payload.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
    })?;
    Ok(FloatVectors { dim, count, data })
}

pub fn read_ivecs(path: &Path) -> io::Result<IdLists> {
    let bytes = std::fs::read(path)?;
    let mut data = Vec::with_capacity(bytes.len() / 4);
    let (dim, count) = parse_records(&bytes, path, |payload| {
        data.extend(payload.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())));
    })?;
    Ok(IdLists { dim, count, data })
}

/// A TEXMEX-style dataset directory: `<prefix>_base.fvecs`,
/// `<prefix>_query.fvecs`, `<prefix>_groundtruth.ivecs`.
pub struct BenchDataset {
    pub base: FloatVectors,
    pub queries: FloatVectors,
    pub ground_truth: IdLists,
}

pub fn load_texmex(dir: &Path, prefix: &str) -> io::Result<BenchDataset> {
    let p = |suffix: &str| -> PathBuf { dir.join(format!("{prefix}_{suffix}")) };
    let base = read_fvecs(&p("base.fvecs"))?;
    let queries = read_fvecs(&p("query.fvecs"))?;
    let ground_truth = read_ivecs(&p("groundtruth.ivecs"))?;
    if base.dim != queries.dim {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("base dim {} != query dim {}", base.dim, queries.dim),
        ));
    }
    if ground_truth.count != queries.count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("gt count {} != query count {}", ground_truth.count, queries.count),
        ));
    }
    Ok(BenchDataset { base, queries, ground_truth })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fvecs_bytes(vectors: &[&[f32]]) -> Vec<u8> {
        let mut out = Vec::new();
        for v in vectors {
            out.extend((v.len() as u32).to_le_bytes());
            for x in *v {
                out.extend(x.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn parses_fvecs() {
        let path = std::env::temp_dir().join(format!("nvvec-fvecs-{}", std::process::id()));
        std::fs::write(&path, fvecs_bytes(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]])).unwrap();
        let v = read_fvecs(&path).unwrap();
        assert_eq!((v.dim, v.count), (3, 2));
        assert_eq!(v.get(1), &[4.0, 5.0, 6.0]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_truncated_file() {
        let path = std::env::temp_dir().join(format!("nvvec-fvecs-bad-{}", std::process::id()));
        let mut bytes = fvecs_bytes(&[&[1.0, 2.0, 3.0]]);
        bytes.pop();
        std::fs::write(&path, bytes).unwrap();
        assert!(read_fvecs(&path).is_err());
        std::fs::remove_file(&path).ok();
    }
}
