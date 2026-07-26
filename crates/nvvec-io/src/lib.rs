//! Block-granular storage I/O abstraction.
//!
//! The disk index reads fixed 4 KiB blocks (one graph node = one block, per
//! the DiskANN layout). Everything above this layer talks in `block_id`s and
//! stays byte-identical across backends:
//!
//! - [`PreadReader`] — portable synchronous positional reads (Windows/macOS
//!   development, and the M2 correctness baseline on Linux);
//! - io_uring backend (M3, Linux) — will implement the same trait, submitting
//!   the whole batch before reaping completions.
//!
//! The batched call is the contract that matters: callers must batch all
//! reads of one beam-search hop into a single `read_batch` so the async
//! backend can keep the device queue deep. Code that loops over
//! `read_block` will work but will never be fast.

use std::fs::File;
use std::io;
use std::path::Path;

#[cfg(target_os = "linux")]
pub mod uring;
#[cfg(target_os = "linux")]
pub use uring::{UringOpts, UringReader};

/// Block size in bytes. 4 KiB matches both NVMe atomic read granularity and
/// the DiskANN node layout (vector + neighbor list packed in one block).
pub const BLOCK_SIZE: usize = 4096;

/// One block-read request: fill `buf` (exactly [`BLOCK_SIZE`]) from `block_id`.
pub struct BlockRead<'a> {
    pub block_id: u64,
    pub buf: &'a mut [u8],
}

pub trait BlockReader: Send + Sync {
    fn num_blocks(&self) -> u64;

    fn read_block(&self, block_id: u64, buf: &mut [u8]) -> io::Result<()>;

    /// Read a batch of blocks. Backends may reorder and parallelize; there is
    /// no ordering guarantee between requests in one batch.
    fn read_batch(&self, batch: &mut [BlockRead<'_>]) -> io::Result<()> {
        for r in batch.iter_mut() {
            self.read_block(r.block_id, r.buf)?;
        }
        Ok(())
    }
}

impl<T: BlockReader + ?Sized> BlockReader for Box<T> {
    fn num_blocks(&self) -> u64 {
        (**self).num_blocks()
    }

    fn read_block(&self, block_id: u64, buf: &mut [u8]) -> io::Result<()> {
        (**self).read_block(block_id, buf)
    }

    fn read_batch(&self, batch: &mut [BlockRead<'_>]) -> io::Result<()> {
        (**self).read_batch(batch)
    }
}

/// Portable synchronous backend using positional reads
/// (`pread` on Unix, `seek_read` on Windows). No shared file cursor, so it
/// is safe to call from many threads at once.
pub struct PreadReader {
    file: File,
    num_blocks: u64,
}

impl PreadReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if !len.is_multiple_of(BLOCK_SIZE as u64) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: size {len} is not a multiple of block size {BLOCK_SIZE}", path.display()),
            ));
        }
        Ok(Self { file, num_blocks: len / BLOCK_SIZE as u64 })
    }
}

impl BlockReader for PreadReader {
    fn num_blocks(&self) -> u64 {
        self.num_blocks
    }

    fn read_block(&self, block_id: u64, buf: &mut [u8]) -> io::Result<()> {
        if buf.len() != BLOCK_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("buffer len {} != block size {BLOCK_SIZE}", buf.len()),
            ));
        }
        if block_id >= self.num_blocks {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("block {block_id} out of range ({} blocks)", self.num_blocks),
            ));
        }
        read_exact_at(&self.file, buf, block_id * BLOCK_SIZE as u64)
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected end of file"));
            }
            Ok(n) => {
                offset += n as u64;
                let rest = buf;
                buf = &mut rest[n..];
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_file(name: &str, content: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("nvvec-io-{}-{name}", std::process::id()));
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn single_and_batch_reads() {
        // 3 blocks, filled with byte 1 / 2 / 3 respectively
        let mut content = vec![0u8; BLOCK_SIZE * 3];
        for (i, b) in content.iter_mut().enumerate() {
            *b = (i / BLOCK_SIZE) as u8 + 1;
        }
        let path = temp_file("roundtrip", &content);
        let reader = PreadReader::open(&path).unwrap();
        assert_eq!(reader.num_blocks(), 3);

        let mut buf = vec![0u8; BLOCK_SIZE];
        reader.read_block(2, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 3));

        let mut b0 = vec![0u8; BLOCK_SIZE];
        let mut b1 = vec![0u8; BLOCK_SIZE];
        let mut batch = [
            BlockRead { block_id: 1, buf: &mut b1[..] },
            BlockRead { block_id: 0, buf: &mut b0[..] },
        ];
        reader.read_batch(&mut batch).unwrap();
        assert!(b1.iter().all(|&b| b == 2));
        assert!(b0.iter().all(|&b| b == 1));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_unaligned_file() {
        let path = temp_file("unaligned", &vec![0u8; BLOCK_SIZE + 1]);
        assert!(PreadReader::open(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_bad_requests() {
        let path = temp_file("range", &vec![0u8; BLOCK_SIZE]);
        let reader = PreadReader::open(&path).unwrap();
        let mut buf = vec![0u8; BLOCK_SIZE];
        assert!(reader.read_block(1, &mut buf).is_err());
        assert!(reader.read_block(0, &mut buf[..BLOCK_SIZE - 1]).is_err());
        std::fs::remove_file(&path).ok();
    }
}
