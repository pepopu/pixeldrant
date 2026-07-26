//! io_uring backend (Linux only) — M3a: per-thread ring, whole batch
//! submitted before reaping, optional O_DIRECT.
//!
//! Design notes:
//! - One `UringReader` per search thread is the intended usage; the internal
//!   mutex only exists to satisfy `BlockReader: Sync` and is uncontended in
//!   that pattern.
//! - O_DIRECT requires 4 KiB-aligned buffers; callers pass ordinary slices,
//!   so reads land in an internal aligned pool and are memcpy'd out (~100 ns
//!   per block, noise next to a ~10 us NVMe read). Registered buffers are a
//!   later optimization.
//! - After an I/O error the ring may still hold in-flight state; discard the
//!   reader rather than reusing it.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::ptr::NonNull;

use io_uring::{IoUring, opcode, types};
use parking_lot::Mutex;

use crate::{BLOCK_SIZE, BlockRead, BlockReader};

#[derive(Clone, Copy)]
pub struct UringOpts {
    /// Ring size; batches larger than this are split.
    pub queue_depth: u32,
    /// Open with O_DIRECT (bypass page cache — honest device numbers).
    pub direct: bool,
}

impl Default for UringOpts {
    fn default() -> Self {
        Self { queue_depth: 128, direct: false }
    }
}

/// 4 KiB-aligned buffer pool, one slot per queue-depth entry.
struct AlignedBufs {
    ptr: NonNull<u8>,
    blocks: usize,
}

impl AlignedBufs {
    fn new(blocks: usize) -> Self {
        assert!(blocks > 0);
        let layout = Self::layout(blocks);
        let ptr = unsafe { alloc_zeroed(layout) };
        Self { ptr: NonNull::new(ptr).expect("aligned alloc failed"), blocks }
    }

    fn layout(blocks: usize) -> Layout {
        Layout::from_size_align(blocks * BLOCK_SIZE, BLOCK_SIZE).unwrap()
    }

    fn slot_ptr(&mut self, i: usize) -> *mut u8 {
        assert!(i < self.blocks);
        unsafe { self.ptr.as_ptr().add(i * BLOCK_SIZE) }
    }

    fn slot(&self, i: usize) -> &[u8] {
        assert!(i < self.blocks);
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(i * BLOCK_SIZE), BLOCK_SIZE) }
    }
}

impl Drop for AlignedBufs {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr(), Self::layout(self.blocks)) }
    }
}

// Raw pointer to memory we exclusively own.
unsafe impl Send for AlignedBufs {}

struct RingState {
    ring: IoUring,
    bufs: AlignedBufs,
}

pub struct UringReader {
    file: File,
    state: Mutex<RingState>,
    queue_depth: usize,
    num_blocks: u64,
}

impl UringReader {
    pub fn open(path: &Path, opts: UringOpts) -> io::Result<Self> {
        assert!(opts.queue_depth > 0);
        let mut oo = std::fs::OpenOptions::new();
        oo.read(true);
        if opts.direct {
            oo.custom_flags(libc::O_DIRECT);
        }
        let file = oo.open(path)?;
        let len = file.metadata()?.len();
        if !len.is_multiple_of(BLOCK_SIZE as u64) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: size {len} is not a multiple of block size {BLOCK_SIZE}", path.display()),
            ));
        }
        let ring = IoUring::new(opts.queue_depth)?;
        let bufs = AlignedBufs::new(opts.queue_depth as usize);
        Ok(Self {
            file,
            state: Mutex::new(RingState { ring, bufs }),
            queue_depth: opts.queue_depth as usize,
            num_blocks: len / BLOCK_SIZE as u64,
        })
    }

    fn validate(&self, r: &BlockRead<'_>) -> io::Result<()> {
        if r.buf.len() != BLOCK_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("buffer len {} != block size {BLOCK_SIZE}", r.buf.len()),
            ));
        }
        if r.block_id >= self.num_blocks {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("block {} out of range ({} blocks)", r.block_id, self.num_blocks),
            ));
        }
        Ok(())
    }
}

impl BlockReader for UringReader {
    fn num_blocks(&self) -> u64 {
        self.num_blocks
    }

    fn read_block(&self, block_id: u64, buf: &mut [u8]) -> io::Result<()> {
        let mut batch = [BlockRead { block_id, buf }];
        self.read_batch(&mut batch)
    }

    fn read_batch(&self, batch: &mut [BlockRead<'_>]) -> io::Result<()> {
        for r in batch.iter() {
            self.validate(r)?;
        }
        let mut st = self.state.lock();
        let RingState { ring, bufs } = &mut *st;
        let fd = types::Fd(self.file.as_raw_fd());

        let mut start = 0;
        while start < batch.len() {
            let end = (start + self.queue_depth).min(batch.len());
            let chunk = &mut batch[start..end];
            for (i, r) in chunk.iter().enumerate() {
                let sqe = opcode::Read::new(fd, bufs.slot_ptr(i), BLOCK_SIZE as u32)
                    .offset(r.block_id * BLOCK_SIZE as u64)
                    .build()
                    .user_data(i as u64);
                // SAFETY: the buffer slot outlives the submission; the ring
                // is drained before the pool can be reused or dropped.
                unsafe {
                    ring.submission().push(&sqe).map_err(|_| {
                        io::Error::other("submission queue full despite chunking")
                    })?;
                }
            }
            ring.submit_and_wait(chunk.len())?;

            let completions: Vec<(u64, i32)> =
                ring.completion().map(|c| (c.user_data(), c.result())).collect();
            if completions.len() != chunk.len() {
                return Err(io::Error::other(format!(
                    "expected {} completions, reaped {}",
                    chunk.len(),
                    completions.len()
                )));
            }
            for (user_data, result) in completions {
                if result < 0 {
                    return Err(io::Error::from_raw_os_error(-result));
                }
                if result as usize != BLOCK_SIZE {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("short read: {result} of {BLOCK_SIZE} bytes"),
                    ));
                }
                let i = user_data as usize;
                chunk[i].buf.copy_from_slice(bufs.slot(i));
            }
            start += chunk.len();
        }
        Ok(())
    }
}

/// Lower-level asynchronous interface for the cross-query pipeline (M3b):
/// stage reads from many query state machines, flush them with one syscall,
/// and drain completions as they arrive — the ring never goes shallow just
/// because one query is between rounds.
///
/// Single-threaded by design (no locking): one executor per scheduler
/// thread. `user_data` layout: high 16 bits = buffer slot, low 48 = caller
/// tag. After an I/O error the executor must be discarded — in-flight slots
/// are not recovered.
pub struct UringExecutor {
    file: File,
    ring: IoUring,
    bufs: AlignedBufs,
    free: Vec<u16>,
    cq_buf: Vec<(u64, i32)>,
    num_blocks: u64,
}

const TAG_MASK: u64 = (1 << 48) - 1;

impl UringExecutor {
    pub fn open(path: &Path, opts: UringOpts) -> io::Result<Self> {
        assert!(opts.queue_depth > 0 && opts.queue_depth <= u16::MAX as u32);
        let mut oo = std::fs::OpenOptions::new();
        oo.read(true);
        if opts.direct {
            oo.custom_flags(libc::O_DIRECT);
        }
        let file = oo.open(path)?;
        let len = file.metadata()?.len();
        if !len.is_multiple_of(BLOCK_SIZE as u64) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: size {len} is not a multiple of block size {BLOCK_SIZE}", path.display()),
            ));
        }
        Ok(Self {
            file,
            ring: IoUring::new(opts.queue_depth)?,
            bufs: AlignedBufs::new(opts.queue_depth as usize),
            free: (0..opts.queue_depth as u16).rev().collect(),
            cq_buf: Vec::new(),
            num_blocks: len / BLOCK_SIZE as u64,
        })
    }

    pub fn capacity(&self) -> usize {
        self.bufs.blocks
    }

    pub fn free_slots(&self) -> usize {
        self.free.len()
    }

    pub fn num_blocks(&self) -> u64 {
        self.num_blocks
    }

    /// Stage one block read tagged with `tag` (< 2^48). Requires a free
    /// slot — size the queue depth to `concurrency * beam_width` so this
    /// never fails in the scheduler's steady state.
    pub fn stage_read(&mut self, block_id: u64, tag: u64) -> io::Result<()> {
        if block_id >= self.num_blocks {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("block {block_id} out of range ({} blocks)", self.num_blocks),
            ));
        }
        if tag > TAG_MASK {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "tag exceeds 48 bits"));
        }
        let Some(slot) = self.free.pop() else {
            return Err(io::Error::other("no free buffer slot (queue depth too small)"));
        };
        let sqe = opcode::Read::new(
            types::Fd(self.file.as_raw_fd()),
            self.bufs.slot_ptr(slot as usize),
            BLOCK_SIZE as u32,
        )
        .offset(block_id * BLOCK_SIZE as u64)
        .build()
        .user_data(((slot as u64) << 48) | tag);
        // SAFETY: the slot buffer stays allocated until the completion is
        // drained; slots are only recycled in drain().
        unsafe {
            self.ring.submission().push(&sqe).map_err(|_| {
                io::Error::other("submission queue full (stage/flush imbalance)")
            })?;
        }
        Ok(())
    }

    /// Submit everything staged since the last flush (one syscall).
    pub fn flush(&mut self) -> io::Result<usize> {
        self.ring.submit()
    }

    /// Wait for at least `min` completions (0 = poll only), then invoke
    /// `f(tag, block_bytes)` for every available completion. Slots are
    /// recycled after each callback. Returns completions processed.
    pub fn drain(&mut self, min: usize, mut f: impl FnMut(u64, &[u8])) -> io::Result<usize> {
        if min > 0 {
            self.ring.submit_and_wait(min)?;
        }
        let Self { ring, cq_buf, bufs, free, .. } = self;
        cq_buf.clear();
        for cqe in ring.completion() {
            cq_buf.push((cqe.user_data(), cqe.result()));
        }
        for &(user_data, result) in cq_buf.iter() {
            if result < 0 {
                return Err(io::Error::from_raw_os_error(-result));
            }
            if result as usize != BLOCK_SIZE {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("short read: {result} of {BLOCK_SIZE} bytes"),
                ));
            }
            let slot = (user_data >> 48) as u16;
            f(user_data & TAG_MASK, bufs.slot(slot as usize));
            free.push(slot);
        }
        Ok(cq_buf.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PreadReader;
    use std::path::PathBuf;

    fn patterned_file(name: &str, blocks: usize) -> PathBuf {
        let path = std::env::temp_dir().join(format!("nvvec-uring-{}-{name}", std::process::id()));
        let mut content = vec![0u8; BLOCK_SIZE * blocks];
        for (i, b) in content.iter_mut().enumerate() {
            *b = ((i / BLOCK_SIZE) % 251) as u8;
        }
        std::fs::write(&path, &content).unwrap();
        path
    }

    #[test]
    fn matches_pread_on_shuffled_batches() {
        let path = patterned_file("parity", 64);
        let uring = UringReader::open(&path, UringOpts::default()).unwrap();
        let pread = PreadReader::open(&path).unwrap();
        assert_eq!(uring.num_blocks(), 64);

        // deliberately unordered, with duplicates, larger than one hop's worth
        let ids: Vec<u64> = (0..200).map(|i| (i * 37 + 11) % 64).collect();
        let mut u_bufs = vec![vec![0u8; BLOCK_SIZE]; ids.len()];
        let mut batch: Vec<BlockRead> = ids
            .iter()
            .zip(u_bufs.iter_mut())
            .map(|(&block_id, buf)| BlockRead { block_id, buf })
            .collect();
        uring.read_batch(&mut batch).unwrap();
        drop(batch);

        let mut p_buf = vec![0u8; BLOCK_SIZE];
        for (&id, u_buf) in ids.iter().zip(&u_bufs) {
            pread.read_block(id, &mut p_buf).unwrap();
            assert_eq!(u_buf, &p_buf, "block {id} mismatch");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn direct_io_roundtrip() {
        let path = patterned_file("direct", 16);
        let uring = match UringReader::open(&path, UringOpts { queue_depth: 16, direct: true }) {
            Ok(r) => r,
            Err(e) => {
                // some filesystems (e.g. tmpfs) reject O_DIRECT — not a failure
                eprintln!("skipping O_DIRECT test: {e}");
                std::fs::remove_file(&path).ok();
                return;
            }
        };
        let mut buf = vec![0u8; BLOCK_SIZE];
        uring.read_block(7, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 7));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_out_of_range() {
        let path = patterned_file("range", 4);
        let uring = UringReader::open(&path, UringOpts::default()).unwrap();
        let mut buf = vec![0u8; BLOCK_SIZE];
        assert!(uring.read_block(4, &mut buf).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn executor_stage_flush_drain() {
        let path = patterned_file("exec", 32);
        let mut exec =
            UringExecutor::open(&path, UringOpts { queue_depth: 8, direct: false }).unwrap();
        assert_eq!(exec.capacity(), 8);

        // two waves of interleaved reads, tags identify the block
        let mut seen = Vec::new();
        for wave in 0..2u64 {
            for i in 0..8u64 {
                let block = (wave * 8 + i) * 2 % 32;
                exec.stage_read(block, block).unwrap();
            }
            assert_eq!(exec.free_slots(), 0);
            exec.flush().unwrap();
            let drained = exec
                .drain(8, |tag, bytes| {
                    assert!(bytes.iter().all(|&b| b == (tag % 251) as u8), "block {tag}");
                    seen.push(tag);
                })
                .unwrap();
            assert_eq!(drained, 8);
            assert_eq!(exec.free_slots(), 8);
        }
        assert_eq!(seen.len(), 16);
        std::fs::remove_file(&path).ok();
    }
}
