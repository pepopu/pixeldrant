# nvvec (working name)

NVMe-native single-machine vector search engine, built from scratch.

**The bet**: modern NVMe delivers 1M+ random 4K IOPS, but mainstream vector
databases still treat disk as "mmap + page cache". With io_uring, a
DiskANN-style block layout, and in-memory quantized codes, a disk-resident
index should reach **~90% of in-memory HNSW performance at ~1/10 the RAM**.

See [ROADMAP.md](ROADMAP.md) for milestones. Non-goals: distributed anything,
complex filtering, Windows/macOS production support (dev works everywhere;
performance work targets Linux).

## Layout

- `crates/nvvec-core` — dataset loaders (fvecs/ivecs), distance kernels,
  exact brute-force search, recall/latency evaluation
- `crates/nvvec-io` — 4 KiB block I/O abstraction: portable pread backend
  today, io_uring backend in M3. The batched read API is the contract.
- `crates/nvvec-bench` — CLI harness; all methods report into one CSV schema
- `baselines/` — reference systems (hnswlib = in-memory upper bound)

## Quickstart

Get SIFT1M (~500 MB unpacked):

```bash
mkdir -p data
curl -o data/sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf data/sift.tar.gz -C data
```

Verify + run the exact-scan sanity check (recall must print ~1.0):

```bash
cargo run --release -p nvvec-bench -- check --data-dir data/sift
cargo run --release -p nvvec-bench -- brute --data-dir data/sift --queries 1000 --csv results/all.csv
```

Run the in-memory HNSW baseline (usearch — prebuilt wheels everywhere;
on Linux also run `bench_hnswlib.py`, same schema):

```bash
python -m pip install numpy usearch
python baselines/bench_usearch.py --data-dir data/sift --csv results/all.csv
```

Note: the SIFT1M mirror rejects plain HTTP; use `ftp://` as shown above.
