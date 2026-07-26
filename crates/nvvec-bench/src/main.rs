//! Evaluation harness CLI. Every index (ours and baselines) reports through
//! the same CSV schema so results stay comparable:
//! `method,params,recall_at_k,qps,p50_us,p99_us`

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use nvvec_core::{brute, dataset, eval};

#[derive(Parser)]
#[command(name = "nvvec-bench", about = "recall/QPS/latency harness for nvvec and baselines")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Validate a dataset directory and print its shape.
    Check {
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long, default_value = "sift")]
        prefix: String,
    },
    /// Exact brute-force scan: recall sanity check (expect ~1.0) and the
    /// honest throughput floor any index must beat.
    Brute {
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long, default_value = "sift")]
        prefix: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Limit query count (a full 10k-query exact scan takes minutes).
        #[arg(long)]
        queries: Option<usize>,
        /// Append a CSV result row to this file.
        #[arg(long)]
        csv: Option<PathBuf>,
    },
    /// In-memory Vamana index (M1): build (or load cached graph), then
    /// sweep ef for recall/QPS/latency.
    Vamana {
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long, default_value = "sift")]
        prefix: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Max out-degree.
        #[arg(long, default_value_t = 32)]
        r: usize,
        /// Candidate list size during construction.
        #[arg(long, default_value_t = 100)]
        l_build: usize,
        /// Pruning slack.
        #[arg(long, default_value_t = 1.2)]
        alpha: f32,
        /// Comma-separated ef values to sweep at query time.
        #[arg(long, default_value = "10,20,40,60,80,120,200,400")]
        efs: String,
        /// Graph cache: load it if it exists, otherwise build and save here.
        #[arg(long)]
        graph: Option<PathBuf>,
        /// Queries used for the single-threaded latency measurement.
        #[arg(long, default_value_t = 1000)]
        latency_queries: usize,
        /// Append CSV result rows to this file.
        #[arg(long)]
        csv: Option<PathBuf>,
    },
    /// On-disk index (M2): pack vectors+graph into 4 KiB blocks, search via
    /// the BlockReader backend, sweep ef and beam width w.
    Disk {
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long, default_value = "sift")]
        prefix: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Graph cache produced by the vamana command (required).
        #[arg(long)]
        graph: PathBuf,
        /// Disk index file; created from the graph if missing.
        #[arg(long)]
        index: PathBuf,
        /// Comma-separated ef values.
        #[arg(long, default_value = "40,60,80,120,200")]
        efs: String,
        /// I/O backend: pread (portable sync) or uring (Linux io_uring).
        #[arg(long, default_value = "pread")]
        backend: String,
        /// Open the index with O_DIRECT (uring backend, Linux only).
        #[arg(long, default_value_t = false)]
        direct: bool,
        /// io_uring queue depth.
        #[arg(long, default_value_t = 128)]
        queue_depth: u32,
        /// Comma-separated beam widths (blocks read per I/O round).
        #[arg(long, default_value = "1,4,8")]
        ws: String,
        /// Limit query count (slow/cold disks can't take the full 10k).
        #[arg(long)]
        queries: Option<usize>,
        #[arg(long, default_value_t = 1000)]
        latency_queries: usize,
        #[arg(long)]
        csv: Option<PathBuf>,
    },
    /// Cross-query pipelined search (M3b, Linux io_uring): one scheduler
    /// thread, N query state machines sharing one deep ring.
    Pipeline {
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long, default_value = "sift")]
        prefix: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Graph cache produced by the vamana command.
        #[arg(long)]
        graph: PathBuf,
        /// Disk index file; created from the graph if missing.
        #[arg(long)]
        index: PathBuf,
        #[arg(long, default_value_t = 80)]
        ef: usize,
        /// Beam width per query per round.
        #[arg(long, default_value_t = 4)]
        w: usize,
        /// Comma-separated in-flight query counts to sweep.
        #[arg(long, default_value = "1,8,16,32,64")]
        concurrency: String,
        /// Open the index with O_DIRECT.
        #[arg(long, default_value_t = false)]
        direct: bool,
        /// Routing scorer: exact (f32 vectors in RAM) or sq8 (M4 quantized
        /// codes, 1/4 RAM; returned distances stay exact via block rerank).
        #[arg(long, default_value = "exact")]
        routing: String,
        /// Limit query count.
        #[arg(long)]
        queries: Option<usize>,
        #[arg(long)]
        csv: Option<PathBuf>,
    },
    /// Convert a TEXMEX .bvecs file (u8 vectors, e.g. BIGANN) to .fvecs,
    /// optionally truncating to the first N vectors.
    Bvecs2fvecs {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        count: Option<usize>,
    },
    /// Train an SQ8 codebook from a dataset and save it (for nvvec-server).
    TrainSq8 {
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long, default_value = "sift")]
        prefix: String,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Check { data_dir, prefix } => check(&data_dir, &prefix),
        Cmd::Brute { data_dir, prefix, k, queries, csv } => {
            brute_cmd(&data_dir, &prefix, k, queries, csv.as_deref())
        }
        Cmd::Vamana {
            data_dir,
            prefix,
            k,
            r,
            l_build,
            alpha,
            efs,
            graph,
            latency_queries,
            csv,
        } => vamana_cmd(VamanaArgs {
            data_dir,
            prefix,
            k,
            r,
            l_build,
            alpha,
            efs,
            graph,
            latency_queries,
            csv,
        }),
        Cmd::Disk {
            data_dir,
            prefix,
            k,
            graph,
            index,
            backend,
            direct,
            queue_depth,
            efs,
            ws,
            queries,
            latency_queries,
            csv,
        } => disk_cmd(DiskArgs {
            data_dir,
            prefix,
            k,
            graph,
            index,
            backend,
            direct,
            queue_depth,
            efs,
            ws,
            queries,
            latency_queries,
            csv,
        }),
        Cmd::Pipeline { data_dir, prefix, k, graph, index, ef, w, concurrency, direct, routing, queries, csv } => {
            pipeline_cmd(&data_dir, &prefix, k, &graph, &index, ef, w, &concurrency, direct, &routing, queries, csv.as_deref())
        }
        Cmd::Bvecs2fvecs { input, output, count } => bvecs2fvecs(&input, &output, count),
        Cmd::TrainSq8 { data_dir, prefix, out } => {
            use nvvec_core::scorer::RouteScorer as _;
            let base = dataset::read_fvecs(&data_dir.join(format!("{prefix}_base.fvecs")))?;
            let t = Instant::now();
            let cb = nvvec_index::Sq8Codebook::train(&base);
            cb.save(&out)?;
            println!(
                "sq8 codebook: {} x {} -> {} ({:.1} MB vs {:.1} MB raw) in {:.1?}",
                cb.count,
                cb.dim,
                out.display(),
                cb.memory_bytes() as f64 / 1e6,
                (base.count * base.dim * 4) as f64 / 1e6,
                t.elapsed()
            );
            Ok(())
        }
    }
}

struct DiskArgs {
    data_dir: PathBuf,
    prefix: String,
    k: usize,
    graph: PathBuf,
    index: PathBuf,
    backend: String,
    direct: bool,
    queue_depth: u32,
    efs: String,
    ws: String,
    queries: Option<usize>,
    latency_queries: usize,
    csv: Option<PathBuf>,
}

/// Open the chosen I/O backend. Each call returns a fresh reader so callers
/// can hold one per thread (the uring backend wants an uncontended ring).
fn open_reader(
    backend: &str,
    path: &Path,
    direct: bool,
    queue_depth: u32,
) -> Result<Box<dyn nvvec_io::BlockReader>> {
    match backend {
        "pread" => {
            if direct {
                bail!("--direct is only supported by the uring backend");
            }
            Ok(Box::new(nvvec_io::PreadReader::open(path)?))
        }
        "uring" => open_uring(path, direct, queue_depth),
        other => bail!("unknown backend '{other}' (expected pread or uring)"),
    }
}

#[cfg(target_os = "linux")]
fn open_uring(path: &Path, direct: bool, queue_depth: u32) -> Result<Box<dyn nvvec_io::BlockReader>> {
    let opts = nvvec_io::UringOpts { queue_depth, direct };
    Ok(Box::new(nvvec_io::UringReader::open(path, opts)?))
}

#[cfg(not(target_os = "linux"))]
fn open_uring(_: &Path, _: bool, _: u32) -> Result<Box<dyn nvvec_io::BlockReader>> {
    bail!("the uring backend is Linux-only; run under WSL2 or Linux")
}

fn disk_cmd(args: DiskArgs) -> Result<()> {
    use nvvec_index::{DiskIndex, SearchScratch};
    use rayon::prelude::*;

    let DiskArgs {
        data_dir,
        prefix,
        k,
        graph: graph_path,
        index: index_path,
        backend,
        direct,
        queue_depth,
        efs,
        ws,
        queries,
        latency_queries,
        csv,
    } = args;
    let (data_dir, prefix) = (data_dir.as_path(), prefix.as_str());
    let (graph_path, index_path) = (graph_path.as_path(), index_path.as_path());
    let (efs, ws) = (efs.as_str(), ws.as_str());
    let csv = csv.as_deref();

    let ds = load(data_dir, prefix)?;
    ensure_index(&ds, graph_path, index_path)?;
    let index = DiskIndex::with_reader(open_reader(&backend, index_path, direct, queue_depth)?)
        .context("opening disk index")?;
    if index.meta.n != ds.base.count as u64 {
        bail!("index has {} nodes, dataset has {}", index.meta.n, ds.base.count);
    }

    let method = format!("disk-{backend}{}", if direct { "-direct" } else { "" });
    let params_tag = format!("R{}", index.meta.r);
    let nq = queries.unwrap_or(ds.queries.count).min(ds.queries.count);
    let query_subset = ds.queries.head(nq);
    for w in parse_efs(ws)? {
        for ef in parse_efs(efs)? {
            if ef < k {
                continue;
            }
            let t = Instant::now();
            let out: Vec<(Vec<u32>, usize)> = (0..nq)
                .collect::<Vec<_>>()
                .par_chunks(64)
                .map_init(
                    // per-thread reader: the uring backend wants an
                    // uncontended ring per thread
                    || {
                        let reader = open_reader(&backend, index_path, direct, queue_depth)
                            .expect("open backend");
                        let idx = DiskIndex::with_reader(reader).expect("open index");
                        (idx, SearchScratch::new(ds.base.count), Vec::new())
                    },
                    |(thread_index, scratch, io_bufs), chunk| {
                        chunk
                            .iter()
                            .map(|&qi| {
                                let (res, reads) = thread_index
                                    .search(&ds.base, query_subset.get(qi), k, ef, w, scratch, io_bufs)
                                    .expect("disk read failed");
                                (res.into_iter().map(|(_, id)| id).collect::<Vec<u32>>(), reads)
                            })
                            .collect::<Vec<_>>()
                    },
                )
                .flatten()
                .collect();
            let qps = nq as f64 / t.elapsed().as_secs_f64();
            let ids: Vec<Vec<u32>> = out.iter().map(|(ids, _)| ids.clone()).collect();
            let mean_reads =
                out.iter().map(|&(_, r)| r).sum::<usize>() as f64 / out.len() as f64;
            let recall = eval::recall_at_k(&ids, &ds.ground_truth, k);

            let mut scratch = SearchScratch::new(ds.base.count);
            let mut io_bufs = Vec::new();
            let mut samples = Vec::with_capacity(latency_queries.min(nq));
            for qi in 0..latency_queries.min(nq) {
                let t = Instant::now();
                index
                    .search(&ds.base, query_subset.get(qi), k, ef, w, &mut scratch, &mut io_bufs)
                    .expect("disk read failed");
                samples.push(t.elapsed().as_secs_f64() * 1e6);
            }
            let lat = eval::latency_stats(samples);

            println!(
                "[{method}] w={w}  ef={ef:4}  recall@{k}={recall:.4}  qps={qps:8.0}  p50={:6.0}us  p99={:6.0}us  reads={mean_reads:.0}",
                lat.p50_us, lat.p99_us
            );
            if let Some(path) = csv {
                append_csv(
                    path,
                    &method,
                    &format!("{params_tag}-ef{ef}-w{w}-qd{queue_depth}"),
                    k,
                    recall,
                    qps,
                    Some(&lat),
                )?;
            }
        }
    }
    Ok(())
}

struct VamanaArgs {
    data_dir: PathBuf,
    prefix: String,
    k: usize,
    r: usize,
    l_build: usize,
    alpha: f32,
    efs: String,
    graph: Option<PathBuf>,
    latency_queries: usize,
    csv: Option<PathBuf>,
}

fn vamana_cmd(args: VamanaArgs) -> Result<()> {
    use nvvec_index::{BuildParams, SearchScratch, VamanaGraph, beam_search};
    use rayon::prelude::*;

    let ds = load(&args.data_dir, &args.prefix)?;
    if args.k > ds.ground_truth.dim {
        bail!("k={} exceeds ground truth depth {}", args.k, ds.ground_truth.dim);
    }

    let graph = match &args.graph {
        Some(path) if path.exists() => {
            let g = VamanaGraph::load(path).context("loading graph cache")?;
            if g.num_nodes() != ds.base.count {
                bail!(
                    "graph cache has {} nodes but dataset has {} — delete {} to rebuild",
                    g.num_nodes(),
                    ds.base.count,
                    path.display()
                );
            }
            eprintln!("loaded graph cache {} (avg degree {:.1})", path.display(), g.avg_degree());
            g
        }
        maybe_path => {
            let params = BuildParams { r: args.r, l_build: args.l_build, alpha: args.alpha };
            eprintln!(
                "building vamana: R={}, L={}, alpha={} ({} threads)...",
                params.r,
                params.l_build,
                params.alpha,
                rayon::current_num_threads()
            );
            let t = Instant::now();
            let g = nvvec_index::build(&ds.base, &params);
            eprintln!("build: {:.1?} (avg degree {:.1})", t.elapsed(), g.avg_degree());
            if let Some(path) = maybe_path {
                g.save(path).context("saving graph cache")?;
                eprintln!("graph cached to {}", path.display());
            }
            g
        }
    };

    let params_tag = format!("R{}-L{}-a{}", args.r, args.l_build, args.alpha);
    let nq = ds.queries.count;
    for ef in parse_efs(&args.efs)? {
        if ef < args.k {
            continue;
        }
        // Batch throughput: all queries across all threads.
        let t = Instant::now();
        let ids: Vec<Vec<u32>> = (0..nq)
            .collect::<Vec<_>>()
            .par_chunks(64)
            .map_init(
                || SearchScratch::new(ds.base.count),
                |scratch, chunk| {
                    chunk
                        .iter()
                        .map(|&qi| {
                            beam_search(
                                &ds.base,
                                ds.queries.get(qi),
                                graph.medoid,
                                ef,
                                |id, buf| {
                                    buf.clear();
                                    buf.extend_from_slice(graph.neighbors_of(id));
                                },
                                scratch,
                            );
                            scratch.top_k(args.k).map(|(_, id)| id).collect::<Vec<u32>>()
                        })
                        .collect::<Vec<_>>()
                },
            )
            .flatten()
            .collect();
        let qps = nq as f64 / t.elapsed().as_secs_f64();
        let recall = eval::recall_at_k(&ids, &ds.ground_truth, args.k);

        // Single-threaded latency + hop count (hops = disk reads in M2).
        let mut scratch = SearchScratch::new(ds.base.count);
        let mut samples = Vec::with_capacity(args.latency_queries.min(nq));
        let mut hops_total = 0usize;
        for qi in 0..args.latency_queries.min(nq) {
            let t = Instant::now();
            hops_total += beam_search(
                &ds.base,
                ds.queries.get(qi),
                graph.medoid,
                ef,
                |id, buf| {
                    buf.clear();
                    buf.extend_from_slice(graph.neighbors_of(id));
                },
                &mut scratch,
            );
            samples.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let lat = eval::latency_stats(samples);
        let mean_hops = hops_total as f64 / args.latency_queries.min(nq) as f64;

        println!(
            "ef={ef:4}  recall@{}={recall:.4}  qps={qps:8.0}  p50={:6.0}us  p99={:6.0}us  hops={mean_hops:.0}",
            args.k, lat.p50_us, lat.p99_us
        );
        if let Some(path) = &args.csv {
            append_csv(
                path,
                "vamana",
                &format!("{params_tag}-ef{ef}"),
                args.k,
                recall,
                qps,
                Some(&lat),
            )?;
        }
    }
    if let Some(path) = &args.csv {
        println!("csv rows appended to {}", path.display());
    }
    Ok(())
}

/// Write the disk index from the graph cache if it doesn't exist yet.
fn ensure_index(ds: &dataset::BenchDataset, graph_path: &Path, index_path: &Path) -> Result<()> {
    use nvvec_index::{VamanaGraph, write_disk_index};
    if index_path.exists() {
        return Ok(());
    }
    if !graph_path.exists() {
        bail!(
            "graph cache {} not found — build it first with the vamana command",
            graph_path.display()
        );
    }
    let graph = VamanaGraph::load(graph_path).context("loading graph cache")?;
    if graph.num_nodes() != ds.base.count {
        bail!("graph has {} nodes, dataset has {}", graph.num_nodes(), ds.base.count);
    }
    let t = Instant::now();
    let meta = write_disk_index(index_path, &ds.base, &graph).context("writing disk index")?;
    let bytes = std::fs::metadata(index_path)?.len();
    eprintln!(
        "disk index written: {} ({:.1} MB, {} B/node, {}/block) in {:.1?}",
        index_path.display(),
        bytes as f64 / 1e6,
        meta.node_len,
        meta.nodes_per_block,
        t.elapsed()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn pipeline_cmd(
    data_dir: &Path,
    prefix: &str,
    k: usize,
    graph_path: &Path,
    index_path: &Path,
    ef: usize,
    w: usize,
    concurrency: &str,
    direct: bool,
    routing: &str,
    queries: Option<usize>,
    csv: Option<&Path>,
) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use nvvec_core::scorer::RouteScorer;
        use nvvec_index::{DiskIndex, Sq8Codebook};

        let ds = load(data_dir, prefix)?;
        if ef < k {
            bail!("ef must be >= k");
        }
        ensure_index(&ds, graph_path, index_path)?;
        let meta = DiskIndex::open(index_path).context("opening disk index")?.meta;
        if meta.n != ds.base.count as u64 {
            bail!("index has {} nodes, dataset has {}", meta.n, ds.base.count);
        }

        let nq = queries.unwrap_or(ds.queries.count).min(ds.queries.count);
        let query_subset = ds.queries.head(nq);
        let method =
            format!("pipeline-uring{}-{routing}", if direct { "-direct" } else { "" });
        let cs = parse_efs(concurrency)?;

        match routing {
            "exact" => {
                eprintln!(
                    "routing: exact f32 in RAM ({:.0} MB resident)",
                    RouteScorer::memory_bytes(&ds.base) as f64 / 1e6
                );
                pipeline_sweep(&ds.base, &meta, index_path, &ds, &query_subset, k, ef, w, &cs, direct, &method, csv)
            }
            "sq8" => {
                let t = Instant::now();
                let codebook = Sq8Codebook::train(&ds.base);
                eprintln!(
                    "routing: sq8 codes ({:.0} MB resident vs {:.0} MB raw, trained in {:.1?})",
                    codebook.memory_bytes() as f64 / 1e6,
                    RouteScorer::memory_bytes(&ds.base) as f64 / 1e6,
                    t.elapsed()
                );
                pipeline_sweep(&codebook, &meta, index_path, &ds, &query_subset, k, ef, w, &cs, direct, &method, csv)
            }
            other => bail!("unknown routing '{other}' (expected exact or sq8)"),
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (data_dir, prefix, k, graph_path, index_path, ef, w, concurrency, direct, routing, queries, csv);
        bail!("the pipeline command requires Linux (io_uring); run under WSL2 or Linux")
    }
}

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn pipeline_sweep<S: nvvec_core::scorer::RouteScorer>(
    scorer: &S,
    meta: &nvvec_index::DiskIndexMeta,
    index_path: &Path,
    ds: &dataset::BenchDataset,
    query_subset: &dataset::FloatVectors,
    k: usize,
    ef: usize,
    w: usize,
    cs: &[usize],
    direct: bool,
    method: &str,
    csv: Option<&Path>,
) -> Result<()> {
    use nvvec_index::pipeline::{PipelineParams, search_pipelined};

    let nq = query_subset.count;
    for &c in cs {
        let opts = nvvec_io::UringOpts { queue_depth: (c * w).max(2) as u32, direct };
        let mut exec = nvvec_io::UringExecutor::open(index_path, opts)?;
        let t = Instant::now();
        let out = search_pipelined(
            meta,
            &mut exec,
            scorer,
            query_subset,
            &PipelineParams { k, ef, w, concurrency: c },
        )?;
        let secs = t.elapsed().as_secs_f64();
        let qps = nq as f64 / secs;
        let ids: Vec<Vec<u32>> =
            out.iter().map(|(res, _)| res.iter().map(|&(_, id)| id).collect()).collect();
        let recall = eval::recall_at_k(&ids, &ds.ground_truth, k);
        let mean_reads = out.iter().map(|&(_, r)| r).sum::<usize>() as f64 / out.len() as f64;
        let iops = mean_reads * qps;

        println!(
            "[{method}] c={c:3}  w={w}  ef={ef}  recall@{k}={recall:.4}  qps={qps:8.0}  reads={mean_reads:.0}  iops={iops:.0}  (1 thread)",
        );
        if let Some(path) = csv {
            append_csv(path, method, &format!("R{}-ef{ef}-w{w}-c{c}", meta.r), k, recall, qps, None)?;
        }
    }
    Ok(())
}

fn bvecs2fvecs(input: &Path, output: &Path, count: Option<usize>) -> Result<()> {
    use std::io::{BufReader, BufWriter, Read, Write};
    let mut r = BufReader::with_capacity(1 << 20, std::fs::File::open(input)?);
    let mut w = BufWriter::with_capacity(1 << 20, std::fs::File::create(output)?);
    let limit = count.unwrap_or(usize::MAX);
    let mut written = 0usize;
    let mut dim_buf = [0u8; 4];
    let mut payload: Vec<u8> = Vec::new();
    while written < limit {
        match r.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;
        if dim == 0 || dim > 65_536 {
            bail!("implausible dimension {dim} at record {written}");
        }
        payload.resize(dim, 0);
        r.read_exact(&mut payload)?;
        w.write_all(&dim_buf)?;
        for &b in &payload {
            w.write_all(&(b as f32).to_le_bytes())?;
        }
        written += 1;
        if written % 1_000_000 == 0 {
            eprintln!("{written} vectors converted...");
        }
    }
    w.flush()?;
    println!("converted {written} vectors ({input:?} -> {output:?})");
    Ok(())
}

fn parse_efs(efs: &str) -> Result<Vec<usize>> {
    efs.split(',')
        .map(|s| s.trim().parse::<usize>().map_err(|e| anyhow::anyhow!("bad ef '{s}': {e}")))
        .collect()
}

fn load(data_dir: &Path, prefix: &str) -> Result<dataset::BenchDataset> {
    let t = Instant::now();
    let ds = dataset::load_texmex(data_dir, prefix)
        .with_context(|| format!("loading dataset '{prefix}' from {}", data_dir.display()))?;
    eprintln!(
        "loaded {prefix}: base {}x{}, {} queries, gt@{} ({:.1?})",
        ds.base.count,
        ds.base.dim,
        ds.queries.count,
        ds.ground_truth.dim,
        t.elapsed()
    );
    Ok(ds)
}

fn check(data_dir: &Path, prefix: &str) -> Result<()> {
    load(data_dir, prefix)?;
    println!("ok");
    Ok(())
}

fn brute_cmd(
    data_dir: &Path,
    prefix: &str,
    k: usize,
    queries: Option<usize>,
    csv: Option<&Path>,
) -> Result<()> {
    let ds = load(data_dir, prefix)?;
    if k > ds.ground_truth.dim {
        bail!("k={k} exceeds ground truth depth {}", ds.ground_truth.dim);
    }
    let nq = queries.unwrap_or(ds.queries.count).min(ds.queries.count);
    let query_subset = ds.queries.head(nq);

    let t = Instant::now();
    let results = brute::search(&ds.base, &query_subset, k);
    let secs = t.elapsed().as_secs_f64();
    let qps = nq as f64 / secs;

    let ids: Vec<Vec<u32>> = results
        .iter()
        .map(|r| r.iter().map(|&(_, id)| id).collect())
        .collect();
    let recall = eval::recall_at_k(&ids, &ds.ground_truth, k);

    println!("brute-force: recall@{k} = {recall:.4}  ({nq} queries in {secs:.2}s, {qps:.0} QPS)");
    if let Some(path) = csv {
        append_csv(path, "brute", &format!("q{nq}"), k, recall, qps, None)?;
        println!("csv row appended to {}", path.display());
    }
    Ok(())
}

fn append_csv(
    path: &Path,
    method: &str,
    params: &str,
    k: usize,
    recall: f64,
    qps: f64,
    latency: Option<&eval::LatencyStats>,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let is_new = !path.exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    if is_new {
        writeln!(f, "method,params,recall_at_{k},qps,p50_us,p99_us")?;
    }
    let (p50, p99) = match latency {
        Some(l) => (format!("{:.0}", l.p50_us), format!("{:.0}", l.p99_us)),
        None => (String::new(), String::new()),
    };
    writeln!(f, "{method},{params},{recall:.4},{qps:.0},{p50},{p99}")?;
    Ok(())
}
