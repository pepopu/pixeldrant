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
        Cmd::Disk { data_dir, prefix, k, graph, index, efs, ws, queries, latency_queries, csv } => {
            disk_cmd(&data_dir, &prefix, k, &graph, &index, &efs, &ws, queries, latency_queries, csv.as_deref())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn disk_cmd(
    data_dir: &Path,
    prefix: &str,
    k: usize,
    graph_path: &Path,
    index_path: &Path,
    efs: &str,
    ws: &str,
    queries: Option<usize>,
    latency_queries: usize,
    csv: Option<&Path>,
) -> Result<()> {
    use nvvec_index::{DiskIndex, SearchScratch, VamanaGraph, write_disk_index};
    use rayon::prelude::*;

    let ds = load(data_dir, prefix)?;
    if !index_path.exists() {
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
    }
    let index = DiskIndex::open(index_path).context("opening disk index")?;
    if index.meta.n != ds.base.count as u64 {
        bail!("index has {} nodes, dataset has {}", index.meta.n, ds.base.count);
    }

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
                    || (SearchScratch::new(ds.base.count), Vec::new()),
                    |(scratch, io_bufs), chunk| {
                        chunk
                            .iter()
                            .map(|&qi| {
                                let (res, reads) = index
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
                "w={w}  ef={ef:4}  recall@{k}={recall:.4}  qps={qps:8.0}  p50={:6.0}us  p99={:6.0}us  reads={mean_reads:.0}",
                lat.p50_us, lat.p99_us
            );
            if let Some(path) = csv {
                append_csv(
                    path,
                    "disk-pread",
                    &format!("{params_tag}-ef{ef}-w{w}"),
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
