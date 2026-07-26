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
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Check { data_dir, prefix } => check(&data_dir, &prefix),
        Cmd::Brute { data_dir, prefix, k, queries, csv } => {
            brute_cmd(&data_dir, &prefix, k, queries, csv.as_deref())
        }
    }
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
