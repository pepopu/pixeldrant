use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use nvvec_server::{AppState, SearchDefaults, build_router, load_scorer};

#[derive(Parser)]
#[command(name = "nvvec-server", about = "HTTP search server over an nvvec disk index")]
struct Cli {
    /// Disk index file (written by nvvec-bench disk/pipeline).
    #[arg(long)]
    index: PathBuf,
    /// Routing scorer: exact | sq8.
    #[arg(long, default_value = "sq8")]
    routing: String,
    /// Base .fvecs (required for --routing exact).
    #[arg(long)]
    base: Option<PathBuf>,
    /// SQ8 codebook file (required for --routing sq8; see bench train-sq8).
    #[arg(long)]
    codes: Option<PathBuf>,
    #[arg(long, default_value = "0.0.0.0:7333")]
    listen: String,
    #[arg(long, default_value_t = 10)]
    k: usize,
    #[arg(long, default_value_t = 80)]
    ef: usize,
    #[arg(long, default_value_t = 4)]
    w: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let scorer = load_scorer(&cli.routing, cli.base.as_deref(), cli.codes.as_deref())?;
    let state = Arc::new(AppState::open(
        &cli.index,
        scorer,
        SearchDefaults { k: cli.k, ef: cli.ef, w: cli.w },
    )?);
    eprintln!(
        "nvvec-server: {} vectors x {} dims, routing memory {:.1} MB, listening on {}",
        state.index.meta.n,
        state.index.meta.dim,
        nvvec_core::scorer::RouteScorer::memory_bytes(&state.scorer) as f64 / 1e6,
        cli.listen
    );
    let listener = tokio::net::TcpListener::bind(&cli.listen).await?;
    axum::serve(listener, build_router(state)).await?;
    Ok(())
}
