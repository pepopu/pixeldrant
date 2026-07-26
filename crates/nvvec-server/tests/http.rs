//! End-to-end HTTP tests: tiny dataset -> graph -> disk index -> codebook,
//! then real requests through the router.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use tower::ServiceExt;

use nvvec_core::dataset::FloatVectors;
use nvvec_index::{BuildParams, Sq8Codebook, build, write_disk_index};
use nvvec_server::{AppState, Scorer, SearchDefaults, build_router};

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn random_vectors(n: usize, dim: usize, seed: u64) -> FloatVectors {
    let mut state = seed;
    let data: Vec<f32> =
        (0..n * dim).map(|_| (splitmix64(&mut state) >> 40) as f32 / (1u64 << 24) as f32).collect();
    FloatVectors::from_raw(dim, data)
}

fn make_state(routing_sq8: bool) -> (Arc<AppState>, FloatVectors, std::path::PathBuf) {
    let (n, dim) = (800, 12);
    let base = random_vectors(n, dim, 71);
    let graph = build(&base, &BuildParams { r: 12, l_build: 30, alpha: 1.2 });
    let path = std::env::temp_dir().join(format!(
        "nvvec-http-{}-{}",
        std::process::id(),
        if routing_sq8 { "sq8" } else { "exact" }
    ));
    write_disk_index(&path, &base, &graph).unwrap();
    let scorer = if routing_sq8 {
        Scorer::Sq8(Sq8Codebook::train(&base))
    } else {
        Scorer::Exact(random_vectors(n, dim, 71))
    };
    let state = Arc::new(
        AppState::open(&path, scorer, SearchDefaults { k: 5, ef: 40, w: 2 }).unwrap(),
    );
    (state, base, path)
}

async fn body_json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn health_reports_shape() {
    let (state, _base, path) = make_state(true);
    let app = build_router(state);
    let resp = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp).await;
    assert_eq!(json["vectors"], 800);
    assert_eq!(json["dim"], 12);
    assert!(json["routing_memory_bytes"].as_u64().unwrap() > 0);
    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn search_finds_self_with_exact_distances() {
    for routing_sq8 in [false, true] {
        let (state, base, path) = make_state(routing_sq8);
        let app = build_router(state);
        let query: Vec<f32> = base.get(123).to_vec();
        let req_body = serde_json::json!({ "vector": query, "k": 3 }).to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/search")
                    .method("POST")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(req_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["ids"][0], 123, "routing_sq8={routing_sq8}");
        assert_eq!(json["distances"][0], 0.0, "self distance must be exactly zero");
        assert!(json["reads"].as_u64().unwrap() > 0);
        std::fs::remove_file(&path).ok();
    }
}

#[tokio::test]
async fn rejects_wrong_dimension() {
    let (state, _base, path) = make_state(true);
    let app = build_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/search")
                .method("POST")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"vector": [1.0, 2.0]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    std::fs::remove_file(&path).ok();
}
