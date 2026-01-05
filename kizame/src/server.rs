//! HTTP server mode for MeCrab API
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Provides a REST API for morphological analysis over HTTP.

use axum::{
    Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
};
use mecrab::MeCrab;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

/// Shared application state
#[derive(Clone)]
struct AppState {
    mecrab: Arc<MeCrab>,
}

/// Parse request body
#[derive(Debug, Deserialize)]
struct ParseRequest {
    /// Text to analyze
    text: String,
    /// Output format (optional, default: "default")
    #[allow(dead_code)]
    #[serde(default)]
    format: Option<String>,
    /// N-best paths (optional)
    #[serde(default)]
    nbest: Option<usize>,
}

/// Batch parse request
#[derive(Debug, Deserialize)]
struct BatchParseRequest {
    /// Texts to analyze
    texts: Vec<String>,
    /// Output format (optional)
    #[allow(dead_code)]
    #[serde(default)]
    format: Option<String>,
}

/// Parse response
#[derive(Debug, Serialize)]
struct ParseResponse {
    /// Analysis result
    result: String,
    /// Processing time in microseconds
    time_us: u64,
}

/// Batch parse response
#[derive(Debug, Serialize)]
struct BatchParseResponse {
    /// Analysis results
    results: Vec<String>,
    /// Total processing time in microseconds
    time_us: u64,
}

/// Error response
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

/// Health check response
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    dictionary_loaded: bool,
}

/// API error type
enum ApiError {
    Parse(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            ApiError::Parse(msg) => (StatusCode::BAD_REQUEST, msg),
        };

        let body = Json(ErrorResponse { error: message });

        (status, body).into_response()
    }
}

/// GET /health - Health check endpoint
async fn health(State(_state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        dictionary_loaded: true,
    })
}

/// POST /parse - Parse single text
async fn parse(
    State(state): State<AppState>,
    Json(req): Json<ParseRequest>,
) -> Result<Json<ParseResponse>, ApiError> {
    let start = std::time::Instant::now();

    // For now, just parse as plain text
    // TODO: Support different output formats properly
    let result = if let Some(_n) = req.nbest {
        // N-best parsing - for now just regular parse
        state
            .mecrab
            .parse(&req.text)
            .map_err(|e| ApiError::Parse(e.to_string()))?
            .to_string()
    } else {
        // Regular parsing
        state
            .mecrab
            .parse(&req.text)
            .map_err(|e| ApiError::Parse(e.to_string()))?
            .to_string()
    };

    let elapsed = start.elapsed();

    Ok(Json(ParseResponse {
        result,
        time_us: elapsed.as_micros() as u64,
    }))
}

/// POST /parse/batch - Parse multiple texts
async fn parse_batch(
    State(state): State<AppState>,
    Json(req): Json<BatchParseRequest>,
) -> Result<Json<BatchParseResponse>, ApiError> {
    let start = std::time::Instant::now();

    let refs: Vec<&str> = req.texts.iter().map(|s| s.as_str()).collect();
    let results = state
        .mecrab
        .parse_batch(&refs)
        .into_iter()
        .map(|r| r.map(|analysis| analysis.to_string()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ApiError::Parse(e.to_string()))?;

    let elapsed = start.elapsed();

    Ok(Json(BatchParseResponse {
        results,
        time_us: elapsed.as_micros() as u64,
    }))
}

/// Query parameters for wakati endpoint
#[derive(Debug, Deserialize)]
struct WakatiQuery {
    text: String,
}

/// GET /wakati - Wakati parsing (space-separated words)
async fn wakati(
    State(state): State<AppState>,
    Query(params): Query<WakatiQuery>,
) -> Result<impl IntoResponse, ApiError> {
    let result = state
        .mecrab
        .wakati(&params.text)
        .map_err(|e| ApiError::Parse(e.to_string()))?;
    Ok(result)
}

/// Build the router with all routes
fn create_router(mecrab: Arc<MeCrab>) -> Router {
    let state = AppState { mecrab };

    Router::new()
        .route("/health", get(health))
        .route("/parse", post(parse))
        .route("/parse/batch", post(parse_batch))
        .route("/wakati", get(wakati))
        .with_state(state)
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(
                    CorsLayer::new()
                        .allow_origin(Any)
                        .allow_methods(Any)
                        .allow_headers(Any),
                ),
        )
}

/// Run the HTTP server
pub async fn run_server(
    mecrab: MeCrab,
    addr: SocketAddr,
) -> Result<(), Box<dyn std::error::Error>> {
    let mecrab = Arc::new(mecrab);
    let app = create_router(mecrab);

    eprintln!("🚀 MeCrab server listening on http://{}", addr);
    eprintln!();
    eprintln!("API Endpoints:");
    eprintln!("  POST /parse          - Parse single text");
    eprintln!("  POST /parse/batch    - Parse multiple texts");
    eprintln!("  GET  /wakati?text=…  - Wakati (space-separated)");
    eprintln!("  GET  /health         - Health check");
    eprintln!();

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
