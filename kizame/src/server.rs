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

/// Output format variants supported by the /parse endpoint
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ParseFormat {
    /// Full token array with surface + feature (default)
    #[default]
    Json,
    /// Space-separated surface forms only
    Wakati,
    /// MeCab-style dump with index, pos_id and wcost per token
    Dump,
    /// CoNLL-U Universal Dependencies format
    Conllu,
}

impl ParseFormat {
    /// Parse a `?format=` query string value (case-insensitive)
    fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "json" => Some(Self::Json),
            "wakati" => Some(Self::Wakati),
            "dump" => Some(Self::Dump),
            "conllu" => Some(Self::Conllu),
            _ => None,
        }
    }
}

/// Parse request body
#[derive(Debug, Deserialize)]
struct ParseRequest {
    /// Text to analyze
    text: String,
    /// Output format: "json" | "wakati" | "dump"  (default: "json")
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
    /// Output format: "json" | "wakati" | "dump"  (default: "json")
    #[serde(default)]
    format: Option<String>,
}

/// Query parameters for format selection on the /parse endpoint
#[derive(Debug, Deserialize, Default)]
struct FormatQuery {
    /// Output format override via query string
    format: Option<String>,
}

/// Token representation for the JSON format response
#[derive(Debug, Serialize)]
struct TokenJson {
    surface: String,
    feature: String,
}

/// Typed parse response that adapts to the requested format
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ParseResponseBody {
    /// `format=json` – full token array
    Tokens {
        tokens: Vec<TokenJson>,
        time_us: u64,
    },
    /// `format=wakati` – space-separated words
    Wakati { wakati: String, time_us: u64 },
    /// `format=dump` – MeCab-style dump string
    Dump { dump: String, time_us: u64 },
    /// `format=conllu` – Universal Dependencies CoNLL-U text
    Conllu { conllu: String, time_us: u64 },
}

/// Batch parse response
#[derive(Debug, Serialize)]
struct BatchParseResponse {
    /// Analysis results (one per input text, format-dependent string)
    results: Vec<serde_json::Value>,
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
    BadRequest(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            ApiError::Parse(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
        };

        let body = Json(ErrorResponse { error: message });

        (status, body).into_response()
    }
}

/// Resolve the effective output format from JSON body field and query param.
/// Query param takes precedence over body field.
fn resolve_format(
    body_format: Option<&str>,
    query_format: Option<&str>,
) -> Result<ParseFormat, ApiError> {
    // Query param wins over body field
    let raw = query_format.or(body_format);
    match raw {
        None => Ok(ParseFormat::default()),
        Some(s) => ParseFormat::from_str_opt(s).ok_or_else(|| {
            ApiError::BadRequest(format!(
                "Unknown format '{}'. Valid values: json, wakati, dump, conllu",
                s
            ))
        }),
    }
}

/// Build the MeCab-style dump string from an `AnalysisResult`
fn build_dump_string(result: &mecrab::AnalysisResult) -> String {
    let mut out = String::new();
    for (i, m) in result.morphemes.iter().enumerate() {
        out.push_str(&format!(
            "[{}] {} (pos_id={}, wcost={})\t{}\n",
            i, m.surface, m.pos_id, m.wcost, m.feature
        ));
    }
    out.push_str("EOS\n");
    out
}

/// Build the wakati (space-separated) string from an `AnalysisResult`
fn build_wakati_string(result: &mecrab::AnalysisResult) -> String {
    result
        .morphemes
        .iter()
        .map(|m| m.surface.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Build token array from an `AnalysisResult`
fn build_tokens(result: &mecrab::AnalysisResult) -> Vec<TokenJson> {
    result
        .morphemes
        .iter()
        .map(|m| TokenJson {
            surface: m.surface.clone(),
            feature: m.feature.clone(),
        })
        .collect()
}

/// Build a CoNLL-U string from an [`AnalysisResult`].
fn build_conllu_string(result: &mecrab::AnalysisResult) -> String {
    result.to_conllu()
}

/// GET /health - Health check endpoint
async fn health(State(_state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        dictionary_loaded: true,
    })
}

/// Convert a single `AnalysisResult` into the appropriate `ParseResponseBody`
/// for the given format.
fn result_to_response_body(
    result: &mecrab::AnalysisResult,
    format: &ParseFormat,
    time_us: u64,
) -> ParseResponseBody {
    match format {
        ParseFormat::Json => ParseResponseBody::Tokens {
            tokens: build_tokens(result),
            time_us,
        },
        ParseFormat::Wakati => ParseResponseBody::Wakati {
            wakati: build_wakati_string(result),
            time_us,
        },
        ParseFormat::Dump => ParseResponseBody::Dump {
            dump: build_dump_string(result),
            time_us,
        },
        ParseFormat::Conllu => ParseResponseBody::Conllu {
            conllu: build_conllu_string(result),
            time_us,
        },
    }
}

/// Serialize an `AnalysisResult` as a `serde_json::Value` matching the
/// `ParseResponseBody` shape, minus the `time_us` field (used in N-best arrays).
fn result_to_json_value(result: &mecrab::AnalysisResult, format: &ParseFormat) -> serde_json::Value {
    match format {
        ParseFormat::Json => {
            let tokens: Vec<serde_json::Value> = result
                .morphemes
                .iter()
                .map(|m| {
                    serde_json::json!({
                        "surface": m.surface,
                        "feature": m.feature,
                    })
                })
                .collect();
            serde_json::json!({ "tokens": tokens })
        }
        ParseFormat::Wakati => serde_json::json!({ "wakati": build_wakati_string(result) }),
        ParseFormat::Dump => serde_json::json!({ "dump": build_dump_string(result) }),
        ParseFormat::Conllu => serde_json::json!({ "conllu": build_conllu_string(result) }),
    }
}

/// POST /parse - Parse single text
///
/// Supports an optional `?format=json|wakati|dump` query parameter and/or a
/// `"format"` field in the JSON body.  Query param takes precedence.
///
/// When `nbest` is set to a value > 1, the response is a JSON object:
/// `{ "results": [...], "time_us": N }` where each element of `results`
/// matches the single-parse response shape (without `time_us`).
async fn parse(
    State(state): State<AppState>,
    Query(query): Query<FormatQuery>,
    Json(req): Json<ParseRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let start = std::time::Instant::now();

    let format = resolve_format(req.format.as_deref(), query.format.as_deref())?;

    // Branch on N-best: if nbest > 1 return an array of results, otherwise
    // fall through to the standard single-parse path.
    if let Some(n) = req.nbest.filter(|&n| n > 1) {
        let paths = state
            .mecrab
            .parse_nbest(&req.text, n)
            .map_err(|e| ApiError::Parse(e.to_string()))?;

        let elapsed = start.elapsed();
        let time_us = elapsed.as_micros() as u64;

        let result_values: Vec<serde_json::Value> = paths
            .iter()
            .map(|(result, cost)| {
                let mut val = result_to_json_value(result, &format);
                // Attach the path cost alongside the morphological data.
                if let serde_json::Value::Object(ref mut map) = val {
                    map.insert("cost".to_owned(), serde_json::Value::Number((*cost).into()));
                }
                val
            })
            .collect();

        let nbest_body = serde_json::json!({
            "results": result_values,
            "time_us": time_us,
        });

        return Ok(Json(serde_json::Value::Object(
            nbest_body
                .as_object()
                .cloned()
                .unwrap_or_default(),
        ))
        .into_response());
    }

    // Standard single-parse path (nbest = None or Some(1)).
    let result = state
        .mecrab
        .parse(&req.text)
        .map_err(|e| ApiError::Parse(e.to_string()))?;

    let elapsed = start.elapsed();
    let time_us = elapsed.as_micros() as u64;

    let body = result_to_response_body(&result, &format, time_us);

    Ok(Json(body).into_response())
}

/// POST /parse/batch - Parse multiple texts
async fn parse_batch(
    State(state): State<AppState>,
    Query(query): Query<FormatQuery>,
    Json(req): Json<BatchParseRequest>,
) -> Result<Json<BatchParseResponse>, ApiError> {
    let start = std::time::Instant::now();

    let format = resolve_format(req.format.as_deref(), query.format.as_deref())?;

    let refs: Vec<&str> = req.texts.iter().map(|s| s.as_str()).collect();
    let analyses = state
        .mecrab
        .parse_batch(&refs)
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ApiError::Parse(e.to_string()))?;

    let results: Vec<serde_json::Value> = analyses
        .iter()
        .map(|analysis| match format {
            ParseFormat::Json => {
                let tokens: Vec<serde_json::Value> = analysis
                    .morphemes
                    .iter()
                    .map(|m| {
                        serde_json::json!({
                            "surface": m.surface,
                            "feature": m.feature,
                        })
                    })
                    .collect();
                serde_json::Value::Array(tokens)
            }
            ParseFormat::Wakati => serde_json::Value::String(build_wakati_string(analysis)),
            ParseFormat::Dump => serde_json::Value::String(build_dump_string(analysis)),
            ParseFormat::Conllu => serde_json::Value::String(build_conllu_string(analysis)),
        })
        .collect();

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

    eprintln!("MeCrab server listening on http://{}", addr);
    eprintln!();
    eprintln!("API Endpoints:");
    eprintln!("  POST /parse             - Parse single text");
    eprintln!("  POST /parse?format=...  - Parse with format: json|wakati|dump");
    eprintln!("  POST /parse/batch       - Parse multiple texts");
    eprintln!("  GET  /wakati?text=...   - Wakati (space-separated)");
    eprintln!("  GET  /health            - Health check");
    eprintln!();

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ParseRequest deserialization ─────────────────────────────────────────

    #[test]
    fn test_parse_request_nbest_none_deserializes() {
        let json = r#"{"text": "東京"}"#;
        let req: ParseRequest = serde_json::from_str(json).expect("deserialize");
        assert!(req.nbest.is_none());
    }

    #[test]
    fn test_parse_request_nbest_some_deserializes() {
        let json = r#"{"text": "東京", "nbest": 5}"#;
        let req: ParseRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.nbest, Some(5));
    }

    #[test]
    fn test_parse_request_nbest_one_deserializes() {
        let json = r#"{"text": "東京", "nbest": 1}"#;
        let req: ParseRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.nbest, Some(1));
    }

    // ── N-best branching logic ───────────────────────────────────────────────

    /// `nbest = None` must NOT trigger the N-best branch (`filter(|&n| n > 1)` → None).
    #[test]
    fn test_nbest_field_ignored_when_none() {
        let nbest: Option<usize> = None;
        // Mirrors the branch condition used in the parse handler.
        let triggers_nbest = nbest.filter(|&n| n > 1).is_some();
        assert!(!triggers_nbest, "nbest=None must not trigger N-best path");
    }

    /// `nbest = Some(1)` must NOT trigger the N-best branch — single-parse path.
    #[test]
    fn test_nbest_one_same_as_single() {
        let nbest: Option<usize> = Some(1);
        let triggers_nbest = nbest.filter(|&n| n > 1).is_some();
        assert!(
            !triggers_nbest,
            "nbest=Some(1) must use the single-parse path"
        );
    }

    /// `nbest = Some(2)` triggers the N-best branch.
    #[test]
    fn test_nbest_two_triggers_nbest_path() {
        let nbest: Option<usize> = Some(2);
        let triggers_nbest = nbest.filter(|&n| n > 1).is_some();
        assert!(triggers_nbest, "nbest=Some(2) must use the N-best path");
    }

    // ── resolve_format ───────────────────────────────────────────────────────

    #[test]
    fn test_resolve_format_defaults_to_json() {
        let fmt = resolve_format(None, None).ok();
        assert_eq!(fmt, Some(ParseFormat::Json));
    }

    #[test]
    fn test_resolve_format_body_wakati() {
        let fmt = resolve_format(Some("wakati"), None).ok();
        assert_eq!(fmt, Some(ParseFormat::Wakati));
    }

    #[test]
    fn test_resolve_format_query_wins_over_body() {
        // Body says "dump", query says "wakati" → query wins.
        let fmt = resolve_format(Some("dump"), Some("wakati")).ok();
        assert_eq!(fmt, Some(ParseFormat::Wakati));
    }

    #[test]
    fn test_resolve_format_unknown_returns_error() {
        let result = resolve_format(Some("unknown_fmt"), None);
        assert!(result.is_err(), "unknown format must produce an error");
    }

    // ── result_to_json_value ─────────────────────────────────────────────────

    fn make_analysis_result() -> mecrab::AnalysisResult {
        let morphemes = vec![mecrab::Morpheme {
            surface: "東京".to_owned(),
            word_id: 1,
            pos_id: 0,
            wcost: 0,
            feature: "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ".to_owned(),
            entities: vec![],
            pronunciation: None,
            embedding: None,
            start_byte: 0,
            end_byte: 6,
        }];
        mecrab::AnalysisResult::new(morphemes, mecrab::OutputFormat::Default)
    }

    #[test]
    fn test_result_to_json_value_json_format_has_tokens_key() {
        let result = make_analysis_result();
        let val = result_to_json_value(&result, &ParseFormat::Json);
        assert!(
            val.get("tokens").is_some(),
            "JSON format must produce a 'tokens' key"
        );
    }

    #[test]
    fn test_result_to_json_value_wakati_format_has_wakati_key() {
        let result = make_analysis_result();
        let val = result_to_json_value(&result, &ParseFormat::Wakati);
        assert!(
            val.get("wakati").is_some(),
            "Wakati format must produce a 'wakati' key"
        );
    }

    #[test]
    fn test_result_to_json_value_dump_format_has_dump_key() {
        let result = make_analysis_result();
        let val = result_to_json_value(&result, &ParseFormat::Dump);
        assert!(
            val.get("dump").is_some(),
            "Dump format must produce a 'dump' key"
        );
    }
}
