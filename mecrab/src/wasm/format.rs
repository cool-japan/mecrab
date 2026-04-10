//! Output formatting helpers for the WASM bindings.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module provides pure functions that transform analysis results into
//! the string formats that JavaScript callers expect (JSON, CSV, wakati, etc.).
//! None of the functions here are `#[wasm_bindgen]` exports.

use crate::viterbi::ViterbiNode;

// ── JSON helpers ─────────────────────────────────────────────────────────────

/// Serialise a slice of [`ViterbiNode`]s into the `{"tokens":[...]}` JSON
/// string returned by `MeCrabWasm::parse`.
pub(super) fn format_json_output(path: &[ViterbiNode]) -> String {
    let tokens: Vec<serde_json::Value> = path
        .iter()
        .map(|node| {
            serde_json::json!({
                "surface": node.surface,
                "feature": node.feature,
                "pos_id":  node.pos_id,
                "wcost":   node.wcost,
            })
        })
        .collect();

    serde_json::json!({ "tokens": tokens }).to_string()
}

/// Serialise a slice of overlay surface-form strings into a JSON array string
/// (no external dependency — hand-built to avoid pulling in serde for a trivial
/// string list).
pub(super) fn format_overlay_surfaces(surfaces: &[String]) -> String {
    format!(
        "[{}]",
        surfaces
            .iter()
            .map(|s| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Build a JSON array string from a slice of static feature-flag names.
pub(super) fn format_features_list(features: &[&str]) -> String {
    format!(
        "[{}]",
        features
            .iter()
            .map(|f| format!("\"{}\"", f))
            .collect::<Vec<_>>()
            .join(",")
    )
}

// ── CSV helper ───────────────────────────────────────────────────────────────

/// Render a Viterbi path as tab-separated CSV lines.
///
/// Each line: `surface\tfeature\tstart_byte\tend_byte`
///
/// EOS tokens are excluded.  Returns an empty string if `path` contains only
/// EOS entries.
pub(super) fn format_csv_output(path: Vec<ViterbiNode>) -> String {
    let lines: Vec<String> = path
        .into_iter()
        .filter(|node| node.surface != "EOS")
        .map(|node| {
            format!(
                "{}\t{}\t{}\t{}",
                node.surface, node.feature, node.start_byte, node.end_byte
            )
        })
        .collect();

    lines.join("\n")
}

// ── Shared Morpheme builder ───────────────────────────────────────────────────

/// Convert a [`ViterbiNode`] into a [`crate::Morpheme`], setting optional
/// fields to their default (empty / `None`) values.
pub(super) fn node_to_morpheme(node: ViterbiNode) -> crate::Morpheme {
    crate::Morpheme {
        surface: node.surface,
        word_id: node.word_id,
        pos_id: node.pos_id,
        wcost: node.wcost,
        feature: node.feature,
        entities: Vec::new(),
        pronunciation: None,
        embedding: None,
        start_byte: node.start_byte,
        end_byte: node.end_byte,
    }
}
