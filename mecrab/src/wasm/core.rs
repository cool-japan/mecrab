//! `MeCrabWasm` struct definition and lifecycle methods.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module owns:
//! - The `MeCrabWasm` struct and its fields.
//! - Construction (`new` / `Default`).
//! - Overlay-dictionary management (`add_word`, `remove_word`, `overlay_size`).
//! - Status queries (`is_initialized`, `get_overlay_surfaces`).
//! - The `load_dictionary` wasm-bindgen entry point (delegates heavy lifting to
//!   [`super::loader::parse_blob`]).

use std::sync::Arc;

use wasm_bindgen::prelude::*;

/// JavaScript-friendly wrapper for MeCrab morphological analyzer.
#[wasm_bindgen]
pub struct MeCrabWasm {
    /// Loaded dictionary (populated after `loadDictionary` succeeds).
    pub(super) inner: Option<Arc<crate::dict::Dictionary>>,
    /// Standalone overlay dictionary for runtime word additions.
    ///
    /// This overlay is kept separate from the inner dictionary so that words
    /// added before `loadDictionary` is called (or across reloads) are not
    /// lost.
    pub(super) overlay: crate::dict::OverlayDictionary,
}

#[wasm_bindgen]
impl MeCrabWasm {
    /// Create a new MeCrab WASM instance.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: None,
            overlay: crate::dict::OverlayDictionary::new(),
        }
    }

    /// Load a packed dictionary blob.
    ///
    /// See the module-level documentation in [`super::loader`] for the blob layout.
    ///
    /// Returns an empty string on success, or an error description on failure.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw bytes of the packed MeCrab dictionary bundle
    #[wasm_bindgen(js_name = "loadDictionary")]
    pub fn load_dictionary(&mut self, data: &[u8]) -> String {
        match super::loader::parse_blob(data) {
            Ok(dict) => {
                // Mirror any pre-existing overlay words into the new dictionary.
                for surface in self.overlay.surfaces() {
                    for entry in self.overlay.lookup(&surface).into_iter().map(|e| {
                        crate::dict::OverlayEntry::with_context(
                            e.left_id, e.right_id, e.wcost, &e.feature,
                        )
                    }) {
                        dict.overlay.add_word(&surface, entry);
                    }
                }
                self.inner = Some(Arc::new(dict));
                String::new()
            }
            Err(e) => e,
        }
    }

    /// Add a word to the overlay dictionary.
    ///
    /// This allows adding custom words (new product names, slang, etc.)
    /// that will be recognized during parsing.
    ///
    /// # Arguments
    ///
    /// * `surface`       - The surface form (the actual text)
    /// * `reading`       - The katakana reading
    /// * `pronunciation` - The pronunciation
    /// * `wcost`         - Word cost (lower = more preferred, typical: 5000-8000)
    #[wasm_bindgen(js_name = addWord)]
    pub fn add_word(&self, surface: &str, reading: &str, pronunciation: &str, wcost: i16) {
        // Store in standalone overlay so it survives dictionary reloads.
        self.overlay
            .add_simple(surface, reading, pronunciation, wcost);
        // Mirror into the loaded dictionary overlay if one is available.
        if let Some(ref dict) = self.inner {
            dict.add_simple_word(surface, reading, pronunciation, wcost);
        }
    }

    /// Remove a word from the overlay dictionary.
    ///
    /// Returns `true` if the word was found and removed from at least one overlay.
    #[wasm_bindgen(js_name = removeWord)]
    pub fn remove_word(&self, surface: &str) -> bool {
        let removed_standalone = self.overlay.remove_word(surface);
        let removed_inner = self.inner.as_ref().is_some_and(|d| d.remove_word(surface));
        removed_standalone || removed_inner
    }

    /// Get the number of words in the standalone overlay dictionary.
    #[wasm_bindgen(js_name = overlaySize)]
    pub fn overlay_size(&self) -> usize {
        self.overlay.len()
    }

    /// Check if the analyzer is initialized (i.e. a dictionary has been loaded).
    #[wasm_bindgen(js_name = isInitialized)]
    pub fn is_initialized(&self) -> bool {
        self.inner.is_some()
    }

    /// Get all surface forms in the overlay dictionary as a JSON array.
    #[wasm_bindgen(js_name = getOverlaySurfaces)]
    pub fn get_overlay_surfaces(&self) -> String {
        super::format::format_overlay_surfaces(&self.overlay.surfaces())
    }
}

impl Default for MeCrabWasm {
    fn default() -> Self {
        Self::new()
    }
}
