//! WebAssembly bindings for MeCrab
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module provides JavaScript bindings for MeCrab using wasm-bindgen.
//! It enables morphological analysis in web browsers and Node.js.
//!
//! # Usage (JavaScript)
//!
//! ```javascript
//! import init, { MeCrabWasm } from 'mecrab';
//!
//! await init();
//!
//! // Load dictionary from ArrayBuffer
//! const response = await fetch('/dict/sys.dic');
//! const dictData = await response.arrayBuffer();
//!
//! const mecrab = new MeCrabWasm();
//! mecrab.loadDictionary(new Uint8Array(dictData));
//!
//! // Parse text
//! const result = mecrab.parse("すもももももももものうち");
//! console.log(result);
//!
//! // Add custom words
//! mecrab.addWord("ChatGPT", "チャットジーピーティー", "チャットジーピーティー", 5000);
//! ```

use wasm_bindgen::prelude::*;

/// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// JavaScript-friendly wrapper for MeCrab morphological analyzer
#[wasm_bindgen]
pub struct MeCrabWasm {
    /// Overlay dictionary for runtime word additions
    overlay: crate::dict::OverlayDictionary,
    /// Whether dictionary is loaded (WASM uses in-memory dictionaries)
    initialized: bool,
}

#[wasm_bindgen]
impl MeCrabWasm {
    /// Create a new MeCrab WASM instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            overlay: crate::dict::OverlayDictionary::new(),
            initialized: false,
        }
    }

    /// Add a word to the overlay dictionary
    ///
    /// This allows adding custom words (new product names, slang, etc.)
    /// that will be recognized during parsing.
    ///
    /// # Arguments
    ///
    /// * `surface` - The surface form (the actual text)
    /// * `reading` - The katakana reading
    /// * `pronunciation` - The pronunciation
    /// * `wcost` - Word cost (lower = more preferred, typical: 5000-8000)
    #[wasm_bindgen(js_name = addWord)]
    pub fn add_word(&self, surface: &str, reading: &str, pronunciation: &str, wcost: i16) {
        self.overlay
            .add_simple(surface, reading, pronunciation, wcost);
    }

    /// Remove a word from the overlay dictionary
    ///
    /// Returns true if the word was found and removed.
    #[wasm_bindgen(js_name = removeWord)]
    pub fn remove_word(&self, surface: &str) -> bool {
        self.overlay.remove_word(surface)
    }

    /// Get the number of words in the overlay dictionary
    #[wasm_bindgen(js_name = overlaySize)]
    pub fn overlay_size(&self) -> usize {
        self.overlay.len()
    }

    /// Check if the analyzer is initialized
    #[wasm_bindgen(js_name = isInitialized)]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get all surface forms in the overlay dictionary as a JSON array
    #[wasm_bindgen(js_name = getOverlaySurfaces)]
    pub fn get_overlay_surfaces(&self) -> String {
        let surfaces = self.overlay.surfaces();
        format!(
            "[{}]",
            surfaces
                .iter()
                .map(|s| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")))
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

impl Default for MeCrabWasm {
    fn default() -> Self {
        Self::new()
    }
}

/// Version information
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get feature flags
#[wasm_bindgen]
pub fn features() -> String {
    let features = ["wasm"];
    // Additional features can be added here with cfg checks
    format!(
        "[{}]",
        features
            .iter()
            .map(|f| format!("\"{}\"", f))
            .collect::<Vec<_>>()
            .join(",")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_overlay() {
        let mecrab = MeCrabWasm::new();

        mecrab.add_word("テスト", "テスト", "テスト", 5000);
        assert_eq!(mecrab.overlay_size(), 1);

        assert!(mecrab.remove_word("テスト"));
        assert_eq!(mecrab.overlay_size(), 0);
    }
}
