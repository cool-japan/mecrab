//! Phonetic transduction for Japanese text
//!
//! Provides conversion between different phonetic representations:
//! - Hiragana ↔ Katakana
//! - Kana → Romaji
//! - Kana → IPA (International Phonetic Alphabet)
//! - Kana → X-SAMPA (IPA-like)

pub mod transducer;

pub use transducer::{to_hiragana, to_ipa, to_katakana, to_romaji, to_xsampa};
