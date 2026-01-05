//! Text normalization for Japanese text processing
//!
//! Provides utilities for normalizing text before morphological analysis,
//! including Unicode normalization, character width conversion, and more.

#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::cast_precision_loss)]

/// Unicode normalization form
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormForm {
    /// No normalization
    None,
    /// Canonical decomposition followed by canonical composition (NFC)
    Nfc,
    /// Canonical decomposition (NFD)
    Nfd,
    /// Compatibility decomposition followed by canonical composition (NFKC)
    Nfkc,
    /// Compatibility decomposition (NFKD)
    Nfkd,
}

impl Default for NormForm {
    fn default() -> Self {
        Self::Nfkc
    }
}

/// Character width for conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharWidth {
    /// Full-width characters (全角)
    Full,
    /// Half-width characters (半角)
    Half,
}

/// Text normalizer with configurable options
#[derive(Debug, Clone)]
pub struct Normalizer {
    /// Unicode normalization form
    pub form: NormForm,
    /// Convert numbers to half-width
    pub halfwidth_numbers: bool,
    /// Convert ASCII letters to half-width
    pub halfwidth_ascii: bool,
    /// Convert katakana to full-width
    pub fullwidth_katakana: bool,
    /// Collapse whitespace
    pub collapse_whitespace: bool,
    /// Trim leading/trailing whitespace
    pub trim: bool,
    /// Convert to lowercase
    pub lowercase: bool,
    /// Remove zero-width characters
    pub remove_zero_width: bool,
}

impl Default for Normalizer {
    fn default() -> Self {
        Self {
            form: NormForm::Nfkc,
            halfwidth_numbers: true,
            halfwidth_ascii: true,
            fullwidth_katakana: true,
            collapse_whitespace: true,
            trim: true,
            lowercase: false,
            remove_zero_width: true,
        }
    }
}

impl Normalizer {
    /// Create a new normalizer with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a minimal normalizer (identity transform)
    pub fn minimal() -> Self {
        Self {
            form: NormForm::None,
            halfwidth_numbers: false,
            halfwidth_ascii: false,
            fullwidth_katakana: false,
            collapse_whitespace: false,
            trim: false,
            lowercase: false,
            remove_zero_width: false,
        }
    }

    /// Set Unicode normalization form
    #[must_use]
    pub fn with_form(mut self, form: NormForm) -> Self {
        self.form = form;
        self
    }

    /// Enable/disable lowercase conversion
    #[must_use]
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Normalize a string
    pub fn normalize(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Apply Unicode normalization first (simplified, not full NFKC)
        if self.form != NormForm::None {
            result = Self::apply_unicode_normalization(&result);
        }

        // Width conversions
        if self.halfwidth_numbers || self.halfwidth_ascii || self.fullwidth_katakana {
            result = self.convert_widths(&result);
        }

        // Remove zero-width characters
        if self.remove_zero_width {
            result = Self::remove_zero_width_chars(&result);
        }

        // Whitespace handling
        if self.collapse_whitespace {
            result = Self::collapse_ws(&result);
        }

        if self.trim {
            result = result.trim().to_string();
        }

        // Case conversion
        if self.lowercase {
            result = result.to_lowercase();
        }

        result
    }

    /// Apply Unicode normalization (simplified version)
    fn apply_unicode_normalization(text: &str) -> String {
        let mut result = String::with_capacity(text.len());

        for c in text.chars() {
            // Simplified NFKC-like normalization
            match c {
                // Full-width variants of ASCII
                '\u{FF01}'..='\u{FF5E}' => {
                    let ascii = ((c as u32) - 0xFF01 + 0x21) as u8 as char;
                    result.push(ascii);
                }
                // Ideographic space to regular space
                '\u{3000}' => result.push(' '),
                // Wave dash to full-width tilde
                '\u{301C}' => result.push('\u{FF5E}'),
                // Other characters pass through
                _ => result.push(c),
            }
        }

        result
    }

    /// Convert character widths
    fn convert_widths(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());

        for c in text.chars() {
            let converted = self.convert_char_width(c);
            result.push(converted);
        }

        result
    }

    /// Convert a single character's width
    fn convert_char_width(&self, c: char) -> char {
        // Full-width ASCII digits to half-width
        if self.halfwidth_numbers {
            if let Some(half) = fullwidth_digit_to_half(c) {
                return half;
            }
        }

        // Full-width ASCII letters to half-width
        if self.halfwidth_ascii {
            if let Some(half) = fullwidth_alpha_to_half(c) {
                return half;
            }
        }

        // Half-width katakana to full-width
        if self.fullwidth_katakana {
            if let Some(full) = halfwidth_kana_to_full(c) {
                return full;
            }
        }

        c
    }

    /// Remove zero-width characters
    fn remove_zero_width_chars(text: &str) -> String {
        text.chars().filter(|&c| !is_zero_width(c)).collect()
    }

    /// Collapse multiple whitespace into single space
    fn collapse_ws(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut prev_space = false;

        for c in text.chars() {
            if c.is_whitespace() {
                if !prev_space {
                    result.push(' ');
                    prev_space = true;
                }
            } else {
                result.push(c);
                prev_space = false;
            }
        }

        result
    }
}

/// Check if character is zero-width
fn is_zero_width(c: char) -> bool {
    matches!(
        c,
        '\u{200B}'  // Zero-width space
        | '\u{200C}' // Zero-width non-joiner
        | '\u{200D}' // Zero-width joiner
        | '\u{FEFF}' // BOM / Zero-width no-break space
        | '\u{2060}' // Word joiner
    )
}

/// Convert full-width digit to half-width
fn fullwidth_digit_to_half(c: char) -> Option<char> {
    match c {
        '\u{FF10}' => Some('0'),
        '\u{FF11}' => Some('1'),
        '\u{FF12}' => Some('2'),
        '\u{FF13}' => Some('3'),
        '\u{FF14}' => Some('4'),
        '\u{FF15}' => Some('5'),
        '\u{FF16}' => Some('6'),
        '\u{FF17}' => Some('7'),
        '\u{FF18}' => Some('8'),
        '\u{FF19}' => Some('9'),
        _ => None,
    }
}

/// Convert full-width ASCII letter to half-width
fn fullwidth_alpha_to_half(c: char) -> Option<char> {
    let code = c as u32;
    // Full-width uppercase A-Z: FF21-FF3A
    if (0xFF21..=0xFF3A).contains(&code) {
        return Some((code - 0xFF21 + 0x41) as u8 as char);
    }
    // Full-width lowercase a-z: FF41-FF5A
    if (0xFF41..=0xFF5A).contains(&code) {
        return Some((code - 0xFF41 + 0x61) as u8 as char);
    }
    None
}

/// Convert half-width katakana to full-width
fn halfwidth_kana_to_full(c: char) -> Option<char> {
    // Half-width katakana range: FF65-FF9F
    let code = c as u32;
    if !(0xFF65..=0xFF9F).contains(&code) {
        return None;
    }

    // Mapping table for basic half-width katakana
    const MAPPING: &[(char, char)] = &[
        ('\u{FF66}', '\u{30F2}'), // ヲ
        ('\u{FF67}', '\u{30A1}'), // ァ
        ('\u{FF68}', '\u{30A3}'), // ィ
        ('\u{FF69}', '\u{30A5}'), // ゥ
        ('\u{FF6A}', '\u{30A7}'), // ェ
        ('\u{FF6B}', '\u{30A9}'), // ォ
        ('\u{FF6C}', '\u{30E3}'), // ャ
        ('\u{FF6D}', '\u{30E5}'), // ュ
        ('\u{FF6E}', '\u{30E7}'), // ョ
        ('\u{FF6F}', '\u{30C3}'), // ッ
        ('\u{FF70}', '\u{30FC}'), // ー
        ('\u{FF71}', '\u{30A2}'), // ア
        ('\u{FF72}', '\u{30A4}'), // イ
        ('\u{FF73}', '\u{30A6}'), // ウ
        ('\u{FF74}', '\u{30A8}'), // エ
        ('\u{FF75}', '\u{30AA}'), // オ
        ('\u{FF76}', '\u{30AB}'), // カ
        ('\u{FF77}', '\u{30AD}'), // キ
        ('\u{FF78}', '\u{30AF}'), // ク
        ('\u{FF79}', '\u{30B1}'), // ケ
        ('\u{FF7A}', '\u{30B3}'), // コ
        ('\u{FF7B}', '\u{30B5}'), // サ
        ('\u{FF7C}', '\u{30B7}'), // シ
        ('\u{FF7D}', '\u{30B9}'), // ス
        ('\u{FF7E}', '\u{30BB}'), // セ
        ('\u{FF7F}', '\u{30BD}'), // ソ
        ('\u{FF80}', '\u{30BF}'), // タ
        ('\u{FF81}', '\u{30C1}'), // チ
        ('\u{FF82}', '\u{30C4}'), // ツ
        ('\u{FF83}', '\u{30C6}'), // テ
        ('\u{FF84}', '\u{30C8}'), // ト
        ('\u{FF85}', '\u{30CA}'), // ナ
        ('\u{FF86}', '\u{30CB}'), // ニ
        ('\u{FF87}', '\u{30CC}'), // ヌ
        ('\u{FF88}', '\u{30CD}'), // ネ
        ('\u{FF89}', '\u{30CE}'), // ノ
        ('\u{FF8A}', '\u{30CF}'), // ハ
        ('\u{FF8B}', '\u{30D2}'), // ヒ
        ('\u{FF8C}', '\u{30D5}'), // フ
        ('\u{FF8D}', '\u{30D8}'), // ヘ
        ('\u{FF8E}', '\u{30DB}'), // ホ
        ('\u{FF8F}', '\u{30DE}'), // マ
        ('\u{FF90}', '\u{30DF}'), // ミ
        ('\u{FF91}', '\u{30E0}'), // ム
        ('\u{FF92}', '\u{30E1}'), // メ
        ('\u{FF93}', '\u{30E2}'), // モ
        ('\u{FF94}', '\u{30E4}'), // ヤ
        ('\u{FF95}', '\u{30E6}'), // ユ
        ('\u{FF96}', '\u{30E8}'), // ヨ
        ('\u{FF97}', '\u{30E9}'), // ラ
        ('\u{FF98}', '\u{30EA}'), // リ
        ('\u{FF99}', '\u{30EB}'), // ル
        ('\u{FF9A}', '\u{30EC}'), // レ
        ('\u{FF9B}', '\u{30ED}'), // ロ
        ('\u{FF9C}', '\u{30EF}'), // ワ
        ('\u{FF9D}', '\u{30F3}'), // ン
        ('\u{FF9E}', '\u{309B}'), // Voiced mark (゛)
        ('\u{FF9F}', '\u{309C}'), // Semi-voiced mark (゜)
    ];

    for &(half, full) in MAPPING {
        if c == half {
            return Some(full);
        }
    }

    None
}

/// Normalize to NFKC (compatibility composition)
pub fn to_nfkc(text: &str) -> String {
    Normalizer::new().normalize(text)
}

/// Remove all whitespace
pub fn remove_whitespace(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Normalize Japanese quotes and brackets
pub fn normalize_quotes(text: &str) -> String {
    let mut result = String::with_capacity(text.len());

    for c in text.chars() {
        let normalized = match c {
            '「' | '『' | '\u{FF62}' => '「',
            '」' | '』' | '\u{FF63}' => '」',
            '(' | '（' => '（',
            ')' | '）' => '）',
            '【' | '〔' => '【',
            '】' | '〕' => '】',
            _ => c,
        };
        result.push(normalized);
    }

    result
}

/// Convert all Japanese periods and commas to standard forms
pub fn normalize_punctuation(text: &str) -> String {
    let mut result = String::with_capacity(text.len());

    for c in text.chars() {
        let normalized = match c {
            '。' | '｡' | '．' => '。',
            '、' | '､' | '，' => '、',
            '!' | '！' => '！',
            '?' | '？' => '？',
            ':' | '：' => '：',
            ';' | '；' => '；',
            _ => c,
        };
        result.push(normalized);
    }

    result
}

/// Check if text contains only hiragana
pub fn is_hiragana_only(text: &str) -> bool {
    text.chars()
        .all(|c| ('\u{3040}'..='\u{309F}').contains(&c) || c.is_whitespace())
}

/// Check if text contains only katakana
pub fn is_katakana_only(text: &str) -> bool {
    text.chars().all(|c| {
        ('\u{30A0}'..='\u{30FF}').contains(&c)
            || ('\u{FF65}'..='\u{FF9F}').contains(&c) // Half-width
            || c.is_whitespace()
    })
}

/// Check if text contains kanji
pub fn contains_kanji(text: &str) -> bool {
    text.chars().any(|c| {
        ('\u{4E00}'..='\u{9FFF}').contains(&c)  // CJK Unified Ideographs
            || ('\u{3400}'..='\u{4DBF}').contains(&c) // CJK Extension A
            || ('\u{F900}'..='\u{FAFF}').contains(&c) // CJK Compatibility
    })
}

/// Count character types in text
#[derive(Debug, Clone, Default)]
pub struct CharTypeCounts {
    /// Hiragana count
    pub hiragana: usize,
    /// Katakana count
    pub katakana: usize,
    /// Kanji count
    pub kanji: usize,
    /// ASCII letters
    pub ascii_letter: usize,
    /// Digits
    pub digit: usize,
    /// Punctuation
    pub punctuation: usize,
    /// Whitespace
    pub whitespace: usize,
    /// Other
    pub other: usize,
}

impl CharTypeCounts {
    /// Count character types in text
    pub fn from_text(text: &str) -> Self {
        let mut counts = Self::default();

        for c in text.chars() {
            if ('\u{3040}'..='\u{309F}').contains(&c) {
                counts.hiragana += 1;
            } else if ('\u{30A0}'..='\u{30FF}').contains(&c)
                || ('\u{FF65}'..='\u{FF9F}').contains(&c)
            {
                counts.katakana += 1;
            } else if ('\u{4E00}'..='\u{9FFF}').contains(&c) {
                counts.kanji += 1;
            } else if c.is_ascii_alphabetic() {
                counts.ascii_letter += 1;
            } else if c.is_ascii_digit() {
                counts.digit += 1;
            } else if c.is_ascii_punctuation() || is_cjk_punctuation(c) {
                counts.punctuation += 1;
            } else if c.is_whitespace() {
                counts.whitespace += 1;
            } else {
                counts.other += 1;
            }
        }

        counts
    }

    /// Total character count
    pub fn total(&self) -> usize {
        self.hiragana
            + self.katakana
            + self.kanji
            + self.ascii_letter
            + self.digit
            + self.punctuation
            + self.whitespace
            + self.other
    }

    /// Japanese character ratio (hiragana + katakana + kanji)
    pub fn japanese_ratio(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        (self.hiragana + self.katakana + self.kanji) as f64 / total as f64
    }
}

/// Check if character is CJK punctuation
fn is_cjk_punctuation(c: char) -> bool {
    matches!(
        c,
        '。' | '、'
            | '！'
            | '？'
            | '「'
            | '」'
            | '『'
            | '』'
            | '（'
            | '）'
            | '【'
            | '】'
            | '・'
            | '：'
            | '；'
            | '〜'
            | '…'
            | '‥'
            | '゛'
            | '゜'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalizer_default() {
        let norm = Normalizer::new();
        let result = norm.normalize("　ＡＢＣＤ　１２３　");
        assert_eq!(result, "ABCD 123");
    }

    #[test]
    fn test_normalizer_minimal() {
        let norm = Normalizer::minimal();
        let input = "　ＡＢＣ　";
        let result = norm.normalize(input);
        assert_eq!(result, input); // No change
    }

    #[test]
    fn test_fullwidth_digit_to_half() {
        assert_eq!(fullwidth_digit_to_half('\u{FF10}'), Some('0'));
        assert_eq!(fullwidth_digit_to_half('\u{FF19}'), Some('9'));
        assert_eq!(fullwidth_digit_to_half('a'), None);
    }

    #[test]
    fn test_fullwidth_alpha_to_half() {
        assert_eq!(fullwidth_alpha_to_half('\u{FF21}'), Some('A'));
        assert_eq!(fullwidth_alpha_to_half('\u{FF3A}'), Some('Z'));
        assert_eq!(fullwidth_alpha_to_half('\u{FF41}'), Some('a'));
        assert_eq!(fullwidth_alpha_to_half('\u{FF5A}'), Some('z'));
    }

    #[test]
    fn test_halfwidth_kana_to_full() {
        assert_eq!(halfwidth_kana_to_full('\u{FF71}'), Some('\u{30A2}')); // ア
        assert_eq!(halfwidth_kana_to_full('\u{FF9D}'), Some('\u{30F3}')); // ン
        assert_eq!(halfwidth_kana_to_full('A'), None);
    }

    #[test]
    fn test_is_zero_width() {
        assert!(is_zero_width('\u{200B}'));
        assert!(is_zero_width('\u{FEFF}'));
        assert!(!is_zero_width('a'));
    }

    #[test]
    fn test_remove_zero_width() {
        let norm = Normalizer::new();
        let input = "te\u{200B}st";
        let result = norm.normalize(input);
        assert_eq!(result, "test");
    }

    #[test]
    fn test_collapse_whitespace() {
        let norm = Normalizer::new();
        let result = norm.normalize("a   b  c");
        assert_eq!(result, "a b c");
    }

    #[test]
    fn test_lowercase() {
        let norm = Normalizer::new().with_lowercase(true);
        let result = norm.normalize("HELLO World");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_to_nfkc() {
        let result = to_nfkc("ＡＢＣＤ１２３");
        assert_eq!(result, "ABCD123");
    }

    #[test]
    fn test_remove_whitespace() {
        let result = remove_whitespace("a b c");
        assert_eq!(result, "abc");
    }

    #[test]
    fn test_normalize_quotes() {
        let result = normalize_quotes("『テスト』");
        assert_eq!(result, "「テスト」");
    }

    #[test]
    fn test_normalize_punctuation() {
        let result = normalize_punctuation("テスト｡これはテスト､です");
        assert_eq!(result, "テスト。これはテスト、です");
    }

    #[test]
    fn test_is_hiragana_only() {
        assert!(is_hiragana_only("ひらがな"));
        assert!(!is_hiragana_only("カタカナ"));
        assert!(!is_hiragana_only("漢字"));
    }

    #[test]
    fn test_is_katakana_only() {
        assert!(is_katakana_only("カタカナ"));
        assert!(!is_katakana_only("ひらがな"));
    }

    #[test]
    fn test_contains_kanji() {
        assert!(contains_kanji("漢字テスト"));
        assert!(!contains_kanji("ひらがなカタカナ"));
    }

    #[test]
    fn test_char_type_counts() {
        // 漢字=2, とひらがな=5, カタカナ=4, ABC=3, 123=3
        let counts = CharTypeCounts::from_text("漢字とひらがなカタカナABC123");
        assert_eq!(counts.kanji, 2);
        assert_eq!(counts.hiragana, 5); // と + ひ + ら + が + な
        assert_eq!(counts.katakana, 4);
        assert_eq!(counts.ascii_letter, 3);
        assert_eq!(counts.digit, 3);
    }

    #[test]
    fn test_char_type_japanese_ratio() {
        let counts = CharTypeCounts::from_text("あいうえお12345");
        let ratio = counts.japanese_ratio();
        assert!((ratio - 0.5).abs() < 0.01);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_char_type_counts_empty() {
        let counts = CharTypeCounts::from_text("");
        assert_eq!(counts.total(), 0);
        assert_eq!(counts.japanese_ratio(), 0.0);
    }

    #[test]
    fn test_ideographic_space() {
        let norm = Normalizer::new();
        let result = norm.normalize("テスト\u{3000}テスト");
        assert_eq!(result, "テスト テスト");
    }
}
