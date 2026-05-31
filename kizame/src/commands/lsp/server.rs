//! MeCrabLanguageServer state struct and supporting helpers.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use std::path::PathBuf;
use std::sync::Arc;

use dashmap::DashMap;
use mecrab::MeCrab;
use tokio::sync::RwLock;
use tower_lsp::lsp_types::{Diagnostic, DiagnosticSeverity, MessageType, Position, Range, Url};
use tower_lsp::Client;

/// MeCrab Language Server state
pub struct MeCrabLanguageServer {
    pub(super) client: Client,
    pub(super) mecrab: Arc<RwLock<Option<MeCrab>>>,
    pub(super) documents: Arc<DashMap<Url, String>>,
    pub(super) dict_path: Arc<RwLock<Option<PathBuf>>>,
}

impl MeCrabLanguageServer {
    /// Create a new language server instance.
    ///
    /// Optionally pre-loads the dictionary from `dict_path` if provided.
    /// During `initialize`, standard paths are also probed if no dict was found.
    pub fn new(client: Client, dict_path: Option<PathBuf>) -> Self {
        Self {
            client,
            mecrab: Arc::new(RwLock::new(None)),
            documents: Arc::new(DashMap::new()),
            dict_path: Arc::new(RwLock::new(dict_path)),
        }
    }

    /// Attempt to load the MeCrab dictionary from the stored path or from well-known system paths.
    ///
    /// Called during `initialize`. Failures are logged as warnings; the server
    /// remains functional (but without analysis) if no dictionary is found.
    pub(super) async fn try_load_dictionary(&self) {
        let candidate = {
            let guard = self.dict_path.read().await;
            guard.clone()
        };

        // User-provided path takes priority; then probe standard locations.
        let search_paths: Vec<PathBuf> = if let Some(p) = candidate {
            vec![p]
        } else {
            Self::standard_dict_paths()
        };

        for path in search_paths {
            if path.is_dir() {
                match MeCrab::builder().dicdir(Some(path.clone())).build() {
                    Ok(instance) => {
                        let mut guard = self.mecrab.write().await;
                        *guard = Some(instance);
                        self.client
                            .log_message(
                                MessageType::INFO,
                                format!("MeCrab: loaded dictionary from {}", path.display()),
                            )
                            .await;
                        return;
                    }
                    Err(e) => {
                        self.client
                            .log_message(
                                MessageType::WARNING,
                                format!(
                                    "MeCrab: failed to load dictionary from {}: {}",
                                    path.display(),
                                    e
                                ),
                            )
                            .await;
                    }
                }
            }
        }

        self.client
            .log_message(
                MessageType::WARNING,
                "MeCrab: no dictionary found; hover/completion/diagnostics are disabled. \
                 Use --dict <path> to specify a dictionary.",
            )
            .await;
    }

    /// Well-known locations where MeCab dictionaries are installed on various platforms.
    pub fn standard_dict_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // User home-relative paths
        if let Some(home) = dirs::home_dir() {
            paths.push(home.join("dic").join("ipadic"));
            paths.push(home.join(".mecab").join("dic").join("ipadic"));
            paths.push(
                home.join(".local")
                    .join("share")
                    .join("mecab")
                    .join("dic")
                    .join("ipadic-utf8"),
            );
        }

        // Common system-wide paths (Linux / macOS Homebrew)
        paths.push(PathBuf::from("/usr/local/lib/mecab/dic/ipadic-utf8"));
        paths.push(PathBuf::from("/usr/local/lib/mecab/dic/ipadic"));
        paths.push(PathBuf::from("/usr/lib/mecab/dic/ipadic-utf8"));
        paths.push(PathBuf::from("/usr/lib/mecab/dic/ipadic"));
        paths.push(PathBuf::from("/opt/homebrew/lib/mecab/dic/ipadic-utf8"));

        paths
    }

    /// Extract the word (or Japanese character run) at `position` from `text`.
    ///
    /// Returns `(word, char_start, char_end)` where the char offsets are
    /// within the line on which the position falls.
    pub(super) fn word_at_position(text: &str, position: Position) -> Option<(String, u32, u32)> {
        let line = text.lines().nth(position.line as usize)?;
        extract_word_at_column(line, position.character as usize)
            .map(|(word, start, end)| (word.to_string(), start as u32, end as u32))
    }

    /// Run diagnostics on a document and publish them to the client.
    pub(super) async fn run_diagnostics(&self, uri: Url, text: &str) {
        let guard = self.mecrab.read().await;
        let instance = match guard.as_ref() {
            Some(m) => m,
            None => return,
        };

        let diagnostics = match instance.parse(text) {
            Ok(result) => build_diagnostics(text, &result),
            Err(e) => {
                self.client
                    .log_message(
                        MessageType::WARNING,
                        format!("MeCrab: parse error for diagnostics: {e}"),
                    )
                    .await;
                return;
            }
        };

        self.client
            .publish_diagnostics(uri, diagnostics, None)
            .await;
    }
}

/// Returns `true` if the character falls inside a recognised CJK or Japanese Unicode block.
///
/// Blocks covered:
/// - CJK Symbols and Punctuation / CJK Unified Ideographs / CJK Compatibility (0x3000–0x9FFF)
/// - Hiragana / Katakana are included in that range (0x3040–0x30FF)
/// - CJK Compatibility Ideographs (0xF900–0xFAFF)
/// - CJK Unified Ideographs Extension B (0x20000–0x2A6DF)
pub(crate) fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3000..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2A6DF
    )
}

/// Returns `true` if `c` should be considered part of a "word" token for extraction purposes.
///
/// A word character is alphanumeric, an underscore, or any CJK / Japanese script character
/// (delegated to [`is_cjk`] for characters whose code point is above U+3000).
#[inline]
pub(crate) fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || is_cjk(c)
}

/// Extract the word at a given *character* column from a single line of text.
///
/// Returns `(word_slice, char_start, char_end)` on success, or `None` when:
/// - `col` is beyond the character count of `line`
/// - the character at `col` is not a word character and neither is the one before it
pub(crate) fn extract_word_at_column(line: &str, col: usize) -> Option<(&str, usize, usize)> {
    let chars: Vec<char> = line.chars().collect();
    let char_count = chars.len();

    if col > char_count {
        return None;
    }

    // Determine whether we are standing *on* a word char or just *after* one.
    // LSP positions point between characters, so col == char_count is valid (end-of-line).
    let at_word = col < char_count && is_word_char(chars[col]);
    let before_word = col > 0 && is_word_char(chars[col - 1]);

    if !at_word && !before_word {
        return None;
    }

    // Walk backwards to find the start of the token.
    let mut start = if at_word { col } else { col - 1 };
    while start > 0 && is_word_char(chars[start - 1]) {
        start -= 1;
    }

    // Walk forwards to find the end of the token.
    // `col` is the correct starting point regardless of whether we are on or just after a word
    // char: in both cases we need to scan forward from `col` to consume any remaining word chars.
    let mut end = col;
    while end < char_count && is_word_char(chars[end]) {
        end += 1;
    }

    if start == end {
        return None;
    }

    // Convert char indices back to byte offsets for the slice.
    let byte_start = line
        .char_indices()
        .nth(start)
        .map(|(b, _)| b)
        .unwrap_or(line.len());
    let byte_end = line
        .char_indices()
        .nth(end)
        .map(|(b, _)| b)
        .unwrap_or(line.len());

    Some((&line[byte_start..byte_end], start, end))
}

/// Collect `DiagnosticSeverity::WARNING` entries for unknown-word morphemes.
///
/// IPADIC marks unknown words as `*,*,*,*,*,*,*,*,*` or with POS "未知語".
/// We also treat morphemes whose feature string starts with "未知語" as unknown.
pub(super) fn build_diagnostics(text: &str, result: &mecrab::AnalysisResult) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    // Build a character-offset → (line, col) index for fast lookup.
    let mut line_starts: Vec<usize> = vec![0];
    for (i, ch) in text.char_indices() {
        if ch == '\n' {
            line_starts.push(i + ch.len_utf8());
        }
    }

    let char_offset_to_lsp_pos = |char_off: usize| -> Position {
        let byte_off = text
            .char_indices()
            .nth(char_off)
            .map(|(b, _)| b)
            .unwrap_or(text.len());

        let line_idx = line_starts
            .partition_point(|&s| s <= byte_off)
            .saturating_sub(1);
        // LSP uses UTF-16 code units for columns
        let col_utf16 = text[line_starts[line_idx]..byte_off]
            .chars()
            .map(|c| c.len_utf16() as u32)
            .sum();
        Position {
            line: line_idx as u32,
            character: col_utf16,
        }
    };

    let mut char_cursor: usize = 0;
    for morpheme in &result.morphemes {
        let surface_char_len = morpheme.surface.chars().count();

        if is_unknown_morpheme(morpheme) {
            let start_pos = char_offset_to_lsp_pos(char_cursor);
            let end_pos = char_offset_to_lsp_pos(char_cursor + surface_char_len);
            diagnostics.push(Diagnostic {
                range: Range {
                    start: start_pos,
                    end: end_pos,
                },
                severity: Some(DiagnosticSeverity::WARNING),
                code: None,
                code_description: None,
                source: Some("mecrab".to_string()),
                message: format!(
                    "Unknown word '{}': not found in dictionary",
                    morpheme.surface
                ),
                related_information: None,
                tags: None,
                data: None,
            });
        }

        char_cursor += surface_char_len;
    }

    diagnostics
}

/// Return `true` if `morpheme` represents an unknown word.
pub(super) fn is_unknown_morpheme(morpheme: &mecrab::Morpheme) -> bool {
    // IPADIC: feature starts with "未知語" for truly unknown tokens
    if morpheme.feature.starts_with("未知語") {
        return true;
    }
    // Another common marker: feature fields are all "*"
    let fields: Vec<&str> = morpheme.feature.splitn(2, ',').collect();
    if let Some(first) = fields.first() {
        if *first == "*" {
            return true;
        }
    }
    false
}

/// Parse the feature string into (pos, sub_pos, base_form, reading).
pub(super) fn parse_ipadic_features(feature: &str) -> (String, String, String, String) {
    let fields: Vec<&str> = feature.split(',').collect();
    let pos = fields.first().copied().unwrap_or("*").to_string();
    let sub_pos = fields.get(1).copied().unwrap_or("*").to_string();
    let base_form = fields.get(6).copied().unwrap_or("*").to_string();
    let reading = fields.get(7).copied().unwrap_or("*").to_string();
    (pos, sub_pos, base_form, reading)
}

/// Map a MeCab POS string to an LSP `CompletionItemKind`.
pub(super) fn pos_to_completion_kind(pos: &str) -> tower_lsp::lsp_types::CompletionItemKind {
    use tower_lsp::lsp_types::CompletionItemKind;
    if pos.starts_with("名詞") {
        CompletionItemKind::TEXT
    } else if pos.starts_with("動詞") {
        CompletionItemKind::FUNCTION
    } else if pos.starts_with("形容詞") || pos.starts_with("形状詞") {
        CompletionItemKind::KEYWORD
    } else if pos.starts_with("助詞") || pos.starts_with("助動詞") {
        CompletionItemKind::OPERATOR
    } else {
        CompletionItemKind::VALUE
    }
}

/// Extract the text on `line` from the beginning of the line up to `position.character`.
pub(super) fn extract_prefix_at(text: &str, position: Position) -> String {
    let line = match text.lines().nth(position.line as usize) {
        Some(l) => l,
        None => return String::new(),
    };
    let char_count = line.chars().count();
    let col = (position.character as usize).min(char_count);
    line.chars().take(col).collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "lsp"))]
mod tests {
    use super::*;
    use tower_lsp::lsp_types::Position;

    // ── extract_word_at_column ────────────────────────────────────────────────

    #[test]
    fn test_extract_word_at_pos_cjk_start() {
        let line = "東京は日本語";
        // col 0 — pointing at 東, should return the full CJK run
        let result = extract_word_at_column(line, 0);
        assert!(result.is_some(), "expected Some at col 0");
        let (word, start, end) = result.unwrap();
        assert_eq!(start, 0);
        assert_eq!(end, 6); // 6 chars
        assert_eq!(word, "東京は日本語");
    }

    #[test]
    fn test_extract_word_at_pos_cjk_mid() {
        let line = "東京は日本語";
        // col 2 — pointing at は, same contiguous word
        let result = extract_word_at_column(line, 2);
        assert!(result.is_some());
        let (word, start, end) = result.unwrap();
        assert_eq!(start, 0);
        assert_eq!(end, 6);
        assert_eq!(word, "東京は日本語");
    }

    #[test]
    fn test_extract_word_at_pos_ascii_second_word() {
        let line = "Hello world";
        // col 6 — pointing at 'w' in "world"
        let result = extract_word_at_column(line, 6);
        assert!(result.is_some());
        let (word, start, end) = result.unwrap();
        assert_eq!(word, "world");
        assert_eq!(start, 6);
        assert_eq!(end, 11);
    }

    #[test]
    fn test_extract_word_at_pos_ascii_first_word() {
        let line = "Hello world";
        // col 0 — first char of "Hello"
        let result = extract_word_at_column(line, 0);
        assert!(result.is_some());
        let (word, start, end) = result.unwrap();
        assert_eq!(word, "Hello");
        assert_eq!(start, 0);
        assert_eq!(end, 5);
    }

    #[test]
    fn test_extract_word_at_pos_mixed_japanese_ascii() {
        let line = "今日はOK";
        // col 0 — on 今 (CJK run: 今日は)
        let result = extract_word_at_column(line, 0);
        assert!(result.is_some());
        let (word, _, _) = result.unwrap();
        assert_eq!(word, "今日はOK"); // entire line is word-chars (CJK + alphanumeric)
    }

    #[test]
    fn test_extract_word_at_pos_mixed_at_ascii_part() {
        // A space separates CJK from ASCII
        let line = "今日は OK";
        // col 4 — pointing at 'O' in "OK"
        let result = extract_word_at_column(line, 4);
        assert!(result.is_some());
        let (word, _, _) = result.unwrap();
        assert_eq!(word, "OK");
    }

    #[test]
    fn test_extract_word_empty_line() {
        let result = extract_word_at_column("", 0);
        assert!(result.is_none(), "empty line should yield None");
    }

    #[test]
    fn test_extract_word_out_of_bounds() {
        let line = "abc";
        // col 100 — far beyond the string length
        let result = extract_word_at_column(line, 100);
        assert!(result.is_none(), "out-of-bounds col should yield None");
    }

    #[test]
    fn test_extract_word_whitespace_only_position() {
        let line = "abc def";
        // col 3 — the space character between the two words
        let result = extract_word_at_column(line, 3);
        // ' ' is not a word char and neither is it after a word char when we
        // are pointing exactly at it (chars[3] = ' ', chars[2] = 'c' *is* a
        // word char), so the "before_word" branch fires and we get "abc".
        assert!(result.is_some());
        let (word, _, _) = result.unwrap();
        assert_eq!(word, "abc");
    }

    // ── word_at_position (integration over extract_word_at_column) ──────────

    #[test]
    fn test_word_at_position_multiline() {
        let text = "first line\n東京都\nthird line";
        // line 1, col 0 → 東京都
        let pos = Position {
            line: 1,
            character: 0,
        };
        let result = MeCrabLanguageServer::word_at_position(text, pos);
        assert!(result.is_some());
        let (word, start, end) = result.unwrap();
        assert_eq!(word, "東京都");
        assert_eq!(start, 0);
        assert_eq!(end, 3);
    }

    #[test]
    fn test_word_at_position_out_of_line_range() {
        let text = "abc";
        let pos = Position {
            line: 5,
            character: 0,
        };
        assert!(MeCrabLanguageServer::word_at_position(text, pos).is_none());
    }

    // ── is_cjk ───────────────────────────────────────────────────────────────

    #[test]
    fn test_cjk_char_range_kanji() {
        for ch in ['東', '京', '大', '阪', '語', '字'] {
            assert!(is_cjk(ch), "expected is_cjk('{ch}') == true");
        }
    }

    #[test]
    fn test_cjk_char_range_hiragana() {
        // Hiragana: U+3040–U+309F (inside 0x3000–0x9FFF)
        for ch in ['あ', 'い', 'う', 'え', 'お', 'ん'] {
            assert!(is_cjk(ch), "hiragana '{ch}' should be detected as CJK");
        }
    }

    #[test]
    fn test_cjk_char_range_katakana() {
        // Katakana: U+30A0–U+30FF (inside 0x3000–0x9FFF)
        for ch in ['ア', 'イ', 'ウ', 'エ', 'オ', 'ン'] {
            assert!(is_cjk(ch), "katakana '{ch}' should be detected as CJK");
        }
    }

    #[test]
    fn test_cjk_char_range_ascii_false() {
        for ch in ['a', 'z', 'A', 'Z', '0', '9', ' ', '!'] {
            assert!(!is_cjk(ch), "ASCII '{ch}' should NOT be detected as CJK");
        }
    }

    // ── is_unknown_morpheme ──────────────────────────────────────────────────

    #[test]
    fn test_is_unknown_morpheme_prefix() {
        let m = mecrab::Morpheme {
            surface: "フガ".to_string(),
            word_id: 0,
            pos_id: 0,
            wcost: 0,
            feature: "未知語,*,*,*,*,*,*,*,*".to_string(),
            entities: vec![],
            pronunciation: None,
            embedding: None,
            start_byte: 0,
            end_byte: 0,
        };
        assert!(is_unknown_morpheme(&m));
    }

    #[test]
    fn test_is_unknown_morpheme_star_fields() {
        let m = mecrab::Morpheme {
            surface: "XYZ".to_string(),
            word_id: 0,
            pos_id: 0,
            wcost: 0,
            feature: "*,*,*,*,*,*,*,*,*".to_string(),
            entities: vec![],
            pronunciation: None,
            embedding: None,
            start_byte: 0,
            end_byte: 0,
        };
        assert!(is_unknown_morpheme(&m));
    }

    #[test]
    fn test_is_unknown_morpheme_known_word_false() {
        let m = mecrab::Morpheme {
            surface: "東京".to_string(),
            word_id: 1,
            pos_id: 38,
            wcost: 100,
            feature: "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ".to_string(),
            entities: vec![],
            pronunciation: None,
            embedding: None,
            start_byte: 0,
            end_byte: 6,
        };
        assert!(!is_unknown_morpheme(&m));
    }

    // ── parse_ipadic_features ─────────────────────────────────────────────────

    #[test]
    fn test_parse_ipadic_features_full() {
        let feature = "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ";
        let (pos, sub_pos, base_form, reading) = parse_ipadic_features(feature);
        assert_eq!(pos, "名詞");
        assert_eq!(sub_pos, "固有名詞");
        assert_eq!(base_form, "東京");
        assert_eq!(reading, "トウキョウ");
    }

    #[test]
    fn test_parse_ipadic_features_all_stars() {
        let feature = "*,*,*,*,*,*,*,*,*";
        let (pos, sub_pos, base_form, reading) = parse_ipadic_features(feature);
        assert_eq!(pos, "*");
        assert_eq!(sub_pos, "*");
        assert_eq!(base_form, "*");
        assert_eq!(reading, "*");
    }

    #[test]
    fn test_parse_ipadic_features_short() {
        // Fewer fields than expected — should not panic, should return "*"
        let feature = "名詞";
        let (pos, sub_pos, base_form, reading) = parse_ipadic_features(feature);
        assert_eq!(pos, "名詞");
        assert_eq!(sub_pos, "*");
        assert_eq!(base_form, "*");
        assert_eq!(reading, "*");
    }

    // ── utf-16 column offset calculation ─────────────────────────────────────

    #[test]
    fn test_utf16_column_calc_ascii() {
        // ASCII chars are all 1 UTF-16 code unit wide
        let line = "Hello world";
        let col_utf16: u32 = line[..5].chars().map(|c| c.len_utf16() as u32).sum();
        assert_eq!(col_utf16, 5);
    }

    #[test]
    fn test_utf16_column_calc_cjk() {
        // BMP CJK chars (U+4E00–U+9FFF) are 1 UTF-16 code unit each
        let line = "東京都";
        let col_utf16: u32 = line.chars().map(|c| c.len_utf16() as u32).sum();
        assert_eq!(col_utf16, 3, "3 BMP kanji = 3 UTF-16 code units");
    }

    #[test]
    fn test_utf16_column_calc_supplementary() {
        // Supplementary characters (emoji / extension-B kanji) need 2 UTF-16 code units
        // U+1F600 GRINNING FACE
        let s = "😀";
        let ch = s.chars().next().unwrap();
        assert_eq!(ch.len_utf16(), 2, "emoji should be 2 UTF-16 code units");
        let col_utf16: u32 = s.chars().map(|c| c.len_utf16() as u32).sum();
        assert_eq!(col_utf16, 2);
    }

    // ── extract_prefix_at ────────────────────────────────────────────────────

    #[test]
    fn test_extract_prefix_at_mid_line() {
        let text = "東京都に行く\nfoo bar";
        let pos = Position {
            line: 0,
            character: 3,
        };
        let prefix = extract_prefix_at(text, pos);
        assert_eq!(prefix, "東京都");
    }

    #[test]
    fn test_extract_prefix_at_nonexistent_line() {
        let text = "only one line";
        let pos = Position {
            line: 99,
            character: 0,
        };
        let prefix = extract_prefix_at(text, pos);
        assert!(prefix.is_empty());
    }

    #[test]
    fn test_extract_prefix_at_clamped_col() {
        let text = "abc";
        // character beyond line length should be clamped, not panic
        let pos = Position {
            line: 0,
            character: 999,
        };
        let prefix = extract_prefix_at(text, pos);
        assert_eq!(prefix, "abc");
    }

    // ── standard_dict_paths ──────────────────────────────────────────────────

    #[test]
    fn test_standard_dict_paths_min_count() {
        let paths = MeCrabLanguageServer::standard_dict_paths();
        assert!(
            paths.len() >= 3,
            "expected at least 3 standard dict paths, got {}",
            paths.len()
        );
    }

    #[test]
    fn test_standard_dict_paths_look_reasonable() {
        let paths = MeCrabLanguageServer::standard_dict_paths();
        for p in &paths {
            let s = p.to_string_lossy();
            // Every path should mention "mecab" or a home-relative prefix
            assert!(
                s.contains("mecab") || s.contains("dic") || s.contains("ipadic"),
                "unexpected dict path: {s}"
            );
        }
    }

    #[test]
    fn test_standard_dict_paths_absolute_or_home() {
        let paths = MeCrabLanguageServer::standard_dict_paths();
        for p in &paths {
            assert!(
                p.is_absolute(),
                "expected absolute path, got: {}",
                p.display()
            );
        }
    }

    // ── pos_to_completion_kind ───────────────────────────────────────────────

    #[test]
    fn test_pos_to_completion_kind_noun() {
        use tower_lsp::lsp_types::CompletionItemKind;
        assert_eq!(pos_to_completion_kind("名詞"), CompletionItemKind::TEXT);
        assert_eq!(
            pos_to_completion_kind("名詞,固有名詞"),
            CompletionItemKind::TEXT
        );
    }

    #[test]
    fn test_pos_to_completion_kind_verb() {
        use tower_lsp::lsp_types::CompletionItemKind;
        assert_eq!(pos_to_completion_kind("動詞"), CompletionItemKind::FUNCTION);
        assert_eq!(
            pos_to_completion_kind("動詞,自立"),
            CompletionItemKind::FUNCTION
        );
    }

    #[test]
    fn test_pos_to_completion_kind_adjective() {
        use tower_lsp::lsp_types::CompletionItemKind;
        assert_eq!(
            pos_to_completion_kind("形容詞"),
            CompletionItemKind::KEYWORD
        );
        assert_eq!(
            pos_to_completion_kind("形状詞"),
            CompletionItemKind::KEYWORD
        );
    }

    #[test]
    fn test_pos_to_completion_kind_particle() {
        use tower_lsp::lsp_types::CompletionItemKind;
        assert_eq!(pos_to_completion_kind("助詞"), CompletionItemKind::OPERATOR);
        assert_eq!(
            pos_to_completion_kind("助動詞"),
            CompletionItemKind::OPERATOR
        );
    }

    #[test]
    fn test_pos_to_completion_kind_fallback() {
        use tower_lsp::lsp_types::CompletionItemKind;
        assert_eq!(pos_to_completion_kind("*"), CompletionItemKind::VALUE);
        assert_eq!(pos_to_completion_kind("感動詞"), CompletionItemKind::VALUE);
    }
}
