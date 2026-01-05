//! User dictionary persistence and management
//!
//! Provides functionality for saving and loading user dictionaries
//! in a portable format.

#![allow(clippy::cast_precision_loss)]

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use super::overlay::OverlayDictionary;

/// User dictionary entry
#[derive(Debug, Clone, PartialEq)]
pub struct UserEntry {
    /// Surface form (the word itself)
    pub surface: String,
    /// Reading in katakana
    pub reading: String,
    /// Pronunciation
    pub pronunciation: String,
    /// Word cost (lower = more likely to be selected)
    pub cost: i16,
    /// Part-of-speech tag
    pub pos: String,
}

impl UserEntry {
    /// Create a new user entry
    pub fn new(surface: String, reading: String, pronunciation: String, cost: i16) -> Self {
        Self {
            surface,
            reading,
            pronunciation,
            cost,
            pos: "名詞,固有名詞,一般,*,*,*".to_string(),
        }
    }

    /// Create entry with custom POS
    pub fn with_pos(mut self, pos: impl Into<String>) -> Self {
        self.pos = pos.into();
        self
    }

    /// Parse from CSV format
    pub fn from_csv(line: &str) -> Option<Self> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 4 {
            return None;
        }

        let surface = parts[0].to_string();
        let reading = (*parts.get(1).unwrap_or(&"")).to_string();
        let pronunciation = (*parts.get(2).unwrap_or(&"")).to_string();
        let cost = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
        let pos = if parts.len() > 4 {
            parts[4..].join(",")
        } else {
            "名詞,固有名詞,一般,*,*,*".to_string()
        };

        Some(Self {
            surface,
            reading,
            pronunciation,
            cost,
            pos,
        })
    }

    /// Convert to CSV format
    pub fn to_csv(&self) -> String {
        format!(
            "{},{},{},{},{}",
            self.surface, self.reading, self.pronunciation, self.cost, self.pos
        )
    }

    /// Convert to MeCab dictionary format (for compilation)
    pub fn to_mecab_format(&self) -> String {
        format!(
            "{},0,0,{},{},{}",
            self.surface, self.cost, self.pos, self.reading
        )
    }
}

/// User dictionary file format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DictFormat {
    /// Simple CSV format: surface,reading,pronunciation,cost[,pos...]
    #[default]
    Csv,
    /// MeCab dictionary format
    MeCab,
    /// JSON format
    Json,
}

/// User dictionary manager
#[derive(Debug)]
pub struct UserDictManager {
    entries: Vec<UserEntry>,
    dirty: bool,
}

impl UserDictManager {
    /// Create a new empty manager
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            dirty: false,
        }
    }

    /// Load entries from a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or read.
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some(entry) = UserEntry::from_csv(trimmed) {
                entries.push(entry);
            }
        }

        Ok(Self {
            entries,
            dirty: false,
        })
    }

    /// Save entries to a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or written to.
    pub fn save<P: AsRef<Path>>(&mut self, path: P, format: DictFormat) -> std::io::Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let mut writer = BufWriter::new(file);

        match format {
            DictFormat::Csv => {
                writeln!(writer, "# MeCrab User Dictionary")?;
                writeln!(writer, "# Format: surface,reading,pronunciation,cost,pos")?;
                for entry in &self.entries {
                    writeln!(writer, "{}", entry.to_csv())?;
                }
            }
            DictFormat::MeCab => {
                for entry in &self.entries {
                    writeln!(writer, "{}", entry.to_mecab_format())?;
                }
            }
            DictFormat::Json => {
                writeln!(writer, "[")?;
                for (i, entry) in self.entries.iter().enumerate() {
                    let comma = if i < self.entries.len() - 1 { "," } else { "" };
                    writeln!(
                        writer,
                        r#"  {{"surface":"{}","reading":"{}","pronunciation":"{}","cost":{},"pos":"{}"}}{}"#,
                        escape_json(&entry.surface),
                        escape_json(&entry.reading),
                        escape_json(&entry.pronunciation),
                        entry.cost,
                        escape_json(&entry.pos),
                        comma
                    )?;
                }
                writeln!(writer, "]")?;
            }
        }

        writer.flush()?;
        self.dirty = false;
        Ok(())
    }

    /// Add an entry
    pub fn add(&mut self, entry: UserEntry) {
        self.entries.push(entry);
        self.dirty = true;
    }

    /// Remove an entry by surface form
    pub fn remove(&mut self, surface: &str) -> bool {
        let len_before = self.entries.len();
        self.entries.retain(|e| e.surface != surface);
        let removed = self.entries.len() < len_before;
        if removed {
            self.dirty = true;
        }
        removed
    }

    /// Find entries by surface
    pub fn find(&self, surface: &str) -> Vec<&UserEntry> {
        self.entries
            .iter()
            .filter(|e| e.surface == surface)
            .collect()
    }

    /// Find entries by prefix
    pub fn find_by_prefix(&self, prefix: &str) -> Vec<&UserEntry> {
        self.entries
            .iter()
            .filter(|e| e.surface.starts_with(prefix))
            .collect()
    }

    /// Get all entries
    pub fn entries(&self) -> &[UserEntry] {
        &self.entries
    }

    /// Get entry count
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check if there are unsaved changes
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.dirty = true;
    }

    /// Apply entries to an overlay dictionary
    pub fn apply_to(&self, overlay: &OverlayDictionary) {
        for entry in &self.entries {
            let overlay_entry = super::OverlayEntry {
                left_id: 0,
                right_id: 0,
                wcost: entry.cost,
                feature: format!("{},{},{}", entry.pos, entry.reading, entry.pronunciation),
            };
            overlay.add_word(&entry.surface, overlay_entry);
        }
    }

    /// Merge another manager's entries
    pub fn merge(&mut self, other: &UserDictManager) {
        for entry in &other.entries {
            // Avoid duplicates
            if !self.entries.iter().any(|e| e.surface == entry.surface) {
                self.entries.push(entry.clone());
                self.dirty = true;
            }
        }
    }

    /// Sort entries by surface form
    pub fn sort(&mut self) {
        self.entries.sort_by(|a, b| a.surface.cmp(&b.surface));
    }

    /// Sort entries by cost (lowest first)
    pub fn sort_by_cost(&mut self) {
        self.entries.sort_by(|a, b| a.cost.cmp(&b.cost));
    }
}

impl Default for UserDictManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Escape a string for JSON
fn escape_json(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c => result.push(c),
        }
    }
    result
}

/// User dictionary validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Number of valid entries
    pub valid_count: usize,
    /// Number of invalid entries
    pub invalid_count: usize,
    /// Validation errors
    pub errors: Vec<ValidationError>,
}

/// Validation error for a single entry
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Line number (1-indexed)
    pub line: usize,
    /// Error message
    pub message: String,
    /// The problematic content
    pub content: String,
}

impl ValidationResult {
    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.invalid_count == 0
    }
}

/// Validate a user dictionary file
///
/// # Errors
///
/// Returns an error if the file cannot be opened or read.
pub fn validate_file<P: AsRef<Path>>(path: P) -> std::io::Result<ValidationResult> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut valid_count = 0;
    let mut invalid_count = 0;
    let mut errors = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if UserEntry::from_csv(trimmed).is_some() {
            valid_count += 1;
        } else {
            invalid_count += 1;
            errors.push(ValidationError {
                line: line_num + 1,
                message: "Invalid CSV format".to_string(),
                content: trimmed.to_string(),
            });
        }
    }

    Ok(ValidationResult {
        valid_count,
        invalid_count,
        errors,
    })
}

/// Statistics about user dictionary
#[derive(Debug, Clone, Default)]
pub struct UserDictStats {
    /// Total entry count
    pub entry_count: usize,
    /// Unique surface forms
    pub unique_surfaces: usize,
    /// Average cost
    pub avg_cost: f64,
    /// Min cost
    pub min_cost: i16,
    /// Max cost
    pub max_cost: i16,
    /// Total characters in surface forms
    pub total_chars: usize,
}

impl UserDictStats {
    /// Calculate statistics for a user dictionary
    pub fn calculate(manager: &UserDictManager) -> Self {
        if manager.is_empty() {
            return Self::default();
        }

        let entries = manager.entries();
        let entry_count = entries.len();

        let mut surfaces = std::collections::HashSet::new();
        let mut total_cost: i64 = 0;
        let mut min_cost = i16::MAX;
        let mut max_cost = i16::MIN;
        let mut total_chars = 0;

        for entry in entries {
            surfaces.insert(&entry.surface);
            total_cost += entry.cost as i64;
            min_cost = min_cost.min(entry.cost);
            max_cost = max_cost.max(entry.cost);
            total_chars += entry.surface.chars().count();
        }

        Self {
            entry_count,
            unique_surfaces: surfaces.len(),
            avg_cost: total_cost as f64 / entry_count as f64,
            min_cost,
            max_cost,
            total_chars,
        }
    }

    /// Average surface length
    pub fn avg_surface_len(&self) -> f64 {
        if self.entry_count == 0 {
            0.0
        } else {
            self.total_chars as f64 / self.entry_count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_entry_from_csv() {
        let entry = UserEntry::from_csv("東京,トウキョウ,トウキョウ,100").unwrap();
        assert_eq!(entry.surface, "東京");
        assert_eq!(entry.reading, "トウキョウ");
        assert_eq!(entry.cost, 100);
    }

    #[test]
    fn test_user_entry_from_csv_with_pos() {
        let entry = UserEntry::from_csv("MeCrab,メクラブ,メクラブ,50,名詞,固有名詞,組織").unwrap();
        assert_eq!(entry.surface, "MeCrab");
        assert_eq!(entry.pos, "名詞,固有名詞,組織");
    }

    #[test]
    fn test_user_entry_from_csv_invalid() {
        assert!(UserEntry::from_csv("incomplete,data").is_none());
        assert!(UserEntry::from_csv("").is_none());
    }

    #[test]
    fn test_user_entry_to_csv() {
        let entry = UserEntry::new(
            "テスト".to_string(),
            "テスト".to_string(),
            "テスト".to_string(),
            50,
        );
        let csv = entry.to_csv();
        assert!(csv.contains("テスト"));
        assert!(csv.contains("50"));
    }

    #[test]
    fn test_user_dict_manager_add_remove() {
        let mut manager = UserDictManager::new();
        assert!(manager.is_empty());

        manager.add(UserEntry::new(
            "テスト".to_string(),
            "テスト".to_string(),
            "テスト".to_string(),
            50,
        ));
        assert_eq!(manager.len(), 1);
        assert!(manager.is_dirty());

        assert!(manager.remove("テスト"));
        assert!(manager.is_empty());
    }

    #[test]
    fn test_user_dict_manager_find() {
        let mut manager = UserDictManager::new();
        manager.add(UserEntry::new(
            "東京".to_string(),
            "トウキョウ".to_string(),
            "トウキョウ".to_string(),
            100,
        ));
        manager.add(UserEntry::new(
            "東京都".to_string(),
            "トウキョウト".to_string(),
            "トウキョウト".to_string(),
            80,
        ));

        let found = manager.find("東京");
        assert_eq!(found.len(), 1);

        let prefix_matches = manager.find_by_prefix("東京");
        assert_eq!(prefix_matches.len(), 2);
    }

    #[test]
    fn test_user_dict_manager_sort() {
        let mut manager = UserDictManager::new();
        manager.add(UserEntry::new(
            "Z".to_string(),
            "Z".to_string(),
            "Z".to_string(),
            100,
        ));
        manager.add(UserEntry::new(
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            50,
        ));

        manager.sort();
        assert_eq!(manager.entries()[0].surface, "A");
        assert_eq!(manager.entries()[1].surface, "Z");
    }

    #[test]
    fn test_user_dict_manager_sort_by_cost() {
        let mut manager = UserDictManager::new();
        manager.add(UserEntry::new(
            "高".to_string(),
            "タカ".to_string(),
            "タカ".to_string(),
            200,
        ));
        manager.add(UserEntry::new(
            "低".to_string(),
            "ヒク".to_string(),
            "ヒク".to_string(),
            50,
        ));

        manager.sort_by_cost();
        assert_eq!(manager.entries()[0].cost, 50);
    }

    #[test]
    fn test_user_dict_manager_merge() {
        let mut manager1 = UserDictManager::new();
        manager1.add(UserEntry::new(
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            10,
        ));

        let mut manager2 = UserDictManager::new();
        manager2.add(UserEntry::new(
            "B".to_string(),
            "B".to_string(),
            "B".to_string(),
            20,
        ));
        manager2.add(UserEntry::new(
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            10,
        )); // Duplicate

        manager1.merge(&manager2);
        assert_eq!(manager1.len(), 2); // No duplicate
    }

    #[test]
    fn test_escape_json() {
        assert_eq!(escape_json("hello"), "hello");
        assert_eq!(escape_json("hello\"world"), "hello\\\"world");
        assert_eq!(escape_json("line1\nline2"), "line1\\nline2");
    }

    #[test]
    fn test_user_dict_stats() {
        let mut manager = UserDictManager::new();
        manager.add(UserEntry::new(
            "短".to_string(),
            "ミジカ".to_string(),
            "ミジカ".to_string(),
            50,
        ));
        manager.add(UserEntry::new(
            "長い".to_string(),
            "ナガイ".to_string(),
            "ナガイ".to_string(),
            100,
        ));

        let stats = UserDictStats::calculate(&manager);
        assert_eq!(stats.entry_count, 2);
        assert_eq!(stats.min_cost, 50);
        assert_eq!(stats.max_cost, 100);
        assert_eq!(stats.total_chars, 3); // 1 + 2 chars
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_user_dict_stats_empty() {
        let manager = UserDictManager::new();
        let stats = UserDictStats::calculate(&manager);
        assert_eq!(stats.entry_count, 0);
        assert_eq!(stats.avg_surface_len(), 0.0);
    }

    #[test]
    fn test_validation_result() {
        let result = ValidationResult {
            valid_count: 10,
            invalid_count: 0,
            errors: vec![],
        };
        assert!(result.is_valid());

        let result2 = ValidationResult {
            valid_count: 8,
            invalid_count: 2,
            errors: vec![],
        };
        assert!(!result2.is_valid());
    }

    #[test]
    fn test_mecab_format() {
        let entry = UserEntry::new(
            "MeCrab".to_string(),
            "メクラブ".to_string(),
            "メクラブ".to_string(),
            100,
        );
        let mecab = entry.to_mecab_format();
        assert!(mecab.starts_with("MeCrab,0,0,100"));
    }
}
