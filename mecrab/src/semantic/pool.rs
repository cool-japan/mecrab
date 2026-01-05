//! SemanticPool - Compact URI storage with 5-byte entries
//!
//! Binary format:
//! - Header: Magic + Version + Counts + Offsets
//! - Entries: 5 bytes each (offset:3, confidence:1, source+flags:1)
//! - Prefix table: Common URI prefixes
//! - String pool: URI suffixes (null-terminated)

use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Write};

/// Magic bytes for SemanticPool binary format
pub const MAGIC: &[u8; 4] = b"MCSP";
/// Current version
pub const VERSION: u32 = 1;

/// Ontology source for semantic entries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum OntologySource {
    /// Wikidata (wd:)
    #[default]
    Wikidata = 0,
    /// DBpedia (dbr:)
    DBpedia = 1,
    /// Schema.org (schema:)
    SchemaOrg = 2,
    /// Custom/other
    Custom = 3,
}

impl OntologySource {
    /// Get the URI prefix for this source
    pub fn prefix(&self) -> &'static str {
        match self {
            OntologySource::Wikidata => "http://www.wikidata.org/entity/",
            OntologySource::DBpedia => "http://dbpedia.org/resource/",
            OntologySource::SchemaOrg => "http://schema.org/",
            OntologySource::Custom => "",
        }
    }

    /// Get the CURIE prefix
    pub fn curie_prefix(&self) -> &'static str {
        match self {
            OntologySource::Wikidata => "wd:",
            OntologySource::DBpedia => "dbr:",
            OntologySource::SchemaOrg => "schema:",
            OntologySource::Custom => "",
        }
    }
}

impl From<u8> for OntologySource {
    fn from(v: u8) -> Self {
        match v & 0x0F {
            0 => OntologySource::Wikidata,
            1 => OntologySource::DBpedia,
            2 => OntologySource::SchemaOrg,
            _ => OntologySource::Custom,
        }
    }
}

/// Header for SemanticPool binary format
#[derive(Debug, Clone, Copy)]
pub struct Header {
    /// Magic bytes to identify file format ("SMPL")
    pub magic: [u8; 4],
    /// Format version number
    pub version: u32,
    /// Total number of entries in the pool
    pub entry_count: u32,
    /// Byte offset to the string pool section
    pub string_pool_offset: u32,
    /// Byte offset to the prefix table section
    pub prefix_table_offset: u32,
}

impl Header {
    /// Size of the header in bytes
    pub const SIZE: usize = 20;

    /// Create a new header with specified offsets
    pub fn new(entry_count: u32, string_pool_offset: u32, prefix_table_offset: u32) -> Self {
        Self {
            magic: *MAGIC,
            version: VERSION,
            entry_count,
            string_pool_offset,
            prefix_table_offset,
        }
    }

    /// Write header to a byte stream
    pub fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_all(&self.magic)?;
        w.write_all(&self.version.to_le_bytes())?;
        w.write_all(&self.entry_count.to_le_bytes())?;
        w.write_all(&self.string_pool_offset.to_le_bytes())?;
        w.write_all(&self.prefix_table_offset.to_le_bytes())?;
        Ok(())
    }

    /// Read header from a byte stream
    pub fn read_from<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;

        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        let version = u32::from_le_bytes(buf);

        r.read_exact(&mut buf)?;
        let entry_count = u32::from_le_bytes(buf);

        r.read_exact(&mut buf)?;
        let string_pool_offset = u32::from_le_bytes(buf);

        r.read_exact(&mut buf)?;
        let prefix_table_offset = u32::from_le_bytes(buf);

        Ok(Self {
            magic,
            version,
            entry_count,
            string_pool_offset,
            prefix_table_offset,
        })
    }
}

/// 5-byte entry in SemanticPool
#[derive(Debug, Clone, Copy)]
pub struct Entry {
    /// Offset into string pool (24-bit)
    pub offset: u32,
    /// Confidence (0-255 → 0.0-1.0)
    pub confidence: u8,
    /// Source (lower 4 bits) + flags (upper 4 bits)
    pub source_flags: u8,
}

impl Entry {
    /// Size of an entry in bytes (5 bytes)
    pub const SIZE: usize = 5;

    /// Create a new entry with the given offset, confidence, and source
    pub fn new(offset: u32, confidence: f32, source: OntologySource) -> Self {
        Self {
            offset: offset & 0x00FF_FFFF,
            confidence: (confidence * 255.0) as u8,
            source_flags: source as u8,
        }
    }

    /// Get the ontology source for this entry
    pub fn source(&self) -> OntologySource {
        OntologySource::from(self.source_flags)
    }

    /// Get the confidence as a float (0.0-1.0)
    pub fn confidence_f32(&self) -> f32 {
        self.confidence as f32 / 255.0
    }

    /// Write entry to a byte stream
    pub fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        let bytes = [
            (self.offset & 0xFF) as u8,
            ((self.offset >> 8) & 0xFF) as u8,
            ((self.offset >> 16) & 0xFF) as u8,
            self.confidence,
            self.source_flags,
        ];
        w.write_all(&bytes)
    }

    /// Read entry from a byte stream
    pub fn read_from<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let mut bytes = [0u8; 5];
        r.read_exact(&mut bytes)?;
        Ok(Self {
            offset: bytes[0] as u32 | ((bytes[1] as u32) << 8) | ((bytes[2] as u32) << 16),
            confidence: bytes[3],
            source_flags: bytes[4],
        })
    }
}

/// Statistics for SemanticPool
#[derive(Debug, Clone, Default)]
pub struct SemanticPoolStats {
    /// Total number of entries in the pool
    pub entry_count: usize,
    /// Number of unique URIs
    pub unique_uris: usize,
    /// Number of Wikidata entries
    pub wikidata_count: usize,
    /// Number of DBpedia entries
    pub dbpedia_count: usize,
    /// Number of Schema.org entries
    pub schemaorg_count: usize,
    /// Number of custom/other entries
    pub custom_count: usize,
    /// Size of the string pool in bytes
    pub string_pool_bytes: usize,
}

impl fmt::Display for SemanticPoolStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SemanticPool Statistics:")?;
        writeln!(f, "  Entries:      {}", self.entry_count)?;
        writeln!(f, "  Unique URIs:  {}", self.unique_uris)?;
        writeln!(f, "  Wikidata:     {}", self.wikidata_count)?;
        writeln!(f, "  DBpedia:      {}", self.dbpedia_count)?;
        writeln!(f, "  Schema.org:   {}", self.schemaorg_count)?;
        writeln!(f, "  Custom:       {}", self.custom_count)?;
        writeln!(f, "  String pool:  {} bytes", self.string_pool_bytes)
    }
}

/// Builder for SemanticPool
#[derive(Debug, Default)]
pub struct SemanticPoolBuilder {
    entries: Vec<Entry>,
    string_pool: Vec<u8>,
    uri_to_offset: HashMap<String, u32>,
    custom_prefixes: Vec<String>,
}

impl SemanticPoolBuilder {
    /// Create a new empty SemanticPoolBuilder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a URI and return its ID (1-based)
    pub fn add(&mut self, uri: &str, confidence: f32, source: OntologySource) -> u32 {
        let suffix = uri.strip_prefix(source.prefix()).unwrap_or(uri);

        let offset = if let Some(&existing) = self.uri_to_offset.get(suffix) {
            existing
        } else {
            let offset = self.string_pool.len() as u32;
            self.string_pool.extend_from_slice(suffix.as_bytes());
            self.string_pool.push(0); // null terminator
            self.uri_to_offset.insert(suffix.to_string(), offset);
            offset
        };

        let entry = Entry::new(offset, confidence, source);
        self.entries.push(entry);
        self.entries.len() as u32
    }

    /// Get statistics
    pub fn stats(&self) -> SemanticPoolStats {
        let mut stats = SemanticPoolStats {
            entry_count: self.entries.len(),
            unique_uris: self.uri_to_offset.len(),
            string_pool_bytes: self.string_pool.len(),
            ..Default::default()
        };

        for entry in &self.entries {
            match entry.source() {
                OntologySource::Wikidata => stats.wikidata_count += 1,
                OntologySource::DBpedia => stats.dbpedia_count += 1,
                OntologySource::SchemaOrg => stats.schemaorg_count += 1,
                OntologySource::Custom => stats.custom_count += 1,
            }
        }

        stats
    }

    /// Write to binary format
    pub fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        let entries_size = self.entries.len() * Entry::SIZE;
        let prefix_table_offset = Header::SIZE + entries_size;
        let string_pool_offset = prefix_table_offset + 2 + self.custom_prefixes.len() * 64;

        let header = Header::new(
            self.entries.len() as u32,
            string_pool_offset as u32,
            prefix_table_offset as u32,
        );
        header.write_to(w)?;

        for entry in &self.entries {
            entry.write_to(w)?;
        }

        // Prefix table (simplified)
        w.write_all(&(self.custom_prefixes.len() as u16).to_le_bytes())?;
        for prefix in &self.custom_prefixes {
            let mut buf = [0u8; 64];
            let bytes = prefix.as_bytes();
            buf[..bytes.len().min(63)].copy_from_slice(&bytes[..bytes.len().min(63)]);
            w.write_all(&buf)?;
        }

        // String pool
        w.write_all(&self.string_pool)?;

        Ok(())
    }
}

/// Immutable SemanticPool for querying
pub struct SemanticPool {
    data: Vec<u8>,
    header: Header,
}

impl SemanticPool {
    /// Load from bytes
    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        let mut cursor = std::io::Cursor::new(data);
        let header = Header::read_from(&mut cursor)?;

        if &header.magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid magic bytes",
            ));
        }

        Ok(Self {
            data: data.to_vec(),
            header,
        })
    }

    /// Get URI by ID (1-based)
    pub fn get(&self, id: u32) -> Option<String> {
        if id == 0 || id > self.header.entry_count {
            return None;
        }

        let entry_offset = Header::SIZE + (id as usize - 1) * Entry::SIZE;
        let mut cursor = std::io::Cursor::new(&self.data[entry_offset..]);
        let entry = Entry::read_from(&mut cursor).ok()?;

        let string_start = self.header.string_pool_offset as usize + entry.offset as usize;
        let string_end = self.data[string_start..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| string_start + p)
            .unwrap_or(self.data.len());

        let suffix = std::str::from_utf8(&self.data[string_start..string_end]).ok()?;
        Some(format!("{}{}", entry.source().prefix(), suffix))
    }

    /// Get confidence by ID (1-based)
    pub fn get_confidence(&self, id: u32) -> Option<f32> {
        if id == 0 || id > self.header.entry_count {
            return None;
        }

        let entry_offset = Header::SIZE + (id as usize - 1) * Entry::SIZE;
        let mut cursor = std::io::Cursor::new(&self.data[entry_offset..]);
        let entry = Entry::read_from(&mut cursor).ok()?;

        Some(entry.confidence_f32())
    }

    /// Get entry count
    pub fn len(&self) -> usize {
        self.header.entry_count as usize
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.header.entry_count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_size() {
        assert_eq!(Entry::SIZE, 5);
    }

    #[test]
    fn test_header_size() {
        assert_eq!(Header::SIZE, 20);
    }

    #[test]
    fn test_entry_encoding() {
        let entry = Entry::new(0x12_3456, 0.75, OntologySource::Wikidata);
        assert_eq!(entry.offset, 0x12_3456 & 0x00FF_FFFF);
        assert_eq!(entry.confidence, 191); // 0.75 * 255
        assert_eq!(entry.source(), OntologySource::Wikidata);
    }

    #[test]
    fn test_entry_roundtrip() {
        let entry = Entry::new(12345, 0.95, OntologySource::DBpedia);
        let mut buf = Vec::new();
        entry.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), 5);

        let mut cursor = std::io::Cursor::new(&buf);
        let read_entry = Entry::read_from(&mut cursor).unwrap();
        assert_eq!(read_entry.offset, entry.offset);
        assert_eq!(read_entry.confidence, entry.confidence);
        assert_eq!(read_entry.source(), entry.source());
    }

    #[test]
    fn test_builder_basic() {
        let mut builder = SemanticPoolBuilder::new();
        let id1 = builder.add(
            "http://www.wikidata.org/entity/Q1490",
            0.95,
            OntologySource::Wikidata,
        );
        let id2 = builder.add(
            "http://dbpedia.org/resource/Tokyo",
            0.90,
            OntologySource::DBpedia,
        );

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_builder_deduplication() {
        let mut builder = SemanticPoolBuilder::new();
        let id1 = builder.add(
            "http://www.wikidata.org/entity/Q1490",
            0.95,
            OntologySource::Wikidata,
        );
        let id2 = builder.add(
            "http://www.wikidata.org/entity/Q1490",
            0.90,
            OntologySource::Wikidata,
        );

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        // Same URI suffix stored once
        assert_eq!(builder.uri_to_offset.len(), 1);
    }

    #[test]
    fn test_builder_custom_prefix() {
        let mut builder = SemanticPoolBuilder::new();
        let id = builder.add(
            "http://example.org/entity/123",
            0.80,
            OntologySource::Custom,
        );
        assert_eq!(id, 1);
    }

    #[test]
    fn test_header_roundtrip() {
        let header = Header::new(100, 1000, 500);
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), Header::SIZE);

        let mut cursor = std::io::Cursor::new(&buf);
        let read_header = Header::read_from(&mut cursor).unwrap();
        assert_eq!(read_header.entry_count, 100);
        assert_eq!(read_header.string_pool_offset, 1000);
    }

    #[test]
    fn test_header_prefix_table() {
        let header = Header::new(50, 2000, 1500);
        assert_eq!(header.prefix_table_offset, 1500);
    }

    #[test]
    fn test_null_terminated_strings() {
        let mut builder = SemanticPoolBuilder::new();
        builder.add(
            "http://www.wikidata.org/entity/Q1",
            0.9,
            OntologySource::Wikidata,
        );
        builder.add(
            "http://www.wikidata.org/entity/Q2",
            0.8,
            OntologySource::Wikidata,
        );

        // Check null terminators in string pool
        #[allow(clippy::naive_bytecount)]
        let null_count = builder.string_pool.iter().filter(|&&b| b == 0).count();
        assert_eq!(null_count, 2);
    }

    #[test]
    fn test_full_roundtrip() {
        let mut builder = SemanticPoolBuilder::new();
        builder.add(
            "http://www.wikidata.org/entity/Q1490",
            0.95,
            OntologySource::Wikidata,
        );
        builder.add(
            "http://dbpedia.org/resource/Tokyo",
            0.90,
            OntologySource::DBpedia,
        );

        let mut buf = Vec::new();
        builder.write_to(&mut buf).unwrap();

        let pool = SemanticPool::from_bytes(&buf).unwrap();
        assert_eq!(pool.len(), 2);
        assert_eq!(
            pool.get(1),
            Some("http://www.wikidata.org/entity/Q1490".to_string())
        );
        assert_eq!(
            pool.get(2),
            Some("http://dbpedia.org/resource/Tokyo".to_string())
        );
    }

    #[test]
    fn test_stats_display() {
        let mut builder = SemanticPoolBuilder::new();
        builder.add(
            "http://www.wikidata.org/entity/Q1",
            0.9,
            OntologySource::Wikidata,
        );
        builder.add(
            "http://dbpedia.org/resource/Test",
            0.8,
            OntologySource::DBpedia,
        );

        let stats = builder.stats();
        let display = format!("{}", stats);
        assert!(display.contains("Entries:"));
        assert!(display.contains("Wikidata:"));
    }
}
