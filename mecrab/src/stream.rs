//! Streaming text processing for morphological analysis
//!
//! This module provides utilities for processing text in a streaming fashion,
//! handling sentence boundaries, and managing incremental tokenization.

use std::io::{BufRead, BufReader, Read};

/// Sentence boundary detection mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentenceBoundary {
    /// Split on newlines only
    Newline,
    /// Split on Japanese sentence-ending punctuation (。！？)
    JapanesePunctuation,
    /// Split on all punctuation (。！？.!?)
    AllPunctuation,
    /// Custom delimiter
    Custom(char),
}

impl Default for SentenceBoundary {
    fn default() -> Self {
        Self::JapanesePunctuation
    }
}

/// Buffer for accumulating partial sentences
#[derive(Debug, Default)]
pub struct SentenceBuffer {
    buffer: String,
    boundary: SentenceBoundary,
}

impl SentenceBuffer {
    /// Create a new sentence buffer with given boundary mode
    pub fn new(boundary: SentenceBoundary) -> Self {
        Self {
            buffer: String::new(),
            boundary,
        }
    }

    /// Check if a character is a sentence boundary
    pub fn is_boundary(&self, c: char) -> bool {
        match self.boundary {
            SentenceBoundary::Newline => c == '\n',
            SentenceBoundary::JapanesePunctuation => matches!(c, '。' | '！' | '？'),
            SentenceBoundary::AllPunctuation => {
                matches!(c, '。' | '！' | '？' | '.' | '!' | '?')
            }
            SentenceBoundary::Custom(delim) => c == delim,
        }
    }

    /// Push text into the buffer and return complete sentences
    pub fn push(&mut self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        for c in text.chars() {
            current.push(c);
            if self.is_boundary(c) {
                let sentence = std::mem::take(&mut self.buffer) + &current;
                if !sentence.trim().is_empty() {
                    sentences.push(sentence);
                }
                current.clear();
            }
        }

        self.buffer.push_str(&current);
        sentences
    }

    /// Flush any remaining content as a sentence
    pub fn flush(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            None
        } else {
            let sentence = std::mem::take(&mut self.buffer);
            if sentence.trim().is_empty() {
                None
            } else {
                Some(sentence)
            }
        }
    }

    /// Check if buffer has pending content
    pub fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get pending content length
    pub fn pending_len(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Streaming text reader that yields sentences
pub struct SentenceReader<R: Read> {
    reader: BufReader<R>,
    buffer: SentenceBuffer,
    line_buffer: String,
    /// Queue of sentences ready to be returned
    pending_sentences: Vec<String>,
}

impl<R: Read> SentenceReader<R> {
    /// Create a new sentence reader
    pub fn new(reader: R, boundary: SentenceBoundary) -> Self {
        Self {
            reader: BufReader::new(reader),
            buffer: SentenceBuffer::new(boundary),
            line_buffer: String::new(),
            pending_sentences: Vec::new(),
        }
    }

    /// Read and yield the next sentence
    pub fn next_sentence(&mut self) -> std::io::Result<Option<String>> {
        // First check if we have pending sentences from previous reads
        if !self.pending_sentences.is_empty() {
            return Ok(Some(self.pending_sentences.remove(0)));
        }

        loop {
            self.line_buffer.clear();
            let bytes_read = self.reader.read_line(&mut self.line_buffer)?;

            if bytes_read == 0 {
                // EOF - flush any pending content
                return Ok(self.buffer.flush());
            }

            let mut sentences = self.buffer.push(&self.line_buffer);
            if !sentences.is_empty() {
                // Take first sentence, queue the rest
                let first = sentences.remove(0);
                self.pending_sentences = sentences;
                return Ok(Some(first));
            }
        }
    }
}

/// Iterator adapter for sentence reading
pub struct SentenceIter<R: Read> {
    reader: SentenceReader<R>,
    done: bool,
}

impl<R: Read> SentenceIter<R> {
    /// Create a new sentence iterator
    pub fn new(reader: R, boundary: SentenceBoundary) -> Self {
        Self {
            reader: SentenceReader::new(reader, boundary),
            done: false,
        }
    }
}

impl<R: Read> Iterator for SentenceIter<R> {
    type Item = std::io::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        match self.reader.next_sentence() {
            Ok(Some(sentence)) => Some(Ok(sentence)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

/// Window-based text processing for context-aware analysis
#[derive(Debug)]
pub struct TextWindow {
    /// Previous sentences (context)
    context: Vec<String>,
    /// Maximum context size
    max_context: usize,
    /// Current sentence being processed
    current: Option<String>,
}

impl TextWindow {
    /// Create a new text window with given context size
    pub fn new(max_context: usize) -> Self {
        Self {
            context: Vec::with_capacity(max_context),
            max_context,
            current: None,
        }
    }

    /// Push a new sentence, returning the sentence to process
    pub fn push(&mut self, sentence: String) -> &str {
        // Move current to context
        if let Some(prev) = self.current.take() {
            self.context.push(prev);
            while self.context.len() > self.max_context {
                self.context.remove(0);
            }
        }

        self.current = Some(sentence);
        self.current.as_ref().unwrap()
    }

    /// Get the current sentence
    pub fn current(&self) -> Option<&str> {
        self.current.as_deref()
    }

    /// Get context sentences
    pub fn context(&self) -> &[String] {
        &self.context
    }

    /// Get combined context as a single string
    pub fn context_string(&self) -> String {
        self.context.join("")
    }
}

/// Chunk-based processing for large texts
#[derive(Debug)]
pub struct TextChunker {
    /// Maximum chunk size in characters
    max_chunk_size: usize,
    /// Overlap between chunks
    overlap: usize,
}

impl TextChunker {
    /// Create a new text chunker
    pub fn new(max_chunk_size: usize, overlap: usize) -> Self {
        Self {
            max_chunk_size,
            overlap: overlap.min(max_chunk_size / 2),
        }
    }

    /// Split text into overlapping chunks
    pub fn chunk<'a>(&self, text: &'a str) -> Vec<&'a str> {
        if text.len() <= self.max_chunk_size {
            return vec![text];
        }

        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + self.max_chunk_size).min(chars.len());

            // Find byte positions
            let byte_start: usize = chars[..start].iter().map(|c| c.len_utf8()).sum();
            let byte_end: usize = chars[..end].iter().map(|c| c.len_utf8()).sum();

            chunks.push(&text[byte_start..byte_end]);

            if end >= chars.len() {
                break;
            }

            start = end - self.overlap;
        }

        chunks
    }

    /// Split text at sentence boundaries when possible
    pub fn chunk_at_boundaries(&self, text: &str, boundary: SentenceBoundary) -> Vec<String> {
        let buffer = SentenceBuffer::new(boundary);
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        for c in text.chars() {
            current_chunk.push(c);

            if buffer.is_boundary(c) && current_chunk.len() >= self.max_chunk_size / 2 {
                if !current_chunk.trim().is_empty() {
                    chunks.push(std::mem::take(&mut current_chunk));
                }
            }

            if current_chunk.len() >= self.max_chunk_size {
                if !current_chunk.trim().is_empty() {
                    chunks.push(std::mem::take(&mut current_chunk));
                }
            }
        }

        if !current_chunk.trim().is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }
}

impl Default for TextChunker {
    fn default() -> Self {
        Self::new(4096, 256)
    }
}

/// Progress tracking for stream processing
#[derive(Debug, Clone, Default)]
pub struct StreamProgress {
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Total sentences processed
    pub sentences_processed: u64,
    /// Total tokens generated
    pub tokens_generated: u64,
    /// Processing errors encountered
    pub errors: u64,
}

impl StreamProgress {
    /// Create a new progress tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a processed sentence
    pub fn record_sentence(&mut self, bytes: usize, tokens: usize) {
        self.bytes_processed += bytes as u64;
        self.sentences_processed += 1;
        self.tokens_generated += tokens as u64;
    }

    /// Record an error
    pub fn record_error(&mut self) {
        self.errors += 1;
    }

    /// Get processing rate (bytes per sentence)
    pub fn bytes_per_sentence(&self) -> f64 {
        if self.sentences_processed == 0 {
            0.0
        } else {
            self.bytes_processed as f64 / self.sentences_processed as f64
        }
    }

    /// Get token density (tokens per sentence)
    pub fn tokens_per_sentence(&self) -> f64 {
        if self.sentences_processed == 0 {
            0.0
        } else {
            self.tokens_generated as f64 / self.sentences_processed as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_buffer_japanese() {
        let mut buffer = SentenceBuffer::new(SentenceBoundary::JapanesePunctuation);
        let sentences = buffer.push("これはテストです。もう一つの文。");
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "これはテストです。");
        assert_eq!(sentences[1], "もう一つの文。");
    }

    #[test]
    fn test_sentence_buffer_partial() {
        let mut buffer = SentenceBuffer::new(SentenceBoundary::JapanesePunctuation);
        let sentences1 = buffer.push("これは");
        assert!(sentences1.is_empty());
        assert!(buffer.has_pending());

        let sentences2 = buffer.push("テストです。");
        assert_eq!(sentences2.len(), 1);
        assert_eq!(sentences2[0], "これはテストです。");
    }

    #[test]
    fn test_sentence_buffer_flush() {
        let mut buffer = SentenceBuffer::new(SentenceBoundary::JapanesePunctuation);
        buffer.push("未完成の文");
        assert!(buffer.has_pending());

        let flushed = buffer.flush();
        assert_eq!(flushed, Some("未完成の文".to_string()));
        assert!(!buffer.has_pending());
    }

    #[test]
    fn test_sentence_buffer_newline() {
        let mut buffer = SentenceBuffer::new(SentenceBoundary::Newline);
        let sentences = buffer.push("行1\n行2\n");
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn test_sentence_buffer_custom() {
        let mut buffer = SentenceBuffer::new(SentenceBoundary::Custom('|'));
        let sentences = buffer.push("part1|part2|");
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn test_text_window() {
        let mut window = TextWindow::new(2);

        window.push("文1。".to_string());
        assert_eq!(window.current(), Some("文1。"));
        assert!(window.context().is_empty());

        window.push("文2。".to_string());
        assert_eq!(window.current(), Some("文2。"));
        assert_eq!(window.context(), &["文1。".to_string()]);

        window.push("文3。".to_string());
        assert_eq!(window.context().len(), 2);
    }

    #[test]
    fn test_text_chunker_small() {
        let chunker = TextChunker::new(100, 10);
        let chunks = chunker.chunk("短いテキスト");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_text_chunker_large() {
        let chunker = TextChunker::new(5, 2);
        let text = "あいうえおかきくけこ";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_text_chunker_boundaries() {
        let chunker = TextChunker::new(20, 5);
        let chunks = chunker.chunk_at_boundaries(
            "短い。もう一つ。さらに。",
            SentenceBoundary::JapanesePunctuation,
        );
        assert!(!chunks.is_empty());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_stream_progress() {
        let mut progress = StreamProgress::new();
        progress.record_sentence(100, 10);
        progress.record_sentence(50, 5);

        assert_eq!(progress.sentences_processed, 2);
        assert_eq!(progress.bytes_processed, 150);
        assert_eq!(progress.tokens_generated, 15);
        assert_eq!(progress.bytes_per_sentence(), 75.0);
        assert_eq!(progress.tokens_per_sentence(), 7.5);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_stream_progress_empty() {
        let progress = StreamProgress::new();
        assert_eq!(progress.bytes_per_sentence(), 0.0);
        assert_eq!(progress.tokens_per_sentence(), 0.0);
    }

    #[test]
    fn test_sentence_reader() {
        use std::io::Cursor;

        let text = "文1。文2。";
        let cursor = Cursor::new(text);
        let mut reader = SentenceReader::new(cursor, SentenceBoundary::JapanesePunctuation);

        let s1 = reader.next_sentence().unwrap();
        assert!(s1.is_some());

        let s2 = reader.next_sentence().unwrap();
        assert!(s2.is_some());
    }

    #[test]
    fn test_sentence_iter() {
        use std::io::Cursor;

        let text = "テスト1。テスト2。テスト3。";
        let cursor = Cursor::new(text);
        let iter = SentenceIter::new(cursor, SentenceBoundary::JapanesePunctuation);

        let sentences: Vec<_> = iter.filter_map(Result::ok).collect();
        assert_eq!(sentences.len(), 3);
    }
}
