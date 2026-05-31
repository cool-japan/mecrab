//! Builder for configuring a [`MeCrab`] instance.
//!
//! Use [`MeCrabBuilder`] (obtained via [`MeCrab::builder()`]) to set the
//! dictionary directory, user dictionary, semantic pool, vector pool, IPA
//! output, and output format before constructing the analyzer.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::MeCrab;
use crate::dict::Dictionary;
use crate::dict::provider::{AutoDetectProvider, DictionaryProvider, IpadicProvider};
use crate::error::Result;
use crate::types::OutputFormat;
use crate::vectors;

/// Builder for configuring MeCrab instance
pub struct MeCrabBuilder {
    pub(crate) dicdir: Option<PathBuf>,
    pub(crate) userdic: Option<PathBuf>,
    pub(crate) semantic_pool: Option<PathBuf>,
    pub(crate) vector_pool: Option<PathBuf>,
    pub(crate) with_semantic: bool,
    pub(crate) with_ipa: bool,
    pub(crate) with_vector: bool,
    pub(crate) output_format: OutputFormat,
    /// Optional custom dictionary provider; defaults to `IpadicProvider`
    pub(crate) provider: Option<Arc<dyn DictionaryProvider>>,
}

impl fmt::Debug for MeCrabBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MeCrabBuilder")
            .field("dicdir", &self.dicdir)
            .field("userdic", &self.userdic)
            .field("semantic_pool", &self.semantic_pool)
            .field("vector_pool", &self.vector_pool)
            .field("with_semantic", &self.with_semantic)
            .field("with_ipa", &self.with_ipa)
            .field("with_vector", &self.with_vector)
            .field("output_format", &self.output_format)
            .field(
                "provider",
                &self.provider.as_ref().map(|_| "<DictionaryProvider>"),
            )
            .finish()
    }
}

impl Default for MeCrabBuilder {
    fn default() -> Self {
        Self {
            dicdir: None,
            userdic: None,
            semantic_pool: None,
            vector_pool: None,
            with_semantic: false,
            with_ipa: false,
            with_vector: false,
            output_format: OutputFormat::default(),
            provider: None,
        }
    }
}

impl MeCrabBuilder {
    /// Create a new builder with default settings
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the dictionary directory
    #[must_use]
    pub fn dicdir(mut self, path: Option<PathBuf>) -> Self {
        self.dicdir = path;
        self
    }

    /// Set the user dictionary path
    #[must_use]
    pub fn userdic(mut self, path: Option<PathBuf>) -> Self {
        self.userdic = path;
        self
    }

    /// Set the semantic pool path
    #[must_use]
    pub fn semantic_pool(mut self, path: Option<PathBuf>) -> Self {
        self.semantic_pool = path;
        self
    }

    /// Enable semantic URI output (requires semantic pool to be loaded)
    #[must_use]
    pub fn with_semantic(mut self, enabled: bool) -> Self {
        self.with_semantic = enabled;
        self
    }

    /// Enable IPA pronunciation output
    #[must_use]
    pub fn with_ipa(mut self, enabled: bool) -> Self {
        self.with_ipa = enabled;
        self
    }

    /// Set the vector pool file path (vectors.bin)
    #[must_use]
    pub fn vector_pool(mut self, path: Option<PathBuf>) -> Self {
        self.vector_pool = path;
        self
    }

    /// Enable vector embedding output
    #[must_use]
    pub fn with_vector(mut self, enabled: bool) -> Self {
        self.with_vector = enabled;
        self
    }

    /// Set the output format
    #[must_use]
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Set a custom `DictionaryProvider` for feature parsing.
    ///
    /// By default, `IpadicProvider` is used.  Use this method to switch to
    /// `UnidicProvider`, `NeologdProvider`, or a custom implementation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mecrab::{MeCrab, UnidicProvider};
    ///
    /// let mecrab = MeCrab::builder()
    ///     .with_provider(UnidicProvider)
    ///     .build()?;
    /// # Ok::<(), mecrab::Error>(())
    /// ```
    #[must_use]
    pub fn with_provider(mut self, provider: impl DictionaryProvider + 'static) -> Self {
        self.provider = Some(Arc::new(provider));
        self
    }

    /// Build the MeCrab instance
    ///
    /// # Errors
    ///
    /// Returns an error if the dictionary cannot be loaded.
    pub fn build(self) -> Result<MeCrab> {
        let dictionary = match (self.dicdir, self.semantic_pool) {
            (Some(dicdir), Some(semantic_path)) => {
                Dictionary::load_with_semantics(&dicdir, &semantic_path)?
            }
            (Some(dicdir), None) => {
                // Try to auto-load semantic.bin from dicdir if it exists
                let semantic_path = dicdir.join("semantic.bin");
                if semantic_path.exists() {
                    Dictionary::load_with_semantics(&dicdir, &semantic_path)?
                } else {
                    Dictionary::load(&dicdir)?
                }
            }
            (None, Some(semantic_path)) => {
                let dict = Dictionary::default_dictionary()?;
                let pool_file = std::fs::File::open(&semantic_path)?;
                let pool_data = unsafe { memmap2::Mmap::map(&pool_file)? };
                let pool = crate::semantic::pool::SemanticPool::from_bytes(&pool_data)?;
                let mut dict_mut = dict;
                dict_mut.semantic_pool = Some(Arc::new(pool));
                dict_mut
            }
            (None, None) => {
                // Try to auto-load from default directory
                Dictionary::default_dictionary()?
            }
        };

        // Load vector store if path provided
        let vector_store = if let Some(vector_path) = self.vector_pool {
            Some(Arc::new(vectors::VectorStore::from_file(&vector_path)?))
        } else {
            None
        };

        // Auto-detect provider from dictionary if no explicit provider was set.
        // We sample the first token's feature field count to choose IpadicProvider
        // vs UnidicProvider vs a fallback, without requiring the caller to know the
        // dictionary format in advance.
        let provider: Arc<dyn DictionaryProvider> = match self.provider {
            Some(explicit) => explicit,
            None => {
                if let Some(feature_count) = dictionary.sample_feature_count() {
                    Arc::new(AutoDetectProvider::detect(feature_count))
                } else {
                    Arc::new(IpadicProvider)
                }
            }
        };

        Ok(MeCrab {
            dictionary: Arc::new(ArcSwap::new(Arc::new(Arc::new(dictionary)))),
            output_format: self.output_format,
            semantic_enabled: self.with_semantic,
            ipa_enabled: self.with_ipa,
            vector_enabled: self.with_vector,
            vector_store,
            provider,
        })
    }
}
