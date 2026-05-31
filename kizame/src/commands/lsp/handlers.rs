//! `impl LanguageServer for MeCrabLanguageServer` — all LSP request handlers.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use super::server::{
    extract_prefix_at, parse_ipadic_features, pos_to_completion_kind, MeCrabLanguageServer,
};
use tower_lsp::jsonrpc::Result as LspResult;
use tower_lsp::lsp_types::{
    CompletionItem, CompletionOptions, CompletionParams, CompletionResponse,
    DiagnosticOptions, DiagnosticServerCapabilities, DidChangeTextDocumentParams,
    DidOpenTextDocumentParams, Hover, HoverContents, HoverParams, HoverProviderCapability,
    InitializeParams, InitializeResult, InitializedParams, MarkupContent, MarkupKind,
    MessageType, ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind,
    WorkDoneProgressOptions,
};
use tower_lsp::LanguageServer;

#[tower_lsp::async_trait]
impl LanguageServer for MeCrabLanguageServer {
    async fn initialize(&self, _params: InitializeParams) -> LspResult<InitializeResult> {
        self.try_load_dictionary().await;

        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec!["@".to_string()]),
                    resolve_provider: Some(false),
                    work_done_progress_options: WorkDoneProgressOptions {
                        work_done_progress: Some(false),
                    },
                    ..Default::default()
                }),
                diagnostic_provider: Some(DiagnosticServerCapabilities::Options(
                    DiagnosticOptions {
                        identifier: Some("mecrab".to_string()),
                        inter_file_dependencies: false,
                        workspace_diagnostics: false,
                        work_done_progress_options: WorkDoneProgressOptions {
                            work_done_progress: Some(false),
                        },
                    },
                )),
                ..Default::default()
            },
            server_info: Some(tower_lsp::lsp_types::ServerInfo {
                name: "mecrab-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _params: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "MeCrab LSP server initialized.")
            .await;
    }

    async fn shutdown(&self) -> LspResult<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        let text = params.text_document.text.clone();
        self.documents.insert(uri.clone(), text.clone());
        self.run_diagnostics(uri, &text).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        // We requested FULL sync, so exactly one change event with the full text.
        if let Some(change) = params.content_changes.into_iter().last() {
            let text = change.text;
            self.documents.insert(uri.clone(), text.clone());
            self.run_diagnostics(uri, &text).await;
        }
    }

    async fn hover(&self, params: HoverParams) -> LspResult<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        // Retrieve document text
        let text = match self.documents.get(uri) {
            Some(entry) => entry.value().clone(),
            None => return Ok(None),
        };

        // Extract the word under the cursor
        let (word, _col_start, _col_end) = match Self::word_at_position(&text, position) {
            Some(w) => w,
            None => return Ok(None),
        };

        // Analyze the extracted word
        let guard = self.mecrab.read().await;
        let instance = match guard.as_ref() {
            Some(m) => m,
            None => return Ok(None),
        };

        let result = match instance.parse(&word) {
            Ok(r) => r,
            Err(_) => return Ok(None),
        };

        if result.morphemes.is_empty() {
            return Ok(None);
        }

        // Build a Markdown hover card showing each morpheme
        let mut md = String::from("**MeCrab Analysis**\n\n");
        md.push_str("| Surface | POS | Sub-POS | Base Form | Reading |\n");
        md.push_str("|---------|-----|---------|-----------|---------|\n");

        for morpheme in &result.morphemes {
            let (pos, sub_pos, base_form, reading) = parse_ipadic_features(&morpheme.feature);
            md.push_str(&format!(
                "| `{}` | {} | {} | {} | {} |\n",
                morpheme.surface, pos, sub_pos, base_form, reading
            ));
        }

        Ok(Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: md,
            }),
            range: None,
        }))
    }

    async fn completion(&self, params: CompletionParams) -> LspResult<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        let text = match self.documents.get(uri) {
            Some(entry) => entry.value().clone(),
            None => return Ok(None),
        };

        let guard = self.mecrab.read().await;
        let instance = match guard.as_ref() {
            Some(m) => m,
            None => return Ok(Some(CompletionResponse::Array(vec![]))),
        };

        // Extract the prefix typed so far on the current line (up to cursor)
        let prefix = extract_prefix_at(&text, position);
        if prefix.is_empty() {
            return Ok(Some(CompletionResponse::Array(vec![])));
        }

        // Analyse the prefix to get completions from its constituent morphemes
        let result = match instance.parse(&prefix) {
            Ok(r) => r,
            Err(_) => return Ok(Some(CompletionResponse::Array(vec![]))),
        };

        let items: Vec<CompletionItem> = result
            .morphemes
            .iter()
            .filter(|m| !m.surface.trim().is_empty())
            .map(|morpheme| {
                let (pos, _sub, base_form, reading) = parse_ipadic_features(&morpheme.feature);
                let label = if base_form != "*" && base_form != morpheme.surface {
                    base_form.clone()
                } else {
                    morpheme.surface.clone()
                };
                let detail = if reading != "*" {
                    format!("{pos} [{reading}]")
                } else {
                    pos.clone()
                };
                CompletionItem {
                    label,
                    kind: Some(pos_to_completion_kind(&pos)),
                    detail: Some(detail),
                    ..Default::default()
                }
            })
            .collect();

        Ok(Some(CompletionResponse::Array(items)))
    }
}
