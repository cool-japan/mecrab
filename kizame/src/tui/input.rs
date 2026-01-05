//! Input handling for the TUI
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Handles keyboard navigation and commands.
//!
//! This module provides reusable command abstractions and key bindings
//! that can be used for future TUI extensions.

use crossterm::event::KeyCode;

/// Input handler for the TUI
pub struct InputHandler;

impl InputHandler {
    /// Create a new input handler
    pub fn new() -> Self {
        Self
    }
}

impl Default for InputHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Command to execute based on user input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Quit the application
    Quit,
    /// Show/hide help
    ToggleHelp,
    /// Move to the next position (right)
    MoveRight,
    /// Move to the previous position (left)
    MoveLeft,
    /// Move to the next node at current position (down)
    MoveDown,
    /// Move to the previous node at current position (up)
    MoveUp,
    /// Go to the start (BOS)
    GotoStart,
    /// Go to the end (EOS)
    GotoEnd,
    /// Jump to next best path node
    NextBestPathNode,
    /// Jump to previous best path node
    PrevBestPathNode,
    /// No command
    None,
}

impl Command {
    /// Parse a key code into a command
    pub fn from_key(key: KeyCode) -> Self {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => Command::Quit,
            KeyCode::Char('?') | KeyCode::F(1) => Command::ToggleHelp,
            KeyCode::Right | KeyCode::Char('l') => Command::MoveRight,
            KeyCode::Left | KeyCode::Char('h') => Command::MoveLeft,
            KeyCode::Down | KeyCode::Char('j') => Command::MoveDown,
            KeyCode::Up | KeyCode::Char('k') => Command::MoveUp,
            KeyCode::Home | KeyCode::Char('g') => Command::GotoStart,
            KeyCode::End | KeyCode::Char('G') => Command::GotoEnd,
            KeyCode::Char('n') => Command::NextBestPathNode,
            KeyCode::Char('p') => Command::PrevBestPathNode,
            _ => Command::None,
        }
    }

    /// Get the description of a command
    pub fn description(&self) -> &'static str {
        match self {
            Command::Quit => "Quit the application",
            Command::ToggleHelp => "Show/hide help",
            Command::MoveRight => "Move to next position",
            Command::MoveLeft => "Move to previous position",
            Command::MoveDown => "Select next node at position",
            Command::MoveUp => "Select previous node at position",
            Command::GotoStart => "Go to start (BOS)",
            Command::GotoEnd => "Go to end (EOS)",
            Command::NextBestPathNode => "Jump to next best path node",
            Command::PrevBestPathNode => "Jump to previous best path node",
            Command::None => "No action",
        }
    }
}

/// Key binding information
#[derive(Debug, Clone)]
pub struct KeyBinding {
    /// Primary key
    pub primary: &'static str,
    /// Alternative key (if any)
    pub alternative: Option<&'static str>,
    /// Description
    pub description: &'static str,
}

/// Get all key bindings
pub fn get_key_bindings() -> Vec<KeyBinding> {
    vec![
        KeyBinding {
            primary: "h",
            alternative: Some("Left"),
            description: "Move to previous position",
        },
        KeyBinding {
            primary: "l",
            alternative: Some("Right"),
            description: "Move to next position",
        },
        KeyBinding {
            primary: "k",
            alternative: Some("Up"),
            description: "Select previous node",
        },
        KeyBinding {
            primary: "j",
            alternative: Some("Down"),
            description: "Select next node",
        },
        KeyBinding {
            primary: "g",
            alternative: Some("Home"),
            description: "Go to start (BOS)",
        },
        KeyBinding {
            primary: "G",
            alternative: Some("End"),
            description: "Go to end (EOS)",
        },
        KeyBinding {
            primary: "n",
            alternative: None,
            description: "Next best path node",
        },
        KeyBinding {
            primary: "p",
            alternative: None,
            description: "Previous best path node",
        },
        KeyBinding {
            primary: "?",
            alternative: Some("F1"),
            description: "Toggle help",
        },
        KeyBinding {
            primary: "q",
            alternative: Some("Esc"),
            description: "Quit",
        },
    ]
}
