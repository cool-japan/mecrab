//! TUI Lattice Debugger for MeCrab
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module provides an interactive terminal UI for exploring
//! the lattice structure and understanding path selection decisions.

mod cost_panel;
#[allow(dead_code)]
mod input;
mod lattice_view;

pub use cost_panel::CostPanel;
pub use lattice_view::LatticeView;

use std::io::{self, Stdout};
use std::time::Duration;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

use mecrab::debug::DebugSession;

/// The TUI application state
pub struct App {
    /// The debug session containing lattice data
    session: DebugSession,
    /// Currently selected position in the lattice
    selected_pos: usize,
    /// Currently selected node index at the position
    selected_idx: usize,
    /// Whether the app should quit
    should_quit: bool,
    /// Scroll offset for the lattice view
    scroll_offset: usize,
    /// Show help overlay
    show_help: bool,
    /// Last error message (if any)
    error_message: Option<String>,
}

impl App {
    /// Create a new App from a DebugSession
    pub fn new(session: DebugSession) -> Self {
        Self {
            session,
            selected_pos: 0,
            selected_idx: 0,
            should_quit: false,
            scroll_offset: 0,
            show_help: false,
            error_message: None,
        }
    }

    /// Run the TUI application
    pub fn run(&mut self) -> io::Result<()> {
        let mut terminal = setup_terminal()?;
        let result = self.main_loop(&mut terminal);
        restore_terminal(&mut terminal)?;
        result
    }

    /// Main event loop
    fn main_loop(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
        while !self.should_quit {
            terminal.draw(|frame| self.draw(frame))?;

            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        self.handle_key(key.code);
                    }
                }
            }
        }
        Ok(())
    }

    /// Handle keyboard input
    fn handle_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Char('?') | KeyCode::F(1) => self.show_help = !self.show_help,
            KeyCode::Right | KeyCode::Char('l') => self.move_right(),
            KeyCode::Left | KeyCode::Char('h') => self.move_left(),
            KeyCode::Down | KeyCode::Char('j') => self.move_down(),
            KeyCode::Up | KeyCode::Char('k') => self.move_up(),
            KeyCode::Home | KeyCode::Char('g') => self.goto_start(),
            KeyCode::End | KeyCode::Char('G') => self.goto_end(),
            KeyCode::Char('n') => self.goto_next_best_path_node(),
            KeyCode::Char('p') => self.goto_prev_best_path_node(),
            _ => {}
        }
    }

    /// Move selection to the right (next position)
    fn move_right(&mut self) {
        if self.selected_pos + 1 < self.session.num_positions() {
            self.selected_pos += 1;
            // Clamp selected_idx to valid range at new position
            let max_idx = self
                .session
                .nodes_at
                .get(self.selected_pos)
                .map(|n| n.len().saturating_sub(1))
                .unwrap_or(0);
            self.selected_idx = self.selected_idx.min(max_idx);
            self.update_scroll();
        }
    }

    /// Move selection to the left (previous position)
    fn move_left(&mut self) {
        if self.selected_pos > 0 {
            self.selected_pos -= 1;
            // Clamp selected_idx to valid range at new position
            let max_idx = self
                .session
                .nodes_at
                .get(self.selected_pos)
                .map(|n| n.len().saturating_sub(1))
                .unwrap_or(0);
            self.selected_idx = self.selected_idx.min(max_idx);
            self.update_scroll();
        }
    }

    /// Move selection down (next node at same position)
    fn move_down(&mut self) {
        let max_idx = self
            .session
            .nodes_at
            .get(self.selected_pos)
            .map(|n| n.len().saturating_sub(1))
            .unwrap_or(0);
        if self.selected_idx < max_idx {
            self.selected_idx += 1;
            self.update_scroll();
        }
    }

    /// Move selection up (previous node at same position)
    fn move_up(&mut self) {
        if self.selected_idx > 0 {
            self.selected_idx -= 1;
            self.update_scroll();
        }
    }

    /// Go to the start of the lattice
    fn goto_start(&mut self) {
        self.selected_pos = 0;
        self.selected_idx = 0;
        self.scroll_offset = 0;
    }

    /// Go to the end of the lattice
    fn goto_end(&mut self) {
        self.selected_pos = self.session.num_positions().saturating_sub(1);
        self.selected_idx = 0;
        self.update_scroll();
    }

    /// Go to the next node on the best path
    fn goto_next_best_path_node(&mut self) {
        for &(pos, idx) in &self.session.best_path {
            if pos > self.selected_pos {
                self.selected_pos = pos;
                self.selected_idx = idx;
                self.update_scroll();
                return;
            }
        }
    }

    /// Go to the previous node on the best path
    fn goto_prev_best_path_node(&mut self) {
        for &(pos, idx) in self.session.best_path.iter().rev() {
            if pos < self.selected_pos {
                self.selected_pos = pos;
                self.selected_idx = idx;
                self.update_scroll();
                return;
            }
        }
    }

    /// Update scroll offset to keep selected position visible
    fn update_scroll(&mut self) {
        // Calculate line number for selected position
        // Each position takes ~3-10 lines (header + nodes)
        // We'll approximate with the position count
        let selected_line = self.calculate_line_for_position(self.selected_pos);

        // Keep selected position in view (with some margin)
        const SCROLL_MARGIN: usize = 3;
        const VISIBLE_LINES: usize = 20; // Approximate visible lines in lattice view

        if selected_line < self.scroll_offset + SCROLL_MARGIN {
            self.scroll_offset = selected_line.saturating_sub(SCROLL_MARGIN);
        } else if selected_line >= self.scroll_offset + VISIBLE_LINES - SCROLL_MARGIN {
            self.scroll_offset = selected_line.saturating_sub(VISIBLE_LINES - SCROLL_MARGIN - 1);
        }
    }

    /// Calculate approximate line number for a given position
    fn calculate_line_for_position(&self, pos: usize) -> usize {
        let mut line = 2; // Header lines
        for p in 0..pos {
            line += 1; // Position header
            if let Some(nodes) = self.session.nodes_at.get(p) {
                line += nodes.len(); // Nodes at this position
            }
            line += 1; // Empty line separator
        }
        line
    }

    /// Draw the UI
    fn draw(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Title bar
                Constraint::Min(10),   // Main content
                Constraint::Length(3), // Status bar
            ])
            .split(frame.area());

        self.draw_title_bar(frame, chunks[0]);
        self.draw_main_content(frame, chunks[1]);
        self.draw_status_bar(frame, chunks[2]);

        if self.show_help {
            self.draw_help_overlay(frame);
        }
    }

    /// Draw the title bar
    fn draw_title_bar(&self, frame: &mut Frame, area: Rect) {
        let title = format!(
            " MeCrab Lattice Debugger | Input: \"{}\" | Positions: {} | Best Cost: {}",
            self.session.text,
            self.session.num_positions(),
            self.session.best_path_cost
        );

        let block = Paragraph::new(title)
            .style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .block(Block::default().borders(Borders::ALL).title("Matrix"));

        frame.render_widget(block, area);
    }

    /// Draw the main content area
    fn draw_main_content(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(65), // Lattice view
                Constraint::Percentage(35), // Cost panel
            ])
            .split(area);

        // Draw lattice view with scroll offset
        let mut lattice_view =
            LatticeView::new(&self.session, self.selected_pos, self.selected_idx);
        lattice_view.set_scroll_offset(self.scroll_offset);
        lattice_view.draw(frame, chunks[0]);

        // Draw cost panel
        let cost_panel = CostPanel::new(&self.session, self.selected_pos, self.selected_idx);
        cost_panel.draw(frame, chunks[1]);
    }

    /// Draw the status bar
    fn draw_status_bar(&self, frame: &mut Frame, area: Rect) {
        let is_on_best_path = self
            .session
            .is_on_best_path(self.selected_pos, self.selected_idx);

        let status = if let Some(ref err) = self.error_message {
            Span::styled(format!(" Error: {}", err), Style::default().fg(Color::Red))
        } else if is_on_best_path {
            Span::styled(
                " ON BEST PATH ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled(" Alternative path ", Style::default().fg(Color::Yellow))
        };

        let help_hint = Span::styled(
            " | Press ? for help, q to quit",
            Style::default().fg(Color::DarkGray),
        );

        let pos_info = Span::styled(
            format!(" | Pos: {} Idx: {} ", self.selected_pos, self.selected_idx),
            Style::default().fg(Color::Cyan),
        );

        let line = Line::from(vec![status, pos_info, help_hint]);

        let paragraph =
            Paragraph::new(line).block(Block::default().borders(Borders::ALL).title("Status"));

        frame.render_widget(paragraph, area);
    }

    /// Draw the help overlay
    fn draw_help_overlay(&self, frame: &mut Frame) {
        let area = centered_rect(60, 70, frame.area());

        let help_text = vec![
            Line::from(Span::styled(
                "Keyboard Shortcuts",
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .fg(Color::Cyan),
            )),
            Line::from(""),
            Line::from(vec![Span::styled(
                "Navigation:",
                Style::default().add_modifier(Modifier::BOLD),
            )]),
            Line::from("  h/Left     Move to previous position"),
            Line::from("  l/Right    Move to next position"),
            Line::from("  k/Up       Select previous node at position"),
            Line::from("  j/Down     Select next node at position"),
            Line::from("  g/Home     Go to start (BOS)"),
            Line::from("  G/End      Go to end (EOS)"),
            Line::from(""),
            Line::from(vec![Span::styled(
                "Best Path:",
                Style::default().add_modifier(Modifier::BOLD),
            )]),
            Line::from("  n          Jump to next best path node"),
            Line::from("  p          Jump to previous best path node"),
            Line::from(""),
            Line::from(vec![Span::styled(
                "Other:",
                Style::default().add_modifier(Modifier::BOLD),
            )]),
            Line::from("  ?/F1       Toggle this help"),
            Line::from("  q/Esc      Quit"),
            Line::from(""),
            Line::from(vec![Span::styled(
                "Legend:",
                Style::default().add_modifier(Modifier::BOLD),
            )]),
            Line::from(vec![
                Span::styled("  [*] ", Style::default().fg(Color::Green)),
                Span::raw("Best path node"),
            ]),
            Line::from(vec![
                Span::styled("  [>] ", Style::default().fg(Color::Yellow)),
                Span::raw("Selected node"),
            ]),
            Line::from(vec![
                Span::styled("  [?] ", Style::default().fg(Color::Red)),
                Span::raw("Unknown word"),
            ]),
        ];

        let paragraph = Paragraph::new(help_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Help")
                    .style(Style::default().bg(Color::DarkGray)),
            )
            .style(Style::default().bg(Color::DarkGray));

        frame.render_widget(ratatui::widgets::Clear, area);
        frame.render_widget(paragraph, area);
    }
}

/// Setup the terminal for TUI
fn setup_terminal() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend)
}

/// Restore the terminal to normal state
fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

/// Create a centered rectangle
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

/// Run the TUI explorer for a given text
///
/// # Errors
///
/// Returns an error if terminal setup fails or dictionary loading fails.
pub fn run_explore(
    text: &str,
    dicdir: Option<std::path::PathBuf>,
    semantic_pool: Option<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let dict = match (dicdir, semantic_pool) {
        (Some(path), Some(semantic_path)) => {
            mecrab::dict::Dictionary::load_with_semantics(&path, &semantic_path)?
        }
        (Some(path), None) => mecrab::dict::Dictionary::load(&path)?,
        (None, _) => mecrab::dict::Dictionary::default_dictionary()?,
    };

    let session = DebugSession::new(text, &dict)?;
    let mut app = App::new(session);
    app.run()?;

    Ok(())
}
