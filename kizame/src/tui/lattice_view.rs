//! Lattice visualization component for the TUI
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Renders the lattice as a navigable graph with nodes arranged by position.

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

use mecrab::debug::DebugSession;

/// Lattice view component
pub struct LatticeView<'a> {
    session: &'a DebugSession,
    selected_pos: usize,
    selected_idx: usize,
    scroll_offset: u16,
}

impl<'a> LatticeView<'a> {
    /// Create a new lattice view
    pub fn new(session: &'a DebugSession, selected_pos: usize, selected_idx: usize) -> Self {
        Self {
            session,
            selected_pos,
            selected_idx,
            scroll_offset: 0,
        }
    }

    /// Set the scroll offset
    pub fn set_scroll_offset(&mut self, offset: usize) {
        self.scroll_offset = offset as u16;
    }

    /// Draw the lattice view
    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        let mut lines: Vec<Line> = Vec::new();

        // Header with input text
        lines.push(Line::from(vec![
            Span::styled("Input: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("\"{}\"", self.session.text),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(""));

        // Render each position as a column
        let visible_positions = self.calculate_visible_positions(area.width as usize);

        for pos in visible_positions {
            lines.push(self.render_position_header(pos));

            if let Some(nodes) = self.session.nodes_at.get(pos) {
                for (idx, node) in nodes.iter().enumerate() {
                    lines.push(self.render_node(pos, idx, node));
                }
            }
            lines.push(Line::from(""));
        }

        // Add connection arrows for selected node
        lines.push(Line::from(""));
        lines.extend(self.render_connections());

        let paragraph = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Lattice Structure"),
            )
            .wrap(Wrap { trim: false })
            .scroll((self.scroll_offset, 0));

        frame.render_widget(paragraph, area);
    }

    /// Calculate which positions to display based on terminal width
    fn calculate_visible_positions(&self, _width: usize) -> std::ops::Range<usize> {
        // For now, show all positions; in a more sophisticated version,
        // we could implement horizontal scrolling
        0..self.session.num_positions()
    }

    /// Render a position header
    fn render_position_header(&self, pos: usize) -> Line<'static> {
        let is_selected = pos == self.selected_pos;
        let style = if is_selected {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        let label = if pos == 0 {
            "BOS".to_string()
        } else if pos == self.session.num_positions() - 1 {
            "EOS".to_string()
        } else {
            format!("Pos {}", pos)
        };

        Line::from(vec![Span::styled(format!("=== {} ===", label), style)])
    }

    /// Render a single node
    fn render_node(
        &self,
        pos: usize,
        idx: usize,
        node: &mecrab::debug::DebugNode,
    ) -> Line<'static> {
        let is_selected = pos == self.selected_pos && idx == self.selected_idx;
        let is_on_best_path = self.session.is_on_best_path(pos, idx);

        // Determine the marker
        let marker = if is_selected && is_on_best_path {
            Span::styled(
                "[*>]",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )
        } else if is_selected {
            Span::styled(
                "[>] ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
        } else if is_on_best_path {
            Span::styled("[*] ", Style::default().fg(Color::Green))
        } else if node.is_unknown {
            Span::styled("[?] ", Style::default().fg(Color::Red))
        } else {
            Span::styled("[ ] ", Style::default().fg(Color::DarkGray))
        };

        // Surface form
        let surface = if node.surface.is_empty() {
            if node.is_bos() {
                "BOS".to_string()
            } else if node.is_eos() {
                "EOS".to_string()
            } else {
                "(empty)".to_string()
            }
        } else {
            node.surface.clone()
        };

        let surface_style = if is_selected {
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD)
        } else if is_on_best_path {
            Style::default().fg(Color::Green)
        } else if node.is_unknown {
            Style::default().fg(Color::Red)
        } else {
            Style::default().fg(Color::White)
        };

        // Cost info
        let cost_str = if let Some(entry) = self.session.get_viterbi_entry(pos, idx) {
            format!(
                " (w:{} c:{} t:{})",
                node.wcost, entry.connection_cost, entry.cumulative_cost
            )
        } else {
            format!(" (w:{})", node.wcost)
        };

        let cost_style = Style::default().fg(Color::DarkGray);

        Line::from(vec![
            marker,
            Span::styled(surface, surface_style),
            Span::styled(cost_str, cost_style),
        ])
    }

    /// Render connection information for the selected node
    fn render_connections(&self) -> Vec<Line<'static>> {
        let mut lines = Vec::new();

        lines.push(Line::from(vec![Span::styled(
            "Connections:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )]));

        // Incoming edges
        let incoming = self.session.edges_to(self.selected_pos, self.selected_idx);
        if !incoming.is_empty() {
            lines.push(Line::from(vec![Span::styled(
                "  Incoming:",
                Style::default().fg(Color::DarkGray),
            )]));

            for edge in incoming.iter().take(5) {
                if let Some(from_node) = self.session.get_node(edge.from.0, edge.from.1) {
                    let surface = if from_node.surface.is_empty() {
                        "BOS".to_string()
                    } else {
                        from_node.surface.clone()
                    };

                    let is_best = self.session.is_on_best_path(edge.from.0, edge.from.1);
                    let style = if is_best {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default().fg(Color::White)
                    };

                    lines.push(Line::from(vec![
                        Span::raw("    "),
                        Span::styled(surface, style),
                        Span::styled(
                            format!(" --[{}]--> ", edge.connection_cost),
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::styled("(selected)", Style::default().fg(Color::Yellow)),
                    ]));
                }
            }
            if incoming.len() > 5 {
                lines.push(Line::from(vec![Span::styled(
                    format!("    ... and {} more", incoming.len() - 5),
                    Style::default().fg(Color::DarkGray),
                )]));
            }
        }

        // Outgoing edges
        let outgoing = self
            .session
            .edges_from(self.selected_pos, self.selected_idx);
        if !outgoing.is_empty() {
            lines.push(Line::from(vec![Span::styled(
                "  Outgoing:",
                Style::default().fg(Color::DarkGray),
            )]));

            for edge in outgoing.iter().take(5) {
                if let Some(to_node) = self.session.get_node(edge.to.0, edge.to.1) {
                    let surface = if to_node.surface.is_empty() {
                        "EOS".to_string()
                    } else {
                        to_node.surface.clone()
                    };

                    let is_best = self.session.is_on_best_path(edge.to.0, edge.to.1);
                    let style = if is_best {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default().fg(Color::White)
                    };

                    lines.push(Line::from(vec![
                        Span::raw("    "),
                        Span::styled("(selected)", Style::default().fg(Color::Yellow)),
                        Span::styled(
                            format!(" --[{}]--> ", edge.connection_cost),
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::styled(surface, style),
                    ]));
                }
            }
            if outgoing.len() > 5 {
                lines.push(Line::from(vec![Span::styled(
                    format!("    ... and {} more", outgoing.len() - 5),
                    Style::default().fg(Color::DarkGray),
                )]));
            }
        }

        if incoming.is_empty() && outgoing.is_empty() {
            lines.push(Line::from(vec![Span::styled(
                "  No connections",
                Style::default().fg(Color::DarkGray),
            )]));
        }

        lines
    }
}
