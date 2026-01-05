//! Cost panel component for the TUI
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Displays detailed cost breakdown for the selected node.

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

use mecrab::debug::DebugSession;

/// Cost panel component
pub struct CostPanel<'a> {
    session: &'a DebugSession,
    selected_pos: usize,
    selected_idx: usize,
}

impl<'a> CostPanel<'a> {
    /// Create a new cost panel
    pub fn new(session: &'a DebugSession, selected_pos: usize, selected_idx: usize) -> Self {
        Self {
            session,
            selected_pos,
            selected_idx,
        }
    }

    /// Draw the cost panel
    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        let mut lines: Vec<Line> = Vec::new();

        // Get the selected node
        if let Some(node) = self.session.get_node(self.selected_pos, self.selected_idx) {
            // Node header
            lines.push(Line::from(vec![Span::styled(
                "Selected Node",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )]));
            lines.push(Line::from(""));

            // Surface form
            let surface = if node.surface.is_empty() {
                if node.is_bos() {
                    "BOS (Begin of Sentence)".to_string()
                } else if node.is_eos() {
                    "EOS (End of Sentence)".to_string()
                } else {
                    "(empty)".to_string()
                }
            } else {
                node.surface.clone()
            };

            lines.push(self.render_field("Surface", &surface, Color::White));

            // Feature string (truncated if too long, respecting UTF-8 boundaries)
            let feature = if node.feature.chars().count() > 40 {
                let truncated: String = node.feature.chars().take(37).collect();
                format!("{}...", truncated)
            } else {
                node.feature.clone()
            };
            lines.push(self.render_field("Feature", &feature, Color::DarkGray));

            lines.push(Line::from(""));

            // Position info
            lines.push(Line::from(vec![Span::styled(
                "Position",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )]));
            lines.push(self.render_field(
                "Byte range",
                &format!("[{}, {})", node.start, node.end),
                Color::White,
            ));
            lines.push(self.render_field(
                "Lattice pos",
                &format!("{}", self.selected_pos),
                Color::White,
            ));
            lines.push(self.render_field(
                "Node index",
                &format!("{}", self.selected_idx),
                Color::White,
            ));

            lines.push(Line::from(""));

            // Cost breakdown
            lines.push(Line::from(vec![Span::styled(
                "Cost Breakdown",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )]));

            lines.push(self.render_cost_field("Word cost", node.wcost as i64, Color::Yellow));

            if let Some(entry) = self
                .session
                .get_viterbi_entry(self.selected_pos, self.selected_idx)
            {
                lines.push(self.render_cost_field(
                    "Connection cost",
                    entry.connection_cost as i64,
                    Color::Magenta,
                ));
                lines.push(self.render_cost_field(
                    "Cumulative cost",
                    entry.cumulative_cost,
                    Color::Green,
                ));

                // Show predecessor
                if let Some((prev_pos, prev_idx)) = entry.prev {
                    lines.push(Line::from(""));
                    lines.push(Line::from(vec![Span::styled(
                        "Predecessor",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )]));

                    if let Some(prev_node) = self.session.get_node(prev_pos, prev_idx) {
                        let prev_surface = if prev_node.surface.is_empty() {
                            "BOS".to_string()
                        } else {
                            prev_node.surface.clone()
                        };
                        lines.push(self.render_field("From", &prev_surface, Color::White));
                        lines.push(self.render_field(
                            "Position",
                            &format!("({}, {})", prev_pos, prev_idx),
                            Color::DarkGray,
                        ));
                    }
                }
            }

            lines.push(Line::from(""));

            // Context IDs
            lines.push(Line::from(vec![Span::styled(
                "Context IDs",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )]));
            lines.push(self.render_field("Left ID", &format!("{}", node.left_id), Color::White));
            lines.push(self.render_field("Right ID", &format!("{}", node.right_id), Color::White));

            // Best path status
            lines.push(Line::from(""));
            let is_on_best = self
                .session
                .is_on_best_path(self.selected_pos, self.selected_idx);
            if is_on_best {
                lines.push(Line::from(vec![Span::styled(
                    "ON BEST PATH",
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                )]));

                // Show cost delta from next best alternative
                if let Some(delta) = self
                    .session
                    .cost_delta_from_alternative(self.selected_pos, self.selected_idx)
                {
                    lines.push(Line::from(vec![
                        Span::styled("Alternative is ", Style::default().fg(Color::DarkGray)),
                        Span::styled(format!("+{}", delta), Style::default().fg(Color::Red)),
                        Span::styled(" more costly", Style::default().fg(Color::DarkGray)),
                    ]));
                }
            } else {
                lines.push(Line::from(vec![Span::styled(
                    "ALTERNATIVE PATH",
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )]));

                // Show how much more costly this is
                if let Some(delta) = self
                    .session
                    .cost_delta_from_alternative(self.selected_pos, self.selected_idx)
                {
                    lines.push(Line::from(vec![
                        Span::styled("This path is ", Style::default().fg(Color::DarkGray)),
                        Span::styled(format!("+{}", delta), Style::default().fg(Color::Red)),
                        Span::styled(" more costly", Style::default().fg(Color::DarkGray)),
                    ]));
                }
            }

            // Unknown word status
            if node.is_unknown {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![Span::styled(
                    "UNKNOWN WORD",
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Red)
                        .add_modifier(Modifier::BOLD),
                )]));
            }

            // Alternatives at this position
            lines.push(Line::from(""));
            lines.push(Line::from(vec![Span::styled(
                "Alternatives at Position",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )]));

            if let Some(nodes_at_pos) = self.session.nodes_at.get(self.selected_pos) {
                let mut alternatives: Vec<_> = nodes_at_pos
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx != self.selected_idx)
                    .filter_map(|(idx, n)| {
                        self.session
                            .get_viterbi_entry(self.selected_pos, idx)
                            .map(|e| (n, e.cumulative_cost))
                    })
                    .collect();

                alternatives.sort_by_key(|(_, cost)| *cost);

                for (alt_node, alt_cost) in alternatives.iter().take(3) {
                    let alt_surface = if alt_node.surface.is_empty() {
                        "(empty)".to_string()
                    } else {
                        alt_node.surface.clone()
                    };

                    let is_best = self
                        .session
                        .is_on_best_path(self.selected_pos, self.selected_idx);
                    let delta = if let Some(entry) = self
                        .session
                        .get_viterbi_entry(self.selected_pos, self.selected_idx)
                    {
                        alt_cost - entry.cumulative_cost
                    } else {
                        0
                    };

                    let delta_str = if delta > 0 {
                        format!("+{}", delta)
                    } else if delta < 0 {
                        format!("{}", delta)
                    } else {
                        "0".to_string()
                    };

                    let color = if !is_best && delta < 0 {
                        Color::Green
                    } else {
                        Color::DarkGray
                    };

                    lines.push(Line::from(vec![
                        Span::styled(format!("  {} ", alt_surface), Style::default().fg(color)),
                        Span::styled(
                            format!("({})", delta_str),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]));
                }

                if alternatives.len() > 3 {
                    lines.push(Line::from(vec![Span::styled(
                        format!("  ... and {} more", alternatives.len() - 3),
                        Style::default().fg(Color::DarkGray),
                    )]));
                }

                if alternatives.is_empty() {
                    lines.push(Line::from(vec![Span::styled(
                        "  (none)",
                        Style::default().fg(Color::DarkGray),
                    )]));
                }
            }
        } else {
            lines.push(Line::from(vec![Span::styled(
                "No node selected",
                Style::default().fg(Color::DarkGray),
            )]));
        }

        let paragraph = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Cost Details"))
            .wrap(Wrap { trim: false });

        frame.render_widget(paragraph, area);
    }

    /// Render a field with label and value
    fn render_field(&self, label: &str, value: &str, color: Color) -> Line<'static> {
        Line::from(vec![
            Span::styled(format!("{}: ", label), Style::default().fg(Color::DarkGray)),
            Span::styled(value.to_string(), Style::default().fg(color)),
        ])
    }

    /// Render a cost field with special formatting
    fn render_cost_field(&self, label: &str, value: i64, color: Color) -> Line<'static> {
        let value_str = value.to_string();

        Line::from(vec![
            Span::styled(format!("{}: ", label), Style::default().fg(Color::DarkGray)),
            Span::styled(value_str, Style::default().fg(color)),
        ])
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_feature_truncation_utf8() {
        // Test that long Japanese feature strings are truncated at character boundaries
        let feature = "名詞,一般,*,*,*,*,すもも,スモモ,スモモ";

        // Simulate the truncation logic
        let truncated = if feature.chars().count() > 40 {
            let result: String = feature.chars().take(37).collect();
            format!("{}...", result)
        } else {
            feature.to_string()
        };

        // Should not panic (would panic if we used byte slicing)
        assert!(truncated.len() <= feature.len() + 3); // +3 for "..."

        // Test with a very long feature string
        let long_feature =
            "名詞,一般,*,*,*,*,すもも,スモモ,スモモ,これは非常に長い文字列です,テスト,TEST";
        let truncated_long = if long_feature.chars().count() > 40 {
            let result: String = long_feature.chars().take(37).collect();
            format!("{}...", result)
        } else {
            long_feature.to_string()
        };

        // Should end with "..."
        assert!(truncated_long.ends_with("..."));
        // Should be valid UTF-8
        assert!(std::str::from_utf8(truncated_long.as_bytes()).is_ok());
    }
}
