//! Deterministic key-fact extraction for compaction checkpoints.
//!
//! A shrinking checkpoint can still lose exact file paths, commands, and error
//! codes. This module extracts a small set of high-value verbatim facts from the
//! source messages, measures whether the summary kept them, and renders a
//! reserve block for the ones that were dropped.

use std::collections::HashSet;

use jcode_message_types::{ContentBlock, Message};

pub const MAX_KEY_FACTS: usize = 24;
pub const MAX_KEY_FACT_CHARS: usize = 2048;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KeyFactKind {
    FilePath,
    Command,
    ErrorCode,
}

impl KeyFactKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::FilePath => "file",
            Self::Command => "command",
            Self::ErrorCode => "error",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyFact {
    pub kind: KeyFactKind,
    pub value: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KeyFactCoverage {
    pub total: usize,
    pub covered: usize,
    pub missing: Vec<KeyFact>,
}

impl KeyFactCoverage {
    pub fn hit_rate(&self) -> f32 {
        if self.total == 0 {
            1.0
        } else {
            self.covered as f32 / self.total as f32
        }
    }
}

/// Extract stable, exact-value facts from the messages being compacted.
pub fn extract_key_facts(messages: &[Message]) -> Vec<KeyFact> {
    let mut facts = Vec::new();
    let mut seen = HashSet::new();
    for message in messages {
        for block in message.content.iter() {
            match block {
                ContentBlock::Text { text, .. } => collect_text_facts(text, &mut facts, &mut seen),
                ContentBlock::ToolUse { name, input, .. } => {
                    if is_command_tool(name)
                        && let Some(command) = input.get("command").and_then(|value| value.as_str())
                    {
                        push_fact(
                            KeyFactKind::Command,
                            sanitize_command(command),
                            &mut facts,
                            &mut seen,
                        );
                    }
                    collect_text_facts(&input.to_string(), &mut facts, &mut seen);
                }
                ContentBlock::ToolResult { content, .. } => {
                    collect_text_facts(content, &mut facts, &mut seen);
                }
                _ => {}
            }
            if facts.len() >= MAX_KEY_FACTS * 4 {
                break;
            }
        }
        if facts.len() >= MAX_KEY_FACTS * 4 {
            break;
        }
    }
    facts
}

/// Compare extracted facts against the final checkpoint text.
pub fn measure_key_fact_coverage(messages: &[Message], summary: &str) -> KeyFactCoverage {
    let facts = extract_key_facts(messages);
    let mut coverage = KeyFactCoverage {
        total: facts.len(),
        ..KeyFactCoverage::default()
    };
    for fact in facts {
        if summary.contains(&fact.value) {
            coverage.covered += 1;
        } else if coverage.missing.len() < MAX_KEY_FACTS {
            coverage.missing.push(fact);
        }
    }
    coverage
}

/// Render missing facts as a compact verbatim block. Returns an empty string
/// when there is nothing to reserve.
pub fn render_key_fact_reserve(missing: &[KeyFact]) -> String {
    if missing.is_empty() {
        return String::new();
    }
    let mut out = String::from("## Key Facts (verbatim)\n\n");
    for fact in missing.iter().take(MAX_KEY_FACTS) {
        let line = format!("- [{}] `{}`\n", fact.kind.label(), fact.value);
        if out.chars().count() + line.chars().count() > MAX_KEY_FACT_CHARS {
            break;
        }
        out.push_str(&line);
    }
    if out == "## Key Facts (verbatim)\n\n" {
        String::new()
    } else {
        out
    }
}

fn is_command_tool(name: &str) -> bool {
    matches!(
        name,
        "bash" | "shell" | "exec_command" | "run_terminal_cmd" | "terminal"
    )
}

fn sanitize_command(command: &str) -> String {
    let line = command.lines().next().unwrap_or_default().trim();
    line.chars().take(240).collect()
}

fn push_fact(
    kind: KeyFactKind,
    value: String,
    facts: &mut Vec<KeyFact>,
    seen: &mut HashSet<(KeyFactKind, String)>,
) {
    if value.is_empty() {
        return;
    }
    let key = (kind.clone(), value.clone());
    if seen.insert(key) {
        facts.push(KeyFact { kind, value });
    }
}

fn collect_text_facts(
    text: &str,
    facts: &mut Vec<KeyFact>,
    seen: &mut HashSet<(KeyFactKind, String)>,
) {
    for raw in text.split_whitespace() {
        let token = raw.trim_matches(|ch: char| {
            matches!(
                ch,
                '`' | '\''
                    | '"'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '<'
                    | '>'
                    | ','
                    | ';'
                    | ':'
                    | '.'
                    | '!'
                    | '?'
            )
        });
        if token.is_empty() {
            continue;
        }
        if let Some(code) = error_code(token) {
            push_fact(KeyFactKind::ErrorCode, code, facts, seen);
            continue;
        }
        if looks_like_file_path(token) && token.chars().count() <= 240 {
            push_fact(KeyFactKind::FilePath, token.to_string(), facts, seen);
        }
    }
}

fn error_code(token: &str) -> Option<String> {
    let candidate = token
        .strip_prefix("error[")
        .map(|rest| rest.trim_end_matches(']'))
        .unwrap_or(token);
    let bytes = candidate.as_bytes();
    if !(3..=10).contains(&bytes.len()) {
        return None;
    }
    let mut chars = candidate.chars();
    let first = chars.next()?;
    if !first.is_ascii_uppercase() {
        return None;
    }
    let mut has_digit = false;
    for ch in chars {
        if ch.is_ascii_digit() {
            has_digit = true;
        } else if !ch.is_ascii_uppercase() {
            return None;
        }
    }
    if has_digit {
        Some(candidate.to_string())
    } else {
        None
    }
}

fn looks_like_file_path(token: &str) -> bool {
    if token.starts_with("http://") || token.starts_with("https://") || token.starts_with("www.") {
        return false;
    }
    if !(token.contains('/') || token.contains('\\')) {
        return false;
    }
    let Some((_, extension)) = token.rsplit_once('.') else {
        return false;
    };
    matches!(
        extension,
        "rs" | "toml"
            | "lock"
            | "md"
            | "json"
            | "yaml"
            | "yml"
            | "ts"
            | "tsx"
            | "js"
            | "jsx"
            | "py"
            | "go"
            | "java"
            | "kt"
            | "c"
            | "h"
            | "cc"
            | "cpp"
            | "hpp"
            | "sh"
            | "ps1"
            | "sql"
            | "html"
            | "css"
            | "scss"
            | "vue"
            | "swift"
            | "rb"
            | "php"
            | "lua"
            | "zig"
            | "ex"
            | "exs"
            | "erl"
            | "hs"
            | "ml"
            | "clj"
            | "proto"
            | "gradle"
            | "cmake"
            | "txt"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jcode_message_types::Role;
    use serde_json::json;

    fn message(content: Vec<ContentBlock>) -> Message {
        Message {
            role: Role::Assistant,
            content: content.into(),
            timestamp: None,
            tool_duration_ms: None,
        }
    }

    #[test]
    fn extracts_paths_commands_and_error_codes() {
        let messages = vec![
            message(vec![ContentBlock::ToolUse {
                id: "call-1".to_string(),
                name: "bash".to_string(),
                input: json!({"command": "cargo test -p jcode"}),
                thought_signature: None,
            }]),
            message(vec![ContentBlock::ToolResult {
                tool_use_id: "call-1".to_string(),
                content: "error[E0425]: failed in crates/jcode-base/src/compaction.rs".to_string(),
                is_error: Some(true),
            }]),
        ];
        let facts = extract_key_facts(&messages);
        assert!(
            facts.iter().any(
                |fact| fact.kind == KeyFactKind::Command && fact.value == "cargo test -p jcode"
            )
        );
        assert!(
            facts
                .iter()
                .any(|fact| fact.kind == KeyFactKind::ErrorCode && fact.value == "E0425")
        );
        assert!(facts.iter().any(|fact| fact.kind == KeyFactKind::FilePath
            && fact.value == "crates/jcode-base/src/compaction.rs"));
    }

    #[test]
    fn coverage_reports_missing_facts() {
        let messages = vec![message(vec![ContentBlock::Text {
            text: "fixed crates/jcode-base/src/compaction.rs after E0425".to_string(),
            cache_control: None,
        }])];
        let coverage = measure_key_fact_coverage(&messages, "only mentions E0425");
        assert_eq!(coverage.total, 2);
        assert_eq!(coverage.covered, 1);
        assert_eq!(coverage.missing.len(), 1);
        assert!(coverage.hit_rate() < 0.9);
    }

    #[test]
    fn reserve_block_is_bounded_and_verbatim() {
        let missing = vec![KeyFact {
            kind: KeyFactKind::FilePath,
            value: "src/lib.rs".to_string(),
        }];
        let block = render_key_fact_reserve(&missing);
        assert!(block.contains("## Key Facts (verbatim)"));
        assert!(block.contains("`src/lib.rs`"));
    }
}
