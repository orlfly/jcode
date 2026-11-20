//! Task contract library for facts that cannot be stored in the memory graph.
//!
//! Checkpoints are a narrative summary and get rewritten on every compaction.
//! Hard user constraints, retracted conclusions, error fixes, decisions, and
//! deferred actions are not graph facts, so they need a separate, verbatim
//! container that survives checkpoint rewrites. Entries are rendered as a
//! compact block after the checkpoint, with `RETRACTED` injected first because
//! reviving a dead conclusion is more dangerous than forgetting a todo.

use jcode_message_types::{ContentBlock, Message, Role};

/// Heading used for the independent task-contract block.
pub const CONTRACT_HEADING: &str = "## Task Contracts";

/// Maximum number of active entries injected into one request.
pub const MAX_INJECTED_CONTRACTS: usize = 8;

/// Contract markers in injection priority order.
pub const CONTRACT_MARKERS: [&str; 5] = [
    "[RETRACTED]",
    "[CONSTRAINT]",
    "[ERROR_FIX]",
    "[DECISION]",
    "[TODO]",
];

/// Kind of non-graph fact carried by a contract entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContractKind {
    Retracted,
    Constraint,
    ErrorFix,
    Decision,
    Todo,
}

impl ContractKind {
    pub fn marker(self) -> &'static str {
        match self {
            Self::Retracted => "[RETRACTED]",
            Self::Constraint => "[CONSTRAINT]",
            Self::ErrorFix => "[ERROR_FIX]",
            Self::Decision => "[DECISION]",
            Self::Todo => "[TODO]",
        }
    }

    pub fn parse(marker: &str) -> Option<Self> {
        CONTRACT_MARKERS
            .iter()
            .position(|candidate| candidate.eq_ignore_ascii_case(marker))
            .map(|index| match index {
                0 => Self::Retracted,
                1 => Self::Constraint,
                2 => Self::ErrorFix,
                3 => Self::Decision,
                _ => Self::Todo,
            })
    }

    /// Lower sorts earlier; `RETRACTED` must always be injected first.
    pub fn priority(self) -> u8 {
        match self {
            Self::Retracted => 0,
            Self::Constraint => 1,
            Self::ErrorFix => 2,
            Self::Decision => 3,
            Self::Todo => 4,
        }
    }
}

/// Whether an entry came from the user's exact words or an operator inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractAuthority {
    Owner,
    Operator,
}

/// Lifecycle of a contract entry. Superseded and expired entries are retained
/// for audit but never injected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractStatus {
    Active,
    Superseded,
    Expired,
}

/// One verbatim non-graph fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractEntry {
    pub kind: ContractKind,
    /// Exact source text after the marker; never paraphrased.
    pub text: String,
    pub status: ContractStatus,
    pub authority: ContractAuthority,
    /// Stable source label, e.g. `message:3`.
    pub provenance: String,
}

impl ContractEntry {
    pub fn active(
        kind: ContractKind,
        text: impl Into<String>,
        authority: ContractAuthority,
        provenance: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            text: text.into(),
            status: ContractStatus::Active,
            authority,
            provenance: provenance.into(),
        }
    }

    fn key(&self) -> (ContractKind, &str) {
        (self.kind, self.text.as_str())
    }

    /// Verbatim guard: the stored text must still occur in its source.
    pub fn is_verbatim_in(&self, source: &str) -> bool {
        source.contains(&self.text)
    }
}

/// Parse a checkpoint-style line into a contract entry.
///
/// Accepts the markdown shapes produced by the checkpoint prompt:
/// `- [CONSTRAINT] text`, ``- `[CONSTRAINT]` text``, and the bare marker form.
fn parse_contract_line(line: &str) -> Option<(ContractKind, String)> {
    let mut line = line.trim();
    for prefix in ["- ", "* ", "+ "] {
        if let Some(rest) = line.strip_prefix(prefix) {
            line = rest.trim_start();
            break;
        }
    }
    line = line.trim_start_matches('`').trim_start();
    for marker in CONTRACT_MARKERS {
        let Some(rest) = line.strip_prefix(marker) else {
            continue;
        };
        let rest = rest.trim_start_matches('`').trim();
        if rest.is_empty() || rest == "..." {
            return None;
        }
        return Some((ContractKind::parse(marker)?, rest.to_string()));
    }
    None
}

/// Extract marked contract lines from conversation messages.
///
/// Extraction is intentionally deterministic and shallow: it never rewrites a
/// line and never asks a model to summarize one. Duplicate kind+text pairs keep
/// the earliest provenance.
pub fn extract_contract_entries(messages: &[Message]) -> Vec<ContractEntry> {
    let mut entries: Vec<ContractEntry> = Vec::new();
    for (turn, message) in messages.iter().enumerate() {
        for block in message.content.iter() {
            let (text, authority) = match block {
                ContentBlock::Text { text, .. } => {
                    let authority = if message.role == Role::User {
                        ContractAuthority::Owner
                    } else {
                        ContractAuthority::Operator
                    };
                    (text.as_str(), authority)
                }
                ContentBlock::ToolResult { content, .. } => {
                    (content.as_str(), ContractAuthority::Operator)
                }
                _ => continue,
            };
            for line in text.lines() {
                let Some((kind, body)) = parse_contract_line(line) else {
                    continue;
                };
                let candidate =
                    ContractEntry::active(kind, body, authority, format!("message:{turn}"));
                if !entries.iter().any(|entry| entry.key() == candidate.key()) {
                    entries.push(candidate);
                }
            }
        }
    }
    entries
}

/// Parse entries from an existing task-contract block.
pub fn parse_contract_block(summary: &str) -> Vec<ContractEntry> {
    let Some(start) = summary.find(CONTRACT_HEADING) else {
        return Vec::new();
    };
    let body_start = start + CONTRACT_HEADING.len();
    let body_end = summary[body_start..]
        .find("\n## ")
        .map(|offset| body_start + offset)
        .unwrap_or(summary.len());
    let mut entries: Vec<ContractEntry> = Vec::new();
    for line in summary[body_start..body_end].lines() {
        let Some((kind, text)) = parse_contract_line(line) else {
            continue;
        };
        let candidate = ContractEntry::active(
            kind,
            text,
            ContractAuthority::Operator,
            "checkpoint:contracts".to_string(),
        );
        if !entries.iter().any(|entry| entry.key() == candidate.key()) {
            entries.push(candidate);
        }
    }
    entries
}

/// Remove a previously injected contract block so it can be rebuilt once.
pub fn strip_contract_block(summary: &str) -> String {
    let Some(start) = summary.find(CONTRACT_HEADING) else {
        return summary.to_string();
    };
    let body_start = start + CONTRACT_HEADING.len();
    let body_end = summary[body_start..]
        .find("\n## ")
        .map(|offset| body_start + offset)
        .unwrap_or(summary.len());
    let mut out = String::with_capacity(summary.len());
    out.push_str(summary[..start].trim_end());
    out.push_str(summary[body_end..].trim_end());
    if out.is_empty() {
        summary.to_string()
    } else {
        out
    }
}

/// Merge existing and freshly extracted entries, preserving the first
/// provenance for duplicate kind+text pairs.
pub fn merge_contract_entries(
    existing: Vec<ContractEntry>,
    new_entries: Vec<ContractEntry>,
) -> Vec<ContractEntry> {
    let mut merged = existing;
    for entry in new_entries {
        if !merged.iter().any(|seen| seen.key() == entry.key()) {
            merged.push(entry);
        }
    }
    merged
}

/// Render the injectable block, active entries only, capped after priority sort.
pub fn render_contract_block(entries: &[ContractEntry], limit: usize) -> String {
    let mut active: Vec<&ContractEntry> = entries
        .iter()
        .filter(|entry| entry.status == ContractStatus::Active)
        .collect();
    active.sort_by(|left, right| {
        left.kind
            .priority()
            .cmp(&right.kind.priority())
            .then_with(|| left.provenance.cmp(&right.provenance))
            .then_with(|| left.text.cmp(&right.text))
    });
    if active.is_empty() || limit == 0 {
        return String::new();
    }
    let mut out = String::from(CONTRACT_HEADING);
    out.push_str("\n\n");
    for entry in active.into_iter().take(limit) {
        out.push_str("- ");
        out.push_str(entry.kind.marker());
        out.push(' ');
        out.push_str(&entry.text);
        out.push('\n');
    }
    out.trim_end().to_string()
}

/// Replace any existing contract block with a freshly rendered one.
pub fn inject_contract_block(summary: String, entries: &[ContractEntry], limit: usize) -> String {
    let block = render_contract_block(entries, limit);
    let without = strip_contract_block(&summary);
    if block.is_empty() {
        return without;
    }
    let mut out = without.trim_end().to_string();
    out.push_str("\n\n");
    out.push_str(&block);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_verbatim_markers_and_keeps_owner_authority() {
        let messages = vec![
            Message::user("Keep this:\n- `[CONSTRAINT]` never force-push to master"),
            Message::assistant_text("[ERROR_FIX] rebuild after protocol change"),
            Message::assistant_text("[TODO] add the regression test"),
        ];
        let entries = extract_contract_entries(&messages);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].kind, ContractKind::Constraint);
        assert_eq!(entries[0].text, "never force-push to master");
        assert_eq!(entries[0].authority, ContractAuthority::Owner);
        assert_eq!(entries[1].kind, ContractKind::ErrorFix);
        assert_eq!(entries[2].kind, ContractKind::Todo);
    }

    #[test]
    fn render_puts_retracted_first_and_caps_entries() {
        let entries = vec![
            ContractEntry::active(
                ContractKind::Todo,
                "todo",
                ContractAuthority::Operator,
                "message:0",
            ),
            ContractEntry::active(
                ContractKind::Constraint,
                "constraint",
                ContractAuthority::Owner,
                "message:1",
            ),
            ContractEntry::active(
                ContractKind::Retracted,
                "retracted",
                ContractAuthority::Owner,
                "message:2",
            ),
        ];
        let block = render_contract_block(&entries, 2);
        let retracted = block.find("[RETRACTED]").expect("retracted");
        let constraint = block.find("[CONSTRAINT]").expect("constraint");
        assert!(retracted < constraint, "{block}");
        assert!(!block.contains("[TODO]"), "{block}");
    }

    #[test]
    fn parse_and_inject_round_trip_without_duplicates() {
        let summary = "## Critical Context\n- keep this\n\n## Task Contracts\n\n- [TODO] old\n";
        let parsed = parse_contract_block(summary);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].text, "old");

        let fresh = vec![ContractEntry::active(
            ContractKind::Retracted,
            "stale conclusion",
            ContractAuthority::Owner,
            "message:4",
        )];
        let merged = merge_contract_entries(parsed, fresh);
        let injected = inject_contract_block(summary.to_string(), &merged, 8);
        assert_eq!(injected.matches(CONTRACT_HEADING).count(), 1, "{injected}");
        assert_eq!(injected.matches("[TODO]").count(), 1, "{injected}");
        assert!(
            injected.contains("[RETRACTED] stale conclusion"),
            "{injected}"
        );
        assert!(
            injected.find("[RETRACTED]").unwrap() < injected.find("[TODO]").unwrap(),
            "{injected}"
        );
    }

    #[test]
    fn verbatim_guard_rejects_rewritten_text() {
        let entry = ContractEntry::active(
            ContractKind::Constraint,
            "never delete the live database",
            ContractAuthority::Owner,
            "message:0",
        );
        assert!(entry.is_verbatim_in("Please never delete the live database."));
        assert!(!entry.is_verbatim_in("never drop the live database"));
    }

    #[test]
    fn protected_repair_skips_the_contract_block() {
        let existing = "## Critical Context\n- `[CONSTRAINT]` keep api stable\n\n## Task Contracts\n\n- [RETRACTED] old conclusion\n";
        let (repaired, preserved) = crate::preserve_protected_checkpoint_lines(
            Some(existing),
            "## Critical Context\n- new".to_string(),
        );
        assert_eq!(preserved, 1);
        assert!(
            repaired.contains("[CONSTRAINT]` keep api stable"),
            "{repaired}"
        );
        assert!(!repaired.contains("[RETRACTED] old conclusion"), "{repaired}");
    }
}
