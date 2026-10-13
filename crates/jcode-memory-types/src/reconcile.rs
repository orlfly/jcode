//! Write-time reconciliation and content hygiene for memory instances.
//!
//! This module is the runtime implementation behind two ontology declarations
//! in [`crate::ontology`]:
//!
//! * `rule.write_reconcile` (`ActivityStep::ReconcileOnWrite`) - classify an
//!   incoming instance against the store it is about to join.
//! * `rule.reject` (`Condition::Custom { expression: "content_is_noise" }`) -
//!   keep tool payloads and bare identifiers out of the graph.
//!
//! ## Why lexical instead of embeddings
//!
//! v1 declared `rule.dedup_reinforce` (>= 0.85) and `rule.supersede` (>= 0.5)
//! but the checks they described never ran: in the live store 4.2k instances
//! produced **zero** `supersedes` / `contradicts` edges while 1.6k pairs sat
//! above the 0.5 similarity threshold, unlinked. The similarity signal has to
//! be available on every machine, including the ones where no embedder is
//! configured (measured: 22.7% of stored instances carry an embedding) and no
//! LLM judge credential exists (measured: 0 real-session judge calls in 15
//! days). Token-level Jaccard over normalized search text always works, costs
//! one pass over the store, and is deterministic - which matters because this
//! runs synchronously on the write path and its decisions create permanent
//! graph edges.

use std::collections::HashSet;

use crate::graph::MemoryGraph;
use crate::{MemoryCategory, MemoryEntry};

/// Similarity at or above this collapses the incoming instance into the
/// existing one (reinforce, do not create a second row).
pub const REINFORCE_AT: f32 = 0.85;
/// Similarity at or above this supersedes the existing instance.
pub const SUPERSEDE_AT: f32 = 0.5;
/// Similarity at or above this links the two instances (relates_to, or
/// contradicts when the content disagrees).
pub const RELATE_AT: f32 = 0.3;

/// Maximum number of related instances recorded per write.
pub const MAX_RELATED: usize = 5;

/// What the runtime should do with an incoming instance.
#[derive(Debug, Clone, PartialEq)]
pub enum ReconcileAction {
    /// Nothing similar exists; insert a new instance.
    Insert,
    /// An equivalent instance exists; reinforce it and keep the existing id.
    Reinforce { existing_id: String },
    /// A closely related but distinct instance exists; the incoming instance
    /// replaces it (`supersedes` edge, older instance archived).
    Supersede { existing_id: String },
    /// The incoming instance disagrees with a related instance
    /// (`contradicts` edge, both stay active).
    Contradict { existing_id: String },
    /// The incoming instance is related but independent (`relates_to` edge).
    Relate { existing_id: String },
}

impl ReconcileAction {
    pub fn existing_id(&self) -> Option<&str> {
        match self {
            ReconcileAction::Insert => None,
            ReconcileAction::Reinforce { existing_id }
            | ReconcileAction::Supersede { existing_id }
            | ReconcileAction::Contradict { existing_id }
            | ReconcileAction::Relate { existing_id } => Some(existing_id),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            ReconcileAction::Insert => "insert",
            ReconcileAction::Reinforce { .. } => "reinforce",
            ReconcileAction::Supersede { .. } => "supersede",
            ReconcileAction::Contradict { .. } => "contradict",
            ReconcileAction::Relate { .. } => "relate",
        }
    }
}

/// Result of reconciling one incoming instance.
#[derive(Debug, Clone, PartialEq)]
pub struct ReconcileOutcome {
    pub action: ReconcileAction,
    /// Similarity of the deciding match (0.0 when inserting).
    pub similarity: f32,
    /// All matches at or above [`RELATE_AT`], best first, capped at
    /// [`MAX_RELATED`]. Used to create `relates_to` edges.
    pub related: Vec<(String, f32)>,
}

impl ReconcileOutcome {
    fn insert() -> Self {
        Self {
            action: ReconcileAction::Insert,
            similarity: 0.0,
            related: Vec::new(),
        }
    }
}

/// Tokenize normalized text into comparable terms.
fn terms(text: &str) -> HashSet<String> {
    crate::normalize_search_text(text)
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() >= 3)
        .map(|t| t.to_string())
        .collect()
}

/// Token-set Jaccard similarity over the searchable text of two entries.
///
/// Returns `0.0` when either side has no comparable terms.
pub fn lexical_similarity(a: &MemoryEntry, b: &MemoryEntry) -> f32 {
    let ta = terms(&a.searchable_text());
    let tb = terms(&b.searchable_text());
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let inter = ta.intersection(&tb).count() as f32;
    let union = ta.union(&tb).count() as f32;
    if union == 0.0 { 0.0 } else { inter / union }
}

/// Whether two entries claim the same subject.
pub fn same_subject(a: &MemoryEntry, b: &MemoryEntry) -> bool {
    match (a.subject_key.as_deref(), b.subject_key.as_deref()) {
        (Some(x), Some(y)) => !x.is_empty() && x.eq_ignore_ascii_case(y),
        _ => false,
    }
}

/// Whether two related entries disagree rather than merely overlap.
///
/// Two independent signals:
///
/// 1. **Negation asymmetry** - one side says "never / not / no longer / 禁止"
///    and the other does not.
/// 2. **Same subject, different concrete value** - both entries are about the
///    same `subject_key` but carry disjoint identifier-like tokens (a
///    different port, host, task id).
///
/// Deliberately conservative: unrelated entries that merely share vocabulary
/// are not reported as conflicting, because a false `contradicts` edge is
/// worse than a missing one.
pub fn detect_conflict(a: &MemoryEntry, b: &MemoryEntry) -> bool {
    let na = crate::normalize_search_text(&a.content);
    let nb = crate::normalize_search_text(&b.content);
    const NEGATIONS: [&str; 10] = [
        "not ",
        "never",
        "no longer",
        "instead of",
        "must not",
        "should not",
        "禁止",
        "不得",
        "不是",
        "不应",
    ];
    let neg = |s: &str| NEGATIONS.iter().any(|n| s.contains(n));
    if neg(&na) != neg(&nb) {
        return true;
    }
    if !same_subject(a, b) {
        return false;
    }
    let ia = identifiers(&na);
    let ib = identifiers(&nb);
    !ia.is_empty() && !ib.is_empty() && ia.is_disjoint(&ib)
}

/// Numbers and id-like tokens that identify a concrete value.
fn identifiers(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != ':' && c != '-' && c != '_' && c != '.')
        .filter(|t| {
            let has_digit = t.chars().any(|c| c.is_ascii_digit());
            let is_addr = t.contains(':') || t.contains('.');
            (has_digit && t.len() >= 3) || is_addr
        })
        .map(|t| t.to_string())
        .collect()
}

/// Classify an incoming instance against `graph`.
///
/// Link nodes (`goal:` / `skill:`) are owned by their subsystem and upserted by
/// id, so they are inserted without reconciliation.
pub fn reconcile(entry: &MemoryEntry, graph: &MemoryGraph) -> ReconcileOutcome {
    if crate::is_link_node(entry) {
        return ReconcileOutcome::insert();
    }

    let mut scored: Vec<(&MemoryEntry, f32)> = graph
        .active_memories()
        .filter(|existing| existing.id != entry.id)
        // Link nodes are indexes of external artifacts, not knowledge peers.
        .filter(|existing| !crate::is_link_node(existing))
        .map(|existing| (existing, lexical_similarity(entry, existing)))
        .filter(|(_, sim)| *sim >= RELATE_AT)
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // An explicit `subject_key` match is a stronger signal than term overlap:
    // two instances about `/api/task/claim` are about the same thing even when
    // one says "claim" and the other says "claiming". Check it independently of
    // the lexical threshold so the anchor is not filtered out first.
    let subject_match = graph
        .active_memories()
        .filter(|existing| existing.id != entry.id && !crate::is_link_node(existing))
        .find(|existing| same_subject(entry, existing))
        .map(|existing| {
            let sim = lexical_similarity(entry, existing);
            (existing, sim)
        });

    let best = match subject_match {
        Some((existing, sim)) => {
            if !scored.iter().any(|(e, _)| e.id == existing.id) {
                scored.push((existing, sim));
                scored.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            (existing, sim)
        }
        None => match scored.first().copied() {
            Some(hit) => hit,
            None => return ReconcileOutcome::insert(),
        },
    };

    let related: Vec<(String, f32)> = scored
        .iter()
        .take(MAX_RELATED)
        .map(|(e, s)| (e.id.clone(), *s))
        .collect();
    let similarity = scored
        .iter()
        .find(|(e, _)| e.id == best.0.id)
        .map(|(_, s)| *s)
        .unwrap_or(0.0);

    // Order matters:
    //   1. an explicit disagreement wins - a contradiction must not be
    //      laundered into a "newer version of the same fact";
    //   2. an equivalent instance (same subject, or near-identical text) is
    //      reinforced instead of duplicated;
    //   3. a close paraphrase supersedes, but only when the incoming instance is
    //      at least as authoritative as the stored one;
    //   4. otherwise the two are related but independent.
    let action = if detect_conflict(entry, best.0) {
        ReconcileAction::Contradict {
            existing_id: best.0.id.clone(),
        }
    } else if same_subject(entry, best.0) || similarity >= REINFORCE_AT {
        ReconcileAction::Reinforce {
            existing_id: best.0.id.clone(),
        }
    } else if similarity >= SUPERSEDE_AT {
        // Never let a weaker source replace a stronger one: an LLM paraphrase
        // must not supersede something the user stated. Equal authority is
        // fine - the incoming instance is the newer observation.
        if entry.effective_provenance().method.authority_rank()
            >= best.0.effective_provenance().method.authority_rank()
        {
            ReconcileAction::Supersede {
                existing_id: best.0.id.clone(),
            }
        } else {
            ReconcileAction::Relate {
                existing_id: best.0.id.clone(),
            }
        }
    } else {
        ReconcileAction::Relate {
            existing_id: best.0.id.clone(),
        }
    };

    ReconcileOutcome {
        action,
        similarity,
        related,
    }
}

// ---------------------------------------------------------------------------
// Noise gate
// ---------------------------------------------------------------------------

/// Content shorter than this is almost always extraction noise (a bare
/// identifier, a truncated path fragment, a lone symbol).
pub const MIN_MEANINGFUL_MEMORY_LEN: usize = 12;

/// Reason an incoming instance must be rejected, if any.
///
/// Empty content returns `None`: that case is already an error in
/// `validation::validate_new_entry` (E20) and the two checks are deliberately
/// disjoint so a candidate trips exactly one gate.
pub fn noise_reason(content: &str) -> Option<&'static str> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.chars().count() < MIN_MEANINGFUL_MEMORY_LEN {
        return Some("content shorter than the minimum meaningful length");
    }
    if !trimmed.chars().any(char::is_alphanumeric) {
        return Some("content has no alphanumeric substance");
    }
    if looks_like_tool_payload(trimmed) {
        return Some("content is a raw tool payload, not knowledge");
    }
    if looks_like_bare_identifier(trimmed) {
        return Some("content is a bare identifier");
    }
    if looks_like_commit_hash(trimmed) {
        return Some("content is a bare commit hash");
    }
    None
}

fn looks_like_tool_payload(content: &str) -> bool {
    const MARKERS: [&str; 8] = [
        "#tool:",
        "#tool ",
        "tool:",
        "\"command\"",
        "input: {",
        "output: {",
        "tool_call",
        "function_call",
    ];
    let lower = content.to_ascii_lowercase();
    MARKERS.iter().any(|m| lower.starts_with(m))
        || lower.starts_with('{') && (lower.contains("\"command\"") || lower.contains("\"input\""))
}

fn looks_like_bare_identifier(content: &str) -> bool {
    if content.contains(char::is_whitespace) {
        return false;
    }
    // A single token that is a path, url or dotted identifier and nothing else.
    content.contains('/') || content.contains("::") || content.contains('.')
}

fn looks_like_commit_hash(content: &str) -> bool {
    let token = content.trim();
    (7..=40).contains(&token.len()) && token.chars().all(|c| c.is_ascii_hexdigit())
}

// ---------------------------------------------------------------------------
// subject_key derivation
// ---------------------------------------------------------------------------

/// Derive a normalized retrieval key for an instance.
///
/// The key is what makes lexical recall and write-time reconciliation reliable
/// without embeddings: `api:/api/task/claim`, `host:localhost:1337`,
/// `file:crates/jcode-base/src/memory.rs`, `task:#157`, `mr:!358`.
///
/// Returns `None` when the content carries no stable subject (pure prose
/// preferences, for example); such instances rely on tag/term overlap instead.
pub fn derive_subject_key(content: &str) -> Option<String> {
    let lower = content.to_ascii_lowercase();

    if let Some(route) = extract_api_route(&lower) {
        return Some(format!("api:{route}"));
    }
    if let Some(host) = extract_host_port(&lower) {
        return Some(format!("host:{host}"));
    }
    if let Some(path) = extract_file_path(content) {
        return Some(format!("file:{path}"));
    }
    if let Some(mr) = extract_prefixed_id(&lower, '!') {
        return Some(format!("mr:{mr}"));
    }
    if let Some(task) = extract_task_id(&lower) {
        return Some(format!("task:{task}"));
    }
    None
}

fn extract_api_route(lower: &str) -> Option<String> {
    let start = lower.find("/api/")?;
    let rest = &lower[start..];
    let end = rest
        .find(|c: char| c.is_whitespace() || matches!(c, '`' | '"' | '\'' | ')' | ',' | '?'))
        .unwrap_or(rest.len());
    let route = rest[..end].trim_end_matches('/');
    if route.len() > 4 && route.len() < 120 {
        Some(route.to_string())
    } else {
        None
    }
}

fn extract_host_port(lower: &str) -> Option<String> {
    for token in lower.split(|c: char| c.is_whitespace() || matches!(c, '`' | '"' | '\'' | '(' | ')')) {
        let candidate = token.trim_matches(|c: char| c == ',' || c == ';' || c == '.');
        let Some((host, port)) = candidate.rsplit_once(':') else {
            continue;
        };
        if host.is_empty() || port.is_empty() || !port.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let host_ok = host == "localhost"
            || host.chars().all(|c| c.is_ascii_digit() || c == '.')
            || host.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.');
        if host_ok {
            return Some(format!("{host}:{port}"));
        }
    }
    None
}

fn extract_file_path(content: &str) -> Option<String> {
    const EXTENSIONS: [&str; 14] = [
        ".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".rb", ".md", ".toml", ".yaml",
        ".yml", ".json",
    ];
    for raw in content.split_whitespace() {
        let token = raw.trim_matches(|c: char| {
            matches!(c, '`' | '"' | '\'' | '(' | ')' | ',' | ';' | ':' | '*')
        });
        let lower = token.to_ascii_lowercase();
        if !EXTENSIONS.iter().any(|ext| lower.ends_with(ext)) {
            continue;
        }
        if token.len() > 160 {
            continue;
        }
        return Some(token.trim_start_matches("./").to_string());
    }
    None
}

fn extract_prefixed_id(lower: &str, prefix: char) -> Option<String> {
    let mut search = lower;
    while let Some(idx) = search.find(prefix) {
        let rest = &search[idx + prefix.len_utf8()..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.len() >= 2 {
            return Some(format!("{prefix}{digits}"));
        }
        search = rest;
    }
    None
}

fn extract_task_id(lower: &str) -> Option<String> {
    for marker in ["task #", "task#", "#"] {
        if let Some(idx) = lower.find(marker) {
            let rest = &lower[idx + marker.len()..];
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.len() >= 2 {
                return Some(format!("#{digits}"));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tag vocabulary
// ---------------------------------------------------------------------------

/// Tags emitted by the extraction/transcript plumbing that describe the *role
/// of the speaker* rather than a concept. They carry no retrieval value and
/// dominated the v1 tag layer (`assistant` alone was 247 of 1395 tag
/// assignments).
pub const TAG_DENY_LIST: [&str; 12] = [
    "assistant",
    "user",
    "system",
    "tool",
    "reply",
    "think",
    "thinking",
    "message",
    "prompt",
    "context",
    "conversation",
    "turn",
];

/// Whether a tag is admitted into the graph.
pub fn is_admitted_tag(tag: &str) -> bool {
    let normalized = tag.trim().to_ascii_lowercase();
    if normalized.len() < 2 || normalized.len() > 64 {
        return false;
    }
    if TAG_DENY_LIST.contains(&normalized.as_str()) {
        return false;
    }
    // Namespaced tags (`goal:x`, `skill:y`, `goal_status:active`) are written by
    // the owning subsystems and always admitted.
    if normalized.contains(':') {
        return true;
    }
    normalized
        .chars()
        .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | '#' | '/'))
}

/// Normalize and filter a tag list for storage.
pub fn sanitize_tags<'a, I>(tags: I) -> Vec<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut out: Vec<String> = Vec::new();
    for tag in tags {
        let trimmed = tag.trim().to_lowercase();
        if !is_admitted_tag(&trimmed) || out.iter().any(|existing| existing == &trimmed) {
            continue;
        }
        out.push(trimmed);
    }
    out
}

/// Prepare an entry for storage: stamp a derived `subject_key` when absent and
/// drop tags that carry no retrieval value.
///
/// Idempotent - safe to call on entries that were already sanitized.
pub fn prepare_for_write(mut entry: MemoryEntry) -> MemoryEntry {
    if entry.subject_key.is_none() {
        entry.subject_key = derive_subject_key(&entry.content);
    }
    let sanitized = sanitize_tags(entry.tags.iter().map(String::as_str));
    if sanitized.len() != entry.tags.len() {
        entry.tags = sanitized;
        entry.search_text.clear();
        entry.search_text = crate::normalize_memory_search_text(&entry.content, &entry.tags);
    }
    // `entity` instances must carry an identity to be resolvable; without one
    // they are demoted to the residual type instead of polluting the entity
    // layer with anonymous rows (v1: 175 entities, 0 identities).
    if entry.category == MemoryCategory::Entity && entry.identity.is_empty() {
        entry.category = MemoryCategory::Fact;
    }
    entry
}

/// Whether an instance is authoritative enough to be superseded by an incoming
/// instance of the given method. Exposed for callers that want the same policy
/// outside [`reconcile`].
pub fn can_supersede(incoming: &MemoryEntry, existing: &MemoryEntry) -> bool {
    incoming.effective_provenance().method.authority_rank()
        >= existing.effective_provenance().method.authority_rank()
}

#[cfg(test)]
#[path = "reconcile_tests.rs"]
mod tests;
