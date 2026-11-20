//! Memory system for cross-session learning
//!
//! Provides persistent memory that survives across sessions, organized by:
//! - Project (per working directory)
//! - Global (user-level preferences)
//!
//! Jev provides typed relevance decisions. Optional text-generating extraction
//! is independent of recall and is never required to read existing memories.

use crate::memory_graph::{GRAPH_VERSION, MemoryGraph};
use crate::memory_types::{
    InjectedMemoryItem, MemoryActivity, MemoryEvent, MemoryEventKind, MemoryState, StepResult,
    StepStatus,
    ranking::{top_k_by_ord, top_k_by_score},
};
use crate::sidecar::Sidecar;
use crate::storage;
use anyhow::Result;
use chrono::{DateTime, Utc};
use jcode_memory_types::layered::{
    OntologyLayer, compose_ontology_layers, diff_ontology_layer, layer_stack_hash,
};
use jcode_memory_types::{GraphBackend, StoreKey};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[path = "memory/activity.rs"]
mod activity;
mod backend;
mod cache;
#[path = "memory/ontology_registry.rs"]
mod ontology_registry;
#[path = "memory/pending.rs"]
mod pending;
#[path = "memory_prompt.rs"]
mod prompt_support;
#[path = "memory/reextract.rs"]
mod reextract;
#[path = "memory/review.rs"]
mod review;
#[path = "memory/source_health.rs"]
mod source_health;

pub use crate::memory_types::{
    ExtractionMethod, MemoryCategory, MemoryEntry, MemoryScope, MemoryStore, ProvenanceRecord,
    Reinforcement, TrustLevel, format_relevant_display_prompt, format_relevant_prompt,
};
use crate::memory_types::{
    collect_skill_query_terms, format_entries_for_prompt, memory_matches_search,
    normalize_memory_search_text, normalize_search_text, skill_retrieval_bonus,
};
pub use activity::{
    activity_snapshot, add_event, apply_remote_activity_snapshot, check_staleness, clear_activity,
    get_activity, pipeline_start, pipeline_update, record_injected_prompt, set_state,
};
use cache::{cache_graph, cache_graph_for_backend, cached_graph};
pub use jcode_memory_types::{
    AdmissionTier, ChangeLevel, ConfirmationGate, MemoryReviewItem, MemoryReviewState,
    ReviewRecommendation, ReviewStatus, admission_tier,
};
use ontology_registry::OntologyRegistry;
pub(crate) use pending::set_pending_memory_for_project_with_selection;
pub use pending::{
    PendingMemory, clear_all_injected_memories, clear_all_pending_memory, clear_injected_memories,
    clear_pending_memory, has_any_pending_memory, has_pending_memory, is_memory_injected,
    is_memory_injected_any, mark_memories_injected, mark_memories_known, set_pending_memory,
    set_pending_memory_for_project, set_pending_memory_with_ids,
    set_pending_memory_with_ids_and_display, sync_injected_memories, take_pending_memory,
    take_pending_memory_for_project,
};
#[cfg(test)]
use pending::{backdate_injected_memory_for_test, insert_pending_memory_for_test};
use pending::{begin_memory_check, finish_memory_check};
pub(crate) use prompt_support::format_context_for_extraction;
pub use prompt_support::{
    focus_query_text, format_context_for_relevance, format_focused_query_for_relevance,
};
pub use reextract::{ReextractItem, ReextractReport, resolve_reextract_sessions};
pub use review::{MemoryHealthReport, ReviewHealth};
pub use source_health::{ExtractionBatch, ExtractionHealth};

const LEGACY_NOTE_CATEGORY: &str = "note";
const MEMORY_RELEVANCE_MAX_CANDIDATES: usize = 30;
const MEMORY_RELEVANCE_MAX_RESULTS: usize = 10;

/// Apply an ontology-derived plan to a freshly added memory inside `graph`.
/// Wraps the graph-level effects from `jcode_memory_types::rule_engine`
/// and additionally binds the graph metadata so the next save records the
/// active ontology id.
// Dormant since the upstream Jev merge rewired remember_project/global to
// remember_in_graph; kept for the planned ontology-driven write path.
#[allow(dead_code)]
fn apply_plan(
    graph: &mut MemoryGraph,
    new_id: &str,
    plan: &jcode_memory_types::rule_engine::RulePlan,
) {
    if let Some(entry) = graph.memories.get_mut(new_id) {
        jcode_memory_types::rule_engine::apply_entry_effects(plan, entry);
    }
    jcode_memory_types::rule_engine::apply_graph_effects(plan, new_id, graph);
}

/// Producer of synthetic [`MemoryEntry`] values contributed by a higher layer.
///
/// Used to invert the legacy `memory -> skill` dependency: the `skill` layer
/// (which already depends on `MemoryEntry`) registers a provider that turns the
/// shared skill registry into synthetic memory entries, instead of `memory`
/// reaching up into `skill::SkillRegistry`.
type SyntheticEntryProvider = fn() -> Vec<MemoryEntry>;

static SYNTHETIC_ENTRY_PROVIDERS: std::sync::LazyLock<
    std::sync::RwLock<Vec<SyntheticEntryProvider>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(Vec::new()));

/// Register a provider of synthetic memory entries (e.g. skills).
///
/// Inverts `memory -> skill`: higher layers register their synthetic-entry
/// source here at startup so `memory` stays free of upward references.
pub fn register_synthetic_entry_provider(provider: SyntheticEntryProvider) {
    SYNTHETIC_ENTRY_PROVIDERS
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push(provider);
}

#[cfg(test)]
fn collect_synthetic_entries() -> Vec<MemoryEntry> {
    let providers = SYNTHETIC_ENTRY_PROVIDERS
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut entries = Vec::new();
    for provider in providers.iter() {
        entries.extend(provider());
    }
    entries
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct LegacyNotesFile {
    #[serde(default)]
    entries: Vec<LegacyNoteEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyNoteEntry {
    id: String,
    content: String,
    created_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tag: Option<String>,
}

pub type MemoryEventSink = Arc<dyn Fn(crate::protocol::ServerEvent) + Send + Sync>;

/// Optional text-generating extraction is independent of Jev recall.
pub fn memory_sidecar_enabled() -> bool {
    crate::config::config().agents.memory_sidecar_enabled
}

/// Availability of the optional extraction sidecar, never used to gate recall.
pub fn memory_llm_judge_available() -> bool {
    memory_sidecar_enabled() && crate::sidecar::Sidecar::llm_backend_available()
}

/// Whether recall should run at all this moment.
///
/// Recall is served by Jev typed Decisions, so it requires a Jev credential
/// route (subscription entitlement is checked by the gateway, not inferred
/// from a cached client tier). Additionally, when the user explicitly opted
/// OUT of the sidecar (`memory_sidecar_enabled = false`) they asked for a
/// fully local experience, so skip remote recall rather than spending their
/// Jev quota behind their back.
pub fn memory_runtime_active() -> bool {
    memory_sidecar_enabled() && crate::jev::JevClient::available()
}

/// Whether local (L1) recall may run when the Jev judge route is unavailable.
///
/// L1 needs no embedder, no LLM and no credential: it scores `subject_key`,
/// tags and term overlap against the stored instances. Before it existed, a
/// machine without a Jev route silently recalled nothing at all (measured: 0
/// judge decisions and 11 injections across 214 real sessions in 15 days).
pub fn l1_recall_enabled() -> bool {
    crate::config::config().agents.memory_l1_recall
}

/// Minimum L1 score for an instance to be injected.
///
/// Chosen so a single keyword collision is not enough: the query has to share
/// a subject anchor or at least two meaningful terms.
pub const L1_MIN_SCORE: f32 = 0.34;
/// Maximum instances L1 injects into one turn.
pub const L1_MAX_RESULTS: usize = 3;
/// Minimum `relates_to` weight used to expand an L1 hit to a neighbour.
pub const L1_LINK_MIN_WEIGHT: f32 = 0.65;

/// Local relevance scoring (L1): deterministic, no network, no embedder.
///
/// Score is the best of three signals, in descending reliability:
/// `subject_key` held by the query (1.0), a query term appearing in a tag
/// (0.6), and term overlap with the instance's searchable text (up to 0.5).
/// Returns entries ordered best-first.
pub fn score_l1_relevance(
    entry: &MemoryEntry,
    query_terms: &[String],
    normalized_query: &str,
) -> f32 {
    if !entry.is_retrievable_now() {
        return 0.0;
    }
    if let Some(subject) = entry.subject_key.as_deref() {
        let subject_head = subject.split(':').nth(1).unwrap_or(subject);
        // Both sides get the query normalization: the normalizer maps `-` /
        // `/` / `:` to spaces, so comparing the raw head (e.g.
        // `api-api-task-param`) against the normalized query would never
        // match, and anchored recall silently lost every dashed subject
        // (measured: the live two-turn check scored 0 where the anchor was
        // spelled the same way).
        let subject_norm = if subject_head.contains(['-', '/', '.', '_']) {
            normalize_search_text(subject_head)
        } else {
            subject_head.to_string()
        };
        if !subject_norm.is_empty() && normalized_query.contains(&subject_norm) {
            return 1.0;
        }
    }
    let mut tag_hits = 0usize;
    for tag in &entry.tags {
        let bare = tag.split(':').next_back().unwrap_or(tag);
        if bare.len() >= 3 && normalized_query.contains(bare) {
            tag_hits += 1;
        }
    }
    if tag_hits > 0 {
        return 0.6;
    }
    if query_terms.is_empty() {
        return 0.0;
    }
    let searchable = entry.searchable_text();
    let hits = query_terms
        .iter()
        .filter(|term| searchable.contains(term.as_str()))
        .count();
    let ratio = hits as f32 / query_terms.len() as f32;
    // Two independent terms are the floor for a lexical match.
    if hits < 2 { 0.0 } else { ratio.min(0.5) }
}

/// Result of a multi-source/multi-target memory path query.
///
/// `paths` contains only successfully resolved routes. The diagnostic fields
/// let the caller distinguish "anchor does not exist" from "anchors exist but
/// the graph is disconnected" or "relation name is unknown" without retrying
/// with a wider search.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MemoryPathQueryOutcome {
    pub paths: Vec<jcode_memory_types::MemoryPath>,
    pub unresolved_from: Vec<String>,
    pub unresolved_to: Vec<String>,
    pub unknown_relations: Vec<String>,
    /// Stable diagnosis codes (E01 anchor unresolved, E03 disconnected,
    /// E05 depth budget exhausted, E10 unknown relation).
    pub diagnosis: Vec<String>,
    /// Deterministic L0 relaxations applied after the first attempt, e.g.
    /// `depth 8 -> 16`. Empty when the original query succeeded.
    pub relaxations: Vec<String>,
    pub stores_searched: usize,
    pub depth: usize,
}

const MAX_INITIAL_PATH_DEPTH: usize = 8;
const MAX_RELAXED_PATH_DEPTH: usize = 16;

/// Explicit bridge between a project-scoped memory and a global memory.
///
/// Bridges are the only sanctioned way to join project and global paths. They
/// are persisted separately so neither store's graph file is rewritten and the
/// alignment can be audited or removed independently.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryBridge {
    pub id: String,
    pub project_store: String,
    pub project_id: String,
    pub global_id: String,
    pub weight: f32,
    pub confidence: f32,
    pub source: String,
    pub created_at: DateTime<Utc>,
}

fn normalize_path_anchors(anchors: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for anchor in anchors {
        let anchor = anchor.trim();
        if anchor.is_empty() || !seen.insert(anchor.to_string()) {
            continue;
        }
        out.push(anchor.to_string());
        if out.len() >= 16 {
            break;
        }
    }
    out
}

fn stitch_paths_with_bridge(
    left: &jcode_memory_types::MemoryPath,
    bridge: jcode_memory_types::MemoryPathEdge,
    right: &jcode_memory_types::MemoryPath,
    bridge_confidence: f32,
) -> Option<jcode_memory_types::MemoryPath> {
    let bridge_from = bridge.from.clone();
    let bridge_to = bridge.to.clone();
    let bridge_cost = bridge.cost;
    if left.nodes.last() != Some(&bridge_from) || right.nodes.first() != Some(&bridge_to) {
        return None;
    }
    let mut nodes = left.nodes.clone();
    nodes.extend(right.nodes.iter().cloned());
    let mut edges = left.edges.clone();
    edges.push(bridge);
    edges.extend(right.edges.iter().cloned());
    Some(jcode_memory_types::MemoryPath {
        nodes,
        edges,
        cost: left.cost + bridge_cost + right.cost,
        confidence: left.confidence.min(bridge_confidence).min(right.confidence),
    })
}

/// Resolve a path endpoint without requiring the caller to know an internal id.
fn resolve_path_anchor(graph: &MemoryGraph, anchor: &str) -> Option<String> {
    let anchor = anchor.trim();
    if anchor.is_empty() {
        return None;
    }
    if graph.get_memory(anchor).is_some() {
        return Some(anchor.to_string());
    }
    if let Some(subject) = anchor.strip_prefix("subject:") {
        let wanted = jcode_memory_types::reconcile::normalize_subject_key(subject);
        return pick_best_path_entry(graph.active_memories().filter(|entry| {
            entry.subject_key.as_deref().is_some_and(|candidate| {
                jcode_memory_types::reconcile::normalize_subject_key(candidate)
                    .eq_ignore_ascii_case(&wanted)
            })
        }));
    }
    if let Some(tag) = anchor.strip_prefix("tag:") {
        return pick_best_path_entry(
            graph
                .get_memories_by_tag(tag)
                .into_iter()
                .filter(|entry| entry.is_retrievable_now()),
        );
    }

    let normalized = normalize_search_text(anchor);
    if normalized.is_empty() {
        return None;
    }
    let normalized_subject = jcode_memory_types::reconcile::normalize_subject_key(anchor);
    pick_best_path_entry(graph.active_memories().filter(|entry| {
        (!normalized_subject.is_empty()
            && entry.subject_key.as_deref().is_some_and(|subject| {
                let subject = jcode_memory_types::reconcile::normalize_subject_key(subject);
                subject.contains(&normalized_subject) || normalized_subject.contains(&subject)
            }))
            || normalize_search_text(&entry.searchable_text()).contains(&normalized)
    }))
}

fn pick_best_path_entry<'a>(entries: impl Iterator<Item = &'a MemoryEntry>) -> Option<String> {
    entries
        .max_by(|left, right| {
            left.confidence
                .total_cmp(&right.confidence)
                .then_with(|| left.updated_at.cmp(&right.updated_at))
                .then_with(|| right.id.cmp(&left.id))
        })
        .map(|entry| entry.id.clone())
}

/// Whether the sidecar is currently auto-disabled due to sustained judge
/// failures. Exposed for the TUI status badge so the user can see why memory
/// is silently running on the no-LLM path.
pub fn memory_sidecar_auto_disabled() -> bool {
    crate::memory_judge_metrics::sidecar_should_auto_disable()
}

/// Scope selector for the GC passes. Local to the manager so the public
/// `MemoryScope` API stays unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GcScope {
    Project,
    Global,
}

/// Outcome of a TTL maintenance pass (`activity.periodic_gc`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TtlGcReport {
    /// Instances archived because their type TTL elapsed.
    pub archived: usize,
    /// Instances physically removed by [`MemoryManager::purge_archived_memories`].
    pub purged: usize,
}

impl ReclassifyReport {
    /// Everything this pass would write, types plus anchors.
    pub fn total_changes(&self) -> usize {
        self.type_changes + self.subject_keys
    }

    /// Human-readable one-liner shared by the CLI and the finalize log.
    pub fn summary(&self) -> String {
        let mut out = format!(
            "{} store(s), {} visited, {} residual, {} type change(s), {} subject key(s)",
            self.stores, self.scanned, self.residual, self.type_changes, self.subject_keys
        );
        if !self.by_type.is_empty() {
            let detail: Vec<String> = self
                .by_type
                .iter()
                .map(|(target, count)| format!("{target}: {count}"))
                .collect();
            out.push_str(&format!(" [{}]", detail.join(", ")));
        }
        out
    }
}

impl TtlGcReport {
    fn merge(&mut self, other: TtlGcReport) {
        self.archived += other.archived;
        self.purged += other.purged;
    }
}

/// Outcome of an all-store ontology data migration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct OntologyMigrationReport {
    /// Stores visited by the pass.
    pub stores: usize,
    /// Stores whose core-only default ontology was rewritten to the compiled
    /// default definition (description, hash and version included).
    #[serde(default)]
    pub ontology_canonicalized: usize,
    /// Whether this was a dry run.
    pub dry_run: bool,
    /// Aggregated per-step migration counts.
    pub migrations: jcode_memory_types::migration::MigrationReport,
}

impl OntologyMigrationReport {
    /// Human-readable one-liner shared by the CLI and maintenance logs.
    pub fn summary(&self) -> String {
        let migrations = &self.migrations;
        format!(
            "{} store(s), {} ontology canonicalized, {} change(s): {} remapped, {} split, {} retired, {} archived, {} anchor(s), {} tag pass(es), {} lifecycle fix(es), {} entity demotion(s), {} dangling edge(s), {} rewired edge(s), {} reclassified",
            self.stores,
            self.ontology_canonicalized,
            migrations.changed(),
            migrations.remapped,
            migrations.split,
            migrations.retired,
            migrations.archived,
            migrations.subject_keys,
            migrations.tags_resanitized,
            migrations.lifecycle_fixed,
            migrations.entities_demoted,
            migrations.edges_dropped,
            migrations.edges_rewired,
            migrations.reclassified,
        )
    }
}

/// Classify every eligible instance in one graph.
///
/// Returns true when the graph was modified (and therefore needs saving).
/// Shared by the all-stores pass and the per-project pass so both agree on the
/// rules and the accounting.
fn classify_graph(
    graph: &mut MemoryGraph,
    ontology: &jcode_memory_types::Ontology,
    stamp_subject_keys: bool,
    apply: bool,
    report: &mut ReclassifyReport,
) -> bool {
    let residual = jcode_memory_types::ontology::TYPE_OBSERVATION;
    let mut touched = false;
    for entry in graph.memories.values_mut() {
        if !entry.active {
            continue;
        }
        // Link nodes are indexes of external artifacts; their category is
        // owned by the artifact subsystem and must not be inferred from the
        // pointer text (`skill:` pointers often mention APIs or "must").
        if jcode_memory_types::is_link_node(entry) {
            continue;
        }
        report.scanned += 1;
        if stamp_subject_keys
            && entry.subject_key.is_none()
            && let Some(key) = jcode_memory_types::reconcile::derive_subject_key(&entry.content)
        {
            report.subject_keys += 1;
            if apply {
                entry.subject_key = Some(key);
                entry.updated_at = Utc::now();
                touched = true;
            }
        }
        // Only residual rows are type candidates: a correction stays a
        // correction, and an already-typed row is left alone.
        let is_residual = ontology
            .type_from_category(&entry.category)
            .is_some_and(|t| t == residual);
        if !is_residual {
            continue;
        }
        report.residual += 1;
        let Some(target) = classify_residual_type(&entry.content) else {
            continue;
        };
        if type_name_of(&entry.category) == target {
            continue;
        }
        *report.by_type.entry(target.to_string()).or_default() += 1;
        report.type_changes += 1;
        if apply {
            // Parse (not `Custom(target)`) so the store converges on real
            // enum variants instead of string-shaped categories.
            entry.category = target.parse().unwrap_or(MemoryCategory::Fact);
            entry.updated_at = Utc::now();
            touched = true;
        }
    }
    touched
}

/// Scope of a stored ontology. Project and global stores each carry their own
/// definition so a reflection pass can tune one workspace without changing the
/// machine-wide knowledge base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OntologyScope {
    Project,
    Global,
}

impl OntologyScope {
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "project" | "local" => Ok(Self::Project),
            "global" | "machine" => Ok(Self::Global),
            other => Err(anyhow::anyhow!(
                "unknown ontology scope '{other}'; use project or global"
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Project => "project",
            Self::Global => "global",
        }
    }
}

/// Outcome of a reclassification pass ([`MemoryManager::reclassify_observations`]).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ReclassifyReport {
    /// Stores visited by the pass.
    pub stores: usize,
    /// Active instances visited across all stores.
    pub scanned: usize,
    /// Of those, instances sitting in the residual (`fact` / `observation`)
    /// bucket and therefore eligible for reclassification.
    pub residual: usize,
    /// Instances whose *type* would change (or did, when not a dry run).
    pub type_changes: usize,
    /// Instances that gained a derived `subject_key`.
    pub subject_keys: usize,
    /// Target type -> number of instances.
    pub by_type: std::collections::BTreeMap<String, usize>,
    /// Whether this was a dry run.
    pub dry_run: bool,
}

fn type_name_of(category: &MemoryCategory) -> &str {
    match category {
        MemoryCategory::Custom(name) => name.as_str(),
        MemoryCategory::Fact => "fact",
        MemoryCategory::Preference => "preference",
        MemoryCategory::Entity => "entity",
        MemoryCategory::Correction => "correction",
        MemoryCategory::EnvFact => "env_fact",
        MemoryCategory::Contract => "contract",
        MemoryCategory::ProcessRule => "process_rule",
        MemoryCategory::Observation => "observation",
    }
}

/// Deterministic classifier for residual (`fact` / `observation`) instances.
///
/// The implementation lives in `jcode_memory_types::migration` so ontology
/// migrations and the runtime classification pass share one definition.
pub use jcode_memory_types::migration::classify_residual_type;

/// Whether an anchor can name a reusable procedure, i.e. whether it is worth
/// turning accumulated evidence about it into a skill.
///
/// `task:*`, `mr:*` and `host:*` are instance or environment identifiers: a
/// skill about "task #117" or "localhost:1337" is not reusable, even though the
/// anchor is an excellent retrieval key for the memory itself. Bare
/// JS-ecosystem filenames are usually dependency or product names ("Next.js"),
/// so a file anchor only counts when it carries a path or names a
/// configuration/source file that a repository actually owns.
fn proposal_eligible_subject(key: &str) -> bool {
    match key.split_once(':') {
        Some(("api", _)) => true,
        Some(("file", rest)) => {
            rest.contains('/')
                || !matches!(
                    rest.rsplit_once('.').map(|(_, ext)| ext),
                    Some("js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs")
                )
        }
        _ => false,
    }
}

/// Minimum number of agreeing evidence instances before a skill is proposed.
const SKILL_DISTILL_MIN_EVIDENCE: usize = 3;
/// Minimum number of distinct stores the evidence must span.
const SKILL_DISTILL_MIN_STORES: usize = 2;
/// Upper bound on emitted proposals per pass. The queue is for review, not a
/// dump of every subject the store has ever mentioned.
const SKILL_DISTILL_MAX_PROPOSALS: usize = 20;
/// Minimum new-session evidence before an automatic ontology reflection call.
const ONTOLOGY_REFLECTION_MIN_EVIDENCE: usize = 5;
/// At most one automatic reflection proposal per project per day.
const ONTOLOGY_REFLECTION_INTERVAL_HOURS: i64 = 24;
/// Repository-owned project ontology overlay, relative to the project root.
const PROJECT_ONTOLOGY_OVERLAY: &str = ".jcode/memory/ontology.yaml";

#[derive(Debug, Serialize, Deserialize)]
struct OntologyReflectionState {
    last_run: DateTime<Utc>,
}

fn ontology_reflection_enabled() -> bool {
    match std::env::var("JCODE_ONTOLOGY_REFLECTION") {
        Ok(value) => !matches!(value.trim(), "0" | "false" | "off" | "no"),
        Err(_) => true,
    }
}

fn layer_is_empty(layer: &OntologyLayer) -> bool {
    layer.types.is_empty()
        && layer.type_patches.is_empty()
        && layer.disabled_types.is_empty()
        && layer.relationships.is_empty()
        && layer.disabled_relationships.is_empty()
        && layer.rules.is_empty()
        && layer.disabled_rules.is_empty()
        && layer.activities.is_empty()
        && layer.disabled_activities.is_empty()
        && layer.category_aliases.is_empty()
        && layer.disabled_aliases.is_empty()
}

fn same_by_id<T: PartialEq, F: Fn(&T) -> &str>(left: &[T], right: &[T], id: F) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let by_id: std::collections::HashMap<&str, &T> =
        left.iter().map(|value| (id(value), value)).collect();
    by_id.len() == right.len()
        && right
            .iter()
            .all(|value| by_id.get(id(value)) == Some(&value))
}

fn is_subset_by_id<T: PartialEq, F: Fn(&T) -> &str>(subset: &[T], superset: &[T], id: F) -> bool {
    let by_id: std::collections::HashMap<&str, &T> =
        superset.iter().map(|value| (id(value), value)).collect();
    subset
        .iter()
        .all(|value| by_id.get(id(value)) == Some(&value))
}

fn same_rules(left: &[jcode_memory_types::Rule], right: &[jcode_memory_types::Rule]) -> bool {
    same_by_id(left, right, |rule| rule.id.as_str())
}

fn same_activities(
    left: &[jcode_memory_types::Activity],
    right: &[jcode_memory_types::Activity],
) -> bool {
    same_by_id(left, right, |activity| activity.id.as_str())
}

fn ontology_reflection_dir() -> Result<PathBuf> {
    Ok(storage::jcode_dir()?.join("memory").join("ontology"))
}

fn ontology_reflection_state_path(project_dir: &Path) -> Result<PathBuf> {
    let key = crate::memory::project_store_key(project_dir).replace(':', "-");
    Ok(ontology_reflection_dir()?.join(format!(".reflection-{key}.json")))
}

fn write_ontology_proposal(
    patch: &jcode_memory_types::migration::OntologyPatch,
) -> Result<PathBuf> {
    let dir = ontology_reflection_dir()?.join(".proposals");
    std::fs::create_dir_all(&dir)?;
    let slug: String = patch
        .base_id
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect();
    let path = dir.join(format!(
        "{}-auto-{}-{:08x}.json",
        Utc::now().format("%Y%m%dT%H%M%S"),
        slug,
        rand::random::<u32>()
    ));
    std::fs::write(&path, serde_json::to_vec_pretty(patch)?)?;
    Ok(path)
}

/// Keep an automatic reflection patch inside the low-risk envelope:
/// type-parameter updates only, evidence from this session only, and a patch
/// that must validate against the current ontology. Anything else is dropped
/// rather than written as a misleading proposal.
fn prepare_reflection_patch(
    ontology: &jcode_memory_types::Ontology,
    mut patch: jcode_memory_types::migration::OntologyPatch,
    memory_ids: &[String],
) -> Option<jcode_memory_types::migration::OntologyPatch> {
    use jcode_memory_types::migration::{OntologyOp, apply_patch, validate_ontology};

    patch.base_id = ontology.id.clone();
    patch.base_version = ontology.version;
    patch.base_hash = ontology.content_hash();
    patch.migrations.clear();
    patch.ops.retain(|op| match op {
        OntologyOp::UpdateType { id, .. } => {
            !jcode_memory_types::is_core_memory_class(id) && ontology.get_type(id).is_some()
        }
        _ => false,
    });
    for op in &mut patch.ops {
        if let OntologyOp::UpdateType { patch, .. } = op {
            // `properties` changes the type's schema, not a runtime parameter.
            patch.properties = None;
        }
    }
    if patch.ops.is_empty() {
        return None;
    }
    let allowed: std::collections::BTreeSet<&str> = memory_ids.iter().map(String::as_str).collect();
    patch.evidence.retain(|id| allowed.contains(id.as_str()));
    if patch.evidence.is_empty() {
        patch.evidence = memory_ids.to_vec();
    }
    if patch.rationale.trim().is_empty() {
        patch.rationale = "automatic reflection from session evidence".to_string();
    }
    let next = apply_patch(ontology, &patch).ok()?;
    if !validate_ontology(&next).is_empty() {
        return None;
    }
    if next.content_hash() == ontology.content_hash() {
        return None;
    }
    Some(patch)
}

/// A procedure that the memory store has accumulated enough evidence for.
///
/// Produced by [`MemoryManager::collect_skill_evidence`]: the memory -> skill
/// half of the v2 linkage. `subject_key` is the anchor shared by the evidence
/// instances (e.g. `api:/api/task/claim`), `weight` double-counts corrections
/// because a corrected procedure is stronger evidence than a stated one.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SkillEvidenceProposal {
    pub subject_key: String,
    pub evidence: Vec<String>,
    pub stores: Vec<String>,
    pub sessions: Vec<String>,
    pub weight: f32,
    /// Composite ranking score (evidence, diversity, outcomes, recency).
    #[serde(default)]
    pub score: f32,
    /// Evidence entries carrying `skill-validated`.
    #[serde(default)]
    pub positive: u32,
    /// Evidence entries carrying `skill-invalidated` or failed outcomes.
    #[serde(default)]
    pub negative: u32,
    /// Most recent evidence update.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_seen: Option<DateTime<Utc>>,
    /// True while the draft body is still a skeleton.
    #[serde(default)]
    pub needs_human_draft: bool,
}

impl SkillEvidenceProposal {
    /// Stable file name for the draft, derived from the subject anchor.
    fn slug(&self) -> String {
        let raw = self
            .subject_key
            .replace([':', '/', '{', '}', '.', ' '], "-");
        let filtered: String = raw
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        // Collapse runs and trim both ends: `api:/api/task/{param}` should read
        // `api-api-task-param`, not `api--api-task--param-`.
        let mut slug = String::with_capacity(filtered.len());
        for ch in filtered.chars() {
            if ch == '-' && slug.ends_with('-') {
                continue;
            }
            slug.push(ch);
        }
        let slug: String = slug.trim_matches('-').chars().take(64).collect();
        let mut slug = slug;
        while slug.ends_with('-') {
            slug.pop();
        }
        if slug.is_empty() {
            "skill".to_string()
        } else {
            slug
        }
    }

    /// Draft `SKILL.md` body for review. Deliberately a skeleton: the concrete
    /// steps have to come from the evidence, and nothing is installed silently.
    fn draft(&self, proposed_name: &str) -> String {
        let mut out = String::new();
        out.push_str("---\n");
        out.push_str(&format!("name: {proposed_name}\n"));
        out.push_str("version: 0.1.0\n");
        out.push_str("status: active\n");
        out.push_str("needs_human_draft: true\n");
        out.push_str("triggers:\n");
        out.push_str(&format!("  - work with {}\n", self.subject_key));
        out.push_str(&format!(
            "description: Apply when working on {}. Derived from {} memory evidence item(s) across {} store(s).\n",
            self.subject_key,
            self.evidence.len(),
            self.stores.len()
        ));
        // `evidence:` is the machine-readable half of accepting this draft:
        // copy the file into `<workspace>/.jcode/skills/<name>/SKILL.md` (or
        // `~/.jcode/skills/`) unchanged and the mirror node will record a
        // `derived_from` edge to each evidence memory that still exists.
        out.push_str("evidence:\n");
        for id in self.evidence.iter().take(20) {
            out.push_str(&format!("  - {id}\n"));
        }
        out.push_str("---\n\n");
        out.push_str(&format!(
            "# {proposed_name}\n\nDerived from accumulated memory evidence.\n\n- subject: `{}`\n- evidence: {} instance(s)\n- stores: {}\n- sessions: {}\n- weight (corrections count double): {:.1}\n- score: {:.1}\n- positive: {}, negative: {}\n\n## Evidence\n",
            self.subject_key,
            self.evidence.len(),
            self.stores.len(),
            self.sessions.len(),
            self.weight,
            self.score,
            self.positive,
            self.negative
        ));
        for id in &self.evidence {
            out.push_str(&format!("- `{id}`\n"));
        }
        out.push_str("\n## TODO\n- [ ] Condense the evidence into concrete, checkable steps.\n- [ ] Set `allowed-tools` for the steps above.\n");
        out
    }
}

/// Rate limiter shared by the local maintenance passes. Returns true when the
/// caller may run (and touches the stamp); false when the window is still open.
fn claim_maintenance_slot(base: &Path, stamp: &str, interval_secs: u64) -> bool {
    let path = base.join(stamp);
    let now = std::time::SystemTime::now();
    if let Ok(metadata) = std::fs::metadata(&path)
        && let Ok(modified) = metadata.modified()
        && let Ok(elapsed) = now.duration_since(modified)
        && elapsed.as_secs() < interval_secs
    {
        return false;
    }
    let _ = std::fs::write(&path, b"");
    true
}

/// Write draft skill proposals for review. Existing drafts are refreshed in
/// place (they are derived, not authored), and nothing outside `.proposals/` is
/// touched - installing a skill stays an explicit act.
fn write_skill_proposals(base: &Path, proposals: &[SkillEvidenceProposal]) -> Result<()> {
    let dir = base.join("skills").join(".proposals");
    std::fs::create_dir_all(&dir)?;
    // The slug is human-readable, not injective: `api:/api/task/:id` and
    // `api:/api/task/<id>` both reduce to `api--api-task--id`. Without a
    // per-batch guard the second draft silently overwrote the first (observed:
    // 26 proposals produced 25 files), so duplicates get a numeric suffix.
    let mut used: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut kept: std::collections::HashSet<String> = std::collections::HashSet::new();
    for proposal in proposals {
        let slug = proposal.slug();
        let seen = used.entry(slug.clone()).or_insert(0);
        *seen += 1;
        let file_slug = if *seen == 1 {
            slug
        } else {
            format!("{slug}-{seen}")
        };
        kept.insert(format!("{file_slug}.md"));
        std::fs::write(
            dir.join(format!("{file_slug}.md")),
            proposal.draft(&file_slug),
        )?;
    }
    // Retire proposals nobody acted on. Keeping stale TODOs forever is how the
    // queue grew to 44 files with one installed skill.
    const PROPOSAL_TTL: std::time::Duration = std::time::Duration::from_secs(30 * 24 * 60 * 60);
    if let Ok(entries) = std::fs::read_dir(&dir) {
        let expired = dir.join(".expired");
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("md") {
                continue;
            }
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if kept.contains(name) {
                continue;
            }
            let stale = std::fs::metadata(&path)
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|modified| std::time::SystemTime::now().duration_since(modified).ok())
                .is_some_and(|age| age > PROPOSAL_TTL);
            if stale {
                std::fs::create_dir_all(&expired)?;
                let _ = std::fs::rename(&path, expired.join(name));
            }
        }
    }
    Ok(())
}

/// Minimum meaningful content length (in chars) for an LLM-extracted memory.
///
/// Kept as a re-export of the v2 constant so there is a single definition;
/// the gate itself now lives in `jcode_memory_types::reconcile` and is applied
/// by `rule.reject` / `validate_new_entry` (E23).
pub const MIN_MEANINGFUL_MEMORY_LEN: usize =
    jcode_memory_types::reconcile::MIN_MEANINGFUL_MEMORY_LEN;

/// Classify a memory's content as low-value extraction noise.
///
/// These are the patterns observed polluting the recall pool (P0-3 audit) plus
/// the ones the v2 ontology makes explicit:
/// - empty content (validation handles this as E20)
/// - content shorter than [`MIN_MEANINGFUL_MEMORY_LEN`] chars (bare ids, paths)
/// - content with no alphabetic/digit substance (only symbols, e.g. "---", "::")
/// - pure git commit hashes
/// - raw tool payloads (`#Tool: bash input: {"command": ...}`)
///
/// Returns true when the content looks like noise that should NOT be stored.
pub fn is_extraction_noise(content: &str) -> bool {
    content.trim().is_empty() || jcode_memory_types::reconcile::noise_reason(content).is_some()
}

fn emit_memory_activity(event_tx: Option<&MemoryEventSink>) {
    let (Some(event_tx), Some(activity)) = (event_tx, activity_snapshot()) else {
        return;
    };
    (event_tx)(crate::protocol::ServerEvent::MemoryActivity { activity });
}

trait MemoryEntryEmbeddingExt {
    fn ensure_embedding(&mut self) -> bool;
}

impl MemoryEntryEmbeddingExt for MemoryEntry {
    /// Generate and set embedding if not already present.
    /// Returns true if embedding was generated, false if already exists or failed.
    fn ensure_embedding(&mut self) -> bool {
        if self.embedding.is_some() {
            return false;
        }

        match crate::embedding_backend::embed_passage_active(&self.content) {
            Ok((embedding, model_id)) => {
                // Tag with the ACTIVE backend's model id so dense search only
                // compares vectors from the same model/vector space. Untagged
                // legacy memories are treated as local MiniLM via
                // effective_embedding_model().
                self.set_embedding(Some(embedding), Some(model_id));
                true
            }
            Err(err) => {
                crate::logging::info(&format!("Failed to generate embedding: {err}"));
                false
            }
        }
    }
}

/// Per-project memory file for `project_dir`. Keyed by the absolute path, so a
/// migrated session that keeps the same repo path keeps the same memories.
pub fn project_memory_file(project_dir: &std::path::Path) -> Result<PathBuf> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    project_dir.hash(&mut hasher);
    let project_hash = format!("{:016x}", hasher.finish());
    let memory_dir = storage::jcode_dir()?.join("memory").join("projects");
    Ok(memory_dir.join(format!("{}.json", project_hash)))
}

/// Recovery snapshot used only when the active backend cannot save. It lives
/// outside `memory/projects/` so the canonical legacy audit never mistakes a
/// failed write for a real JSON-mode store.
pub fn project_recovery_file(project_dir: &std::path::Path) -> Result<PathBuf> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    project_dir.hash(&mut hasher);
    let project_hash = format!("{:016x}", hasher.finish());
    let recovery_dir = storage::jcode_dir()?
        .join("memory")
        .join("recovery")
        .join("projects");
    Ok(recovery_dir.join(format!("{}.json", project_hash)))
}

/// Optional `<jcode_dir>/memory/artifact-kinds.json` override.
///
/// The file adds or overrides artifact kinds without a rebuild:
/// `{"kinds":[{"id":"workflow","prefix":"workflow:"}]}`.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct ArtifactKindConfig {
    #[serde(default)]
    pub kinds: Vec<jcode_memory_types::ArtifactKind>,
}

/// Reload artifact kinds from disk. Public so embedding apps and tests can
/// force a refresh; normal startup uses [`ensure_artifact_kind_config`].
pub fn load_artifact_kind_config() {
    let Ok(dir) = crate::storage::jcode_dir() else {
        return;
    };
    let path = dir.join("memory").join("artifact-kinds.json");
    if !path.exists() {
        return;
    }
    match crate::storage::read_json::<ArtifactKindConfig>(&path) {
        Ok(config) => jcode_memory_types::configure_artifact_kinds(config.kinds),
        Err(error) => crate::logging::warn(&format!(
            "artifact kind config {} is unreadable: {error}",
            path.display()
        )),
    }
}

/// Load the artifact-kind config once per process.
pub fn ensure_artifact_kind_config() {
    static LOADED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    LOADED.get_or_init(load_artifact_kind_config);
}

/// One deterministic link-discovery candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkSuggestion {
    pub from: String,
    pub to: String,
    pub weight: f32,
    pub reason: String,
}

/// Result of a link-discovery pass over every memory store.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LinkDiscoveryReport {
    pub stores: usize,
    pub memories: usize,
    pub candidates: usize,
    pub existing: usize,
    pub applied: usize,
    pub capped: bool,
    pub suggestions: Vec<LinkSuggestion>,
}

/// Result of applying an already-reviewed list of link suggestions.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LinkApplyReport {
    pub suggestions: usize,
    pub matched: usize,
    pub existing: usize,
    pub applicable: usize,
    pub applied: usize,
    pub missing: usize,
    pub stores_changed: usize,
}

/// One legacy `projects/<hash>.json` snapshot compared with its SQLite graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyJsonAudit {
    pub path: String,
    pub store_key: String,
    pub json_memories: usize,
    pub json_edges: usize,
    pub backend_memories: Option<usize>,
    pub backend_edges: Option<usize>,
    pub json_only_memories: usize,
    pub backend_only_memories: usize,
    pub common_memories: usize,
    pub stale: bool,
    pub only_copy: bool,
    pub quarantined_to: Option<String>,
}

/// Result of auditing legacy JSON snapshots against the active backend.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LegacyJsonAuditReport {
    pub files: Vec<LegacyJsonAudit>,
    pub stale: usize,
    pub only_copy: usize,
    pub quarantined: usize,
}

/// One legacy JSON snapshot merged into its backend graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyJsonMerge {
    pub path: String,
    pub store_key: String,
    pub json_only_memories: usize,
    pub added_memories: usize,
    pub added_edges: usize,
    pub added_tags: usize,
    pub backend_memories_before: usize,
    pub backend_memories_after: usize,
    pub applied: bool,
}

/// Result of merging legacy JSON snapshots into the active backend.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LegacyJsonMergeReport {
    pub files: Vec<LegacyJsonMerge>,
    pub added_memories: usize,
    pub added_edges: usize,
    pub applied: usize,
}

/// Grouped view of the pending review queue.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReviewTriageReport {
    pub pending: usize,
    pub oldest_pending_days: i64,
    pub by_action: std::collections::BTreeMap<String, usize>,
    pub by_gate: std::collections::BTreeMap<String, usize>,
    pub by_project: Vec<(String, usize)>,
}

/// Result of accepting a filtered slice of the pending review queue.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReviewBatchReport {
    pub pending: usize,
    pub selected: usize,
    pub accepted: usize,
    pub skipped_missing_target: usize,
    pub skipped_existing: usize,
    pub failed: usize,
    pub by_action: std::collections::BTreeMap<String, usize>,
    pub applied: bool,
}

/// Result of rejecting review items that can no longer be applied.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReviewPruneReport {
    pub pending: usize,
    pub stale_existing: usize,
    pub stale_missing_target: usize,
    pub rejected: usize,
    pub applied: bool,
}

pub struct MemoryManager {
    project_dir: Option<PathBuf>,
    /// When true, use isolated test storage instead of real memory
    test_mode: bool,
    include_skills: bool,
    /// Ontology registry consulted for rule dispatch + activity scheduling.
    /// Held by `Arc` so cloning the manager is cheap and tests can substitute
    /// a custom registry without touching global state.
    ontology_registry: OntologyRegistry,
    /// Set when a sqlite load failed and the graph was served from the legacy
    /// JSON snapshot; the next legacy save on this instance must be a no-op so
    /// the snapshot cannot overwrite the newer sqlite store
    /// (`self-suppress`, the fallback is read-only by construction).
    suppress_next_legacy_save: std::sync::atomic::AtomicBool,
}

/// Result of a governed memory write.
///
/// A write can be stored immediately, stored with a review badge, or parked in
/// the durable review queue without touching the graph. Callers that need to
/// tell the difference should use the `remember_*_governed` methods; the legacy
/// `remember_*` methods return the stored id or the review id as a string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryWriteOutcome {
    Stored {
        id: String,
        needs_review: bool,
    },
    PendingReview {
        review_id: String,
        target_id: Option<String>,
    },
}

/// High-level retrieval route chosen from the query shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRoute {
    /// Fact or state lookup.
    Fact,
    /// How-to / procedure lookup; skills should lead.
    Procedure,
    /// Why/which/compare question; graph traversal matters.
    MultiHop,
    /// A requested operation; skill contracts and safety matter.
    Operation,
}

impl MemoryRoute {
    pub fn label(self) -> &'static str {
        match self {
            MemoryRoute::Fact => "fact",
            MemoryRoute::Procedure => "procedure",
            MemoryRoute::MultiHop => "multi_hop",
            MemoryRoute::Operation => "operation",
        }
    }
}

/// Classify a query into a retrieval route with cheap lexical signals.
pub fn route_memory_query(query: &str) -> MemoryRoute {
    let lower = query.to_lowercase();
    let procedure = [
        "how to",
        "how do i",
        "how can i",
        "steps",
        "workflow",
        "procedure",
        "walk me through",
        "怎么",
        "如何",
        "步骤",
        "流程",
    ];
    if procedure.iter().any(|signal| lower.contains(signal)) {
        return MemoryRoute::Procedure;
    }
    let operation = [
        "install", "deploy", "delete", "remove", "create", "run ", "execute", "apply", "migrate",
        "rollback", "安装", "部署", "删除", "执行", "迁移", "回滚",
    ];
    if operation.iter().any(|signal| lower.contains(signal)) {
        return MemoryRoute::Operation;
    }
    let multi_hop = [
        "why",
        "which",
        "compare",
        "difference",
        "relationship",
        "depends on",
        "为什么",
        "哪个",
        "比较",
        "区别",
        "关系",
        "依赖",
    ];
    if multi_hop.iter().any(|signal| lower.contains(signal)) {
        return MemoryRoute::MultiHop;
    }
    MemoryRoute::Fact
}

impl MemoryWriteOutcome {
    pub fn id(&self) -> &str {
        match self {
            MemoryWriteOutcome::Stored { id, .. } => id,
            MemoryWriteOutcome::PendingReview { review_id, .. } => review_id,
        }
    }

    pub fn is_pending_review(&self) -> bool {
        matches!(self, MemoryWriteOutcome::PendingReview { .. })
    }
}

/// Mark durable user-stated preferences, corrections and contracts critical.
///
/// This is deliberately conservative: only an authoritative `UserStated`
/// provenance qualifies, so ordinary LLM observations and agent tool writes do
/// not turn every row into a confirmation gate.
fn apply_critical_policy(entry: &mut MemoryEntry) {
    if entry.critical {
        return;
    }
    let method = entry.effective_provenance().method;
    if method == ExtractionMethod::UserStated
        && matches!(
            entry.category,
            MemoryCategory::Preference | MemoryCategory::Correction | MemoryCategory::Contract
        )
    {
        entry.critical = true;
    }
}

/// G4 threshold: more than five changed records in one batch, or more than
/// 20% of the category's existing active rows.
const BATCH_CHANGE_ABSOLUTE_LIMIT: usize = 5;

fn batch_change_categories(
    candidates: &[MemoryCategory],
    existing_counts: &std::collections::HashMap<MemoryCategory, usize>,
) -> std::collections::HashSet<MemoryCategory> {
    let mut counts = std::collections::HashMap::new();
    for category in candidates {
        *counts.entry(category.clone()).or_insert(0usize) += 1;
    }
    counts
        .into_iter()
        .filter_map(|(category, count)| {
            let existing = existing_counts.get(&category).copied().unwrap_or(0);
            let over_absolute = count > BATCH_CHANGE_ABSOLUTE_LIMIT;
            let over_share = existing > 0 && count * 5 > existing;
            (over_absolute || over_share).then_some(category)
        })
        .collect()
}

/// Confirmation gates for a single proposed memory write.
fn write_confirmation_gates(
    old: Option<&MemoryEntry>,
    new: &MemoryEntry,
) -> Vec<jcode_memory_types::ConfirmationGate> {
    use jcode_memory_types::ConfirmationGate;

    let Some(old) = old else {
        return Vec::new();
    };
    let changed = old.content != new.content
        || old.tags != new.tags
        || old.category != new.category
        || old.properties != new.properties
        || old.lifecycle.status != new.lifecycle.status
        || old.active != new.active;
    if !changed {
        return Vec::new();
    }

    let mut gates = Vec::new();
    // Only an *existing* critical memory makes a destructive update sensitive.
    // A new critical row replacing a non-critical one is an upgrade, not a
    // loss of protected data.
    if old.critical {
        gates.push(ConfirmationGate::CriticalData);
    }
    let authoritative = |entry: &MemoryEntry| {
        entry
            .provenance
            .as_ref()
            .map(|provenance| {
                provenance.method.authority_rank()
                    >= ExtractionMethod::StructuredMapping.authority_rank()
            })
            .unwrap_or(false)
    };
    if authoritative(old) {
        gates.push(ConfirmationGate::AuthoritativeOverwrite);
    }
    if old.lifecycle.status != new.lifecycle.status || (old.active && !new.active) {
        gates.push(ConfirmationGate::StatusTransition);
    }
    if authoritative(old)
        && authoritative(new)
        && jcode_memory_types::reconcile::detect_conflict(old, new)
    {
        gates.push(ConfirmationGate::Contradiction);
    }
    gates.dedup();
    gates
}

impl std::fmt::Debug for MemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryManager")
            .field("project_dir", &self.project_dir)
            .field("test_mode", &self.test_mode)
            .field("include_skills", &self.include_skills)
            .finish()
    }
}

impl Clone for MemoryManager {
    fn clone(&self) -> Self {
        Self {
            project_dir: self.project_dir.clone(),
            test_mode: self.test_mode,
            include_skills: self.include_skills,
            ontology_registry: self.ontology_registry.clone(),
            suppress_next_legacy_save: std::sync::atomic::AtomicBool::new(
                self.suppress_next_legacy_save
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Recall output retains the exact entries supplied to the relevance judge so
/// publication can reject even non-rendered metadata changes during inference.
#[derive(Default)]
pub struct MemoryRelevanceResult {
    pub prompt: Option<String>,
    pub display_prompt: Option<String>,
    pub selected_entries: Vec<MemoryEntry>,
}

impl MemoryManager {
    pub fn new() -> Self {
        ensure_artifact_kind_config();
        Self {
            project_dir: None,
            test_mode: false,
            include_skills: true,
            ontology_registry: OntologyRegistry::with_default(),
            suppress_next_legacy_save: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Override the ontology registry.  Mainly useful for tests; the default
    /// registry loads the bundled `jcode/default/v1` ontology which mirrors
    /// the historical hardcoded behavior.
    pub fn with_ontology_registry(mut self, registry: OntologyRegistry) -> Self {
        self.ontology_registry = registry;
        self
    }

    /// Borrow the ontology registry.  Call sites use this to dispatch rule
    /// plans and schedule activities through the data-driven engine.
    pub fn ontology_registry(&self) -> &OntologyRegistry {
        &self.ontology_registry
    }

    /// Borrow the active ontology id stored on `graph` (or the default if the
    /// graph was authored before ontology metadata existed).
    // Dormant since the upstream Jev merge; kept for ontology-drift checks
    // when the ontology-driven write path is re-enabled.
    #[allow(dead_code)]
    fn active_ontology_id<'a>(&self, graph: &'a MemoryGraph) -> &'a str {
        if !graph.metadata.ontology_id.is_empty() {
            graph.metadata.ontology_id.as_str()
        } else {
            jcode_memory_types::ontology::DEFAULT_ONTOLOGY_ID
        }
    }

    pub fn with_project_dir(mut self, project_dir: impl Into<PathBuf>) -> Self {
        self.project_dir = Some(project_dir.into());
        self
    }

    pub fn with_skills(mut self, include_skills: bool) -> Self {
        self.include_skills = include_skills;
        self
    }

    /// Create a memory manager in test mode (isolated storage)
    pub fn new_test() -> Self {
        Self {
            project_dir: None,
            test_mode: true,
            include_skills: true,
            ontology_registry: OntologyRegistry::with_default(),
            suppress_next_legacy_save: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Check if running in test mode
    pub fn is_test_mode(&self) -> bool {
        self.test_mode
    }

    /// Set test mode (for debug sessions)
    pub fn set_test_mode(&mut self, test_mode: bool) {
        self.test_mode = test_mode;
    }

    /// Clear all test memories (only works in test mode)
    pub fn clear_test_storage(&self) -> Result<()> {
        if !self.test_mode {
            anyhow::bail!("clear_test_storage only allowed in test mode");
        }

        let test_dir = storage::jcode_dir()?.join("memory").join("test");
        if test_dir.exists() {
            std::fs::remove_dir_all(&test_dir)?;
            crate::logging::info("Cleared test memory storage");
        }
        Ok(())
    }

    fn get_project_dir(&self) -> Option<PathBuf> {
        self.project_dir.clone()
    }

    fn project_memory_path(&self) -> Result<Option<PathBuf>> {
        // In test mode, use test directory
        if self.test_mode {
            let test_dir = storage::jcode_dir()?.join("memory").join("test");
            std::fs::create_dir_all(&test_dir)?;
            return Ok(Some(test_dir.join("test_project.json")));
        }

        let project_dir = match self.get_project_dir() {
            Some(d) => d,
            None => return Ok(None),
        };

        project_memory_file(&project_dir).map(Some)
    }

    fn legacy_notes_path(&self) -> Result<Option<PathBuf>> {
        if self.test_mode {
            let test_dir = storage::jcode_dir()?.join("notes").join("test");
            std::fs::create_dir_all(&test_dir)?;
            return Ok(Some(test_dir.join("test_notes.json")));
        }

        let project_dir = match self.get_project_dir() {
            Some(d) => d,
            None => return Ok(None),
        };

        let project_hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            project_dir.hash(&mut hasher);
            format!("{:016x}", hasher.finish())
        };

        Ok(Some(
            storage::jcode_dir()?
                .join("notes")
                .join(format!("{}.json", project_hash)),
        ))
    }

    fn normalize_graph_search_text(graph: &mut MemoryGraph) -> bool {
        let mut changed = false;
        for memory in graph.memories.values_mut() {
            // Link nodes keep the match text their owning subsystem built
            // (a skill's full body, a goal's bounded summary). Recomputing it
            // from the pointer would silently drop that signal.
            if jcode_memory_types::is_link_node(memory) {
                continue;
            }
            let expected = normalize_memory_search_text(&memory.content, &memory.tags);
            if memory.search_text != expected {
                memory.search_text = expected;
                changed = true;
            }
        }
        changed
    }

    fn import_legacy_notes_into_graph(&self, graph: &mut MemoryGraph) -> Result<bool> {
        let Some(path) = self.legacy_notes_path()? else {
            return Ok(false);
        };
        if !path.exists() {
            return Ok(false);
        }

        let legacy: LegacyNotesFile = storage::read_json(&path)?;
        if legacy.entries.is_empty() {
            return Ok(false);
        }

        let mut changed = false;
        for note in legacy.entries {
            if graph.memories.contains_key(&note.id) {
                continue;
            }

            let mut entry = MemoryEntry::new(
                MemoryCategory::Custom(LEGACY_NOTE_CATEGORY.to_string()),
                note.content,
            );
            entry.id = note.id;
            entry.created_at = note.created_at;
            entry.updated_at = note.created_at;
            entry.source = Some("legacy_remember_migration".to_string());
            if let Some(tag) = note.tag {
                entry.tags.push(tag);
            }
            graph.add_memory(entry);
            changed = true;
        }

        Ok(changed)
    }

    fn global_memory_path(&self) -> Result<PathBuf> {
        if self.test_mode {
            let test_dir = storage::jcode_dir()?.join("memory").join("test");
            std::fs::create_dir_all(&test_dir)?;
            Ok(test_dir.join("test_global.json"))
        } else {
            Ok(storage::jcode_dir()?.join("memory").join("global.json"))
        }
    }

    pub fn load_project(&self) -> Result<MemoryStore> {
        match self.project_memory_path()? {
            Some(path) if path.exists() => storage::read_json(&path),
            _ => Ok(MemoryStore::new()),
        }
    }

    pub fn load_global(&self) -> Result<MemoryStore> {
        let path = self.global_memory_path()?;
        if path.exists() {
            storage::read_json(&path)
        } else {
            Ok(MemoryStore::new())
        }
    }

    pub fn save_project(&self, store: &MemoryStore) -> Result<()> {
        if let Some(path) = self.project_memory_path()? {
            storage::write_json(&path, store)?;
        }
        Ok(())
    }

    pub fn save_global(&self, store: &MemoryStore) -> Result<()> {
        let path = self.global_memory_path()?;
        storage::write_json(&path, store)
    }

    /// Resolve the ontology a specific store is authored against.
    ///
    /// A store that has an installed ontology uses it. Older stores fall back
    /// to the process registry default, which keeps pre-v2-install data
    /// working unchanged.
    pub fn ontology_for_graph(
        &self,
        graph: &MemoryGraph,
    ) -> std::sync::Arc<jcode_memory_types::Ontology> {
        match &graph.metadata.ontology {
            Some(stored) => std::sync::Arc::new(stored.clone()),
            None => self.ontology_registry().default_ontology(),
        }
    }

    fn project_ontology_overlay_path(&self) -> Option<PathBuf> {
        let dir = self.get_project_dir()?;
        if dir.is_dir() {
            return Some(dir.join(PROJECT_ONTOLOGY_OVERLAY));
        }
        // Synthetic/test project paths and deleted checkouts have no writable
        // repository root. Keep the overlay beside the project's memory store.
        let key = project_store_key(&dir).replace(':', "-");
        storage::jcode_dir().ok().map(|base| {
            base.join("memory")
                .join("projects")
                .join(format!("{key}.ontology.yaml"))
        })
    }

    /// Read the global ontology without counting a retrieval or advancing
    /// global graph metrics. Composition runs on every project load, so it must
    /// not look like a user-facing global recall.
    fn global_ontology_for_composition(
        &self,
    ) -> Result<std::sync::Arc<jcode_memory_types::Ontology>> {
        if crate::memory::active_backend_name() == "sqlite-gvec" {
            let backend = if self.test_mode {
                crate::memory::test_backend()
            } else {
                crate::memory::graph_backend()
            };
            let key = StoreKey::new(crate::memory::global_store_key());
            if let Ok(graph) = backend.load(&key) {
                return Ok(self.ontology_for_graph(&graph));
            }
            return Ok(self.ontology_registry().default_ontology());
        }
        let path = self.global_memory_path()?;
        if let Some(graph) = cached_graph(&path) {
            return Ok(self.ontology_for_graph(&graph));
        }
        if path.exists()
            && let Ok(graph) = storage::read_json::<MemoryGraph>(&path)
            && graph.graph_version == GRAPH_VERSION
        {
            cache_graph(path, &graph);
            return Ok(self.ontology_for_graph(&graph));
        }
        Ok(self.ontology_registry().default_ontology())
    }

    /// Load the repository-owned project overlay.
    ///
    /// A cloned repository is untrusted input, so rules and activities are
    /// deliberately stripped. Only types, aliases and relationship definitions
    /// can affect the effective ontology.
    fn load_project_ontology_layer(&self) -> Result<Option<OntologyLayer>> {
        let Some(path) = self.project_ontology_overlay_path() else {
            return Ok(None);
        };
        if !path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(&path)?;
        let mut layer: OntologyLayer = serde_yaml::from_str(&raw)
            .map_err(|err| anyhow::anyhow!("invalid {}: {err}", path.display()))?;
        layer.rules.clear();
        layer.activities.clear();
        layer.disabled_rules.clear();
        layer.disabled_activities.clear();
        layer.priority = layer.priority.max(100);
        Ok(Some(layer))
    }

    /// Compose the project overlay onto the effective global ontology.
    ///
    /// The graph keeps a materialized copy for runtime use. The parent and layer
    /// hashes make a changed global ontology or overlay invalidate that copy on
    /// the next load, without treating recomposition as a data migration.
    fn compose_project_ontology(&self, graph: &mut MemoryGraph) -> Result<()> {
        let parent = self.global_ontology_for_composition()?;
        let mut layer = self.load_project_ontology_layer()?;
        let had_layer = !graph.metadata.ontology_layer_hash.is_empty()
            || !graph.metadata.ontology_parent_hash.is_empty();
        if layer.is_none() && !had_layer {
            if self.adopt_legacy_project_overlay(&parent, graph)? {
                layer = self.load_project_ontology_layer()?;
            } else if !self.project_snapshot_is_disposable(&parent, graph) {
                // A store that predates overlays keeps its own snapshot unless
                // it is safely expressible as additive project concepts or is
                // only a copy of the compiled default.
                return Ok(());
            }
        }
        let layers: Vec<OntologyLayer> = layer.into_iter().collect();
        let layer_hash = layer_stack_hash(&layers);
        let parent_hash = parent.content_hash();
        if layers.is_empty() {
            // No overlay: the effective ontology is exactly the global parent.
            // Default parents are stored as a reference, not a snapshot.
            let effective_is_default = *parent == *self.ontology_registry().default_ontology();
            let metadata_matches = graph.metadata.ontology_parent_hash == parent_hash
                && graph.metadata.ontology_layer_hash == layer_hash
                && graph.metadata.ontology_id == parent.id
                && graph.metadata.ontology_version == parent.version
                && graph.metadata.ontology_hash == parent_hash
                && graph.metadata.ontology.is_some() != effective_is_default;
            if metadata_matches {
                return Ok(());
            }
            self.set_effective_ontology_metadata(
                graph,
                &parent,
                parent_hash,
                layer_hash,
                Vec::new(),
            );
            return Ok(());
        }

        let effective = compose_ontology_layers(&parent, &layers).map_err(|errors| {
            anyhow::anyhow!(
                "project ontology layer rejected:\n- {}",
                errors.join("\n- ")
            )
        })?;
        let layer_ids = layers.iter().map(|layer| layer.id.clone()).collect();
        self.set_effective_ontology_metadata(graph, &effective, parent_hash, layer_hash, layer_ids);
        Ok(())
    }

    /// Whether a project's stored ontology carries no project-specific content.
    ///
    /// This is true when it is exactly the compiled default, or when every type,
    /// relationship, rule, activity and alias it contains is also present with
    /// the same definition in the current global ontology. Such a snapshot is a
    /// stale copy, not an override, so the project can inherit global changes.
    fn project_snapshot_is_disposable(
        &self,
        parent: &jcode_memory_types::Ontology,
        graph: &MemoryGraph,
    ) -> bool {
        let Some(stored) = graph.metadata.ontology.as_ref() else {
            return true;
        };
        let default = self.ontology_registry().default_ontology();
        if stored == &*default {
            return true;
        }
        if stored.id != parent.id {
            return false;
        }
        for (id, type_def) in &stored.types {
            if parent.types.get(id) != Some(type_def) {
                return false;
            }
        }
        for (id, relationship) in &stored.relationships {
            if parent.relationships.get(id) != Some(relationship) {
                return false;
            }
        }
        if !is_subset_by_id(&stored.rules, &parent.rules, |rule| rule.id.as_str()) {
            return false;
        }
        if !is_subset_by_id(&stored.activities, &parent.activities, |activity| {
            activity.id.as_str()
        }) {
            return false;
        }
        stored
            .category_aliases
            .iter()
            .all(|(alias, target)| parent.category_aliases.get(alias) == Some(target))
    }

    /// Adopt a legacy project snapshot as an overlay only when it is provably
    /// additive over the current global ontology.
    ///
    /// Older stores persisted a full ontology copy. Turning that copy into an
    /// overlay is safe when every global type, relationship, rule and activity
    /// is unchanged and the store only adds extra project types. If any core
    /// definition differs, the snapshot may include stale compiled defaults or
    /// an offline core update, so it is left alone instead of pinning old
    /// behavior in the repository.
    fn adopt_legacy_project_overlay(
        &self,
        parent: &jcode_memory_types::Ontology,
        graph: &MemoryGraph,
    ) -> Result<bool> {
        let Some(stored) = graph.metadata.ontology.as_ref() else {
            return Ok(false);
        };
        if stored.id != parent.id {
            return Ok(false);
        }
        for (id, type_def) in &parent.types {
            if stored.types.get(id) != Some(type_def) {
                return Ok(false);
            }
        }
        if stored.relationships != parent.relationships
            || !same_rules(&stored.rules, &parent.rules)
            || !same_activities(&stored.activities, &parent.activities)
        {
            return Ok(false);
        }
        for (alias, target) in &parent.category_aliases {
            if stored.category_aliases.get(alias) != Some(target) {
                return Ok(false);
            }
        }

        let extra: std::collections::BTreeMap<_, _> = stored
            .types
            .iter()
            .filter(|(id, _)| !parent.types.contains_key(*id))
            .map(|(id, type_def)| (id.clone(), type_def.clone()))
            .collect();
        if extra.is_empty() {
            return Ok(false);
        }

        let mut layer =
            OntologyLayer::new("project/legacy-adopted", stored.version.max(parent.version))
                .with_priority(100);
        layer.types = extra;
        for (alias, target) in &stored.category_aliases {
            if !parent.category_aliases.contains_key(alias) && layer.types.contains_key(target) {
                layer.category_aliases.insert(alias.clone(), target.clone());
            }
        }
        // Validate before writing so a broken legacy parent chain or alias can
        // never turn into an overlay that fails on every later load.
        compose_ontology_layers(parent, std::slice::from_ref(&layer)).map_err(|errors| {
            anyhow::anyhow!(
                "legacy project ontology cannot be adopted:\n- {}",
                errors.join("\n- ")
            )
        })?;
        self.write_project_ontology_layer(&layer)?;
        crate::logging::info(&format!(
            "Adopted legacy project ontology as overlay: {} type(s)",
            layer.types.len()
        ));
        Ok(true)
    }

    fn write_project_ontology_layer(&self, layer: &OntologyLayer) -> Result<()> {
        let Some(path) = self.project_ontology_overlay_path() else {
            anyhow::bail!("project ontology overlay requires a project directory");
        };
        if layer_is_empty(layer) {
            if path.exists() {
                std::fs::remove_file(&path)?;
            }
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let yaml = serde_yaml::to_string(layer)?;
        let temp = path.with_extension("yaml.tmp");
        std::fs::write(&temp, yaml)?;
        if let Err(first) = std::fs::rename(&temp, &path) {
            // Windows cannot rename over an existing file.
            let _ = std::fs::remove_file(&path);
            if let Err(second) = std::fs::rename(&temp, &path) {
                let _ = std::fs::remove_file(&temp);
                anyhow::bail!(
                    "write {} failed: {first}; retry failed: {second}",
                    path.display()
                );
            }
        }
        Ok(())
    }

    /// Store without embedding inference. Exact duplicates reinforce an
    /// existing entry only within the requested scope, never mutate a
    /// different project. Inadmissible provenance (below the extraction
    /// method's admission threshold) is rejected instead of committed.
    pub fn remember_project(&self, entry: MemoryEntry) -> Result<String> {
        Ok(self.remember_project_governed(entry)?.id().to_string())
    }

    /// Governed project write: confidence admission, confirmation gates and
    /// review-queue parking. Returns `PendingReview` without mutating the graph
    /// when a human decision is required.
    pub fn remember_project_governed(&self, entry: MemoryEntry) -> Result<MemoryWriteOutcome> {
        self.remember_project_governed_with_policy(entry, &[], None)
    }

    fn remember_project_governed_with_policy(
        &self,
        entry: MemoryEntry,
        forced_gates: &[ConfirmationGate],
        forced_reason: Option<&str>,
    ) -> Result<MemoryWriteOutcome> {
        anyhow::ensure!(
            self.project_memory_path()?.is_some(),
            "Project memory requires a working directory; use global scope explicitly"
        );
        let project_dir = self.get_project_dir();
        let mut graph = self.load_project_graph()?;
        let ontology = self.ontology_for_graph(&graph);
        let outcome = self.governed_write(
            &mut graph,
            entry,
            &ontology,
            "project",
            project_dir.as_deref(),
            forced_gates,
            forced_reason,
        )?;
        if let MemoryWriteOutcome::Stored { .. } = outcome {
            self.save_project_graph(&graph)?;
        }
        Ok(outcome)
    }

    pub fn remember_global(&self, entry: MemoryEntry) -> Result<String> {
        Ok(self.remember_global_governed(entry)?.id().to_string())
    }

    /// Governed global write, including the ontology global-scope gate.
    pub fn remember_global_governed(&self, entry: MemoryEntry) -> Result<MemoryWriteOutcome> {
        let mut graph = self.load_global_graph()?;
        let ontology = self.ontology_for_graph(&graph);
        // Ontology gate: global memory must stay conceptual / generalized /
        // cross-project. Dispatch the scope-gate rule and abort environment-
        // specific content before it ever reaches the graph. The tool layer
        // additionally steers in-conversation writes toward project scope;
        // generalized entries from background extraction pass the gate via
        // their extraction provenance and content shape.
        if let Some(verdict) = self.evaluate_global_gate(&entry, &ontology) {
            return Err(anyhow::anyhow!(
                "global memory rejected by scope gate: {verdict}; \
                 write environment-specific facts to project scope instead"
            ));
        }
        let outcome =
            self.governed_write(&mut graph, entry, &ontology, "global", None, &[], None)?;
        if let MemoryWriteOutcome::Stored { .. } = outcome {
            self.save_global_graph(&graph)?;
        }
        Ok(outcome)
    }

    /// Run the ontology global-scope gate for `entry`. `Some(reason)` means
    /// the write must be aborted; `None` admits it. Uses the same rule
    /// engine as the rest of the ontology so the policy stays declarative.
    fn evaluate_global_gate(
        &self,
        entry: &MemoryEntry,
        ontology: &jcode_memory_types::Ontology,
    ) -> Option<String> {
        let registry = self.ontology_registry();
        let type_id = entry.category.to_string();
        let mut ctx = registry.make_context(
            &ontology.id,
            entry.clone(),
            jcode_memory_types::ontology::EVENT_REMEMBER_GLOBAL,
        );
        ctx.ontology = std::sync::Arc::new(ontology.clone());
        ctx.type_id = type_id.clone();
        ctx.scope = "global".to_string();
        let plan = registry.dispatch(&ctx);
        let generalized = jcode_memory_types::rule_engine::is_generalized_content(&entry.content);
        if !generalized {
            return Some(format!(
                "content is environment-specific (type_id={type_id})"
            ));
        }
        if plan
            .effects
            .iter()
            .any(|effect| matches!(effect, jcode_memory_types::ontology::Effect::Abort { .. }))
        {
            return Some("aborted by ontology rule".to_string());
        }
        None
    }

    /// Apply a precomputed reconcile outcome to a graph.
    fn apply_reconcile(
        graph: &mut MemoryGraph,
        entry: MemoryEntry,
        outcome: jcode_memory_types::reconcile::ReconcileOutcome,
    ) -> String {
        let source_label = entry
            .source
            .clone()
            .unwrap_or_else(|| "reconcile".to_string());
        match outcome.action {
            jcode_memory_types::reconcile::ReconcileAction::Insert => graph.add_memory(entry),
            jcode_memory_types::reconcile::ReconcileAction::Reinforce { existing_id } => {
                if let Some(existing) = graph.get_memory_mut(&existing_id) {
                    existing.reinforce(&source_label, 0);
                }
                // Fold in tags the incoming instance carried but the stored one
                // lacks, so a re-statement can still enrich the existing row.
                for tag in entry.tags.clone() {
                    graph.tag_memory(&existing_id, &tag);
                }
                existing_id
            }
            jcode_memory_types::reconcile::ReconcileAction::Supersede { existing_id } => {
                let new_id = graph.add_memory(entry);
                graph.supersede(&new_id, &existing_id);
                new_id
            }
            jcode_memory_types::reconcile::ReconcileAction::Contradict { existing_id } => {
                let new_id = graph.add_memory(entry);
                graph.mark_contradiction(&new_id, &existing_id);
                new_id
            }
            jcode_memory_types::reconcile::ReconcileAction::Relate { .. } => {
                let new_id = graph.add_memory(entry);
                for (related_id, similarity) in &outcome.related {
                    graph.link_memories(&new_id, related_id, *similarity);
                }
                new_id
            }
        }
    }

    /// Shared governed-write implementation for project and global scopes.
    fn governed_write(
        &self,
        graph: &mut MemoryGraph,
        entry: MemoryEntry,
        ontology: &jcode_memory_types::Ontology,
        scope: &str,
        project_dir: Option<&std::path::Path>,
        forced_gates: &[ConfirmationGate],
        forced_reason: Option<&str>,
    ) -> Result<MemoryWriteOutcome> {
        let mut entry = jcode_memory_types::reconcile::prepare_for_write(entry);
        apply_critical_policy(&mut entry);

        let (reinforce_at, supersede_at, relate_at) =
            jcode_memory_types::migration::reconcile_thresholds(ontology);
        let outcome = jcode_memory_types::reconcile::reconcile_with_thresholds(
            &entry,
            graph,
            reinforce_at,
            supersede_at,
            relate_at,
        );
        let old = outcome
            .action
            .existing_id()
            .and_then(|id| graph.get_memory(id))
            .cloned();
        let provenance = entry.effective_provenance().into_owned();
        let tier = admission_tier(provenance.method, provenance.confidence);

        let validation = crate::memory_types::validate_new_entry(&entry);
        if let Err(issues) = validation {
            let only_admission = issues.iter().all(|issue| issue.code == "E22");
            if !(tier == AdmissionTier::PendingOnly && only_admission) {
                return Err(anyhow::anyhow!("memory validation failed: {:?}", issues));
            }
        }

        let mut gates = write_confirmation_gates(old.as_ref(), &entry);
        gates.extend_from_slice(forced_gates);
        gates.dedup();
        if tier == AdmissionTier::PendingOnly || !gates.is_empty() {
            let reason = if tier == AdmissionTier::PendingOnly {
                format!(
                    "confidence {:.2} is below the {:.2} admission floor for {:?}",
                    provenance.confidence,
                    jcode_memory_types::REVIEW_BADGE_CONFIDENCE,
                    provenance.method
                )
            } else if let Some(reason) = forced_reason {
                reason.to_string()
            } else {
                gates
                    .iter()
                    .map(|gate| gate.description())
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            let mut item = MemoryReviewItem::new(scope, old.clone(), entry.clone(), gates, reason)
                .with_action(outcome.action.label());
            item.similarity = outcome.similarity;
            item.related = outcome.related.clone();
            if let Some(dir) = project_dir {
                item = item.with_project_dir(dir.to_string_lossy().to_string());
            }
            self.enqueue_review_item(&item)?;
            return Ok(MemoryWriteOutcome::PendingReview {
                review_id: item.id,
                target_id: old.map(|entry| entry.id),
            });
        }

        let needs_review = tier == AdmissionTier::NeedsReview;
        if needs_review {
            entry.review.needs_review = true;
        }
        let id = Self::apply_reconcile(graph, entry, outcome);
        Ok(MemoryWriteOutcome::Stored { id, needs_review })
    }

    fn enqueue_review_item(&self, item: &MemoryReviewItem) -> Result<()> {
        let path = review::queue_path(self.test_mode)?;
        review::append_item(&path, item)
    }

    /// List review items, newest first. `status = None` lists everything.
    pub fn list_reviews(&self, status: Option<ReviewStatus>) -> Result<Vec<MemoryReviewItem>> {
        let path = review::queue_path(self.test_mode)?;
        let mut items = review::load_items(&path)?;
        if let Some(status) = status {
            items.retain(|item| item.status == status);
        }
        items.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(items)
    }

    pub fn get_review(&self, id: &str) -> Result<Option<MemoryReviewItem>> {
        let path = review::queue_path(self.test_mode)?;
        let items = review::load_items(&path)?;
        Ok(review::find_item(&items, id).cloned())
    }

    /// Summarize the pending queue by action, gate and project so a large
    /// backlog can be triaged without loading every item into the terminal.
    pub fn review_triage(&self) -> Result<ReviewTriageReport> {
        let items = self.list_reviews(Some(ReviewStatus::Pending))?;
        let now = Utc::now();
        let mut report = ReviewTriageReport {
            pending: items.len(),
            ..ReviewTriageReport::default()
        };
        let mut projects: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for item in &items {
            *report.by_action.entry(item.action.clone()).or_default() += 1;
            if item.gates.is_empty() {
                *report.by_gate.entry("<none>".to_string()).or_default() += 1;
            } else {
                for gate in &item.gates {
                    *report
                        .by_gate
                        .entry(format!("{gate:?}").to_lowercase())
                        .or_default() += 1;
                }
            }
            let project = item
                .project_dir
                .clone()
                .unwrap_or_else(|| format!("<{}>", item.scope));
            *projects.entry(project).or_default() += 1;
            let days = now.signed_duration_since(item.created_at).num_days().max(0);
            report.oldest_pending_days = report.oldest_pending_days.max(days);
        }
        report.by_project = projects.into_iter().collect();
        report
            .by_project
            .sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
        report.by_project.truncate(10);
        Ok(report)
    }

    /// Accept a pending review and apply its proposed change.
    pub fn accept_review(
        &self,
        id: &str,
        reviewer: &str,
        note: Option<&str>,
    ) -> Result<MemoryReviewItem> {
        let path = review::queue_path(self.test_mode)?;
        let mut items = review::load_items(&path)?;
        let item = review::find_item(&items, id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Review item not found: {id}"))?;
        anyhow::ensure!(item.is_pending(), "Review item {id} is not pending");

        let now = Utc::now();
        let provenance = item
            .new
            .effective_provenance()
            .into_owned()
            .with_confirmation(reviewer, now)
            .with_confidence(item.new.effective_provenance().confidence.max(0.95));
        let mut entry = item.new.clone();
        entry.provenance = Some(provenance);
        entry.review = MemoryReviewState::confirmed(now);
        entry.updated_at = now;

        let manager = self.manager_for_review(&item)?;
        manager.apply_reviewed_change(&item, entry)?;

        let updated =
            review::update_status(&mut items, id, ReviewStatus::Accepted, Some(reviewer), note)?;
        review::save_items(&path, &items)?;
        Ok(updated)
    }

    /// Accept pending items matching `actions`, in queue order, up to `max`.
    /// Dry run unless `apply` is true. Items whose target disappeared or whose
    /// new id is already present are skipped rather than overwriting anything.
    pub fn accept_reviews_batch(
        &self,
        actions: &[String],
        project: Option<&str>,
        max: usize,
        apply: bool,
    ) -> Result<ReviewBatchReport> {
        let path = review::queue_path(self.test_mode)?;
        let mut all_items = review::load_items(&path)?;
        let pending = all_items
            .iter()
            .filter(|item| item.status == ReviewStatus::Pending)
            .count();
        let wanted: std::collections::HashSet<&str> =
            actions.iter().map(String::as_str).collect();
        let selected: Vec<MemoryReviewItem> = all_items
            .iter()
            .filter(|item| {
                item.status == ReviewStatus::Pending
                    && (wanted.is_empty() || wanted.contains(item.action.as_str()))
                    && project.is_none_or(|project| {
                        item.project_dir.as_deref() == Some(project)
                    })
            })
            .take(max)
            .cloned()
            .collect();
        let mut report = ReviewBatchReport {
            pending,
            selected: selected.len(),
            applied: apply,
            ..ReviewBatchReport::default()
        };
        let mut groups: std::collections::BTreeMap<
            (String, Option<String>),
            Vec<MemoryReviewItem>,
        > = std::collections::BTreeMap::new();
        for item in selected {
            groups
                .entry((item.scope.clone(), item.project_dir.clone()))
                .or_default()
                .push(item);
        }
        let now = Utc::now();
        for items in groups.values() {
            let Some(first) = items.first() else {
                continue;
            };
            let manager = match self.manager_for_review(first) {
                Ok(manager) => manager,
                Err(_) => {
                    report.failed += items.len();
                    continue;
                }
            };
            let mut graph = match manager.load_review_graph(first) {
                Ok(graph) => graph,
                Err(_) => {
                    report.failed += items.len();
                    continue;
                }
            };
            let mut accepted_ids = Vec::new();
            for item in items {
                if graph.memories.contains_key(&item.new.id) {
                    report.skipped_existing += 1;
                    continue;
                }
                if matches!(
                    item.action.as_str(),
                    "relate" | "supersede" | "contradict" | "reinforce"
                ) && item
                    .old
                    .as_ref()
                    .is_none_or(|old| !graph.memories.contains_key(&old.id))
                {
                    report.skipped_missing_target += 1;
                    continue;
                }
                let provenance = item
                    .new
                    .effective_provenance()
                    .into_owned()
                    .with_confirmation("batch-review", now)
                    .with_confidence(item.new.effective_provenance().confidence.max(0.95));
                let mut entry = item.new.clone();
                entry.provenance = Some(provenance);
                entry.review = MemoryReviewState::confirmed(now);
                entry.updated_at = now;
                if Self::apply_reviewed_change_to_graph(&mut graph, item, entry).is_err() {
                    report.failed += 1;
                    continue;
                }
                report.accepted += 1;
                *report.by_action.entry(item.action.clone()).or_default() += 1;
                accepted_ids.push(item.id.clone());
            }
            if apply && !accepted_ids.is_empty() {
                if manager.save_review_graph(first, &graph).is_err() {
                    report.failed += accepted_ids.len();
                    report.accepted = report.accepted.saturating_sub(accepted_ids.len());
                    continue;
                }
                for id in accepted_ids {
                    let _ = review::update_status(
                        &mut all_items,
                        &id,
                        ReviewStatus::Accepted,
                        Some("batch-review"),
                        Some("batch accepted"),
                    );
                }
            }
        }
        if apply {
            review::save_items(&path, &all_items)?;
        }
        Ok(report)
    }

    /// Reject pending items whose new id already exists or whose required old
    /// target disappeared. No memory graph is changed; only queue statuses.
    pub fn prune_stale_reviews(&self, apply: bool) -> Result<ReviewPruneReport> {
        let path = review::queue_path(self.test_mode)?;
        let mut all_items = review::load_items(&path)?;
        let pending: Vec<MemoryReviewItem> = all_items
            .iter()
            .filter(|item| item.status == ReviewStatus::Pending)
            .cloned()
            .collect();
        let mut report = ReviewPruneReport {
            pending: pending.len(),
            applied: apply,
            ..ReviewPruneReport::default()
        };
        let mut groups: std::collections::BTreeMap<
            (String, Option<String>),
            Vec<MemoryReviewItem>,
        > = std::collections::BTreeMap::new();
        for item in pending {
            groups
                .entry((item.scope.clone(), item.project_dir.clone()))
                .or_default()
                .push(item);
        }
        let mut stale_ids = Vec::new();
        for items in groups.values() {
            let Some(first) = items.first() else {
                continue;
            };
            let Ok(manager) = self.manager_for_review(first) else {
                continue;
            };
            let Ok(graph) = manager.load_review_graph(first) else {
                continue;
            };
            for item in items {
                if graph.memories.contains_key(&item.new.id) {
                    report.stale_existing += 1;
                    stale_ids.push(item.id.clone());
                    continue;
                }
                if matches!(
                    item.action.as_str(),
                    "relate" | "supersede" | "contradict" | "reinforce"
                ) && item
                    .old
                    .as_ref()
                    .is_none_or(|old| !graph.memories.contains_key(&old.id))
                {
                    report.stale_missing_target += 1;
                    stale_ids.push(item.id.clone());
                }
            }
        }
        report.rejected = stale_ids.len();
        if apply && !stale_ids.is_empty() {
            for id in stale_ids {
                let _ = review::update_status(
                    &mut all_items,
                    &id,
                    ReviewStatus::Rejected,
                    Some("stale-prune"),
                    Some("target missing or already applied"),
                );
            }
            review::save_items(&path, &all_items)?;
        }
        Ok(report)
    }

    pub fn reject_review(
        &self,
        id: &str,
        reviewer: &str,
        note: Option<&str>,
    ) -> Result<MemoryReviewItem> {
        let path = review::queue_path(self.test_mode)?;
        let mut items = review::load_items(&path)?;
        anyhow::ensure!(
            review::find_item(&items, id).is_some_and(MemoryReviewItem::is_pending),
            "Review item {id} is not pending"
        );
        let updated =
            review::update_status(&mut items, id, ReviewStatus::Rejected, Some(reviewer), note)?;
        review::save_items(&path, &items)?;
        Ok(updated)
    }

    pub fn defer_review(
        &self,
        id: &str,
        reviewer: &str,
        note: Option<&str>,
    ) -> Result<MemoryReviewItem> {
        let path = review::queue_path(self.test_mode)?;
        let mut items = review::load_items(&path)?;
        anyhow::ensure!(
            review::find_item(&items, id).is_some_and(MemoryReviewItem::is_pending),
            "Review item {id} is not pending"
        );
        let updated =
            review::update_status(&mut items, id, ReviewStatus::Deferred, Some(reviewer), note)?;
        review::save_items(&path, &items)?;
        Ok(updated)
    }

    /// Re-open a deferred review so it can be accepted, rejected, or deferred again.
    pub fn reopen_review(
        &self,
        id: &str,
        reviewer: &str,
        note: Option<&str>,
    ) -> Result<MemoryReviewItem> {
        let path = review::queue_path(self.test_mode)?;
        let mut items = review::load_items(&path)?;
        anyhow::ensure!(
            review::find_item(&items, id)
                .is_some_and(|item| item.status == ReviewStatus::Deferred),
            "Review item {id} is not deferred"
        );
        let updated =
            review::update_status(&mut items, id, ReviewStatus::Pending, Some(reviewer), note)?;
        review::save_items(&path, &items)?;
        Ok(updated)
    }

    /// Re-open a deferred review with a corrected admission action.
    pub fn retarget_review(
        &self,
        id: &str,
        new_action: &str,
        reviewer: &str,
        note: Option<&str>,
    ) -> Result<MemoryReviewItem> {
        let new_action = new_action.trim().to_ascii_lowercase();
        anyhow::ensure!(
            matches!(new_action.as_str(), "insert" | "relate"),
            "unsupported review retarget action '{new_action}'; use insert or relate"
        );

        let path = review::queue_path(self.test_mode)?;
        let mut items = review::load_items(&path)?;
        let item = items
            .iter_mut()
            .find(|item| item.id == id)
            .ok_or_else(|| anyhow::anyhow!("Review item not found: {id}"))?;
        anyhow::ensure!(
            item.status == ReviewStatus::Deferred,
            "Review item {id} is not deferred"
        );
        if new_action == "relate" {
            anyhow::ensure!(
                item.old.is_some(),
                "cannot retarget review {id} to relate without an old target"
            );
        }

        item.action = new_action.clone();
        item.status = ReviewStatus::Pending;
        item.confirmed_by = Some(reviewer.to_string());
        item.confirmed_at = Some(Utc::now());
        if let Some(note) = note {
            item.note = Some(note.to_string());
        }
        if new_action == "insert" {
            item.old = None;
            item.related.clear();
            item.diff = vec!["new".to_string()];
        }
        let updated = item.clone();
        review::save_items(&path, &items)?;
        Ok(updated)
    }

    fn manager_for_review(&self, item: &MemoryReviewItem) -> Result<MemoryManager> {
        match item.project_dir.as_deref() {
            Some(dir) if !dir.trim().is_empty() => Ok(self.clone().with_project_dir(dir)),
            _ => {
                anyhow::ensure!(
                    item.scope != "project" || self.get_project_dir().is_some(),
                    "review item {} targets project scope but has no project directory; run it from that project",
                    item.id
                );
                Ok(self.clone())
            }
        }
    }

    fn load_review_graph(&self, item: &MemoryReviewItem) -> Result<MemoryGraph> {
        if item.scope == "global" {
            self.load_global_graph()
        } else {
            anyhow::ensure!(
                self.project_memory_path()?.is_some(),
                "project memory requires a working directory"
            );
            self.load_project_graph()
        }
    }

    fn save_review_graph(&self, item: &MemoryReviewItem, graph: &MemoryGraph) -> Result<()> {
        if item.scope == "global" {
            self.save_global_graph(graph)
        } else {
            self.save_project_graph(graph)
        }
    }

    fn apply_reviewed_change(&self, item: &MemoryReviewItem, entry: MemoryEntry) -> Result<()> {
        let mut graph = self.load_review_graph(item)?;
        Self::apply_reviewed_change_to_graph(&mut graph, item, entry)?;
        self.save_review_graph(item, &graph)
    }

    fn apply_reviewed_change_to_graph(
        graph: &mut MemoryGraph,
        item: &MemoryReviewItem,
        entry: MemoryEntry,
    ) -> Result<()> {
        match item.action.as_str() {
            "reinforce" => {
                let target_id = item
                    .old
                    .as_ref()
                    .map(|old| old.id.clone())
                    .ok_or_else(|| anyhow::anyhow!("reinforce review has no target"))?;
                let existing = graph
                    .get_memory_mut(&target_id)
                    .ok_or_else(|| anyhow::anyhow!("target memory {target_id} no longer exists"))?;
                existing.reinforce("review", 0);
                for tag in &entry.tags {
                    existing.tags.push(tag.clone());
                }
                existing.tags.sort();
                existing.tags.dedup();
                existing.refresh_search_text();
            }
            "supersede" => {
                let old_id = item
                    .old
                    .as_ref()
                    .map(|old| old.id.clone())
                    .ok_or_else(|| anyhow::anyhow!("supersede review has no target"))?;
                let new_id = graph.add_memory(entry);
                graph.supersede(&new_id, &old_id);
            }
            "contradict" => {
                let old_id = item
                    .old
                    .as_ref()
                    .map(|old| old.id.clone())
                    .ok_or_else(|| anyhow::anyhow!("contradict review has no target"))?;
                let new_id = graph.add_memory(entry);
                graph.mark_contradiction(&new_id, &old_id);
            }
            "relate" => {
                let new_id = graph.add_memory(entry);
                let mut linked = std::collections::HashSet::new();
                if let Some(old_id) = item.old.as_ref().map(|old| old.id.clone()) {
                    let weight = if item.similarity > 0.0 {
                        item.similarity.clamp(0.5, 0.95)
                    } else {
                        0.7
                    };
                    if graph.memories.contains_key(&old_id) && linked.insert(old_id.clone()) {
                        graph.link_memories(&new_id, &old_id, weight);
                    }
                }
                for (target, weight) in &item.related {
                    if graph.memories.contains_key(target) && linked.insert(target.clone()) {
                        graph.link_memories(&new_id, target, *weight);
                    }
                }
            }
            "insert" => {
                graph.add_memory(entry);
            }
            _ => {
                graph.add_memory(entry);
            }
        }
        Ok(())
    }

    /// Aggregate memory and review-queue health across every persisted store.
    pub fn health_report(&self) -> Result<MemoryHealthReport> {
        use std::collections::HashMap;

        let mut report = MemoryHealthReport::default();
        let mut subjects: HashMap<String, usize> = HashMap::new();
        let visited = self.for_each_store(|key, graph| {
            for entry in graph.all_memories() {
                // Skill/goal mirrors are link nodes, not user memories. They
                // have no provenance or locator by design, so counting them
                // inflated every provenance gap and type distribution.
                if jcode_memory_types::is_link_id(&entry.id) {
                    continue;
                }
                report.memories += 1;
                match entry.lifecycle.status {
                    jcode_memory_types::MemoryStatus::Active if entry.active => report.active += 1,
                    jcode_memory_types::MemoryStatus::Active => report.superseded += 1,
                    jcode_memory_types::MemoryStatus::Archived => report.archived += 1,
                    jcode_memory_types::MemoryStatus::Expired => report.expired += 1,
                    jcode_memory_types::MemoryStatus::Disputed => report.disputed += 1,
                }
                if entry.critical {
                    report.critical += 1;
                }
                if entry.review.needs_review {
                    report.needs_review += 1;
                }
                match entry.provenance.as_ref() {
                    None => report.missing_provenance += 1,
                    Some(provenance) if provenance.locator.is_none() => {
                        report.missing_locator += 1;
                    }
                    Some(_) => {}
                }
                let provenance = entry.effective_provenance();
                if !matches!(
                    provenance.method,
                    ExtractionMethod::UserStated | ExtractionMethod::StructuredMapping
                ) && provenance.confidence < jcode_memory_types::REVIEW_BADGE_CONFIDENCE
                {
                    report.low_confidence += 1;
                }
                *report
                    .categories
                    .entry(entry.category.to_string())
                    .or_default() += 1;
                if let Some(subject) = entry.subject_key.as_deref() {
                    *subjects
                        .entry(format!(
                            "{key}:{}",
                            jcode_memory_types::reconcile::normalize_subject_key(subject)
                        ))
                        .or_default() += 1;
                }
            }
            for edge in graph.edges.values().flatten() {
                if matches!(edge.kind, jcode_memory_types::EdgeKind::Contradicts) {
                    report.contradictions += 1;
                }
            }
            false
        })?;
        report.stores = visited;
        report.duplicate_subjects = subjects.values().filter(|count| **count > 1).count();

        let path = review::queue_path(self.test_mode)?;
        let now = Utc::now();
        for item in review::load_items(&path)? {
            report.reviews.observe(&item, now);
        }
        Ok(report)
    }

    /// Current extraction source health.
    pub fn extraction_health(&self) -> Result<ExtractionHealth> {
        let path = source_health::health_path(self.test_mode)?;
        source_health::load(&path)
    }

    pub fn extraction_is_paused(&self) -> bool {
        self.extraction_health()
            .map(|health| health.paused)
            .unwrap_or(false)
    }

    /// Resume automatic extraction after a health pause.
    pub fn resume_extraction(&self) -> Result<ExtractionHealth> {
        let path = source_health::health_path(self.test_mode)?;
        let mut health = source_health::load(&path)?;
        health.paused = false;
        health.pause_reason = None;
        source_health::save(&path, &health)?;
        Ok(health)
    }

    /// Record one extraction batch and pause the source when it degrades.
    #[allow(clippy::too_many_arguments)]
    pub fn record_extraction_batch(
        &self,
        session_id: &str,
        candidates: usize,
        admitted: usize,
        queued: usize,
        failed: usize,
        avg_confidence: f32,
        duplicate_ratio: f32,
    ) -> Result<ExtractionHealth> {
        let path = source_health::health_path(self.test_mode)?;
        let mut health = source_health::load(&path)?;
        health.batches.push(ExtractionBatch {
            at: Utc::now(),
            session_id: session_id.to_string(),
            candidates,
            admitted,
            queued,
            failed,
            avg_confidence,
            duplicate_ratio,
        });
        if health.batches.len() > 100 {
            health.batches.drain(0..health.batches.len() - 100);
        }

        let recent = health.recent_batches().to_vec();
        let total_candidates = recent.iter().map(|batch| batch.candidates).sum::<usize>();
        let total_failed = recent.iter().map(|batch| batch.failed).sum::<usize>();
        let total = total_candidates.max(total_failed);
        if recent.len() >= 3 && total >= source_health::MIN_CANDIDATES_FOR_PAUSE {
            let failure_rate =
                recent.iter().map(|batch| batch.failed as f32).sum::<f32>() / total as f32;
            if failure_rate > source_health::MAX_FAILURE_RATE {
                health.paused = true;
                health.pause_reason = Some(format!(
                    "extraction failure rate {:.1}% exceeds {:.0}%",
                    failure_rate * 100.0,
                    source_health::MAX_FAILURE_RATE * 100.0
                ));
            }
            // A single low-confidence batch is often an extraction outlier, not
            // a degraded source. Require two consecutive degraded batches
            // before pausing so one bad transcript cannot stop the pipeline.
            if recent.len() >= 3 {
                let baseline = recent[..recent.len() - 2]
                    .iter()
                    .map(|batch| batch.avg_confidence)
                    .sum::<f32>()
                    / (recent.len() - 2) as f32;
                let recent_two = &recent[recent.len() - 2..];
                let current = recent_two
                    .iter()
                    .map(|batch| batch.avg_confidence)
                    .sum::<f32>()
                    / recent_two.len() as f32;
                if baseline > 0.0
                    && recent_two.iter().all(|batch| {
                        baseline - batch.avg_confidence > source_health::MAX_CONFIDENCE_DROP
                    })
                    && !health.paused
                {
                    health.paused = true;
                    health.pause_reason = Some(format!(
                        "extraction confidence dropped from {:.2} to {:.2}",
                        baseline, current
                    ));
                }
            }
        }
        source_health::save(&path, &health)?;
        Ok(health)
    }

    /// Insert or update a memory with a stable ID in the project graph.
    /// Preserves existing inbound/outbound graph relationships while refreshing
    /// content and tags.
    pub fn upsert_project_memory(&self, entry: MemoryEntry) -> Result<String> {
        crate::memory_types::validate_new_entry(&entry)
            .map_err(|issues| anyhow::anyhow!("memory validation failed: {:?}", issues))?;
        let mut graph = self.load_project_graph()?;
        let id = self.upsert_memory_in_graph(&mut graph, entry);
        self.save_project_graph(&graph)?;
        Ok(id)
    }

    /// Insert or update a memory with a stable ID in the global graph.
    /// Preserves existing inbound/outbound graph relationships while refreshing
    /// content and tags.
    pub fn upsert_global_memory(&self, entry: MemoryEntry) -> Result<String> {
        crate::memory_types::validate_new_entry(&entry)
            .map_err(|issues| anyhow::anyhow!("memory validation failed: {:?}", issues))?;
        let mut graph = self.load_global_graph()?;
        let id = self.upsert_memory_in_graph(&mut graph, entry);
        self.save_global_graph(&graph)?;
        Ok(id)
    }

    fn upsert_memory_in_graph(
        &self,
        graph: &mut crate::memory_graph::MemoryGraph,
        entry: MemoryEntry,
    ) -> String {
        // Same sanitization as the insert path: link nodes (goals, skills) and
        // explicit upserts still get a derived `subject_key` and a clean tag
        // list, they just skip reconciliation because their id is authoritative.
        let entry = jcode_memory_types::reconcile::prepare_for_write(entry);
        let id = entry.id.clone();

        let Some(existing_snapshot) = graph.get_memory(&id).cloned() else {
            return graph.add_memory(entry);
        };

        let old_tags: std::collections::HashSet<String> =
            existing_snapshot.tags.iter().cloned().collect();
        let new_tags: std::collections::HashSet<String> = entry.tags.iter().cloned().collect();
        let is_link_node = jcode_memory_types::is_link_node(&entry);

        for tag in old_tags.difference(&new_tags) {
            graph.untag_memory(&id, tag);
        }
        for tag in new_tags.difference(&old_tags) {
            graph.tag_memory(&id, tag);
        }

        if let Some(existing) = graph.get_memory_mut(&id) {
            let content_changed = existing.content != entry.content;
            existing.category = entry.category;
            existing.content = entry.content;
            existing.tags = entry.tags;
            existing.properties = entry.properties;
            existing.updated_at = entry.updated_at;
            existing.source = entry.source;
            existing.trust = entry.trust;
            existing.active = entry.active;
            existing.superseded_by = entry.superseded_by;
            existing.confidence = entry.confidence;
            existing.provenance = entry.provenance;
            existing.lifecycle = entry.lifecycle;
            existing.review = entry.review;
            existing.critical = entry.critical;
            existing.identity = entry.identity;
            existing.subject_key = entry.subject_key;
            if is_link_node {
                // The subsystem that owns the artifact is authoritative for
                // its match text and versioned metadata.
                existing.search_text = entry.search_text;
                existing.artifact = entry.artifact;
            }
            if content_changed {
                existing.set_embedding(None, None);
            }
            if !is_link_node && (content_changed || old_tags != new_tags) {
                existing.refresh_search_text();
            }
        }

        id
    }

    pub fn find_similar(
        &self,
        text: &str,
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<(MemoryEntry, f32)>> {
        // Generate embedding for query text
        let query_embedding = match crate::embedding_backend::embed_query_active(text) {
            Ok((emb, _model)) => emb,
            Err(e) => {
                crate::logging::info(&format!(
                    "Embedding failed, falling back to keyword search: {}",
                    e
                ));
                return Ok(Vec::new());
            }
        };

        self.find_similar_with_embedding(&query_embedding, threshold, limit)
    }

    pub fn find_similar_scoped(
        &self,
        text: &str,
        threshold: f32,
        limit: usize,
        scope: MemoryScope,
    ) -> Result<Vec<(MemoryEntry, f32)>> {
        let query_embedding = match crate::embedding_backend::embed_query_active(text) {
            Ok((emb, _model)) => emb,
            Err(e) => {
                crate::logging::info(&format!(
                    "Embedding failed, falling back to keyword search: {}",
                    e
                ));
                return Ok(Vec::new());
            }
        };

        self.find_similar_with_embedding_scoped(&query_embedding, threshold, limit, scope)
    }

    /// Find memories similar to the given embedding
    pub fn find_similar_with_embedding(
        &self,
        query_embedding: &[f32],
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<(MemoryEntry, f32)>> {
        let entries_with_emb = self.collect_all_memories_with_embeddings()?;
        Self::score_and_filter(entries_with_emb, query_embedding, "", threshold, limit)
    }

    pub fn find_similar_with_embedding_scoped(
        &self,
        query_embedding: &[f32],
        threshold: f32,
        limit: usize,
        scope: MemoryScope,
    ) -> Result<Vec<(MemoryEntry, f32)>> {
        let entries_with_emb = self.collect_memories_with_embeddings_scoped(scope)?;
        Self::score_and_filter(entries_with_emb, query_embedding, "", threshold, limit)
    }

    /// Hybrid retrieval: fuse dense (embedding cosine) and sparse (BM25 over
    /// memory search text) rankings with Reciprocal Rank Fusion.
    ///
    /// This is the recall-oriented live retrieval path. Unlike
    /// `find_similar_with_embedding`, it does NOT apply a hard cosine floor
    /// (which benchmarking showed zeroes out recall): instead it pulls a
    /// generous candidate pool from each retriever and lets RRF + the
    /// downstream sidecar/rerank decide. Lexical signal is essential for the
    /// identifier/path/term-heavy memories agents store.
    pub fn find_similar_hybrid(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<(MemoryEntry, f32)>> {
        self.find_similar_hybrid_scoped(query_text, query_embedding, limit, MemoryScope::All)
    }

    pub fn find_similar_hybrid_scoped(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        limit: usize,
        scope: MemoryScope,
    ) -> Result<Vec<(MemoryEntry, f32)>> {
        let entries = self.collect_memories_with_embeddings_scoped(scope)?;
        Ok(Self::hybrid_fuse(
            entries,
            query_text,
            query_embedding,
            limit,
        ))
    }

    /// Pull pool, rank by dense and BM25 separately, fuse with RRF.
    fn hybrid_fuse(
        entries: Vec<MemoryEntry>,
        query_text: &str,
        query_embedding: &[f32],
        limit: usize,
    ) -> Vec<(MemoryEntry, f32)> {
        let entries: Vec<MemoryEntry> = entries
            .into_iter()
            .filter(|e| e.embedding.is_some())
            .collect();
        if entries.is_empty() {
            return Vec::new();
        }

        // Generous per-retriever pool so fusion has signal to work with.
        let pool = (limit * 5).max(HYBRID_POOL_MIN);

        // Dense ranking (no hard threshold; just take the top by cosine).
        // Vector-space gate: only entries embedded by the ACTIVE backend share a
        // comparable space, so dense scores are computed over those only. Other
        // entries (different model, e.g. not-yet-re-embedded local memories when
        // OpenAI is active) still participate via the BM25 lexical half below, so
        // they remain reachable rather than disappearing on a backend switch.
        let active_model = crate::embedding_backend::active_model_id();
        let dense_eligible: Vec<usize> = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.effective_embedding_model() == active_model)
            .map(|(i, _)| i)
            .collect();
        let emb_refs: Vec<&[f32]> = dense_eligible
            .iter()
            .filter_map(|&i| entries[i].embedding.as_deref())
            .collect();
        let dense_scores = crate::embedding::batch_cosine_similarity(query_embedding, &emb_refs);
        let mut dense: Vec<(usize, f32)> =
            dense_eligible.iter().copied().zip(dense_scores).collect();
        dense.sort_by(|a, b| b.1.total_cmp(&a.1));
        dense.truncate(pool);

        // Sparse (BM25) ranking over memory search text.
        let sparse = bm25_rank(&entries, query_text, pool);

        // RRF fusion.
        const RRF_K: f32 = 60.0;
        let mut fused: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        for (rank, (idx, _)) in dense.iter().enumerate() {
            *fused.entry(*idx).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0);
        }
        for (rank, (idx, _)) in sparse.iter().enumerate() {
            *fused.entry(*idx).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0);
        }

        // Strength / recency prior. RRF scores live in ~[0, 0.03]; a logarithmic
        // strength term plus a bounded recency bonus gives frequently-reinforced
        // and recently-used memories a gentle lift without letting a strength-30
        // veteran drown every strength-1 newcomer (the 87% strength=1 long tail
        // needs the log to keep the prior sub-dominant to relevance).
        let now = chrono::Utc::now();
        let prior = |e: &MemoryEntry| -> f32 {
            let strength_term = (e.strength as f32).ln() * 0.002;
            let age_days = (now - e.updated_at).num_seconds() as f32 / 86_400.0;
            let recency_term = (-age_days / 14.0).exp() * 0.004;
            strength_term + recency_term
        };

        let mut entries: Vec<Option<MemoryEntry>> = entries.into_iter().map(Some).collect();
        top_k_by_score(
            fused.into_iter().filter_map(|(idx, score)| {
                entries[idx].take().map(|e| {
                    let s = prior(&e);
                    (e, score + s)
                })
            }),
            limit,
        )
    }

    fn collect_all_memories_with_embeddings(&self) -> Result<Vec<MemoryEntry>> {
        self.collect_memories_with_embeddings_scoped(MemoryScope::All)
    }

    fn collect_memories_with_embeddings_scoped(
        &self,
        scope: MemoryScope,
    ) -> Result<Vec<MemoryEntry>> {
        let mut entries: Vec<MemoryEntry> = Vec::new();
        if scope.includes_project()
            && let Ok(project) = self.load_project_graph()
        {
            entries.extend(
                project
                    .active_memories()
                    .filter(|m| m.embedding.is_some())
                    .cloned(),
            );
        }
        if scope.includes_global()
            && let Ok(global) = self.load_global_graph()
        {
            entries.extend(
                global
                    .active_memories()
                    .filter(|m| m.embedding.is_some())
                    .cloned(),
            );
        }
        Ok(entries)
    }

    fn collect_memories_scoped(&self, scope: MemoryScope) -> Result<Vec<MemoryEntry>> {
        let mut entries = Vec::new();
        if scope.includes_project()
            && let Ok(project) = self.load_project_graph()
        {
            entries.extend(project.all_memories().cloned());
        }
        if scope.includes_global()
            && let Ok(global) = self.load_global_graph()
        {
            entries.extend(global.all_memories().cloned());
        }
        Ok(entries)
    }

    #[cfg(test)]
    fn synthetic_skill_entries(&self) -> Vec<MemoryEntry> {
        if !self.include_skills {
            return Vec::new();
        }

        collect_synthetic_entries()
    }

    #[cfg(test)]
    fn collect_retrieval_candidates_scoped(&self, scope: MemoryScope) -> Result<Vec<MemoryEntry>> {
        let mut entries = self.collect_memories_scoped(scope)?;
        if scope.includes_global() {
            entries.extend(self.synthetic_skill_entries());
        }
        Ok(entries)
    }

    fn score_and_filter(
        entries: Vec<MemoryEntry>,
        query_embedding: &[f32],
        query_text: &str,
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<(MemoryEntry, f32)>> {
        if entries.is_empty() {
            return Ok(Vec::new());
        }

        let mut filtered_entries = Vec::with_capacity(entries.len());
        let mut skipped_missing_embeddings = 0usize;
        // Vector-space gate: only compare embeddings produced by the ACTIVE
        // backend (same model id). When the active backend differs from an
        // entry's stored model (e.g. user switched to OpenAI but this memory was
        // embedded with local MiniLM, not yet re-embedded), the cosine would be
        // meaningless, so we exclude it from dense scoring. Such memories remain
        // reachable via the lexical/BM25 path in hybrid retrieval.
        let active_model = crate::embedding_backend::active_model_id();
        let mut skipped_model_mismatch = 0usize;
        for entry in entries {
            if entry.embedding.is_none() {
                skipped_missing_embeddings += 1;
            } else if entry.effective_embedding_model() != active_model {
                skipped_model_mismatch += 1;
            } else {
                filtered_entries.push(entry);
            }
        }
        if skipped_missing_embeddings > 0 {
            crate::logging::warn(&format!(
                "Skipped {} retrieval candidate(s) without embeddings during similarity scoring",
                skipped_missing_embeddings
            ));
        }
        if skipped_model_mismatch > 0 {
            crate::logging::info(&format!(
                "Skipped {} retrieval candidate(s) embedded with a different model than the active backend ({})",
                skipped_model_mismatch, active_model
            ));
        }
        if filtered_entries.is_empty() {
            return Ok(Vec::new());
        }
        let emb_refs: Vec<&[f32]> = filtered_entries
            .iter()
            .filter_map(|entry| entry.embedding.as_deref())
            .collect();
        let scores = crate::embedding::batch_cosine_similarity(query_embedding, &emb_refs);
        let skill_query_terms = collect_skill_query_terms(query_text);

        let scored = top_k_by_score(
            filtered_entries
                .into_iter()
                .zip(scores)
                .map(|(entry, sim)| {
                    let adjusted = sim + skill_retrieval_bonus(&entry, &skill_query_terms);
                    (entry, adjusted)
                })
                .filter(|(_, sim)| *sim >= threshold),
            limit,
        );

        let scored = Self::apply_gap_filter(scored);

        Ok(scored)
    }

    /// Drop trailing low-relevance results by detecting natural gaps in the
    /// score distribution. If the top hit is 0.85 and the next cluster is
    /// 0.40-0.42, the 0.15+ gap tells us those lower results are noise.
    ///
    /// Algorithm: walk the sorted scores and cut when the drop from one score
    /// to the next exceeds `GAP_FACTOR` of the range (top - floor_threshold).
    fn apply_gap_filter(scored: Vec<(MemoryEntry, f32)>) -> Vec<(MemoryEntry, f32)> {
        if scored.len() <= 1 {
            return scored;
        }

        const GAP_FACTOR: f32 = 0.25;
        const MIN_KEEP: usize = 1;

        let top_score = scored[0].1;
        let range = (top_score - EMBEDDING_SIMILARITY_THRESHOLD).max(0.01);
        let max_gap = range * GAP_FACTOR;

        let mut keep = scored.len();
        for i in 1..scored.len() {
            let drop = scored[i - 1].1 - scored[i].1;
            if drop > max_gap && i >= MIN_KEEP {
                keep = i;
                break;
            }
        }

        scored.into_iter().take(keep).collect()
    }

    /// Ensure all memories have embeddings (backfill for existing memories)
    pub fn backfill_embeddings(&self) -> Result<(usize, usize)> {
        let mut generated = 0;
        let mut failed = 0;

        // Process project memories
        if let Ok(mut graph) = self.load_project_graph() {
            let mut changed = false;
            for entry in graph.memories.values_mut() {
                if entry.embedding.is_none() {
                    if entry.ensure_embedding() {
                        generated += 1;
                        changed = true;
                    } else {
                        failed += 1;
                    }
                }
            }
            if changed {
                self.save_project_graph(&graph)?;
            }
        }

        // Process global memories
        if let Ok(mut graph) = self.load_global_graph() {
            let mut changed = false;
            for entry in graph.memories.values_mut() {
                if entry.embedding.is_none() {
                    if entry.ensure_embedding() {
                        generated += 1;
                        changed = true;
                    } else {
                        failed += 1;
                    }
                }
            }
            if changed {
                self.save_global_graph(&graph)?;
            }
        }

        Ok((generated, failed))
    }

    /// Backfill at most `limit` missing embeddings.
    ///
    /// This is the incremental variant used at session finalize and by the
    /// `memory embed` tool action. It avoids turning a large historical store
    /// into a synchronous full re-embed. A store that already contains vectors
    /// from a different model is skipped rather than mixing vector spaces.
    pub fn backfill_embeddings_bounded(&self, limit: usize) -> Result<(usize, usize, usize)> {
        if limit == 0 {
            return Ok((0, 0, self.count_missing_embeddings()?));
        }
        let active_model = crate::embedding_backend::active_model_id();
        let remaining_before = self.count_missing_embeddings()?;
        let mut generated = 0;
        let mut failed = 0;
        let mut attempted = 0usize;

        let project = self.load_project_graph()?;
        let global = self.load_global_graph()?;
        for (mut graph, is_project) in [(project, true), (global, false)] {
            let mixed_models = graph
                .memories
                .values()
                .any(|entry| entry.embedding.is_some() && !entry.embedding_matches_model(&active_model));
            if mixed_models {
                crate::logging::warn(
                    "embedding backfill skipped a store with mixed embedding models",
                );
                continue;
            }
            let mut ids: Vec<String> = graph
                .memories
                .values()
                .filter(|entry| entry.embedding.is_none() && entry.is_retrievable_now())
                .map(|entry| entry.id.clone())
                .collect();
            ids.sort();
            let available = limit.saturating_sub(attempted);
            ids.truncate(available);

            let mut changed = false;
            // Remote embedders are charged per HTTP request, so backfill in
            // small batches. If a batch fails, retry its rows individually so
            // one oversized or malformed memory cannot strand its neighbours.
            const BATCH: usize = 32;
            for chunk in ids.chunks(BATCH) {
                let chunk_ids: Vec<String> = chunk
                    .iter()
                    .filter(|id| graph.memories.contains_key(*id))
                    .cloned()
                    .collect();
                if chunk_ids.is_empty() {
                    continue;
                }
                attempted += chunk_ids.len();
                let texts: Vec<String> = chunk_ids
                    .iter()
                    .filter_map(|id| graph.memories.get(id).map(|entry| entry.content.clone()))
                    .collect();
                let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
                let batch = crate::embedding_backend::embed_passages_active(&text_refs);
                let mut stored = false;
                match batch {
                    Ok((vectors, model)) if vectors.len() == chunk_ids.len() => {
                        for (id, embedding) in chunk_ids.iter().zip(vectors) {
                            if let Some(entry) = graph.memories.get_mut(id) {
                                entry.set_embedding(Some(embedding), Some(model.clone()));
                                generated += 1;
                                changed = true;
                            }
                        }
                        stored = true;
                    }
                    Ok((vectors, _)) => {
                        crate::logging::info(&format!(
                            "bounded embedding backfill batch returned {} vectors for {} passages",
                            vectors.len(),
                            chunk_ids.len()
                        ));
                    }
                    Err(error) => {
                        crate::logging::info(&format!(
                            "bounded embedding backfill batch failed: {error}"
                        ));
                    }
                }
                if stored {
                    continue;
                }
                for id in chunk_ids {
                    let Some(entry) = graph.memories.get_mut(&id) else {
                        continue;
                    };
                    match crate::embedding_backend::embed_passage_active(&entry.content) {
                        Ok((embedding, model)) => {
                            entry.set_embedding(Some(embedding), Some(model));
                            generated += 1;
                            changed = true;
                        }
                        Err(error) => {
                            failed += 1;
                            crate::logging::info(&format!(
                                "bounded embedding backfill failed for {id}: {error}"
                            ));
                        }
                    }
                }
            }
            if changed {
                if is_project {
                    self.save_project_graph(&graph)?;
                } else {
                    self.save_global_graph(&graph)?;
                }
            }
            if attempted >= limit {
                break;
            }
        }
        Ok((
            generated,
            failed,
            remaining_before.saturating_sub(generated),
        ))
    }

    fn count_missing_embeddings(&self) -> Result<usize> {
        let project = self.load_project_graph()?;
        let global = self.load_global_graph()?;
        Ok(project
            .memories
            .values()
            .chain(global.memories.values())
            .filter(|entry| entry.embedding.is_none() && entry.is_retrievable_now())
            .count())
    }

    /// Mirror-side half of accepting a skill proposal: write the mirror link
    /// node and one `skill:<name> -derived_from-> mem_*` edge per evidence id
    /// that exists, then record a first invocation so the loop is observable.
    /// Returns the number of evidence ids actually linked.
    ///
    /// Skill mirrors always live in the **global** store even when the draft
    /// was accepted from a project's `.proposals/`: the file is installed into
    /// the global skills dir (or a workspace's project-local dir, see
    /// `accept_skill_proposal`), and the mirror's scope must match where the
    /// skill file lives, not where the evidence came from.
    pub fn record_accepted_skill(&self, skill_name: &str, evidence_ids: &[String]) -> usize {
        use jcode_memory_types::ontology::TYPE_OBSERVATION;
        let name = skill_name.trim();
        if name.is_empty() {
            return 0;
        }
        let mut graph = match self.load_global_graph() {
            Ok(g) => g,
            Err(e) => {
                crate::logging::warn(&format!("skill acceptance mirror failed: {e}"));
                return 0;
            }
        };
        // Idempotent mirror (same id refreshes, reconciler skips link nodes).
        let mut mirror = crate::memory::MemoryEntry::new(
            MemoryCategory::Custom(TYPE_OBSERVATION.to_string()),
            format!("Skill `/{name}` distilled from memory evidence; invoke with `/{name}`."),
        )
        .with_id(format!("skill:{name}"))
        .with_tags(vec![format!("skill:{name}"), "skill-accepted".to_string()])
        .with_source("skill_registry")
        .with_trust(crate::memory::TrustLevel::High);
        mirror.subject_key = Some(format!("skill:{name}"));
        graph.add_memory(mirror);

        let mut linked = 0usize;
        for id in evidence_ids {
            if !graph.memories.contains_key(id) {
                continue;
            }
            graph.add_edge(
                &format!("skill:{name}"),
                id,
                jcode_memory_types::EdgeKind::DerivedFrom,
            );
            linked += 1;
        }
        if let Err(e) = self.save_global_graph(&graph) {
            crate::logging::warn(&format!("skill acceptance mirror save failed: {e}"));
            return linked;
        }
        // Evidence typically lives in project stores while the mirror lives in
        // the global store. Same-store edges cover the global-share; the rest
        // are recorded on the project mirrors by scanning every store once.
        let _ = self.link_evidence_in_project_stores(name, evidence_ids);
        // One invocation row: the acceptance itself is evidence the skill is in
        // use (the reviewer just acted on this workspace's own evidence). It
        // goes into the same store as the mirror: invocation without a working
        // dir cannot reach the project store (`remember_skill_invocation`
        // routes there), so write the row directly alongside the mirror.
        if graph
            .get_memory(&format!("skill-invocation:{name}"))
            .is_none()
        {
            let mut invocation = crate::memory::MemoryEntry::new(
                MemoryCategory::Fact,
                format!("Skill `/{name}` was accepted from distilled memory evidence."),
            )
            .with_id(format!("skill-invocation:{name}"))
            .with_tags(vec![format!("skill:{name}"), "skill-invoked".to_string()])
            .with_source(format!("skill:{name}"))
            .with_provenance(
                crate::memory::ProvenanceRecord::new(
                    format!("skill:{name}"),
                    crate::memory::ExtractionMethod::SkillInvocation,
                )
                .with_confidence(1.0),
            );
            invocation.subject_key = Some(format!("skill:{name}"));
            let mut graph = self.load_global_graph().unwrap_or_default();
            graph.add_memory(invocation);
            let _ = self.save_global_graph(&graph);
        }
        linked
    }

    /// Record `skill:<name> -derived_from-> mem_*` edges on each project
    /// store's mirror node for evidence that lives in that store.
    ///
    /// The global share is handled by [`Self::record_accepted_skill`]; this
    /// covers the rest: evidence ids written by project sessions live in
    /// project stores, and the edge must connect nodes within one graph.
    fn link_evidence_in_project_stores(&self, skill_name: &str, evidence_ids: &[String]) {
        let name = skill_name.trim();
        if name.is_empty() || evidence_ids.is_empty() {
            return;
        }
        let wanted: std::collections::HashSet<&str> =
            evidence_ids.iter().map(|id| id.as_str()).collect();
        let _ = self.for_each_store(|_key, graph| {
            if !graph.memories.keys().any(|id| wanted.contains(id.as_str())) {
                return false;
            }
            // The mirror in a project store is created by finalize's
            // sync pass; if it is not there yet, create it so the
            // edges have a local endpoint.
            if graph.get_memory(&format!("skill:{name}")).is_none() {
                let mut mirror = MemoryEntry::new(
                    MemoryCategory::Custom("observation".to_string()),
                    format!(
                        "Skill `/{name}` distilled from memory evidence; invoke with `/{name}`."
                    ),
                )
                .with_id(format!("skill:{name}"))
                .with_tags(vec![format!("skill:{name}"), "skill-accepted".to_string()])
                .with_source("skill_registry")
                .with_trust(crate::memory::TrustLevel::High);
                mirror.subject_key = Some(format!("skill:{name}"));
                graph.add_memory(mirror);
            }
            let mut changed = false;
            for id in evidence_ids {
                if !graph.memories.contains_key(id) {
                    continue;
                }
                graph.add_edge(
                    &format!("skill:{name}"),
                    id,
                    jcode_memory_types::EdgeKind::DerivedFrom,
                );
                changed = true;
            }
            changed
        });
    }

    /// Count one retrieval for each ID (project and global stores).
    ///
    /// Unlike [`Self::touch_entries`] this is deliberately quiet: it fires on
    /// every injection, so it neither logs nor needs a return report. Entries
    /// that do not exist are silently skipped (the injection validator already
    /// guarantees they existed at selection time).
    pub fn record_retrievals(&self, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let id_set: std::collections::HashSet<&str> = ids.iter().map(|id| id.as_str()).collect();

        let mut project = self.load_project_graph_for_writeback()?;
        let mut project_changed = false;
        for entry in project.memories.values_mut() {
            if id_set.contains(entry.id.as_str()) {
                entry.record_injection();
                project_changed = true;
            }
        }
        if project_changed {
            self.save_project_graph(&project)?;
        }

        let mut global = self.load_global_graph_for_writeback()?;
        let mut global_changed = false;
        for entry in global.memories.values_mut() {
            if id_set.contains(entry.id.as_str()) {
                entry.record_injection();
                global_changed = true;
            }
        }
        if global_changed {
            self.save_global_graph(&global)?;
        }
        Ok(())
    }

    /// Load a JSON graph from disk instead of the process cache.
    ///
    /// Metric writeback is a read-modify-write of a whole graph. Using the
    /// cache here can persist content that was changed on disk after the cache
    /// was filled (the scoped-pending tests deliberately rewrite while
    /// preserving mtime). The sqlite backend is authoritative, so it needs no
    /// special path.
    fn load_project_graph_for_writeback(&self) -> Result<MemoryGraph> {
        if active_backend_name() == "sqlite-gvec" {
            return self.load_project_graph();
        }
        match self.project_memory_path()? {
            Some(path) if path.exists() => match storage::read_json::<MemoryGraph>(&path) {
                Ok(graph) if graph.graph_version == GRAPH_VERSION => Ok(graph),
                _ => self.load_project_graph(),
            },
            _ => self.load_project_graph(),
        }
    }

    fn load_global_graph_for_writeback(&self) -> Result<MemoryGraph> {
        if active_backend_name() == "sqlite-gvec" {
            return self.load_global_graph();
        }
        let path = self.global_memory_path()?;
        if path.exists() {
            if let Ok(graph) = storage::read_json::<MemoryGraph>(&path)
                && graph.graph_version == GRAPH_VERSION
            {
                return Ok(graph);
            }
        }
        self.load_global_graph()
    }

    /// Count one retrieval *selection* for each ID.
    ///
    /// This runs when a payload becomes pending, before the model actually
    /// receives it. It deliberately does not touch `updated_at` or
    /// `access_count`, so selection cannot refresh TTL or invalidate the
    /// publication snapshot. Comparing this with `record_retrievals` gives the
    /// real selection-to-injection conversion rate.
    pub fn record_retrieval_candidates(&self, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let id_set: std::collections::HashSet<&str> = ids.iter().map(|id| id.as_str()).collect();

        let mut project = self.load_project_graph_for_writeback()?;
        let mut project_changed = false;
        for entry in project.memories.values_mut() {
            if id_set.contains(entry.id.as_str()) {
                entry.record_retrieval_candidate();
                project_changed = true;
            }
        }
        if project_changed {
            self.save_project_graph(&project)?;
        }

        let mut global = self.load_global_graph_for_writeback()?;
        let mut global_changed = false;
        for entry in global.memories.values_mut() {
            if id_set.contains(entry.id.as_str()) {
                entry.record_retrieval_candidate();
                global_changed = true;
            }
        }
        if global_changed {
            self.save_global_graph(&global)?;
        }
        Ok(())
    }

    pub fn touch_entries(&self, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }

        let id_set: std::collections::HashSet<&str> = ids.iter().map(|id| id.as_str()).collect();

        let mut project = self.load_project_graph()?;
        let mut project_changed = false;
        for entry in project.memories.values_mut() {
            if id_set.contains(entry.id.as_str()) {
                entry.touch();
                project_changed = true;
            }
        }
        if project_changed {
            self.save_project_graph(&project)?;
        }

        let mut global = self.load_global_graph()?;
        let mut global_changed = false;
        for entry in global.memories.values_mut() {
            if id_set.contains(entry.id.as_str()) {
                entry.touch();
                global_changed = true;
            }
        }
        if global_changed {
            self.save_global_graph(&global)?;
        }

        Ok(())
    }

    pub fn get_prompt_memories(&self, limit: usize) -> Option<String> {
        self.get_prompt_memories_scoped(limit, MemoryScope::All)
    }

    pub fn get_prompt_memories_scoped(&self, limit: usize, scope: MemoryScope) -> Option<String> {
        let all_entries: Vec<_> = top_k_by_ord(
            self.collect_memories_scoped(scope)
                .ok()?
                .into_iter()
                .map(|entry| {
                    let updated_at = entry.updated_at.timestamp_millis();
                    (entry, updated_at)
                }),
            limit,
        )
        .into_iter()
        .map(|(entry, _)| entry)
        .collect();

        if all_entries.is_empty() {
            return None;
        }

        format_entries_for_prompt(&all_entries, limit)
    }

    pub async fn relevant_prompt_for_messages(
        &self,
        messages: &[crate::message::Message],
    ) -> Result<Option<String>> {
        let context = format_context_for_relevance(messages);
        if context.is_empty() {
            return Ok(None);
        }
        self.relevant_prompt_for_context(
            &context,
            MEMORY_RELEVANCE_MAX_CANDIDATES,
            MEMORY_RELEVANCE_MAX_RESULTS,
        )
        .await
    }

    pub async fn relevant_prompt_for_context(
        &self,
        context: &str,
        max_candidates: usize,
        limit: usize,
    ) -> Result<Option<String>> {
        let relevant = self
            .get_relevant_for_context(context, max_candidates)
            .await?;
        if relevant.is_empty() {
            return Ok(None);
        }
        Ok(format_relevant_prompt(&relevant, limit))
    }

    pub fn search(&self, query: &str) -> Result<Vec<MemoryEntry>> {
        self.search_scoped(query, MemoryScope::All)
    }

    pub fn search_scoped(&self, query: &str, scope: MemoryScope) -> Result<Vec<MemoryEntry>> {
        let query_lower = normalize_search_text(query);
        if query_lower.is_empty() {
            return Ok(Vec::new());
        }

        // Fast path: when the sqlite-gvec backend is active, run an
        // FTS5 BM25 search per StoreKey and merge the ranked hits.
        // JSON paths (and any backend that does not implement
        // text_search) keep the existing in-memory substring scan.
        if crate::memory::active_backend_name() == "sqlite-gvec" {
            if let Some(hits) = self.fts_search_scoped(&query_lower, scope) {
                return Ok(hits);
            }
        }

        let mut results = Vec::new();

        for memory in self.collect_memories_scoped(scope)? {
            if memory_matches_search(&memory, &query_lower) {
                results.push(memory);
            }
        }

        Ok(results)
    }

    /// Hybrid memory retrieval over FTS, gvec vector search, and exact tag
    /// lookup.
    ///
    /// The three channels are fused with reciprocal rank fusion (RRF), then a
    /// small strength/recency/confidence prior is added. This keeps lexical
    /// precision for paths and identifiers while recovering semantic neighbours
    /// that FTS misses, and lets shared tags pull related memories into the
    /// candidate pool without scanning the whole graph.
    pub fn hybrid_search_scoped(
        &self,
        query: &str,
        limit: usize,
        scope: MemoryScope,
    ) -> Result<Vec<(MemoryEntry, f32)>> {
        if limit == 0 || query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut entries: std::collections::HashMap<String, MemoryEntry> =
            std::collections::HashMap::new();
        if scope.includes_project() {
            for (id, entry) in &self.load_project_graph()?.memories {
                entries.insert(id.clone(), entry.clone());
            }
        }
        if scope.includes_global() {
            for (id, entry) in &self.load_global_graph()?.memories {
                entries.entry(id.clone()).or_insert_with(|| entry.clone());
            }
        }
        if entries.is_empty() {
            return Ok(Vec::new());
        }

        const RRF_K: f32 = 60.0;
        let mut fused: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();
        let mut add_rank = |id: &str, rank: usize| {
            if entries.contains_key(id) {
                *fused.entry(id.to_string()).or_insert(0.0) +=
                    1.0 / (RRF_K + rank as f32 + 1.0);
            }
        };

        // Lexical channel. `search_scoped` already uses the gvec FTS5 backend
        // when available and falls back to the in-memory scan otherwise.
        for (rank, entry) in self.search_scoped(query, scope)?.iter().enumerate() {
            add_rank(&entry.id, rank);
        }

        // Dense channel. Failures (missing model, dimension mismatch, no
        // embeddings) degrade to the lexical channel instead of failing recall.
        let mut store_keys = Vec::new();
        if scope.includes_project()
            && let Some(project) = self.get_project_dir()
        {
            store_keys.push(StoreKey::new(project_store_key(&project)));
        }
        if scope.includes_global() {
            store_keys.push(StoreKey::new(global_store_key().to_string()));
        }
        let backend = if self.test_mode {
            test_backend()
        } else {
            graph_backend()
        };
        if let Ok((query_embedding, _model)) =
            crate::embedding_backend::embed_query_active(query)
        {
            for key in &store_keys {
                match backend.vector_search(key, &query_embedding, (limit * 5).max(24)) {
                    Ok(hits) => {
                        for (rank, (id, _score)) in hits.iter().enumerate() {
                            add_rank(id, rank);
                        }
                    }
                    Err(error) => crate::logging::info(&format!(
                        "Hybrid vector search failed for {}: {error}",
                        key.as_str()
                    )),
                }
            }
        }

        // Exact tag channel. Query terms are used as tag anchors; this is
        // deliberately limited to a handful of terms so a long query does not
        // explode into hundreds of tag probes.
        let terms: Vec<String> = normalize_search_text(query)
            .split_whitespace()
            .filter(|term| term.len() >= 3)
            .take(8)
            .map(str::to_string)
            .collect();
        for key in &store_keys {
            for term in &terms {
                match backend.tag_search(key, term, (limit * 2).max(8)) {
                    Ok(hits) => {
                        for (rank, (id, _score)) in hits.iter().enumerate() {
                            add_rank(id, rank);
                        }
                    }
                    Err(error) => crate::logging::info(&format!(
                        "Hybrid tag search failed for {}: {error}",
                        key.as_str()
                    )),
                }
            }
        }

        let now = Utc::now();
        let mut scored: Vec<(MemoryEntry, f32)> = fused
            .into_iter()
            .filter_map(|(id, score)| {
                let entry = entries.remove(&id)?;
                if !entry.is_retrievable_now() {
                    return None;
                }
                let age_days = (now - entry.updated_at).num_seconds() as f32 / 86_400.0;
                let prior = (entry.strength as f32).ln() * 0.002
                    + (-age_days / 14.0).exp() * 0.004
                    + entry.confidence.clamp(0.0, 1.0) * 0.05;
                Some((entry, score + prior))
            })
            .collect();
        scored.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.id.cmp(&right.0.id))
        });
        scored.truncate(limit);
        Ok(scored)
    }

    pub fn list_all(&self) -> Result<Vec<MemoryEntry>> {
        self.list_all_scoped(MemoryScope::All)
    }

    /// Run an FTS5 search across the active stores selected by `scope`,
    /// returning the matching `MemoryEntry` rows ordered by their
    /// merged BM25 score (descending).
    ///
    /// Returns `None` when the backend is not sqlite-gvec or when the
    /// per-store search yields no hits — callers should fall back to
    /// the in-memory scan in that case so a partial backend failure
    /// is not visible to the user.
    fn fts_search_scoped(&self, query: &str, scope: MemoryScope) -> Option<Vec<MemoryEntry>> {
        use jcode_memory_types::StoreKey;

        let backend = if self.test_mode {
            crate::memory::test_backend()
        } else {
            crate::memory::graph_backend()
        };

        // FTS5 query syntax: we treat each whitespace-separated token
        // as a `term` and join them with AND so a substring like
        // "rust ownership" matches both words. We also quote each
        // term to neutralise punctuation that FTS5 would otherwise
        // interpret as operators.
        let mut fts_query = String::new();
        for (i, tok) in query.split_whitespace().enumerate() {
            if i > 0 {
                fts_query.push_str(" AND ");
            }
            // Strip the FTS5 quoting chars from the user's token.
            let cleaned: String = tok
                .chars()
                .filter(|c| !matches!(c, '"' | '*' | '(' | ')' | ':' | '-' | '+' | '^'))
                .collect();
            if cleaned.is_empty() {
                continue;
            }
            fts_query.push('"');
            fts_query.push_str(&cleaned);
            fts_query.push('"');
        }
        if fts_query.is_empty() {
            return None;
        }

        // Reasonable default cap; callers can re-rank above this if
        // they want.
        let k: usize = 32;

        // Load the project + global graphs once and use them as a
        // id→MemoryEntry index for the merge. We still call into the
        // trait only for the FTS5 search — the surrounding graph
        // objects come from the existing in-memory cache path so the
        // cost stays bounded.
        let mut by_id: std::collections::HashMap<String, MemoryEntry> =
            std::collections::HashMap::new();
        if scope.includes_project()
            && let Ok(graph) = self.load_project_graph()
        {
            for (id, entry) in &graph.memories {
                by_id.insert(id.clone(), entry.clone());
            }
        }
        if scope.includes_global()
            && let Ok(graph) = self.load_global_graph()
        {
            for (id, entry) in &graph.memories {
                by_id.entry(id.clone()).or_insert_with(|| entry.clone());
            }
        }
        if by_id.is_empty() {
            return Some(Vec::new());
        }

        // Collect BM25 hits across the relevant stores, weighting the
        // global hits slightly higher (matching the legacy behaviour
        // where global memory is more "trusted" than per-project).
        let mut scored: Vec<(f32, String)> = Vec::new();
        let mut saw_any = false;

        if scope.includes_project()
            && let Some(project) = self.get_project_dir()
        {
            let key = StoreKey::new(crate::memory::project_store_key(&project));
            match backend.text_search(&key, &fts_query, k) {
                Ok(hits) if !hits.is_empty() => {
                    saw_any = true;
                    for (id, score) in hits {
                        scored.push((score, id));
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    crate::logging::warn(&format!("fts_search_scoped project: {e}"));
                }
            }
        }

        if scope.includes_global() {
            let key = StoreKey::new(crate::memory::global_store_key());
            match backend.text_search(&key, &fts_query, k) {
                Ok(hits) if !hits.is_empty() => {
                    saw_any = true;
                    // Small global boost so a global memory outranks
                    // an equally-scored project memory.
                    const GLOBAL_BOOST: f32 = 1.1;
                    for (id, score) in hits {
                        scored.push((score * GLOBAL_BOOST, id));
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    crate::logging::warn(&format!("fts_search_scoped global: {e}"));
                }
            }
        }

        if !saw_any {
            // Either the FTS5 index is empty (database has no Memory
            // nodes) or both queries failed — fall back to the
            // in-memory scan so the caller still sees a result.
            return None;
        }

        // Sort by score desc and dedupe by id, keeping the best
        // score for any duplicate (shouldn't happen since the two
        // stores have disjoint ids, but be defensive).
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut out = Vec::with_capacity(scored.len());
        for (_score, id) in scored {
            if !seen.insert(id.clone()) {
                continue;
            }
            if let Some(entry) = by_id.remove(&id) {
                out.push(entry);
            }
        }
        Some(out)
    }

    pub fn list_all_scoped(&self, scope: MemoryScope) -> Result<Vec<MemoryEntry>> {
        let mut all = self.collect_memories_scoped(scope)?;
        all.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(all)
    }

    /// Local (L1) recall: rank stored instances against `query` with no
    /// network, no embedder and no credential.
    ///
    /// This is the recall path that remains available when the Jev judge route
    /// is missing, which used to mean "recall returns nothing at all".
    pub fn l1_relevant(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        let normalized_query = normalize_search_text(query);
        let terms: Vec<String> = collect_skill_query_terms(query).into_iter().collect();
        let mut scored: Vec<(f32, MemoryEntry)> = self
            .collect_memories_scoped(MemoryScope::All)?
            .into_iter()
            .filter_map(|entry| {
                let score = score_l1_relevance(&entry, &terms, &normalized_query);
                (score >= L1_MIN_SCORE).then_some((score, entry))
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(limit.min(L1_MAX_RESULTS));
        // Link-node ids are deliberately mirrored into every scope that uses
        // the artifact, so the same id (e.g. `skill:<name>`) legitimately
        // appears once per store in the scan. One row per id: publication
        // rejects duplicate ids outright ("duplicate selected memory ID"),
        // which previously dropped the whole selection.
        // A link node's id may come from several scopes with slightly different
        // tag sets. The publication snapshot resolves an id duplicate to the
        // GLOBAL copy (`snapshot_selected_memories`), so the dedupe must keep
        // that same copy: collect order is project-then-global, and a stable
        // sort on score keeps that relative order for equal scores, so keeping
        // the LAST occurrence per id selects the global copy. (Regular memories
        // have unique ids; this only rewrites duplicated link-node rows.)
        let mut by_id: Vec<(String, MemoryEntry)> = Vec::new();
        for (_, entry) in scored {
            match by_id.iter_mut().find(|(id, _)| *id == entry.id) {
                Some(slot) => slot.1 = entry,
                None => by_id.push((entry.id.clone(), entry)),
            }
        }
        let entries = by_id.into_iter().map(|(_, entry)| entry).collect();
        let entries = self.expand_l1_links(entries, limit);
        Ok(self.expand_skill_parts(entries, limit))
    }

    /// Fill unused L1 slots with high-weight semantic neighbours.
    ///
    /// Link discovery only creates edges between memories that already share a
    /// subject key or enough tags. Reusing those edges here lets a strongly
    /// matching memory pull in a directly related one without widening the
    /// injection budget or traversing tag edges.
    fn expand_l1_links(&self, mut entries: Vec<MemoryEntry>, limit: usize) -> Vec<MemoryEntry> {
        let max_total = limit.min(L1_MAX_RESULTS);
        if entries.is_empty() || entries.len() >= max_total {
            return entries;
        }
        let graphs: Vec<MemoryGraph> = [self.load_project_graph(), self.load_global_graph()]
            .into_iter()
            .filter_map(Result::ok)
            .collect();
        let mut seen: std::collections::HashSet<String> =
            entries.iter().map(|entry| entry.id.clone()).collect();
        let mut candidates: Vec<(f32, MemoryEntry)> = Vec::new();
        for seed in &entries {
            for graph in &graphs {
                for edge in graph.get_edges(&seed.id) {
                    let jcode_memory_types::EdgeKind::RelatesTo { weight } = &edge.kind else {
                        continue;
                    };
                    if *weight < L1_LINK_MIN_WEIGHT {
                        continue;
                    }
                    let Some(target) = graph.get_memory(&edge.target) else {
                        continue;
                    };
                    if !target.is_retrievable_now() || seen.contains(&target.id) {
                        continue;
                    }
                    seen.insert(target.id.clone());
                    candidates.push((*weight, target.clone()));
                }
            }
        }
        candidates.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.id.cmp(&right.1.id))
        });
        for (_, entry) in candidates {
            if entries.len() >= max_total {
                break;
            }
            entries.push(entry);
        }
        entries
    }

    /// Add the root pointer of any selected skill part before the part itself.
    ///
    /// Part nodes are independently retrievable, but a model that sees only a
    /// step or code block does not know which skill owns it. The root pointer
    /// is bounded and cheap, so attach it once per skill.
    fn expand_skill_parts(&self, entries: Vec<MemoryEntry>, limit: usize) -> Vec<MemoryEntry> {
        if entries.is_empty() {
            return entries;
        }
        let mut contexts: std::collections::HashMap<String, Option<(MemoryGraph, MemoryEntry)>> =
            std::collections::HashMap::new();
        let mut out = Vec::with_capacity(entries.len());
        let mut seen = std::collections::HashSet::new();

        for entry in entries {
            let mut adjacent_entries = Vec::new();
            if let Some(root_id) = skill_root_id_for_part(&entry.id) {
                let context = contexts
                    .entry(root_id.clone())
                    .or_insert_with(|| self.load_skill_root_context(&root_id));
                if let Some((graph, root)) = context.as_ref() {
                    if seen.insert(root.id.clone()) {
                        out.push(root.clone());
                    }
                    if entry
                        .property("skill.part_kind")
                        .and_then(|value| value.as_str())
                        == Some("step")
                    {
                        let next = graph
                            .get_edges(&entry.id)
                            .iter()
                            .find(|edge| {
                                matches!(edge.kind, jcode_memory_types::EdgeKind::NextStep)
                            })
                            .and_then(|edge| graph.get_memory(&edge.target));
                        let previous = graph
                            .get_incoming(&entry.id)
                            .into_iter()
                            .find(|source| {
                                graph.get_edges(source).iter().any(|edge| {
                                    edge.target == entry.id
                                        && matches!(
                                            edge.kind,
                                            jcode_memory_types::EdgeKind::NextStep
                                        )
                                })
                            })
                            .and_then(|source| graph.get_memory(source));
                        adjacent_entries.extend([previous, next].into_iter().flatten().cloned());
                    }
                }
            }
            if seen.insert(entry.id.clone()) {
                out.push(entry);
            }
            for adjacent in adjacent_entries {
                if seen.insert(adjacent.id.clone()) {
                    out.push(adjacent);
                }
            }
        }
        out.truncate(limit.max(1));
        out
    }

    fn load_skill_root_context(&self, root_id: &str) -> Option<(MemoryGraph, MemoryEntry)> {
        if let Ok(graph) = self.load_project_graph()
            && let Some(root) = graph.get_memory(root_id).cloned()
        {
            return Some((graph, root));
        }
        let graph = self.load_global_graph().ok()?;
        let root = graph.get_memory(root_id).cloned()?;
        Some((graph, root))
    }

    /// L1 recall shaped as a [`MemoryRelevanceResult`] so the memory agent can
    /// use it as a drop-in fallback for the judged path.
    pub fn l1_relevance_result(&self, query: &str) -> MemoryRelevanceResult {
        let entries = match self.l1_relevant(query, L1_MAX_RESULTS) {
            Ok(entries) => entries,
            Err(e) => {
                crate::logging::info(&format!("L1 recall failed: {e}"));
                Vec::new()
            }
        };
        // Mirror nodes make skills recallable by the ordinary memory path; the
        // pointer line closes the loop the linkage design calls "retrieval side
        // linkage": when the recalled context (or the query itself) matches a
        // skill that exists on disk, say so in one bounded line instead of
        // silently remembering that a skill is available.
        let skill_mirror = self.skill_pointer_hint(query, &entries);
        let skill_hint = skill_mirror.as_ref().map(Self::skill_hint_text);
        if entries.is_empty() {
            // Hint-only hit: the mirror itself becomes the selected entry so
            // the pending publication has an ID to validate (an empty ID set
            // would be discarded at publication).
            if let Some(mirror) = skill_mirror {
                let hint = Self::skill_hint_text(&mirror);
                return MemoryRelevanceResult {
                    display_prompt: Some(hint.clone()),
                    prompt: Some(hint),
                    selected_entries: vec![mirror],
                };
            }
            return MemoryRelevanceResult {
                prompt: None,
                display_prompt: None,
                selected_entries: Vec::new(),
            };
        }
        let prompt = format_relevant_prompt(&entries, L1_MAX_RESULTS);
        let prompt = match (prompt, skill_hint) {
            (Some(p), Some(hint)) => Some(format!("{p}\n{hint}")),
            (p, hint) => p.or(hint),
        };
        let display_prompt = format_relevant_display_prompt(&entries, L1_MAX_RESULTS);
        MemoryRelevanceResult {
            prompt,
            display_prompt,
            selected_entries: entries,
        }
    }

    /// The matching un-selected `skill:` mirror and its bounded pointer line,
    /// or `None`.
    ///
    /// Fires when the query matches a `skill:<name>` mirror that L1 recall did
    /// *not* already surface as a selected entry. The hint text is bounded
    /// (mirror contents are bounded the same way), so adding it cannot blow up
    /// the injection budget.
    fn skill_pointer_hint(&self, query: &str, selected: &[MemoryEntry]) -> Option<MemoryEntry> {
        let normalized = normalize_search_text(query);
        if normalized.is_empty() {
            return None;
        }
        let selected_ids: std::collections::HashSet<&str> =
            selected.iter().map(|entry| entry.id.as_str()).collect();
        let terms = collect_skill_query_terms(query)
            .into_iter()
            .collect::<Vec<_>>();
        let skill = self
            .collect_memories_scoped(MemoryScope::All)
            .ok()?
            .into_iter()
            .find(|entry| {
                is_skill_root_id(&entry.id)
                    && !selected_ids.contains(entry.id.as_str())
                    && score_l1_relevance(entry, &terms, &normalized) >= L1_MIN_SCORE
            })?;
        Some(skill)
    }

    /// The one-line hint body for a mirror (bounded pointer text).
    fn skill_hint_text(mirror: &MemoryEntry) -> String {
        let mut pointer = mirror.content.trim().to_string();
        if pointer.chars().count() > 200 {
            pointer = pointer.chars().take(200).collect::<String>();
        }
        format!("\n# Available Skill\n\n{pointer}")
    }

    /// TTL maintenance pass (`activity.periodic_gc`).
    ///
    /// Instances whose *type* declares a `ttl_days` and that have gone that
    /// long without being touched are archived (not deleted), which removes
    /// them from recall while keeping them recoverable. "Touched" means the
    /// entry was written, reinforced or read through `touch()`, i.e. the TTL is
    /// measured from `updated_at` exactly as the ontology describes it.
    ///
    /// Link nodes (`goal:` / `skill:`) are exempt: their lifecycle belongs to the
    /// subsystem that owns the artifact.
    pub fn gc_expired_memories(&self) -> Result<TtlGcReport> {
        let mut report = self.gc_scope(GcScope::Project)?;
        report.merge(self.gc_scope(GcScope::Global)?);
        Ok(report)
    }

    fn gc_scope(&self, scope: GcScope) -> Result<TtlGcReport> {
        let mut report = TtlGcReport::default();
        let mut graph = match scope {
            GcScope::Project => match self.load_project_graph() {
                Ok(g) => g,
                Err(_) => return Ok(report),
            },
            GcScope::Global => self.load_global_graph()?,
        };
        let ontology = self.ontology_for_graph(&graph);
        let now = chrono::Utc::now();
        let mut changed = false;
        let candidates: Vec<(String, f32, String)> = graph
            .active_memories()
            .filter(|entry| !jcode_memory_types::is_link_node(entry))
            .filter_map(|entry| {
                let type_id = ontology.type_from_category(&entry.category)?;
                let ttl = ontology.get_type(type_id)?.ttl_days?;
                let age_days = (now - entry.updated_at).num_seconds().max(0) as f32 / 86_400.0;
                (age_days > ttl).then(|| (entry.id.clone(), age_days, type_id.to_string()))
            })
            .collect();
        for (id, age_days, type_id) in candidates {
            let reason = format!("ttl elapsed: {type_id} idle {age_days:.1}d");
            if graph.archive_memory(&id, reason).is_some() {
                report.archived += 1;
                changed = true;
            }
        }
        if changed {
            match scope {
                GcScope::Project => self.save_project_graph(&graph)?,
                GcScope::Global => self.save_global_graph(&graph)?,
            }
        }
        Ok(report)
    }

    /// Physically drop instances that have been archived for longer than
    /// `grace_days`. Not called by [`Self::gc_expired_memories`]: deletion is
    /// opt-in so a TTL decision stays reversible by default.
    pub fn purge_archived_memories(&self, grace_days: f32) -> Result<TtlGcReport> {
        let mut report = TtlGcReport::default();
        for scope in [GcScope::Project, GcScope::Global] {
            let mut graph = match scope {
                GcScope::Project => match self.load_project_graph() {
                    Ok(g) => g,
                    Err(_) => continue,
                },
                GcScope::Global => self.load_global_graph()?,
            };
            let now = chrono::Utc::now();
            let doomed: Vec<String> = graph
                .memories
                .values()
                .filter(|entry| {
                    entry.lifecycle.status == crate::memory_types::MemoryStatus::Archived
                        && !jcode_memory_types::is_link_node(entry)
                        && (now - entry.updated_at).num_seconds().max(0) as f32 / 86_400.0
                            > grace_days
                })
                .map(|entry| entry.id.clone())
                .collect();
            if doomed.is_empty() {
                continue;
            }
            for id in &doomed {
                if graph.remove_memory(id).is_some() {
                    report.purged += 1;
                }
            }
            match scope {
                GcScope::Project => self.save_project_graph(&graph)?,
                GcScope::Global => self.save_global_graph(&graph)?,
            }
        }
        Ok(report)
    }

    /// Discover conservative `relates_to` edges from shared `subject_key`
    /// values and shared tags.
    ///
    /// This is intentionally deterministic and proposal-first: `apply=false`
    /// only reports candidates, while `apply=true` writes `RelatesTo` edges and
    /// bumps `link_discovery_count`. No LLM or embedding call is involved.
    pub fn discover_links(&self, apply: bool) -> Result<LinkDiscoveryReport> {
        use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

        const MAX_TOTAL: usize = 200;
        const MAX_PER_MEMORY: usize = 3;

        let mut report = LinkDiscoveryReport::default();
        report.stores = self.for_each_store(|_store, graph| {
            let active: Vec<(String, Option<String>, Vec<String>)> = graph
                .all_memories()
                .filter(|entry| {
                    entry.is_retrievable_now() && !jcode_memory_types::is_link_id(&entry.id)
                })
                .map(|entry| {
                    (
                        entry.id.clone(),
                        entry.subject_key.clone(),
                        entry.tags.clone(),
                    )
                })
                .collect();
            report.memories += active.len();
            if active.len() < 2 {
                return false;
            }

            let active_ids: HashSet<String> = active.iter().map(|(id, _, _)| id.clone()).collect();
            let mut existing_pairs: HashSet<(String, String)> = HashSet::new();
            for id in &active_ids {
                for edge in graph.get_edges(id) {
                    if active_ids.contains(&edge.target) {
                        existing_pairs.insert(link_pair(id, &edge.target));
                    }
                }
            }

            let mut candidates: BTreeMap<(String, String), LinkSuggestion> = BTreeMap::new();

            // High precision: the same normalized subject key.
            let mut subject_groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for (id, subject_key, _) in &active {
                if let Some(subject_key) = subject_key.as_deref()
                    && !subject_key.trim().is_empty()
                {
                    subject_groups
                        .entry(subject_key.to_string())
                        .or_default()
                        .push(id.clone());
                }
            }
            for ids in subject_groups.values() {
                for left in 0..ids.len() {
                    for right in left + 1..ids.len() {
                        insert_link_suggestion(
                            &mut candidates,
                            &ids[left],
                            &ids[right],
                            0.9,
                            "same subject_key",
                        );
                    }
                }
            }

            // Medium precision: at least two shared tags and Jaccard >= 0.5.
            let tags_by_id: BTreeMap<String, BTreeSet<String>> = active
                .iter()
                .filter_map(|(id, _, tags)| {
                    let tags: BTreeSet<String> = tags
                        .iter()
                        .map(|tag| tag.trim().to_lowercase())
                        .filter(|tag| !tag.is_empty())
                        .collect();
                    (!tags.is_empty()).then(|| (id.clone(), tags))
                })
                .collect();
            let mut tag_index: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for (id, tags) in &tags_by_id {
                for tag in tags {
                    tag_index.entry(tag.clone()).or_default().push(id.clone());
                }
            }
            let mut shared_counts: BTreeMap<(String, String), usize> = BTreeMap::new();
            for ids in tag_index.values() {
                for left in 0..ids.len() {
                    for right in left + 1..ids.len() {
                        *shared_counts
                            .entry(link_pair(&ids[left], &ids[right]))
                            .or_default() += 1;
                    }
                }
            }
            for ((from, to), shared) in shared_counts {
                if shared < 2 {
                    continue;
                }
                let (Some(from_tags), Some(to_tags)) = (tags_by_id.get(&from), tags_by_id.get(&to))
                else {
                    continue;
                };
                let union = from_tags.union(to_tags).count();
                if union == 0 {
                    continue;
                }
                let jaccard = shared as f32 / union as f32;
                if jaccard < 0.5 {
                    continue;
                }
                insert_link_suggestion(
                    &mut candidates,
                    &from,
                    &to,
                    0.5 + 0.3 * jaccard,
                    &format!(
                        "shared tags ({shared} shared, {:.0}% Jaccard)",
                        jaccard * 100.0
                    ),
                );
            }

            let mut suggestions: Vec<LinkSuggestion> = candidates
                .into_values()
                .filter(|suggestion| {
                    let key = link_pair(&suggestion.from, &suggestion.to);
                    if existing_pairs.contains(&key) {
                        report.existing += 1;
                        false
                    } else {
                        true
                    }
                })
                .collect();
            report.candidates += suggestions.len();
            suggestions.sort_by(|left, right| {
                right
                    .weight
                    .partial_cmp(&left.weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.from.cmp(&right.from))
                    .then_with(|| left.to.cmp(&right.to))
            });

            let mut per_memory: HashMap<String, usize> = HashMap::new();
            let mut changed = false;
            for suggestion in suggestions {
                if report.suggestions.len() >= MAX_TOTAL {
                    report.capped = true;
                    break;
                }
                if per_memory.get(&suggestion.from).copied().unwrap_or(0) >= MAX_PER_MEMORY
                    || per_memory.get(&suggestion.to).copied().unwrap_or(0) >= MAX_PER_MEMORY
                {
                    continue;
                }
                *per_memory.entry(suggestion.from.clone()).or_default() += 1;
                *per_memory.entry(suggestion.to.clone()).or_default() += 1;
                if apply {
                    graph.link_memories(&suggestion.from, &suggestion.to, suggestion.weight);
                    report.applied += 1;
                    changed = true;
                }
                report.suggestions.push(suggestion);
            }
            changed
        })?;
        Ok(report)
    }

    /// Apply an already-reviewed list of link suggestions instead of
    /// recomputing them. A suggestion is applied to the first store that
    /// contains both endpoints; missing endpoints are counted, not guessed at.
    pub fn apply_link_proposals(
        &self,
        suggestions: &[LinkSuggestion],
        apply: bool,
    ) -> Result<LinkApplyReport> {
        use std::collections::HashSet;

        let mut report = LinkApplyReport {
            suggestions: suggestions.len(),
            ..LinkApplyReport::default()
        };
        let mut seen: HashSet<(String, String)> = HashSet::new();
        report.stores_changed = self.for_each_store(|_store, graph| {
            let mut changed = false;
            for suggestion in suggestions {
                let key = link_pair(&suggestion.from, &suggestion.to);
                if seen.contains(&key) {
                    continue;
                }
                if !graph.memories.contains_key(&suggestion.from)
                    || !graph.memories.contains_key(&suggestion.to)
                {
                    continue;
                }
                seen.insert(key);
                report.matched += 1;
                if graph
                    .get_edges(&suggestion.from)
                    .iter()
                    .any(|edge| edge.target == suggestion.to)
                {
                    report.existing += 1;
                    continue;
                }
                report.applicable += 1;
                if apply {
                    graph.link_memories(&suggestion.from, &suggestion.to, suggestion.weight);
                    report.applied += 1;
                    changed = true;
                }
            }
            changed
        })?;
        report.missing = suggestions.len().saturating_sub(report.matched);
        Ok(report)
    }

    /// Audit legacy `memory/projects/*.json` snapshots against the active
    /// backend. With `quarantine=true`, only snapshots that are strictly
    /// smaller than an existing backend graph are renamed aside; JSON files
    /// that are the only surviving copy are never touched.
    pub fn audit_legacy_json(&self, quarantine: bool) -> Result<LegacyJsonAuditReport> {
        let backend = if self.test_mode {
            test_backend()
        } else {
            graph_backend()
        };
        self.audit_legacy_json_with_backend(quarantine, backend.as_ref())
    }

    fn audit_legacy_json_with_backend(
        &self,
        quarantine: bool,
        backend: &dyn GraphBackend,
    ) -> Result<LegacyJsonAuditReport> {
        let dir = storage::jcode_dir()?.join("memory").join("projects");
        let mut report = LegacyJsonAuditReport::default();
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Ok(report);
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            let Ok(json_graph) = storage::read_json::<MemoryGraph>(&path) else {
                continue;
            };
            let store_key = format!("project:{stem}");
            let backend_graph = backend.load(&StoreKey::new(&store_key)).ok();
            let backend_memories = backend_graph.as_ref().map(MemoryGraph::memory_count);
            let backend_edges = backend_graph.as_ref().map(MemoryGraph::edge_count);
            let json_ids: std::collections::BTreeSet<&str> =
                json_graph.memories.keys().map(String::as_str).collect();
            let backend_ids: std::collections::BTreeSet<&str> = backend_graph
                .as_ref()
                .map(|graph| graph.memories.keys().map(String::as_str).collect())
                .unwrap_or_default();
            let common_memories = json_ids.intersection(&backend_ids).count();
            let json_only_memories = json_ids.len().saturating_sub(common_memories);
            let backend_only_memories = backend_ids.len().saturating_sub(common_memories);
            let only_copy = match backend_graph.as_ref() {
                None => true,
                Some(graph) => {
                    graph.memory_count() == 0
                        && graph.edge_count() == 0
                        && (json_graph.memory_count() > 0 || json_graph.edge_count() > 0)
                }
            };
            let stale = backend_graph.as_ref().is_some_and(|graph| {
                graph.memory_count() >= json_graph.memory_count()
                    && graph.edge_count() >= json_graph.edge_count()
                    && (graph.memory_count() > json_graph.memory_count()
                        || graph.edge_count() > json_graph.edge_count())
            });
            let mut quarantined_to = None;
            if quarantine && stale {
                let target = path
                    .with_extension(format!("json.legacy.{}", Utc::now().format("%Y%m%d%H%M%S")));
                if !target.exists() {
                    std::fs::rename(&path, &target)?;
                    report.quarantined += 1;
                    quarantined_to = Some(target.display().to_string());
                }
            }
            if stale {
                report.stale += 1;
            }
            if only_copy {
                report.only_copy += 1;
            }
            report.files.push(LegacyJsonAudit {
                path: path.display().to_string(),
                store_key,
                json_memories: json_graph.memory_count(),
                json_edges: json_graph.edge_count(),
                backend_memories,
                backend_edges,
                json_only_memories,
                backend_only_memories,
                common_memories,
                stale,
                only_copy,
                quarantined_to,
            });
        }
        Ok(report)
    }

    /// Merge memories that exist only in legacy `projects/<hash>.json`
    /// snapshots into the active backend. Existing backend memories are never
    /// overwritten; only missing memories and their incident edges are added.
    pub fn merge_legacy_json(&self, apply: bool) -> Result<LegacyJsonMergeReport> {
        self.merge_legacy_json_with_options(apply, false)
    }

    /// Merge legacy snapshots. With `repair_edges`, also backfill missing
    /// non-tag edges and tags for memories that already exist in the backend.
    pub fn merge_legacy_json_with_options(
        &self,
        apply: bool,
        repair_edges: bool,
    ) -> Result<LegacyJsonMergeReport> {
        let backend = if self.test_mode {
            test_backend()
        } else {
            graph_backend()
        };
        self.merge_legacy_json_with_backend(apply, repair_edges, backend.as_ref())
    }

    fn merge_legacy_json_with_backend(
        &self,
        apply: bool,
        repair_edges: bool,
        backend: &dyn GraphBackend,
    ) -> Result<LegacyJsonMergeReport> {
        let dir = storage::jcode_dir()?.join("memory").join("projects");
        let mut report = LegacyJsonMergeReport::default();
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Ok(report);
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            let Ok(json_graph) = storage::read_json::<MemoryGraph>(&path) else {
                continue;
            };
            let store_key = format!("project:{stem}");
            let mut backend_graph = backend
                .load(&StoreKey::new(&store_key))
                .unwrap_or_else(|_| MemoryGraph::new());
            let backend_memories_before = backend_graph.memory_count();
            let json_only_memories = json_graph
                .memories
                .keys()
                .filter(|id| !backend_graph.memories.contains_key(*id))
                .count();
            let (added_memories, added_edges, added_tags) =
                merge_json_graph(&mut backend_graph, &json_graph, repair_edges);
            let backend_memories_after = backend_graph.memory_count();
            let mut applied = false;
            if apply && (added_memories > 0 || added_edges > 0 || added_tags > 0) {
                backend.save(&StoreKey::new(&store_key), &backend_graph)?;
                applied = true;
                report.added_memories += added_memories;
                report.added_edges += added_edges;
                report.applied += 1;
            }
            report.files.push(LegacyJsonMerge {
                path: path.display().to_string(),
                store_key,
                json_only_memories,
                added_memories,
                added_edges,
                added_tags,
                backend_memories_before,
                backend_memories_after,
                applied,
            });
        }
        Ok(report)
    }

    /// Visit every persisted store graph, saving the ones the visitor changed.
    ///
    /// Two layouts have to be covered: the trait backend's own store root
    /// (sqlite named graphs, JSON backend files) and the legacy
    /// `<jcode_dir>/memory/projects/<hash>.json` files that JSON-mode writes
    /// use. Cross-store passes need both, otherwise an install looks empty
    /// depending on which backend it happens to run.
    ///
    /// The visitor returns `true` when it modified the graph; only those are
    /// written back. Each store is saved immediately after it is visited so a
    /// maintenance pass cannot hold a stale snapshot for the whole scan and
    /// overwrite rows another session wrote in the meantime.
    ///
    /// Returns the number of stores visited.
    fn for_each_store(
        &self,
        mut visit: impl FnMut(&str, &mut MemoryGraph) -> bool,
    ) -> Result<usize> {
        let backend = if self.test_mode {
            test_backend()
        } else {
            graph_backend()
        };
        let mut visited = 0usize;
        let mut seen_keys = std::collections::HashSet::new();
        for key in backend.list_keys().unwrap_or_default() {
            let Ok(mut graph) = backend.load(&key) else {
                continue;
            };
            seen_keys.insert(key.as_str().to_string());
            visited += 1;
            if visit(key.as_str(), &mut graph) {
                match backend.save(&key, &graph) {
                    Ok(()) => {
                        if !self.test_mode {
                            cache_graph_for_backend(backend.name(), &key, &graph, 0);
                        }
                    }
                    Err(e) => {
                        crate::logging::warn(&format!(
                            "store save after maintenance failed for {}: {e}",
                            key.as_str()
                        ));
                    }
                }
            }
        }
        if let Ok(base) = crate::storage::jcode_dir() {
            let dir = base.join("memory").join("projects");
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) != Some("json") {
                        continue;
                    }
                    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                        continue;
                    };
                    // The backend may already own this store. Visiting the
                    // legacy snapshot as well double-counted every health
                    // metric and reran maintenance twice on the same key.
                    let legacy_key = format!("project:{stem}");
                    if seen_keys.contains(&legacy_key) {
                        continue;
                    }
                    let Ok(mut graph) = crate::storage::read_json::<MemoryGraph>(&path) else {
                        continue;
                    };
                    visited += 1;
                    if visit(&legacy_key, &mut graph) {
                        crate::storage::write_json(&path, &graph)?;
                        if !self.test_mode {
                            cache_graph(path.clone(), &graph);
                        }
                    }
                }
            }
        }
        Ok(visited)
    }

    /// Resolve declared evidence ids against every persisted store.
    ///
    /// A skill's `evidence:` list is global even when the memories it points at
    /// live in different project stores. Resolving only against the store that
    /// holds the mirror therefore reports a false `partial`/`needs_review` for
    /// cross-store skills. This pass builds one state map over all graphs and
    /// follows `superseded_by` chains to the live successor. It is read-only and
    /// only called when the local/global graph cannot resolve a declaration.
    pub fn resolve_evidence_across_stores(
        &self,
        ids: &[String],
    ) -> std::collections::HashMap<String, String> {
        use std::collections::{HashMap, HashSet};

        let mut resolved = HashMap::new();
        if ids.is_empty() {
            return resolved;
        }

        // id -> (active_anywhere, superseded_by)
        let mut states: HashMap<String, (bool, Option<String>)> = HashMap::new();
        let _ = self.for_each_store(|_key, graph| {
            for (id, entry) in &graph.memories {
                let state = states.entry(id.clone()).or_insert((false, None));
                state.0 |= entry.active;
                if state.1.is_none() {
                    state.1 = entry.superseded_by.clone();
                }
            }
            false
        });

        for id in ids {
            let mut current = id.clone();
            let mut seen = HashSet::new();
            for _ in 0..16 {
                if !seen.insert(current.clone()) {
                    break;
                }
                let Some((active, next)) = states.get(&current) else {
                    break;
                };
                if *active {
                    resolved.insert(id.clone(), current);
                    break;
                }
                match next {
                    Some(next) if !next.is_empty() => current = next.clone(),
                    _ => break,
                }
            }
        }
        resolved
    }

    /// Reclassify residual instances into the v2 type split.
    ///
    /// Everything written before v2 is `fact`, which the ontology aliases to
    /// `observation` (the residual bucket). The split only pays off once the
    /// stored rows say `env_fact` / `contract` / `process_rule` themselves -
    /// otherwise `env_fact` has no instances and no TTL applies to it.
    ///
    /// Deterministic rules only (no LLM): procedures, then contracts, then
    /// perishable environment facts; everything else stays `observation`.
    /// `apply = false` reports what would change without writing.
    ///
    /// `stamp_subject_keys` additionally derives a retrieval anchor for every
    /// instance that has none yet - rows written before the field existed, which
    /// on a real install is nearly all of them (measured: 3 of 4378). Without
    /// anchors L1 recall falls back to term overlap and write-time
    /// reconciliation cannot see "same subject".
    pub fn reclassify_observations(
        &self,
        apply: bool,
        stamp_subject_keys: bool,
    ) -> Result<ReclassifyReport> {
        let mut report = ReclassifyReport {
            dry_run: !apply,
            ..Default::default()
        };
        report.stores = self.for_each_store(|_key, graph| {
            let ontology = self.ontology_for_graph(graph);
            classify_graph(graph, &ontology, stamp_subject_keys, apply, &mut report)
        })?;
        Ok(report)
    }

    /// Same pass, restricted to the current project store.
    ///
    /// This is the one that runs at session finalize: it is bounded by a single
    /// store (milliseconds) instead of walking every store on the machine
    /// (seconds to minutes), and it is what keeps newly extracted rows typed
    /// without waiting for the extraction prompt to emit v2 type names.
    pub fn reclassify_project_store(&self, apply: bool) -> Result<ReclassifyReport> {
        let mut report = ReclassifyReport {
            dry_run: !apply,
            ..Default::default()
        };
        let mut graph = match self.load_project_graph() {
            Ok(graph) => graph,
            Err(_) => return Ok(report),
        };
        report.stores = 1;
        let ontology = self.ontology_for_graph(&graph);
        if classify_graph(&mut graph, &ontology, true, apply, &mut report) {
            self.save_project_graph(&graph)?;
        }
        Ok(report)
    }

    /// Apply the deterministic ontology data plan to every persisted store.
    ///
    /// Dry-run clones each graph before migrating, so callers can inspect the
    /// exact counts without writing. The real pass reuses `for_each_store`,
    /// which covers the SQLite named graphs and legacy JSON project files.
    ///
    /// The apply pass rewrites whole store snapshots. Callers must ensure no
    /// other session is writing the same store concurrently, otherwise a
    /// read-modify-write snapshot can overwrite newer rows.
    pub fn apply_default_ontology_migrations(
        &self,
        apply: bool,
    ) -> Result<OntologyMigrationReport> {
        use jcode_memory_types::migration::{apply_migrations, default_data_migrations};

        let steps = default_data_migrations();
        let mut report = OntologyMigrationReport {
            dry_run: !apply,
            ..Default::default()
        };
        report.stores = self.for_each_store(|_key, graph| {
            if apply {
                let ontology_changed = self.canonicalize_default_ontology(graph);
                if ontology_changed {
                    report.ontology_canonicalized += 1;
                }
                let ontology = self.ontology_for_graph(graph);
                let migrated = apply_migrations(graph, &steps, &ontology);
                let changed = migrated.changed() > 0;
                report.migrations.merge(migrated);
                ontology_changed || changed
            } else {
                let mut probe = graph.clone();
                let ontology_changed = self.canonicalize_default_ontology(&mut probe);
                if ontology_changed {
                    report.ontology_canonicalized += 1;
                }
                let ontology = self.ontology_for_graph(&probe);
                let migrated = apply_migrations(&mut probe, &steps, &ontology);
                report.migrations.merge(migrated);
                false
            }
        })?;
        Ok(report)
    }

    /// Replace a store's core-only default ontology with the compiled default.
    ///
    /// Old stores persisted the memory classes directly in `ontology.types`
    /// and may carry stale descriptions or hashes. Deserialization already
    /// strips those classes, so the remaining value is a default ontology with
    /// no business concepts. Rewriting it to the compiled default makes the
    /// stored version, description and content hash match the running code.
    ///
    /// Stores with project business types, custom rules/relationships/activities
    /// or layered overlays are left untouched.
    fn canonicalize_default_ontology(&self, graph: &mut MemoryGraph) -> bool {
        let default = self.ontology_registry().default_ontology();
        let Some(stored) = graph.metadata.ontology.as_ref() else {
            return false;
        };
        if !self.is_core_only_default(stored) {
            return false;
        }

        let default_hash = default.content_hash();
        let history_is_core_only = graph.metadata.ontology_history.iter().all(|snapshot| {
            snapshot.ontology.id == default.id
                && snapshot.ontology.types.is_empty()
                && snapshot.ontology.category_aliases.is_empty()
                && snapshot.ontology.relationships == default.relationships
                && snapshot.ontology.rules == default.rules
                && snapshot.ontology.activities == default.activities
        });
        graph.metadata.ontology = None;
        graph.metadata.ontology_id = default.id.clone();
        graph.metadata.ontology_version = default.version;
        graph.metadata.ontology_hash = default_hash;
        graph.metadata.ontology_parent_hash.clear();
        graph.metadata.ontology_layer_hash.clear();
        graph.metadata.ontology_layer_ids.clear();
        if history_is_core_only {
            graph.metadata.ontology_history.clear();
        }
        true
    }

    /// Record that a skill was invoked, closing the skill -> memory loop.
    ///
    /// Writes an `observation` instance keyed to the skill
    /// (`subject_key = skill:<name>`), so "which skill ran, where and how often"
    /// becomes a queryable fact instead of nothing (measured before this: 0
    /// memories carried a `skill_registry` source or a `skill:` tag).
    pub fn remember_skill_invocation(&self, skill_name: &str) -> Result<String> {
        self.record_skill_outcome(
            skill_name,
            jcode_memory_types::actions::ActionOutcome::Started,
            &[],
        )
    }

    /// Record an action/skill execution outcome and let the ontology rules
    /// validate or invalidate the knowledge behind it.
    ///
    /// The action contract (`MemoryActionSpec::skill`) and the outcome are
    /// dispatched through `event.action_outcome`; matching rules set the
    /// artifact health, tag the mirror and emit evidence effects. Evidence
    /// memories are then reinforced or weakened in place, which is the
    /// "execution result verifies cognition" half of the loop.
    pub fn record_skill_outcome(
        &self,
        skill_name: &str,
        outcome: jcode_memory_types::actions::ActionOutcome,
        evidence_ids: &[String],
    ) -> Result<String> {
        let name = skill_name.trim();
        anyhow::ensure!(!name.is_empty(), "skill invocation needs a skill name");
        let invocation_id = format!("skill-invocation:{name}");
        let outcome_tag = format!("skill-outcome:{}", outcome.as_str());
        let confidence = if outcome.is_success() { 1.0 } else { 0.6 };
        let entry = MemoryEntry::new(
            MemoryCategory::Fact,
            format!(
                "Skill `/{name}` was invoked in this workspace ({}).",
                outcome.as_str()
            ),
        )
        .with_id(invocation_id.clone())
        .with_tags(vec![
            format!("skill:{name}"),
            "skill-invoked".to_string(),
            outcome_tag,
        ])
        .with_source(format!("skill:{name}"))
        .with_provenance(
            ProvenanceRecord::new(format!("skill:{name}"), ExtractionMethod::SkillInvocation)
                .with_confidence(confidence),
        );
        // Idempotent by id: invoking the same skill again refreshes the row
        // (and its timestamp) instead of piling up one memory per run.
        let mut entry = entry;
        entry.subject_key = Some(format!("skill:{name}"));
        let mut project = self.load_project_graph()?;
        Self::upsert_invocation_entry(&mut project, entry);
        self.apply_skill_outcome_to_graph(&mut project, name, outcome, evidence_ids);
        self.save_project_graph(&project)?;

        // A global skill's mirror lives in the global store even when the
        // invocation ran in a project. Keep its health in sync too.
        if let Ok(mut global) = self.load_global_graph()
            && self.apply_skill_outcome_to_graph(&mut global, name, outcome, evidence_ids)
        {
            self.save_global_graph(&global)?;
        }
        Ok(invocation_id)
    }

    fn upsert_invocation_entry(graph: &mut MemoryGraph, entry: MemoryEntry) {
        let id = entry.id.clone();
        if let Some(existing) = graph.get_memory_mut(&id) {
            existing.content = entry.content;
            existing.tags = entry.tags;
            existing.updated_at = entry.updated_at;
            existing.confidence = entry.confidence;
            existing.source = entry.source;
            existing.provenance = entry.provenance;
            existing.subject_key = entry.subject_key;
            existing.refresh_search_text();
        } else {
            graph.add_memory(entry);
        }
    }

    /// Apply the ontology outcome rules to one graph's skill mirror.
    ///
    /// Returns true when a mirror existed and the graph changed.
    fn apply_skill_outcome_to_graph(
        &self,
        graph: &mut MemoryGraph,
        name: &str,
        outcome: jcode_memory_types::actions::ActionOutcome,
        evidence_ids: &[String],
    ) -> bool {
        use jcode_memory_types::actions::MemoryActionSpec;
        use jcode_memory_types::ontology::Effect;
        use jcode_memory_types::rule_engine::{RuleContext, apply_entry_effects, dispatch_event};

        let mirror_id = format!("skill:{name}");
        let Some(mut mirror) = graph.get_memory(&mirror_id).cloned() else {
            return false;
        };
        let implementation = mirror
            .artifact
            .as_ref()
            .map(|artifact| artifact.path.clone());
        // Stores author their own ontology snapshot. Add the action-outcome
        // rules when an older snapshot predates them, so execution feedback
        // works without forcing an ontology version bump or migration.
        let mut ontology = (*self.ontology_for_graph(graph)).clone();
        for rule in jcode_memory_types::ontology::skill_outcome_rules() {
            if !ontology.rules.iter().any(|existing| existing.id == rule.id) {
                ontology.rules.push(rule);
            }
        }
        let ontology = std::sync::Arc::new(ontology);
        let context = RuleContext {
            entry: mirror.clone(),
            type_id: "custom".to_string(),
            existing_id: Some(mirror_id.clone()),
            similarity: None,
            graph: std::sync::Arc::new(graph.clone()),
            ontology,
            source_label: "skill_outcome".to_string(),
            event: jcode_memory_types::ontology::EVENT_ACTION_OUTCOME.to_string(),
            scope: String::new(),
            action: Some(MemoryActionSpec::skill(name, implementation.as_deref())),
            outcome: Some(outcome),
        };
        let plan = dispatch_event(&context);
        let deferred = apply_entry_effects(&plan, &mut mirror);

        let resolved: Vec<String> = mirror
            .artifact
            .as_ref()
            .map(|artifact| {
                if evidence_ids.is_empty() {
                    artifact.resolved_evidence.clone()
                } else {
                    evidence_ids.to_vec()
                }
            })
            .unwrap_or_else(|| evidence_ids.to_vec());

        graph.memories.insert(mirror_id, mirror);
        for tag in ["skill-validated", "skill-invalidated"] {
            if graph
                .get_memory(&format!("skill:{name}"))
                .is_some_and(|entry| entry.tags.iter().any(|existing| existing == tag))
            {
                graph.tag_memory(&format!("skill:{name}"), tag);
            }
        }

        for effect in deferred {
            let (delta, tag) = match effect {
                Effect::ReinforceEvidence { amount } => (amount, "skill-validated"),
                Effect::InvalidateEvidence { amount } => (-amount, "skill-invalidated"),
                _ => continue,
            };
            for id in &resolved {
                if let Some(evidence) = graph.get_memory_mut(id) {
                    if outcome.is_success() {
                        evidence.record_outcome(true);
                    } else if outcome.is_failure() {
                        evidence.record_outcome(false);
                    }
                    evidence.confidence = (evidence.confidence + delta).clamp(0.0, 1.0);
                    if delta > 0.0 {
                        evidence.strength = evidence.strength.saturating_add(1);
                    }
                    evidence.updated_at = chrono::Utc::now();
                    evidence.refresh_search_text();
                }
                if graph.memories.contains_key(id) {
                    graph.tag_memory(id, tag);
                }
            }
        }

        true
    }

    /// Aggregate evidence memories into skill proposals (memory -> skill).
    ///
    /// This is the learning path the v2 ontology declares as
    /// `activity.skill_distill`: `process_rule` and `correction` instances are
    /// evidence that a procedure exists; when enough of them agree on one
    /// `subject_key`, a draft `SKILL.md` is written to
    /// `<jcode_dir>/skills/.proposals/` for review. Nothing is ever installed
    /// silently - the proposal is a file a human or agent accepts.
    ///
    /// Rate-limited to one pass per day per machine (same stamp pattern as the
    /// session-backup prune) because it walks every project store.
    pub fn distill_skill_evidence(&self) -> Result<Vec<SkillEvidenceProposal>> {
        let Ok(base) = crate::storage::jcode_dir() else {
            return Ok(Vec::new());
        };
        if !claim_maintenance_slot(&base, "skill-distill.stamp", 24 * 60 * 60) {
            return Ok(Vec::new());
        }
        self.write_skill_proposals_now()
    }

    /// Run the distillation pass without the once-a-day rate limit.
    ///
    /// Used both by [`Self::distill_skill_evidence`] (after claiming the slot)
    /// and by `jcode memory distill`, where the user asking for it explicitly is
    /// the intent: waiting for a session finalize to regenerate drafts is not a
    /// usable workflow for reviewing what the store has learned so far.
    pub fn write_skill_proposals_now(&self) -> Result<Vec<SkillEvidenceProposal>> {
        let Ok(base) = crate::storage::jcode_dir() else {
            return Ok(Vec::new());
        };
        let proposals = self.collect_skill_evidence()?;
        write_skill_proposals(&base, &proposals)?;
        Ok(proposals)
    }

    /// Evidence scan behind [`Self::distill_skill_evidence`].
    pub fn collect_skill_evidence(&self) -> Result<Vec<SkillEvidenceProposal>> {
        let evidence_types = [
            jcode_memory_types::ontology::TYPE_PROCESS_RULE,
            jcode_memory_types::ontology::TYPE_CORRECTION,
        ];
        let mut groups: std::collections::BTreeMap<String, SkillEvidenceProposal> =
            std::collections::BTreeMap::new();
        let mut consider = |store: &str, graph: &MemoryGraph| {
            let ontology = self.ontology_for_graph(graph);
            for entry in graph.active_memories() {
                if jcode_memory_types::is_link_node(entry) {
                    continue;
                }
                let Some(type_id) = ontology.type_from_category(&entry.category) else {
                    continue;
                };
                // A correction is stronger evidence than a plain rule: it means
                // the procedure was tried and corrected.
                let weight = if type_id == jcode_memory_types::ontology::TYPE_CORRECTION {
                    2.0
                } else if evidence_types.contains(&type_id) {
                    1.0
                } else {
                    continue;
                };
                let Some(subject) = entry.subject_key.clone() else {
                    continue;
                };
                // Group on the canonical key so anchors written by older
                // extractors (param spelling, trailing punctuation) still land
                // in the same cluster as today's rows.
                let subject = jcode_memory_types::reconcile::normalize_subject_key(&subject);
                if !proposal_eligible_subject(&subject) {
                    continue;
                }
                let proposal =
                    groups
                        .entry(subject.clone())
                        .or_insert_with(|| SkillEvidenceProposal {
                            subject_key: subject.clone(),
                            ..Default::default()
                        });
                if !proposal.evidence.contains(&entry.id) {
                    proposal.evidence.push(entry.id.clone());
                }
                if !proposal.stores.contains(&store.to_string()) {
                    proposal.stores.push(store.to_string());
                }
                if let Some(source) = entry.source.as_deref()
                    && source.starts_with("session_")
                    && !proposal.sessions.contains(&source.to_string())
                {
                    proposal.sessions.push(source.to_string());
                }
                if entry.tags.iter().any(|tag| tag == "skill-validated") {
                    proposal.positive += 1;
                }
                if entry.tags.iter().any(|tag| tag == "skill-invalidated")
                    || entry.metrics.outcome_failures > 0
                {
                    proposal.negative += 1;
                }
                proposal.last_seen = Some(
                    proposal
                        .last_seen
                        .map_or(entry.updated_at, |seen| seen.max(entry.updated_at)),
                );
                proposal.weight += weight;
            }
        };

        // Cross-store scan: the design's promotion rule ("the same procedure
        // learned in >=2 projects") is only observable by walking every store,
        // and on this corpus it is the normal case (185 near-duplicate pairs
        // across stores). `list_keys` enumerates them from durable state, so a
        // store written weeks ago still counts.
        let scanned = self.for_each_store(|key, graph| {
            consider(key, graph);
            false
        })?;
        // Last resort: the manager's own scope (an in-memory-only store, or a
        // schema neither layout above could read).
        if scanned == 0 {
            if let Some(dir) = self.get_project_dir()
                && let Ok(graph) = self.load_project_graph()
            {
                consider(&project_store_key(&dir), &graph);
            }
            if let Ok(global) = self.load_global_graph() {
                consider(crate::memory::global_store_key(), &global);
            }
        }

        let mut out: Vec<SkillEvidenceProposal> = groups
            .into_values()
            .filter(|p| {
                p.evidence.len() >= SKILL_DISTILL_MIN_EVIDENCE
                    && (p.stores.len() >= SKILL_DISTILL_MIN_STORES
                        || p.sessions.len() >= SKILL_DISTILL_MIN_EVIDENCE)
            })
            .collect();
        let now = Utc::now();
        for proposal in &mut out {
            let age_days = proposal
                .last_seen
                .map(|seen| (now - seen).num_days().max(0) as f32)
                .unwrap_or(365.0);
            proposal.score = proposal.weight
                + proposal.evidence.len() as f32 * 0.5
                + proposal.stores.len() as f32 * 1.5
                + proposal.sessions.len() as f32 * 0.3
                + proposal.positive as f32 * 0.5
                - proposal.negative as f32 * 2.0
                - (age_days / 90.0).min(3.0);
            proposal.needs_human_draft = true;
        }
        // A procedure whose evidence is entirely negative should be archived,
        // not promoted. Mixed evidence stays visible with a lower score.
        out.retain(|proposal| proposal.negative < proposal.evidence.len() as u32);
        out.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| b.weight.total_cmp(&a.weight))
                .then_with(|| b.evidence.len().cmp(&a.evidence.len()))
        });
        out.truncate(SKILL_DISTILL_MAX_PROPOSALS);
        Ok(out)
    }

    // === Runtime ontology evolution ===

    fn load_scope_graph(&self, scope: OntologyScope) -> Result<MemoryGraph> {
        match scope {
            OntologyScope::Project => {
                anyhow::ensure!(
                    self.project_memory_path()?.is_some(),
                    "project ontology requires a working directory; use global scope explicitly"
                );
                self.load_project_graph()
            }
            OntologyScope::Global => self.load_global_graph(),
        }
    }

    fn save_scope_graph(&self, scope: OntologyScope, graph: &MemoryGraph) -> Result<()> {
        match scope {
            OntologyScope::Project => self.save_project_graph(graph),
            OntologyScope::Global => self.save_global_graph(graph),
        }
    }

    /// The ontology a store currently uses, whether stored or compiled-in.
    pub fn stored_ontology(
        &self,
        scope: OntologyScope,
    ) -> Result<std::sync::Arc<jcode_memory_types::Ontology>> {
        let graph = self.load_scope_graph(scope)?;
        Ok(self.ontology_for_graph(&graph))
    }

    /// Layer ids and parent hash that produced a store's effective ontology.
    /// Empty values mean the store uses a plain snapshot.
    pub fn ontology_layer_status(&self, scope: OntologyScope) -> Result<(Vec<String>, String)> {
        let graph = self.load_scope_graph(scope)?;
        Ok((
            graph.metadata.ontology_layer_ids.clone(),
            graph.metadata.ontology_parent_hash.clone(),
        ))
    }

    /// Installed ontology snapshots for a store, newest last.
    pub fn ontology_history(
        &self,
        scope: OntologyScope,
    ) -> Result<Vec<jcode_memory_types::OntologySnapshot>> {
        let graph = self.load_scope_graph(scope)?;
        Ok(graph.metadata.ontology_history.clone())
    }

    /// Validate and (unless `dry_run`) apply an ontology patch to one store.
    ///
    /// This is the runtime entry point a reflection pass uses. It:
    /// 1. resolves the store's current ontology,
    /// 2. checks base id/version/hash and validates the result,
    /// 3. runs the patch's data migrations over the graph,
    /// 4. persists the new ontology, hash, version and history,
    /// 5. installs it into the process registry so subsequent turns use it.
    pub fn apply_ontology_patch(
        &self,
        scope: OntologyScope,
        patch: &jcode_memory_types::OntologyPatch,
        dry_run: bool,
    ) -> Result<jcode_memory_types::OntologyApplyReport> {
        if scope == OntologyScope::Project {
            return self.apply_project_ontology_patch(patch, dry_run);
        }
        use jcode_memory_types::migration::{
            MigrationState, OntologyApplyReport, OntologySnapshot, apply_migrations, apply_patch,
        };

        let mut graph = self.load_scope_graph(scope)?;
        let base = self.ontology_for_graph(&graph);
        let next = match apply_patch(&base, patch) {
            Ok(next) => next,
            Err(errors) => {
                anyhow::bail!("ontology patch rejected:\n- {}", errors.join("\n- "));
            }
        };
        let old_hash = base.content_hash();
        let new_hash = next.content_hash();
        let validation_ok = jcode_memory_types::migration::validate_ontology(&next).is_empty();

        let migrations = if dry_run {
            let mut probe = graph.clone();
            apply_migrations(&mut probe, &patch.migrations, &next)
        } else {
            // Keep a bounded history so rollback does not need external files.
            graph.metadata.ontology_history.push(OntologySnapshot::new(
                (*base).clone(),
                patch.rationale.clone(),
            ));
            const MAX_HISTORY: usize = 10;
            if graph.metadata.ontology_history.len() > MAX_HISTORY {
                let drain = graph.metadata.ontology_history.len() - MAX_HISTORY;
                graph.metadata.ontology_history.drain(..drain);
            }
            graph.metadata.migration_state = Some(MigrationState {
                from_version: base.version,
                to_version: next.version,
                started_at: chrono::Utc::now(),
                completed_steps: 0,
                total_steps: patch.migrations.len(),
            });
            let report = apply_migrations(&mut graph, &patch.migrations, &next);
            graph.metadata.migration_state = None;
            graph.metadata.applied_migrations.push(format!(
                "{}:{}->{}",
                patch.base_id, base.version, next.version
            ));
            self.set_effective_ontology_metadata(
                &mut graph,
                &next,
                String::new(),
                String::new(),
                Vec::new(),
            );
            self.save_scope_graph(scope, &graph)?;
            // Only the global ontology becomes the process-wide fallback for
            // stores that carry none. A project-specific patch must not leak
            // into other projects.
            if scope == OntologyScope::Global {
                self.ontology_registry().install(next.clone());
            }
            report
        };

        Ok(OntologyApplyReport {
            store: format!("{}:{}", scope.label(), patch.base_id),
            old_version: base.version,
            new_version: next.version,
            old_hash,
            new_hash,
            dry_run,
            validation_ok,
            migrations,
            patch: patch.clone(),
        })
    }

    /// Apply a project patch by rewriting the repository overlay.
    ///
    /// The effective ontology is the global ontology plus the overlay. Writing
    /// the diff, rather than a full project snapshot, keeps future global changes
    /// flowing for every field the project did not override.
    fn apply_project_ontology_patch(
        &self,
        patch: &jcode_memory_types::OntologyPatch,
        dry_run: bool,
    ) -> Result<jcode_memory_types::OntologyApplyReport> {
        use jcode_memory_types::migration::OntologyOp;
        use jcode_memory_types::migration::{
            MigrationState, OntologyApplyReport, OntologySnapshot, apply_migrations, apply_patch,
        };

        if patch.ops.iter().any(|op| {
            matches!(
                op,
                OntologyOp::AddRule { .. }
                    | OntologyOp::RemoveRule { .. }
                    | OntologyOp::AddActivity { .. }
                    | OntologyOp::RemoveActivity { .. }
            )
        }) {
            anyhow::bail!(
                "project ontology patches cannot change rules or activities; use global scope"
            );
        }

        let parent = self.global_ontology_for_composition()?;
        let overlay = self
            .load_project_ontology_layer()?
            .unwrap_or_else(|| OntologyLayer::new("project/overlay", parent.version));
        let base =
            compose_ontology_layers(&parent, std::slice::from_ref(&overlay)).map_err(|errors| {
                anyhow::anyhow!(
                    "current project ontology layer rejected:\n- {}",
                    errors.join("\n- ")
                )
            })?;
        let next = match apply_patch(&base, patch) {
            Ok(next) => next,
            Err(errors) => {
                anyhow::bail!("ontology patch rejected:\n- {}", errors.join("\n- "));
            }
        };
        let next_overlay = diff_ontology_layer(&parent, &next);
        let old_hash = base.content_hash();
        let new_hash = next.content_hash();
        let validation_ok = jcode_memory_types::migration::validate_ontology(&next).is_empty();
        let parent_hash = parent.content_hash();
        let layer_hash = layer_stack_hash(std::slice::from_ref(&next_overlay));

        let mut graph = self.load_project_graph_inner()?;
        let migrations = if dry_run {
            let mut probe = graph.clone();
            apply_migrations(&mut probe, &patch.migrations, &next)
        } else {
            self.write_project_ontology_layer(&next_overlay)?;
            graph
                .metadata
                .ontology_history
                .push(OntologySnapshot::new(base.clone(), patch.rationale.clone()));
            const MAX_HISTORY: usize = 10;
            if graph.metadata.ontology_history.len() > MAX_HISTORY {
                let drain = graph.metadata.ontology_history.len() - MAX_HISTORY;
                graph.metadata.ontology_history.drain(..drain);
            }
            graph.metadata.migration_state = Some(MigrationState {
                from_version: base.version,
                to_version: next.version,
                started_at: chrono::Utc::now(),
                completed_steps: 0,
                total_steps: patch.migrations.len(),
            });
            let report = apply_migrations(&mut graph, &patch.migrations, &next);
            graph.metadata.migration_state = None;
            graph.metadata.applied_migrations.push(format!(
                "{}:{}->{}",
                patch.base_id, base.version, next.version
            ));
            self.set_effective_ontology_metadata(
                &mut graph,
                &next,
                parent_hash,
                layer_hash,
                vec![next_overlay.id.clone()],
            );
            self.save_project_graph(&graph)?;
            report
        };

        Ok(OntologyApplyReport {
            store: format!("{}:{}", OntologyScope::Project.label(), patch.base_id),
            old_version: base.version,
            new_version: next.version,
            old_hash,
            new_hash,
            dry_run,
            validation_ok,
            migrations,
            patch: patch.clone(),
        })
    }

    /// Automatically reflect session evidence into an ontology patch proposal.
    ///
    /// This is proposal-only: it never calls [`Self::apply_ontology_patch`].
    /// The patch is validated against the project ontology, evidence is
    /// restricted to the supplied memory ids, and only low-risk type parameter
    /// updates are kept. Writes go to `memory/ontology/.proposals/`.
    ///
    /// Returns `Ok(None)` when reflection is disabled, throttled, or produced
    /// no valid change.
    pub async fn propose_ontology_from_session(
        &self,
        transcript: &str,
        session_id: &str,
        memory_ids: &[String],
    ) -> Result<Option<PathBuf>> {
        if !ontology_reflection_enabled() || memory_ids.len() < ONTOLOGY_REFLECTION_MIN_EVIDENCE {
            return Ok(None);
        }
        let graph = self.load_project_graph()?;
        if !self.ontology_reflection_due()? {
            return Ok(None);
        }
        let ontology = self.ontology_for_graph(&graph);
        let wanted: std::collections::BTreeSet<&str> =
            memory_ids.iter().map(String::as_str).collect();
        let evidence: Vec<jcode_memory_types::MemoryEntry> = graph
            .all_memories()
            .filter(|entry| wanted.contains(entry.id.as_str()))
            .cloned()
            .collect();
        if evidence.len() < ONTOLOGY_REFLECTION_MIN_EVIDENCE {
            return Ok(None);
        }

        let sidecar = crate::session::Session::load_startup_stub(session_id)
            .map(|session| Sidecar::for_session(&session))
            .unwrap_or_else(|_| Sidecar::new());
        let proposed = sidecar
            .propose_ontology_patch(&ontology, &evidence, transcript)
            .await?;
        let Some(patch) = proposed else {
            self.mark_ontology_reflection_run()?;
            return Ok(None);
        };
        let Some(patch) = prepare_reflection_patch(&ontology, patch, memory_ids) else {
            self.mark_ontology_reflection_run()?;
            return Ok(None);
        };

        let path = write_ontology_proposal(&patch)?;
        self.mark_ontology_reflection_run()?;
        Ok(Some(path))
    }

    fn ontology_reflection_due(&self) -> Result<bool> {
        let project_dir = self
            .get_project_dir()
            .unwrap_or_else(|| PathBuf::from("no-project"));
        let path = ontology_reflection_state_path(&project_dir)?;
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return Ok(true);
        };
        let Ok(state) = serde_json::from_str::<OntologyReflectionState>(&raw) else {
            return Ok(true);
        };
        Ok(Utc::now() - state.last_run
            >= chrono::Duration::hours(ONTOLOGY_REFLECTION_INTERVAL_HOURS))
    }

    fn mark_ontology_reflection_run(&self) -> Result<()> {
        let project_dir = self
            .get_project_dir()
            .unwrap_or_else(|| PathBuf::from("no-project"));
        let path = ontology_reflection_state_path(&project_dir)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let state = OntologyReflectionState {
            last_run: Utc::now(),
        };
        std::fs::write(&path, serde_json::to_vec_pretty(&state)?)?;
        Ok(())
    }

    /// Restore an ontology snapshot from a store's history.
    ///
    /// Only the ontology definition is restored. Data migrations are not
    /// reversed; use a store backup when a migration itself must be undone.
    pub fn rollback_ontology(
        &self,
        scope: OntologyScope,
        version: u32,
    ) -> Result<jcode_memory_types::OntologySnapshot> {
        if scope == OntologyScope::Project {
            return self.rollback_project_ontology(version);
        }
        use jcode_memory_types::OntologySnapshot;

        let mut graph = self.load_scope_graph(scope)?;
        let snapshot = graph
            .metadata
            .ontology_history
            .iter()
            .rev()
            .find(|snapshot| snapshot.version == version)
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no ontology version {version} in {} history; available: {:?}",
                    scope.label(),
                    graph
                        .metadata
                        .ontology_history
                        .iter()
                        .map(|snapshot| snapshot.version)
                        .collect::<Vec<_>>()
                )
            })?;

        if let Some(current) = graph.metadata.ontology.clone() {
            graph.metadata.ontology_history.push(OntologySnapshot::new(
                current,
                format!("before rollback to v{version}"),
            ));
        }
        graph
            .metadata
            .ontology_history
            .retain(|entry| entry.version != version);
        let restored = snapshot.ontology.clone();
        self.set_effective_ontology_metadata(
            &mut graph,
            &restored,
            String::new(),
            String::new(),
            Vec::new(),
        );
        self.save_scope_graph(scope, &graph)?;
        if scope == OntologyScope::Global {
            self.ontology_registry().install(restored);
        }
        Ok(snapshot)
    }

    fn rollback_project_ontology(
        &self,
        version: u32,
    ) -> Result<jcode_memory_types::OntologySnapshot> {
        use jcode_memory_types::OntologySnapshot;

        let mut graph = self.load_project_graph_inner()?;
        let snapshot = graph
            .metadata
            .ontology_history
            .iter()
            .rev()
            .find(|snapshot| snapshot.version == version)
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no ontology version {version} in project history; available: {:?}",
                    graph
                        .metadata
                        .ontology_history
                        .iter()
                        .map(|snapshot| snapshot.version)
                        .collect::<Vec<_>>()
                )
            })?;

        let parent = self.global_ontology_for_composition()?;
        let overlay = diff_ontology_layer(&parent, &snapshot.ontology);
        self.write_project_ontology_layer(&overlay)?;
        let layers = if layer_is_empty(&overlay) {
            Vec::new()
        } else {
            vec![overlay]
        };
        let restored = if layers.is_empty() {
            (*parent).clone()
        } else {
            compose_ontology_layers(&parent, &layers).map_err(|errors| {
                anyhow::anyhow!(
                    "rolled-back project ontology rejected:\n- {}",
                    errors.join("\n- ")
                )
            })?
        };
        let parent_hash = parent.content_hash();
        let layer_hash = layer_stack_hash(&layers);
        let layer_ids = layers.iter().map(|layer| layer.id.clone()).collect();

        if let Some(current) = graph.metadata.ontology.clone() {
            graph.metadata.ontology_history.push(OntologySnapshot::new(
                current,
                format!("before rollback to v{version}"),
            ));
        }
        graph
            .metadata
            .ontology_history
            .retain(|entry| entry.version != version);
        self.set_effective_ontology_metadata(
            &mut graph,
            &restored,
            parent_hash,
            layer_hash,
            layer_ids,
        );
        self.save_project_graph(&graph)?;
        Ok(snapshot)
    }

    pub fn forget(&self, id: &str) -> Result<bool> {
        // Try graph-based removal first (new format)
        let mut project_graph = self.load_project_graph()?;
        if project_graph.remove_memory(id).is_some() {
            self.save_project_graph(&project_graph)?;
            return Ok(true);
        }

        let mut global_graph = self.load_global_graph()?;
        if global_graph.remove_memory(id).is_some() {
            self.save_global_graph(&global_graph)?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Returns true if a memory with the given id exists in the requested
    /// scope. Used by the memory tool to reject `remember` calls that would
    /// otherwise silently overwrite an existing entry via `add_memory`'s
    /// `HashMap::insert` semantics.
    pub fn has_memory(&self, id: &str, scope: &str) -> Result<bool> {
        match scope {
            "project" => Ok(self.load_project_graph()?.get_memory(id).is_some()),
            "global" => Ok(self.load_global_graph()?.get_memory(id).is_some()),
            _ => Err(anyhow::anyhow!(
                "Unknown scope for has_memory: {}. Use project or global",
                scope
            )),
        }
    }

    // === Sidecar Integration ===

    /// Extract memories from a session transcript using the Haiku sidecar
    pub async fn extract_from_transcript(
        &self,
        transcript: &str,
        session_id: &str,
    ) -> Result<Vec<String>> {
        let sidecar = crate::session::Session::load_startup_stub(session_id)
            .map(|session| Sidecar::for_session(&session))
            .unwrap_or_else(|_| Sidecar::new());
        self.extract_with_sidecar(transcript, session_id, &sidecar)
            .await
    }

    /// Extract through a caller-supplied sidecar.
    ///
    /// Live session finalization already holds the exact sidecar/provider for
    /// the session; passing it through avoids re-resolving credentials from a
    /// startup stub.
    pub async fn extract_with_sidecar(
        &self,
        transcript: &str,
        session_id: &str,
        sidecar: &Sidecar,
    ) -> Result<Vec<String>> {
        if !memory_llm_judge_available() {
            crate::logging::info("Memory transcript extraction skipped: LLM judge unavailable");
            return Ok(Vec::new());
        }
        if self.extraction_is_paused() {
            crate::logging::info("Memory transcript extraction skipped: source health is paused");
            return Ok(Vec::new());
        }
        // The effective project ontology supplies the extra type catalog for
        // the extractor. A load failure must not block extraction; fall back to
        // the static core type list in that case.
        let graph = self.load_project_graph().ok();
        let ontology = graph.as_ref().map(|graph| self.ontology_for_graph(graph));
        let existing_counts = graph
            .as_ref()
            .map(|graph| {
                let mut counts = std::collections::HashMap::new();
                for entry in graph.active_memories() {
                    *counts.entry(entry.category.clone()).or_insert(0usize) += 1;
                }
                counts
            })
            .unwrap_or_default();
        let extracted = sidecar
            .extract_memories_with_ontology(transcript, &[], ontology.as_deref())
            .await?;

        let mut ids = Vec::new();
        let candidates = extracted.len();
        let candidate_categories: Vec<MemoryCategory> = extracted
            .iter()
            .map(|memory| memory.category.parse().unwrap_or(MemoryCategory::Fact))
            .collect();
        let batch_gated_categories =
            batch_change_categories(&candidate_categories, &existing_counts);
        let mut unique_contents = std::collections::HashSet::new();
        let mut confidence_sum = 0.0f32;
        let mut admitted = 0usize;
        let mut queued = 0usize;
        let mut failed = 0usize;
        for (memory, category) in extracted.into_iter().zip(candidate_categories) {
            let batch_gated = batch_gated_categories.contains(&category);
            let trust = match memory.trust.as_str() {
                "high" => TrustLevel::High,
                "medium" => TrustLevel::Medium,
                _ => TrustLevel::Low,
            };

            let confidence = trust.provenance_confidence();
            confidence_sum += confidence;
            unique_contents.insert(normalize_memory_search_text(&memory.content, &[]));
            let method = if trust == TrustLevel::High {
                ExtractionMethod::UserStated
            } else {
                ExtractionMethod::LlmExtraction
            };
            let locator = memory
                .locator
                .clone()
                .unwrap_or_else(|| format!("session:{session_id}"));
            let mut entry = MemoryEntry::new(category, memory.content)
                .with_source(session_id)
                .with_trust(trust)
                .with_provenance(
                    ProvenanceRecord::new(session_id, method)
                        .with_confidence(confidence)
                        .with_locator(locator),
                );
            for (key, value) in memory.properties {
                entry = entry.with_property(key, value);
            }

            // Store in project scope by default. A single bad candidate must
            // not abort the batch: the failure rate feeds source health.
            let outcome = if batch_gated {
                self.remember_project_governed_with_policy(
                    entry,
                    &[ConfirmationGate::BatchChange],
                    Some("extraction batch exceeds the G4 bulk-change threshold for this category"),
                )
            } else {
                self.remember_project_governed(entry)
            };
            match outcome {
                Ok(MemoryWriteOutcome::Stored { id, .. }) => {
                    ids.push(id);
                    admitted += 1;
                }
                Ok(MemoryWriteOutcome::PendingReview { .. }) => {
                    queued += 1;
                }
                Err(error) => {
                    failed += 1;
                    crate::logging::warn(&format!("Memory extraction candidate rejected: {error}"));
                }
            }
        }

        let duplicate_ratio = if candidates == 0 {
            0.0
        } else {
            (candidates - unique_contents.len()) as f32 / candidates as f32
        };
        let avg_confidence = if candidates == 0 {
            0.0
        } else {
            confidence_sum / candidates as f32
        };
        if let Ok(health) = self.record_extraction_batch(
            session_id,
            candidates,
            admitted,
            queued,
            failed,
            avg_confidence,
            duplicate_ratio,
        ) && health.paused
        {
            crate::logging::warn(&format!(
                "Memory extraction paused: {}",
                health
                    .pause_reason
                    .as_deref()
                    .unwrap_or("source health degraded")
            ));
        }

        Ok(ids)
    }

    /// Recall directly through Jev. The legacy `max_candidates` argument now
    /// limits output, not the input pool: old memories must remain discoverable.
    pub async fn get_relevant_for_context(
        &self,
        context: &str,
        max_candidates: usize,
    ) -> Result<Vec<MemoryEntry>> {
        let route = route_memory_query(context);
        let mut entries: Vec<MemoryEntry> =
            crate::memory_jev::recall(self, context, max_candidates, MemoryScope::All)
                .await?
                .into_iter()
                .map(|(entry, _)| entry)
                .collect();
        if matches!(route, MemoryRoute::Procedure | MemoryRoute::Operation) {
            // Stable sort: skills first, preserve relevance order within each
            // group. Route selection is a cheap prior, not a hard filter.
            entries.sort_by_key(|entry| !jcode_memory_types::is_skill_memory(entry));
        }
        crate::logging::info(&format!(
            "Memory route={} candidates={} for query: {}",
            route.label(),
            entries.len(),
            crate::util::truncate_str(context, 120)
        ));
        Ok(self.expand_skill_parts(entries, max_candidates))
    }

    /// Local keyword lookup, available without a remote decision provider.
    pub fn get_relevant_keywords(
        &self,
        keywords: &[&str],
        limit: usize,
    ) -> Result<Vec<MemoryEntry>> {
        let normalized_keywords: Vec<String> = keywords
            .iter()
            .map(|keyword| normalize_search_text(keyword))
            .filter(|keyword| !keyword.is_empty())
            .collect();
        if normalized_keywords.is_empty() {
            return Ok(Vec::new());
        }

        let matches: Vec<_> = top_k_by_ord(
            self.collect_memories_scoped(MemoryScope::All)?
                .into_iter()
                .filter(|entry| {
                    let content_lower = normalize_search_text(&entry.content);
                    normalized_keywords
                        .iter()
                        .any(|kw| content_lower.contains(kw))
                })
                .map(|entry| {
                    let updated_at = entry.updated_at.timestamp_millis();
                    (entry, updated_at)
                }),
            limit,
        )
        .into_iter()
        .map(|(entry, _)| entry)
        .collect();

        Ok(matches)
    }

    // === Async Memory Checking ===

    /// Spawn a background task to check memory relevance for a specific session.
    /// Results are stored in PENDING_MEMORY keyed by session_id and can be retrieved
    /// with take_pending_memory(session_id).
    /// This method returns immediately and never blocks the caller.
    /// Only ONE memory check runs at a time per session - additional calls are ignored.
    pub fn spawn_relevance_check(
        &self,
        session_id: &str,
        messages: std::sync::Arc<[crate::message::Message]>,
        event_tx: Option<MemoryEventSink>,
    ) {
        let sid = session_id.to_string();

        if !begin_memory_check(&sid) {
            return;
        }

        let manager = self.clone();

        tokio::spawn(async move {
            match manager
                .get_relevant_parallel(&sid, &messages, event_tx.clone())
                .await
            {
                Ok(MemoryRelevanceResult {
                    prompt: Some(prompt),
                    display_prompt,
                    selected_entries,
                }) => {
                    let count = selected_entries.len();
                    set_pending_memory_for_project_with_selection(
                        &sid,
                        prompt,
                        count,
                        &selected_entries,
                        display_prompt,
                        manager
                            .project_dir
                            .as_deref()
                            .and_then(|path| path.to_str()),
                    );
                    emit_memory_activity(event_tx.as_ref());
                }
                Ok(MemoryRelevanceResult { prompt: None, .. }) => {
                    clear_pending_memory(&sid);
                    set_state(MemoryState::Idle);
                    emit_memory_activity(event_tx.as_ref());
                }
                Err(e) => {
                    clear_pending_memory(&sid);
                    crate::logging::error(&format!("Background memory check failed: {}", e));
                    add_event(MemoryEventKind::Error {
                        message: e.to_string(),
                    });
                    set_state(MemoryState::Idle);
                    emit_memory_activity(event_tx.as_ref());
                }
            }

            finish_memory_check(&sid);
        });
    }

    /// Automatic recall is globally disabled. Explicit recall remains available.
    pub async fn get_relevant_parallel(
        &self,
        session_id: &str,
        _messages: &[crate::message::Message],
        _event_tx: Option<MemoryEventSink>,
    ) -> Result<MemoryRelevanceResult> {
        // Automatic recall is disabled for everyone, including existing configs.
        // Explicit memory-tool recall still uses memory_jev::recall directly.
        clear_pending_memory(session_id);
        Ok(MemoryRelevanceResult::default())
    }

    /// Retained Jev relevance pipeline for potential future opt-in recall.
    #[allow(dead_code)]
    pub(crate) async fn get_relevant_parallel_enabled(
        &self,
        session_id: &str,
        messages: &[crate::message::Message],
        event_tx: Option<MemoryEventSink>,
    ) -> Result<MemoryRelevanceResult> {
        let query = format_focused_query_for_relevance(messages);
        let query = crate::util::truncate_str(&query, crate::memory_jev::MAX_QUERY_BYTES);
        if query.trim().is_empty() {
            return Ok(MemoryRelevanceResult::default());
        }
        pipeline_start();
        let entries = match crate::memory_jev::collect_scoped(self, MemoryScope::All) {
            Ok(entries) => entries,
            Err(error) => {
                clear_pending_memory(session_id);
                pipeline_update(|p| {
                    p.search = StepStatus::Error;
                    p.verify = StepStatus::Skipped;
                    p.inject = StepStatus::Skipped;
                });
                set_state(MemoryState::Idle);
                emit_memory_activity(event_tx.as_ref());
                return Err(error);
            }
        };
        let entries: Vec<_> = entries
            .into_iter()
            .filter(|entry| {
                entry.is_retrievable_now() && !is_memory_injected(session_id, &entry.id)
            })
            .collect();
        let entries = prefilter_for_jev(entries, query);
        pipeline_update(|p| {
            p.search = StepStatus::Done;
            p.search_result = Some(StepResult {
                summary: format!("{} local memories", entries.len()),
                latency_ms: 0,
            });
            p.verify = StepStatus::Running;
            p.maintain = StepStatus::Skipped;
        });
        set_state(MemoryState::SidecarChecking {
            count: entries.len(),
        });
        emit_memory_activity(event_tx.as_ref());
        let started = Instant::now();
        let result = async {
            if entries.is_empty() {
                return Ok(Vec::new());
            }
            let client = crate::jev::JevClient::new()?;
            crate::memory_jev::select(&client, query, entries, 5).await
        }
        .await;
        let relevant: Vec<MemoryEntry> = match result {
            Ok(results) => results.into_iter().map(|(entry, _)| entry).collect(),
            Err(error) => {
                clear_pending_memory(session_id);
                pipeline_update(|p| {
                    p.verify = StepStatus::Error;
                    p.inject = StepStatus::Skipped;
                });
                set_state(MemoryState::Idle);
                emit_memory_activity(event_tx.as_ref());
                return Err(error);
            }
        };
        let count = relevant.len();
        pipeline_update(|p| {
            p.verify = StepStatus::Done;
            p.verify_result = Some(StepResult {
                summary: format!("Jev: {count} relevant"),
                latency_ms: started.elapsed().as_millis() as u64,
            });
            p.inject = if count == 0 {
                StepStatus::Skipped
            } else {
                StepStatus::Pending
            };
        });
        let prompt = format_relevant_prompt(&relevant, 5);
        let display = format_relevant_display_prompt(&relevant, 5);
        set_state(if count == 0 {
            MemoryState::Idle
        } else {
            MemoryState::FoundRelevant { count }
        });
        emit_memory_activity(event_tx.as_ref());
        Ok(MemoryRelevanceResult {
            prompt,
            display_prompt: display,
            selected_entries: relevant,
        })
    }

    // ==================== Graph-Based Operations ====================

    /// Load the legacy JSON project graph for a one-time migration into
    /// the sqlite backend, if such a graph exists and holds content.
    fn load_legacy_project_json_for_migration(&self) -> Result<Option<MemoryGraph>> {
        let Some(path) = self.project_memory_path()? else {
            return Ok(None);
        };
        Self::read_legacy_graph_for_migration(&path)
    }

    /// Load the legacy JSON global graph for a one-time migration into
    /// the sqlite backend, if such a graph exists and holds content.
    fn load_legacy_global_json_for_migration(&self) -> Result<Option<MemoryGraph>> {
        Self::read_legacy_graph_for_migration(&self.global_memory_path()?)
    }

    /// Read a legacy JSON graph snapshot for migration. Returns `None`
    /// when the file is absent, unreadable, or has no content. On a
    /// successful read the source file is renamed to `*.json.migrated`
    /// (kept as a backup) so a later empty sqlite store does not
    /// re-import and resurrect memories the user deletes after switching
    /// to the sqlite backend.
    fn read_legacy_graph_for_migration(path: &Path) -> Result<Option<MemoryGraph>> {
        if !path.exists() {
            return Ok(None);
        }
        let graph = match storage::read_json::<MemoryGraph>(path) {
            Ok(g) if g.graph_version == GRAPH_VERSION => g,
            _ => match storage::read_json::<MemoryStore>(path) {
                Ok(store) => MemoryGraph::from_legacy_store(store),
                Err(_) => return Ok(None),
            },
        };
        let has_content = !graph.memories.is_empty()
            || !graph.tags.is_empty()
            || !graph.clusters.is_empty()
            || !graph.edges.is_empty();
        if !has_content {
            return Ok(None);
        }
        let backup = path.with_extension("json.migrated");
        if std::fs::rename(path, &backup).is_err() {
            // Keep the original in place if the rename fails; the
            // migration still applies from the in-memory graph.
            crate::logging::info(&format!(
                "kept legacy JSON at {} (could not rename to {})",
                path.display(),
                backup.display()
            ));
        }
        Ok(Some(graph))
    }

    /// Stamp the active ontology onto `graph` and warn when the stored binding
    /// disagrees with it.
    ///
    /// Also bumps `retrieval_count` so an eagerly-loaded store records that it
    /// was read - the counter existed in `GraphMetadata` from the start but had
    /// no writer.
    fn bind_and_check_ontology(
        &self,
        key: &StoreKey,
        backend: &std::sync::Arc<dyn jcode_memory_types::GraphBackend>,
        graph: &mut MemoryGraph,
    ) {
        if self.test_mode {
            return;
        }
        if self.bind_ontology_metadata(key.as_str(), graph)
            && let Err(e) = backend.save(key, graph)
        {
            crate::logging::warn(&format!("ontology binding persist failed: {e}"));
        }
    }

    /// Bind the stored or compiled ontology into `graph.metadata`.
    ///
    /// Returns true when the metadata changed and needs to be persisted. The
    /// caller decides how (backend save, JSON write, cache update).
    fn bind_ontology_metadata(&self, key: &str, graph: &mut MemoryGraph) -> bool {
        graph.metadata.retrieval_count = graph.metadata.retrieval_count.saturating_add(1);
        let mut changed = false;
        let default = self.ontology_registry().default_ontology();

        // A core-only default snapshot is redundant: the compiled definition
        // is already the fallback. Drop it so stores do not carry a stale copy
        // of descriptions, rules or core classes.
        let stored_is_default = graph
            .metadata
            .ontology
            .as_ref()
            .is_some_and(|stored| self.is_core_only_default(stored));
        if stored_is_default {
            graph.metadata.ontology = None;
            changed = true;
        }

        let (id, version, hash) = match graph.metadata.ontology.as_ref() {
            Some(stored) => (
                stored.id.clone(),
                stored.version,
                jcode_memory_types::migration::ontology_content_hash(stored),
            ),
            None => (default.id.clone(), default.version, default.content_hash()),
        };
        if graph.metadata.ontology_hash != hash {
            if graph.metadata.ontology.is_some() {
                crate::logging::warn(&format!(
                    "Memory ontology hash mismatch for {key}: metadata {} -> {hash}",
                    graph.metadata.ontology_hash
                ));
            }
            graph.metadata.ontology_hash = hash;
            changed = true;
        }
        if graph.metadata.ontology_id != id {
            graph.metadata.ontology_id = id;
            changed = true;
        }
        if graph.metadata.ontology_version != version {
            graph.metadata.ontology_version = version;
            changed = true;
        }

        changed
    }

    /// Whether `ontology` is semantically the compiled default, even when it
    /// still carries an older description or a redundant default snapshot.
    fn is_core_only_default(&self, ontology: &jcode_memory_types::Ontology) -> bool {
        let default = self.ontology_registry().default_ontology();
        ontology.id == default.id
            && ontology.types.is_empty()
            && ontology.category_aliases.is_empty()
            && ontology.relationships == default.relationships
            && ontology.rules == default.rules
            && ontology.activities == default.activities
    }

    /// Store the effective ontology as a reference or a materialized snapshot.
    ///
    /// The compiled default is represented by `ontology = None` plus
    /// id/version/hash metadata. Only non-default effective ontologies need a
    /// full snapshot. Returns true when any metadata field changed.
    fn set_effective_ontology_metadata(
        &self,
        graph: &mut MemoryGraph,
        effective: &jcode_memory_types::Ontology,
        parent_hash: String,
        layer_hash: String,
        layer_ids: Vec<String>,
    ) -> bool {
        let default = self.ontology_registry().default_ontology();
        let is_default = effective == &*default;
        let snapshot = if is_default {
            None
        } else {
            Some(effective.clone())
        };
        let hash = effective.content_hash();
        let mut changed = false;
        if graph.metadata.ontology != snapshot {
            graph.metadata.ontology = snapshot;
            changed = true;
        }
        if graph.metadata.ontology_id != effective.id {
            graph.metadata.ontology_id = effective.id.clone();
            changed = true;
        }
        if graph.metadata.ontology_version != effective.version {
            graph.metadata.ontology_version = effective.version;
            changed = true;
        }
        if graph.metadata.ontology_hash != hash {
            graph.metadata.ontology_hash = hash;
            changed = true;
        }
        if graph.metadata.ontology_parent_hash != parent_hash {
            graph.metadata.ontology_parent_hash = parent_hash;
            changed = true;
        }
        if graph.metadata.ontology_layer_hash != layer_hash {
            graph.metadata.ontology_layer_hash = layer_hash;
            changed = true;
        }
        if graph.metadata.ontology_layer_ids != layer_ids {
            graph.metadata.ontology_layer_ids = layer_ids;
            changed = true;
        }
        changed
    }

    /// Bind the ontology into a JSON-backed graph and report whether the
    /// caller must write it back.
    fn bind_json_ontology(&self, path: &std::path::Path, graph: &mut MemoryGraph) -> bool {
        if self.test_mode {
            return false;
        }
        self.bind_ontology_metadata(&path.display().to_string(), graph)
    }

    /// Load project memories as a MemoryGraph without generating embeddings.
    ///
    /// Applies the repository-owned ontology overlay before returning. A broken
    /// overlay is logged and ignored so memory access never depends on a
    /// project file being well formed.
    pub fn load_project_graph(&self) -> Result<MemoryGraph> {
        let mut graph = self.load_project_graph_inner()?;
        if let Err(err) = self.compose_project_ontology(&mut graph) {
            crate::logging::warn(&format!("Project ontology overlay ignored: {err}"));
        }
        Ok(graph)
    }

    fn load_project_graph_inner(&self) -> Result<MemoryGraph> {
        // When the sqlite-gvec backend is active, route everything
        // through the trait. The legacy JSON path is preserved below
        // for users on the default backend or when the trait open
        // fails for any reason.
        if crate::memory::active_backend_name() == "sqlite-gvec" {
            let backend = if self.test_mode {
                crate::memory::test_backend()
            } else {
                crate::memory::graph_backend()
            };
            let key = StoreKey::new(match self.get_project_dir() {
                Some(d) => crate::memory::project_store_key(&d),
                None => "project:none".to_string(),
            });
            match backend.load(&key) {
                Ok(mut graph) => {
                    // One-time migration from the legacy JSON snapshot:
                    // when switching the default backend to sqlite, an
                    // empty sqlite store would otherwise hide existing
                    // JSON memory. Import only when the sqlite store is
                    // empty and a legacy graph exists; the source file
                    // is renamed so a later empty store does not
                    // resurrect memories the user deletes after the
                    // switch.
                    if graph.memory_count() == 0
                        && graph.tags.is_empty()
                        && graph.clusters.is_empty()
                        && graph.edges.is_empty()
                        && let Some(legacy) = self.load_legacy_project_json_for_migration()?
                    {
                        graph = legacy;
                        let _ = backend.save(&key, &graph);
                    }
                    if Self::normalize_graph_search_text(&mut graph) {
                        let _ = backend.save(&key, &graph);
                    }
                    if self.import_legacy_notes_into_graph(&mut graph)? {
                        let _ = backend.save(&key, &graph);
                    }
                    // Ontology binding + drift detection. Cheap here (the graph
                    // was just allocated) and it makes the binding durable, which
                    // is what lets a later load notice that the ontology the
                    // store was authored against has changed.
                    self.bind_and_check_ontology(&key, &backend, &mut graph);
                    if !self.test_mode {
                        cache_graph_for_backend(backend.name(), &key, &graph, 0);
                    }
                    return Ok(graph);
                }
                Err(e) => {
                    crate::logging::warn(&format!(
                        "load_project_graph via {e}; returning the legacy JSON snapshot read-only"
                    ));
                    // Fall through to legacy JSON, but never save this snapshot
                    // back: it is staler than whatever the sqlite store holds
                    // (the sqlite row read failed, not the store), and
                    // persisting that stale graph would roll the whole store
                    // back — measured: it erased generalized rules and
                    // skill-invocation records written after the last JSON
                    // export.
                    self.suppress_next_legacy_save
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }

        let Some(path) = self.project_memory_path()? else {
            return Ok(MemoryGraph::new());
        };

        if !self.test_mode
            && let Some(mut graph) = cached_graph(&path)
        {
            let normalized = Self::normalize_graph_search_text(&mut graph);
            let bound = self.bind_json_ontology(&path, &mut graph);
            if normalized || bound {
                if bound {
                    storage::write_json(&path, &graph)?;
                }
                cache_graph(path.clone(), &graph);
            }
            return Ok(graph);
        }

        if path.exists() {
            // Try loading as MemoryGraph first
            if let Ok(graph) = storage::read_json::<MemoryGraph>(&path)
                && graph.graph_version == GRAPH_VERSION
            {
                let mut graph = graph;
                let normalized = Self::normalize_graph_search_text(&mut graph);
                let imported = self.import_legacy_notes_into_graph(&mut graph)?;
                let bound = self.bind_json_ontology(&path, &mut graph);
                if imported {
                    self.save_project_graph(&graph)?;
                } else if normalized || bound {
                    storage::write_json(&path, &graph)?;
                }
                if !self.test_mode {
                    cache_graph(path, &graph);
                }
                return Ok(graph);
            }

            // Fall back to legacy MemoryStore and migrate
            let store: MemoryStore = storage::read_json(&path)?;
            let mut graph = MemoryGraph::from_legacy_store(store);
            let _ = self.import_legacy_notes_into_graph(&mut graph)?;
            let _ = self.bind_json_ontology(&path, &mut graph);

            // Save migrated format (create backup first)
            let backup_path = path.with_extension("json.bak");
            if !backup_path.exists() {
                let _ = std::fs::copy(&path, &backup_path);
            }
            storage::write_json(&path, &graph)?;

            crate::logging::info(&format!(
                "Migrated memory store to graph format: {}",
                path.display()
            ));
            if !self.test_mode {
                cache_graph(path, &graph);
            }
            Ok(graph)
        } else {
            let mut graph = MemoryGraph::new();
            let imported = self.import_legacy_notes_into_graph(&mut graph)?;
            let bound = self.bind_json_ontology(&path, &mut graph);
            if imported || bound {
                self.save_project_graph(&graph)?;
            }
            if !self.test_mode {
                cache_graph(path, &graph);
            }
            Ok(graph)
        }
    }

    /// Load global memories as a MemoryGraph with automatic migration
    pub fn load_global_graph(&self) -> Result<MemoryGraph> {
        if crate::memory::active_backend_name() == "sqlite-gvec" {
            let backend = if self.test_mode {
                crate::memory::test_backend()
            } else {
                crate::memory::graph_backend()
            };
            let key = StoreKey::new(crate::memory::global_store_key());
            match backend.load(&key) {
                Ok(mut graph) => {
                    // One-time migration from the legacy JSON snapshot,
                    // mirroring the project path above.
                    if graph.memory_count() == 0
                        && graph.tags.is_empty()
                        && graph.clusters.is_empty()
                        && graph.edges.is_empty()
                        && let Some(legacy) = self.load_legacy_global_json_for_migration()?
                    {
                        graph = legacy;
                        let _ = backend.save(&key, &graph);
                    }
                    if Self::normalize_graph_search_text(&mut graph) {
                        let _ = backend.save(&key, &graph);
                    }
                    self.bind_and_check_ontology(&key, &backend, &mut graph);
                    if !self.test_mode {
                        cache_graph_for_backend(backend.name(), &key, &graph, 0);
                    }
                    return Ok(graph);
                }
                Err(e) => {
                    crate::logging::warn(&format!(
                        "load_global_graph via {e}; returning the legacy JSON snapshot read-only"
                    ));
                    self.suppress_next_legacy_save
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }

        let path = self.global_memory_path()?;
        if !self.test_mode
            && let Some(mut graph) = cached_graph(&path)
        {
            let normalized = Self::normalize_graph_search_text(&mut graph);
            let bound = self.bind_json_ontology(&path, &mut graph);
            if normalized || bound {
                if bound {
                    storage::write_json(&path, &graph)?;
                }
                cache_graph(path.clone(), &graph);
            }
            return Ok(graph);
        }

        if path.exists() {
            // Try loading as MemoryGraph first
            if let Ok(graph) = storage::read_json::<MemoryGraph>(&path)
                && graph.graph_version == GRAPH_VERSION
            {
                let mut graph = graph;
                let normalized = Self::normalize_graph_search_text(&mut graph);
                let bound = self.bind_json_ontology(&path, &mut graph);
                if normalized || bound {
                    storage::write_json(&path, &graph)?;
                }
                if !self.test_mode {
                    cache_graph(path, &graph);
                }
                return Ok(graph);
            }

            // Fall back to legacy MemoryStore and migrate
            let store: MemoryStore = storage::read_json(&path)?;
            let mut graph = MemoryGraph::from_legacy_store(store);
            let _ = self.bind_json_ontology(&path, &mut graph);

            // Save migrated format (create backup first)
            let backup_path = path.with_extension("json.bak");
            if !backup_path.exists() {
                let _ = std::fs::copy(&path, &backup_path);
            }
            storage::write_json(&path, &graph)?;

            crate::logging::info(&format!(
                "Migrated global memory store to graph format: {}",
                path.display()
            ));
            if !self.test_mode {
                cache_graph(path, &graph);
            }
            Ok(graph)
        } else {
            let mut graph = MemoryGraph::new();
            if self.bind_json_ontology(&path, &mut graph) {
                storage::write_json(&path, &graph)?;
            }
            if !self.test_mode {
                cache_graph(path, &graph);
            }
            Ok(graph)
        }
    }

    /// Save project memories as a MemoryGraph
    pub fn save_project_graph(&self, graph: &MemoryGraph) -> Result<()> {
        if crate::memory::active_backend_name() == "sqlite-gvec" {
            let backend = if self.test_mode {
                crate::memory::test_backend()
            } else {
                crate::memory::graph_backend()
            };
            let key = StoreKey::new(match self.get_project_dir() {
                Some(d) => crate::memory::project_store_key(&d),
                None => "project:none".to_string(),
            });
            if let Err(e) = backend.save(&key, graph) {
                // Same suppression as the load fallback: a manager that served
                // its graph from the staler JSON snapshot must not persist it.
                if self
                    .suppress_next_legacy_save
                    .swap(false, std::sync::atomic::Ordering::Relaxed)
                {
                    crate::logging::warn(
                        "save_project_graph suppressed: legacy JSON snapshot may be staler than the sqlite store",
                    );
                    return Ok(());
                }
                let recovery = self.write_recovery_graph(graph, true)?;
                anyhow::bail!(
                    "save_project_graph failed: {e}; recovery snapshot saved to {}",
                    recovery
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "<none>".to_string())
                );
            }
            if !self.test_mode {
                cache_graph_for_backend(backend.name(), &key, graph, 0);
            }
            return Ok(());
        }
        self.save_project_graph_json(graph)
    }

    /// Internal: write the project graph to the legacy JSON snapshot.
    /// Kept separate so `save_project_graph` can route to either the
    /// trait backend or this path without duplicating the validation /
    /// cache-update code below.
    fn save_project_graph_json(&self, graph: &MemoryGraph) -> Result<()> {
        if let Some(path) = self.project_memory_path()? {
            let report = graph.validate();
            if !report.is_valid() {
                crate::logging::info(&format!(
                    "Memory graph validation found {} error(s) before saving {}: {:?}",
                    report.errors().len(),
                    path.display(),
                    report.errors()
                ));
            }
            storage::write_json(&path, graph)?;
            if !self.test_mode {
                cache_graph(path, graph);
            }
        }
        Ok(())
    }

    fn project_recovery_path(&self) -> Result<Option<PathBuf>> {
        if self.test_mode {
            let dir = storage::jcode_dir()?.join("memory").join("test").join("recovery");
            return Ok(Some(dir.join("test_project.json")));
        }
        let Some(project_dir) = self.get_project_dir() else {
            return Ok(None);
        };
        project_recovery_file(&project_dir).map(Some)
    }

    fn global_recovery_path(&self) -> Result<PathBuf> {
        Ok(storage::jcode_dir()?
            .join("memory")
            .join("recovery")
            .join("global.json"))
    }

    /// Persist a failed backend write outside the canonical JSON store.
    fn write_recovery_graph(
        &self,
        graph: &MemoryGraph,
        project: bool,
    ) -> Result<Option<PathBuf>> {
        let path = if project {
            self.project_recovery_path()?
        } else {
            Some(self.global_recovery_path()?)
        };
        let Some(path) = path else {
            return Ok(None);
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        storage::write_json(&path, graph)?;
        crate::logging::warn(&format!(
            "memory backend save failed; recovery snapshot written to {}",
            path.display()
        ));
        Ok(Some(path))
    }

    /// Save global memories as a MemoryGraph
    pub fn save_global_graph(&self, graph: &MemoryGraph) -> Result<()> {
        if crate::memory::active_backend_name() == "sqlite-gvec" {
            let backend = if self.test_mode {
                crate::memory::test_backend()
            } else {
                crate::memory::graph_backend()
            };
            let key = StoreKey::new(crate::memory::global_store_key());
            if let Err(e) = backend.save(&key, graph) {
                // Same suppression as the load fallback (see
                // `save_project_graph`).
                if self
                    .suppress_next_legacy_save
                    .swap(false, std::sync::atomic::Ordering::Relaxed)
                {
                    crate::logging::warn(
                        "save_global_graph suppressed: legacy JSON snapshot may be staler than the sqlite store",
                    );
                    return Ok(());
                }
                let recovery = self.write_recovery_graph(graph, false)?;
                anyhow::bail!(
                    "save_global_graph failed: {e}; recovery snapshot saved to {}",
                    recovery
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "<none>".to_string())
                );
            }
            if !self.test_mode {
                cache_graph_for_backend(backend.name(), &key, graph, 0);
            }
            return Ok(());
        }
        self.save_global_graph_json(graph)
    }

    /// Internal: write the global graph to the legacy JSON snapshot.
    fn save_global_graph_json(&self, graph: &MemoryGraph) -> Result<()> {
        let path = self.global_memory_path()?;
        let report = graph.validate();
        if !report.is_valid() {
            crate::logging::info(&format!(
                "Memory graph validation found {} error(s) before saving {}: {:?}",
                report.errors().len(),
                path.display(),
                report.errors()
            ));
        }
        storage::write_json(&path, graph)?;
        if !self.test_mode {
            cache_graph(path, graph);
        }
        Ok(())
    }

    /// Add a tag to a memory
    pub fn tag_memory(&self, memory_id: &str, tag: &str) -> Result<()> {
        // Try project first
        let mut graph = self.load_project_graph()?;
        if graph.memories.contains_key(memory_id) {
            graph.tag_memory(memory_id, tag);
            return self.save_project_graph(&graph);
        }

        // Try global
        let mut graph = self.load_global_graph()?;
        if graph.memories.contains_key(memory_id) {
            graph.tag_memory(memory_id, tag);
            return self.save_global_graph(&graph);
        }

        Err(anyhow::anyhow!("Memory not found: {}", memory_id))
    }

    /// Link two memories with a RelatesTo edge
    pub fn link_memories(&self, from_id: &str, to_id: &str, weight: f32) -> Result<()> {
        // Try project first
        let mut graph = self.load_project_graph()?;
        if graph.memories.contains_key(from_id) && graph.memories.contains_key(to_id) {
            graph.link_memories(from_id, to_id, weight);
            return self.save_project_graph(&graph);
        }

        // Try global
        let mut graph = self.load_global_graph()?;
        if graph.memories.contains_key(from_id) && graph.memories.contains_key(to_id) {
            graph.link_memories(from_id, to_id, weight);
            return self.save_global_graph(&graph);
        }

        // Cross-store links not supported for now
        Err(anyhow::anyhow!(
            "Both memories must be in the same store (project or global)"
        ))
    }

    /// Get memories related to a given memory via graph traversal
    pub fn get_related(&self, memory_id: &str, depth: usize) -> Result<Vec<MemoryEntry>> {
        // Find which store contains the memory
        let (mut graph, _is_project) = {
            let project_graph = self.load_project_graph()?;
            if project_graph.memories.contains_key(memory_id) {
                (project_graph, true)
            } else {
                let global_graph = self.load_global_graph()?;
                if global_graph.memories.contains_key(memory_id) {
                    (global_graph, false)
                } else {
                    return Err(anyhow::anyhow!("Memory not found: {}", memory_id));
                }
            }
        };

        // Use cascade retrieval to find related memories
        let results = graph.cascade_retrieve(&[memory_id.to_string()], &[1.0], depth, 20);

        // Collect memory entries (excluding the seed)
        let entries: Vec<MemoryEntry> = results
            .into_iter()
            .filter(|(id, _)| id != memory_id)
            .filter_map(|(id, _)| graph.get_memory(&id).cloned())
            .collect();

        Ok(entries)
    }

    /// Cheapest k memory-graph paths between two anchors.
    ///
    /// Anchors accept a stable memory id, `subject:<key>`, `tag:<name>`, or a
    /// normalized text fragment. Both endpoints must resolve inside the same
    /// store; cross-store links are intentionally not synthesized here because
    /// the graph does not yet persist bridge edges between project and global
    /// stores.
    pub fn path_between(
        &self,
        from_anchor: &str,
        to_anchor: &str,
        depth: usize,
        limit: usize,
        scope: MemoryScope,
    ) -> Result<Vec<jcode_memory_types::MemoryPath>> {
        Ok(self
            .paths_between_anchors(
                &[from_anchor.to_string()],
                &[to_anchor.to_string()],
                depth,
                limit,
                scope,
                &[],
            )?
            .paths)
    }

    fn bridge_store_path(&self) -> Result<PathBuf> {
        let base = if self.test_mode {
            storage::jcode_dir()?.join("memory").join("test")
        } else {
            storage::jcode_dir()?.join("memory")
        };
        std::fs::create_dir_all(&base)?;
        Ok(base.join("bridges.json"))
    }

    pub fn list_bridges(&self) -> Result<Vec<MemoryBridge>> {
        let path = self.bridge_store_path()?;
        if !path.exists() {
            return Ok(Vec::new());
        }
        storage::read_json(&path)
    }

    fn save_bridges(&self, bridges: &[MemoryBridge]) -> Result<()> {
        storage::write_json(&self.bridge_store_path()?, &bridges)
    }

    /// Link one project memory to one global memory as an explicit bridge.
    pub fn link_bridge(
        &self,
        project_id: &str,
        global_id: &str,
        weight: f32,
        source: &str,
    ) -> Result<MemoryBridge> {
        let project_dir = self
            .get_project_dir()
            .ok_or_else(|| anyhow::anyhow!("bridge requires a project directory"))?;
        let project_store = project_store_key(&project_dir);
        let project = self.load_project_graph()?;
        let project_entry = project
            .get_memory(project_id)
            .ok_or_else(|| anyhow::anyhow!("project memory not found: {project_id}"))?;
        let global = self.load_global_graph()?;
        let global_entry = global
            .get_memory(global_id)
            .ok_or_else(|| anyhow::anyhow!("global memory not found: {global_id}"))?;
        let bridge = MemoryBridge {
            id: format!("bridge:{project_id}:{global_id}"),
            project_store,
            project_id: project_id.to_string(),
            global_id: global_id.to_string(),
            weight: weight.clamp(0.0, 1.0),
            confidence: project_entry
                .confidence
                .min(global_entry.confidence)
                .clamp(0.0, 1.0),
            source: source.to_string(),
            created_at: Utc::now(),
        };
        let mut bridges = self.list_bridges()?;
        bridges.retain(|existing| existing.id != bridge.id);
        bridges.push(bridge.clone());
        bridges.sort_by(|left, right| left.id.cmp(&right.id));
        self.save_bridges(&bridges)?;
        Ok(bridge)
    }

    /// Remove a bridge by id. Returns whether a row was removed.
    pub fn unlink_bridge(&self, id: &str) -> Result<bool> {
        let mut bridges = self.list_bridges()?;
        let before = bridges.len();
        bridges.retain(|bridge| bridge.id != id);
        if bridges.len() == before {
            return Ok(false);
        }
        self.save_bridges(&bridges)?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn stitched_bridge_paths(
        &self,
        project: &MemoryGraph,
        global: &MemoryGraph,
        froms: &[String],
        tos: &[String],
        depth: usize,
        limit: usize,
        allowed_relations: &[&str],
    ) -> Result<Vec<jcode_memory_types::MemoryPath>> {
        if !allowed_relations.is_empty()
            && !allowed_relations
                .iter()
                .any(|relation| relation.eq_ignore_ascii_case("bridge"))
        {
            return Ok(Vec::new());
        }
        let Some(project_dir) = self.get_project_dir() else {
            return Ok(Vec::new());
        };
        let project_store = project_store_key(&project_dir);
        let bridges: Vec<MemoryBridge> = self
            .list_bridges()?
            .into_iter()
            .filter(|bridge| bridge.project_store == project_store)
            .collect();
        if bridges.is_empty() {
            return Ok(Vec::new());
        }

        let mut out = Vec::new();
        for bridge in bridges {
            let bridge_cost = 0.2 + 0.5 * (1.0 - bridge.confidence.clamp(0.0, 1.0));
            for from_anchor in froms {
                let Some(from_project) = resolve_path_anchor(project, from_anchor) else {
                    continue;
                };
                let left_paths = project.k_shortest_paths_with_relations(
                    &from_project,
                    &bridge.project_id,
                    depth,
                    limit,
                    allowed_relations,
                );
                if left_paths.is_empty() {
                    continue;
                }
                for to_anchor in tos {
                    let Some(to_global) = resolve_path_anchor(global, to_anchor) else {
                        continue;
                    };
                    let right_paths = global.k_shortest_paths_with_relations(
                        &bridge.global_id,
                        &to_global,
                        depth,
                        limit,
                        allowed_relations,
                    );
                    for left in &left_paths {
                        for right in &right_paths {
                            if left.edges.len() + 1 + right.edges.len() > depth {
                                continue;
                            }
                            let edge = jcode_memory_types::MemoryPathEdge {
                                from: bridge.project_id.clone(),
                                to: bridge.global_id.clone(),
                                kind: jcode_memory_types::EdgeKind::Bridge,
                                cost: bridge_cost,
                            };
                            if let Some(path) = stitch_paths_with_bridge(
                                left,
                                edge,
                                right,
                                bridge.confidence,
                            ) {
                                out.push(path);
                            }
                        }
                    }
                }
            }

            // Reverse direction: start in global, cross into project.
            for from_anchor in froms {
                let Some(from_global) = resolve_path_anchor(global, from_anchor) else {
                    continue;
                };
                let left_paths = global.k_shortest_paths_with_relations(
                    &from_global,
                    &bridge.global_id,
                    depth,
                    limit,
                    allowed_relations,
                );
                if left_paths.is_empty() {
                    continue;
                }
                for to_anchor in tos {
                    let Some(to_project) = resolve_path_anchor(project, to_anchor) else {
                        continue;
                    };
                    let right_paths = project.k_shortest_paths_with_relations(
                        &bridge.project_id,
                        &to_project,
                        depth,
                        limit,
                        allowed_relations,
                    );
                    for left in &left_paths {
                        for right in &right_paths {
                            if left.edges.len() + 1 + right.edges.len() > depth {
                                continue;
                            }
                            let edge = jcode_memory_types::MemoryPathEdge {
                                from: bridge.global_id.clone(),
                                to: bridge.project_id.clone(),
                                kind: jcode_memory_types::EdgeKind::Bridge,
                                cost: bridge_cost,
                            };
                            if let Some(path) = stitch_paths_with_bridge(
                                left,
                                edge,
                                right,
                                bridge.confidence,
                            ) {
                                out.push(path);
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Cheapest k memory-graph paths between two anchor sets.
    ///
    /// This is the deterministic half of the design document's constrained
    /// multi-source/multi-target path query: anchors are resolved per store,
    /// every source-target pair is solved, and an optional relation allow-list
    /// restricts traversal to canonical `EdgeKind` names. Unknown relation
    /// names are reported instead of silently widening the search.
    ///
    /// Paths cross project/global stores only through explicitly registered
    /// bridges; no implicit cross-store edges are created.
    pub fn paths_between_anchors(
        &self,
        from_anchors: &[String],
        to_anchors: &[String],
        depth: usize,
        limit: usize,
        scope: MemoryScope,
        relations: &[String],
    ) -> Result<MemoryPathQueryOutcome> {
        let froms = normalize_path_anchors(from_anchors);
        let tos = normalize_path_anchors(to_anchors);
        let depth = depth.clamp(1, MAX_INITIAL_PATH_DEPTH);
        let limit = limit.min(20);

        let mut allowed_owned = Vec::new();
        let mut unknown_relations = Vec::new();
        for relation in relations {
            let relation = relation.trim();
            if relation.is_empty() {
                continue;
            }
            if jcode_memory_types::EdgeKind::parse(relation).is_some() {
                allowed_owned.push(relation.to_string());
            } else {
                unknown_relations.push(relation.to_string());
            }
        }
        unknown_relations.sort();
        unknown_relations.dedup();
        let allowed: Vec<&str> = allowed_owned.iter().map(String::as_str).collect();

        let mut outcome = self.paths_between_anchors_inner(
            &froms,
            &tos,
            depth,
            limit,
            scope,
            &allowed,
            &unknown_relations,
        )?;

        // L0 depth relaxation: E05 means anchors are connected but the first
        // budget was too small. Retry once and record the parameter change so
        // callers can tell a real path from a widened search.
        if outcome.paths.is_empty()
            && outcome.diagnosis.iter().any(|code| code == "E05")
            && depth < MAX_RELAXED_PATH_DEPTH
        {
            let relaxed_depth = MAX_RELAXED_PATH_DEPTH;
            let mut relaxed = self.paths_between_anchors_inner(
                &froms,
                &tos,
                relaxed_depth,
                limit,
                scope,
                &allowed,
                &unknown_relations,
            )?;
            relaxed
                .relaxations
                .push(format!("depth {depth} -> {relaxed_depth}"));
            relaxed.depth = relaxed_depth;
            outcome = relaxed;
        }
        Ok(outcome)
    }

    #[allow(clippy::too_many_arguments)]
    fn paths_between_anchors_inner(
        &self,
        froms: &[String],
        tos: &[String],
        depth: usize,
        limit: usize,
        scope: MemoryScope,
        allowed: &[&str],
        unknown_relations: &[String],
    ) -> Result<MemoryPathQueryOutcome> {
        let mut outcome = MemoryPathQueryOutcome {
            depth,
            unknown_relations: unknown_relations.to_vec(),
            ..MemoryPathQueryOutcome::default()
        };

        if froms.is_empty() || tos.is_empty() || limit == 0 {
            outcome.unresolved_from = froms.to_vec();
            outcome.unresolved_to = tos.to_vec();
            outcome.diagnosis.push("E01".to_string());
            return Ok(outcome);
        }
        // A filter made only of unknown names must match nothing. Treating it
        // as an empty filter would turn a typo into an unrestricted query.
        if !unknown_relations.is_empty() && allowed.is_empty() {
            outcome.diagnosis.push("E10".to_string());
            return Ok(outcome);
        }

        let mut graphs = Vec::new();
        if scope.includes_project() {
            graphs.push(("project", self.load_project_graph()?));
        }
        if scope.includes_global() {
            graphs.push(("global", self.load_global_graph()?));
        }
        outcome.stores_searched = graphs.len();

        let mut resolved_from = std::collections::HashSet::new();
        let mut resolved_to = std::collections::HashSet::new();
        for (_, graph) in &graphs {
            for anchor in froms {
                if resolve_path_anchor(graph, anchor).is_some() {
                    resolved_from.insert(anchor.clone());
                }
            }
            for anchor in tos {
                if resolve_path_anchor(graph, anchor).is_some() {
                    resolved_to.insert(anchor.clone());
                }
            }
        }
        outcome.unresolved_from = froms
            .iter()
            .filter(|anchor| !resolved_from.contains(anchor.as_str()))
            .cloned()
            .collect();
        outcome.unresolved_to = tos
            .iter()
            .filter(|anchor| !resolved_to.contains(anchor.as_str()))
            .cloned()
            .collect();
        if !outcome.unresolved_from.is_empty() || !outcome.unresolved_to.is_empty() {
            outcome.diagnosis.push("E01".to_string());
        }
        if !outcome.unknown_relations.is_empty() {
            outcome.diagnosis.push("E10".to_string());
        }

        let mut paths = Vec::new();
        for (_, graph) in &graphs {
            let from_nodes: Vec<String> = froms
                .iter()
                .filter_map(|anchor| resolve_path_anchor(graph, anchor))
                .collect();
            let to_nodes: Vec<String> = tos
                .iter()
                .filter_map(|anchor| resolve_path_anchor(graph, anchor))
                .collect();
            for from in &from_nodes {
                for to in &to_nodes {
                    paths.extend(
                        graph.k_shortest_paths_with_relations(from, to, depth, limit, &allowed),
                    );
                }
            }
        }
        let project = graphs
            .iter()
            .find(|(label, _)| *label == "project")
            .map(|(_, graph)| graph);
        let global = graphs
            .iter()
            .find(|(label, _)| *label == "global")
            .map(|(_, graph)| graph);
        if let (Some(project), Some(global)) = (project, global) {
            paths.extend(
                self.stitched_bridge_paths(project, global, froms, tos, depth, limit, &allowed)?,
            );
        }
        paths.sort_by(|left, right| {
            left.cost
                .total_cmp(&right.cost)
                .then_with(|| left.nodes.cmp(&right.nodes))
        });
        let mut seen = std::collections::HashSet::new();
        paths.retain(|path| seen.insert(path.nodes.clone()));
        paths.truncate(limit);
        if paths.is_empty() && outcome.diagnosis.is_empty() {
            outcome.diagnosis.push("E03".to_string());
            if depth >= 8 {
                outcome.diagnosis.push("E05".to_string());
            }
        }
        outcome.paths = paths;
        Ok(outcome)
    }

    /// Fetch one memory by id from the requested scope.
    pub fn get_by_id(&self, id: &str, scope: MemoryScope) -> Result<Option<MemoryEntry>> {
        if scope.includes_project()
            && let Some(entry) = self.load_project_graph()?.get_memory(id)
        {
            return Ok(Some(entry.clone()));
        }
        if scope.includes_global()
            && let Some(entry) = self.load_global_graph()?.get_memory(id)
        {
            return Ok(Some(entry.clone()));
        }
        Ok(None)
    }

    /// Find similar memories with cascade retrieval through the graph
    ///
    /// This extends the basic embedding search by also traversing through
    /// tags to find related memories that might not have direct embedding similarity.
    pub fn find_similar_with_cascade(
        &self,
        text: &str,
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<(MemoryEntry, f32)>> {
        self.find_similar_with_cascade_scoped(text, threshold, limit, MemoryScope::All)
    }

    pub fn find_similar_with_cascade_scoped(
        &self,
        text: &str,
        threshold: f32,
        limit: usize,
        scope: MemoryScope,
    ) -> Result<Vec<(MemoryEntry, f32)>> {
        // First, do basic embedding search
        let embedding_hits = self.find_similar_scoped(text, threshold, limit, scope)?;

        if embedding_hits.is_empty() {
            return Ok(Vec::new());
        }

        // Get seed IDs and scores
        let seed_ids: Vec<String> = embedding_hits.iter().map(|(e, _)| e.id.clone()).collect();
        let seed_scores: Vec<f32> = embedding_hits.iter().map(|(_, s)| *s).collect();

        // Load graphs and perform cascade retrieval
        let mut project_graph = if scope.includes_project() {
            Some(self.load_project_graph()?)
        } else {
            None
        };
        let mut global_graph = if scope.includes_global() {
            Some(self.load_global_graph()?)
        } else {
            None
        };

        // Cascade through project graph
        let project_cascade = project_graph
            .as_mut()
            .map(|graph| graph.cascade_retrieve(&seed_ids, &seed_scores, 2, limit * 2))
            .unwrap_or_default();

        // Cascade through global graph
        let global_cascade = global_graph
            .as_mut()
            .map(|graph| graph.cascade_retrieve(&seed_ids, &seed_scores, 2, limit * 2))
            .unwrap_or_default();

        // Merge results, keeping highest score for each memory
        let mut merged: std::collections::HashMap<String, f32> = std::collections::HashMap::new();

        for (id, score) in embedding_hits.iter() {
            merged.insert(id.id.clone(), *score);
        }
        for (id, score) in project_cascade {
            let existing = merged.get(&id).copied().unwrap_or(0.0);
            if score > existing {
                merged.insert(id, score);
            }
        }
        for (id, score) in global_cascade {
            let existing = merged.get(&id).copied().unwrap_or(0.0);
            if score > existing {
                merged.insert(id, score);
            }
        }

        // Look up entries and keep only the top-scoring results
        let results: Vec<(MemoryEntry, f32)> = top_k_by_score(
            merged.into_iter().filter_map(|(id, score)| {
                project_graph
                    .as_ref()
                    .and_then(|graph| graph.get_memory(&id))
                    .or_else(|| {
                        global_graph
                            .as_ref()
                            .and_then(|graph| graph.get_memory(&id))
                    })
                    .cloned()
                    .map(|entry| (entry, score))
            }),
            limit,
        );

        Ok(results)
    }

    /// Get graph statistics for display
    pub fn graph_stats(&self) -> Result<(usize, usize, usize, usize)> {
        let project = self.load_project_graph()?;
        let global = self.load_global_graph()?;

        let memories = project.memories.len() + global.memories.len();
        let tags = project.tags.len() + global.tags.len();
        let edges = project.edge_count() + global.edge_count();
        let clusters = project.clusters.len() + global.clusters.len();

        Ok((memories, tags, edges, clusters))
    }
}

/// Embedding similarity threshold (0.0 - 1.0)
/// Lower = more candidates, higher = fewer but more relevant
pub const EMBEDDING_SIMILARITY_THRESHOLD: f32 = 0.5;

/// Maximum embedding hits to verify with sidecar
pub const EMBEDDING_MAX_HITS: usize = 10;

/// Minimum per-retriever candidate pool size for hybrid fusion.
const HYBRID_POOL_MIN: usize = 50;

/// Most memories Jcode sends to Jev per automatic recall. Every 24 entries costs
/// one Jev decision, so judging a whole store (thousands of memories) each turn
/// exhausted the daily plan allowance within hours. Lexical relevance narrows
/// the field first; Jev still judges every candidate that reaches the prompt.
pub(crate) const MAX_JEV_RECALL_CANDIDATES: usize = 72;

#[path = "memory/ranking.rs"]
mod ranking;
use ranking::{bm25_rank, prefilter_for_jev};

// ==================== Pluggable Graph Backend ====================

/// Name of the active graph backend, used by diagnostics and tests.
///
/// Resolved once per process from the `JCODE_MEMORY_BACKEND`
/// environment variable:
///
/// - `json` (default) — write the entire `MemoryGraph` to a single
///   atomic JSON snapshot under `~/.jcode/memory/projects/<hash>.json`.
/// - `sqlite-gvec` — use the SQLite + sqlite-vec engine
///   (`SqliteGvecBackend`). Requires the `sqlite-gvec` Cargo feature.
///
/// When the requested backend is unavailable (e.g. SQLite feature not
/// compiled in, or `SqliteGvecBackend::open_default` failed), the
/// selector logs a warning and falls back to JSON so the rest of the
/// app continues to work.
pub fn active_backend_name() -> &'static str {
    static NAME: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        if test_build_pins_json_backend() {
            return "json";
        }
        let raw = std::env::var("JCODE_MEMORY_BACKEND").unwrap_or_default();
        match raw.to_lowercase().as_str() {
            "" => default_backend_name(),
            "json" => "json",
            "sqlite" | "sqlite-gvec" | "gvec" => "sqlite-gvec",
            other => {
                crate::logging::warn(&format!(
                    "JCODE_MEMORY_BACKEND={other:?} is not recognised; falling back to 'json'"
                ));
                "json"
            }
        }
    })
}

/// A skill root mirror has the exact `skill:<name>` shape. Derived index nodes
/// add `::<kind>:...` segments after the name.
pub(crate) fn is_skill_root_id(id: &str) -> bool {
    id.starts_with("skill:") && !id.contains("::")
}

pub(crate) fn skill_root_id_for_part(id: &str) -> Option<String> {
    if !id.starts_with("skill:") || !id.contains("::") {
        return None;
    }
    let name = id.strip_prefix("skill:")?.split("::").next()?.trim();
    (!name.is_empty()).then(|| format!("skill:{name}"))
}

/// Whether this build is a test build that must ignore an ambient backend.
///
/// Test targets — this crate's own, and dependent crates' compiled with the
/// `test-support` feature — sandbox `JCODE_HOME` per case, but the active
/// backend is resolved once per process and cached. Inheriting a *runtime*
/// backend choice from the ambient environment (a developer shell exporting
/// `JCODE_MEMORY_BACKEND=sqlite-gvec`) freezes the first case's store into every
/// later case, so such builds are pinned to the JSON backend, whose path is
/// resolved per call. SQLite-only cases must skip in those builds instead of
/// opting in through the environment.
fn test_build_pins_json_backend() -> bool {
    cfg!(any(test, feature = "test-support"))
}

/// Resolve the backend used when `JCODE_MEMORY_BACKEND` is unset.
///
/// - In test builds the default is pinned to `json` (see
///   [`test_build_pins_json_backend`]).
/// - At runtime, existing installs keep their `sqlite-gvec` store when its
///   database file is present (the pre-upstream-merge default); fresh installs
///   with no database start on `json`.
fn default_backend_name() -> &'static str {
    if test_build_pins_json_backend() {
        return "json";
    }
    if let Ok(dir) = crate::storage::jcode_dir() {
        if dir
            .join("memory")
            .join("backend-sqlite")
            .join("gvec.sqlite")
            .exists()
        {
            return "sqlite-gvec";
        }
    }
    "json"
}

/// Try to construct the SQLite backend. Returns `None` if the feature
/// is off or the open call fails (the caller should fall back to JSON
/// and log why).
///
/// `test_dir` switches to the per-test database so `new_test()` does
/// not see the real user memory.
#[cfg(feature = "sqlite-gvec")]
fn try_open_sqlite_backend(test_dir: bool) -> Option<Arc<dyn GraphBackend>> {
    use crate::memory::backend::SqliteGvecBackend;
    let result = if test_dir {
        SqliteGvecBackend::open_test_dir()
    } else {
        SqliteGvecBackend::open_default()
    };
    match result {
        Ok(b) => Some(Arc::new(b)),
        Err(e) => {
            crate::logging::warn(&format!(
                "SqliteGvecBackend::open failed: {e}; falling back to JSON"
            ));
            None
        }
    }
}

#[cfg(not(feature = "sqlite-gvec"))]
fn try_open_sqlite_backend(_test_dir: bool) -> Option<Arc<dyn GraphBackend>> {
    None
}

/// Return the active `GraphBackend` for a given `StoreKey`.
///
/// The first call per process initialises a process-wide singleton. The
/// default is the SQLite backend (shared via `Arc` so all threads see
/// the same connection); when SQLite is unavailable it falls back to the
/// JSON backend.
pub fn graph_backend() -> Arc<dyn GraphBackend> {
    use std::sync::OnceLock;
    static BACKEND: OnceLock<Arc<dyn GraphBackend>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            if active_backend_name() == "sqlite-gvec"
                && let Some(b) = try_open_sqlite_backend(false)
            {
                return b;
            }
            // Default / fallback: JSON backend rooted under
            // `~/.jcode/memory/backend-json/`.
            let root = crate::memory::backend::JsonBackend::default_root()
                .unwrap_or_else(|_| PathBuf::from("."));
            Arc::new(crate::memory::backend::JsonBackend::new(root))
        })
        .clone()
}

/// Return the per-test backend. Always returns a fresh, isolated
/// instance (no caching) so a single test run cannot leak state into
/// the next.
pub fn test_backend() -> Arc<dyn GraphBackend> {
    if active_backend_name() == "sqlite-gvec"
        && let Some(b) = try_open_sqlite_backend(true)
    {
        return b;
    }
    let root =
        crate::memory::backend::JsonBackend::default_root().unwrap_or_else(|_| PathBuf::from("."));
    Arc::new(crate::memory::backend::JsonBackend::new(root))
}

fn link_pair(left: &str, right: &str) -> (String, String) {
    if left <= right {
        (left.to_string(), right.to_string())
    } else {
        (right.to_string(), left.to_string())
    }
}

fn merge_json_graph(
    backend: &mut MemoryGraph,
    json: &MemoryGraph,
    repair_edges: bool,
) -> (usize, usize, usize) {
    use std::collections::HashSet;

    let edges_before = backend.edge_count();
    let tags_before = backend.tags.len();
    let mut added = HashSet::new();
    for (id, entry) in &json.memories {
        if backend.memories.contains_key(id) {
            continue;
        }
        added.insert(backend.add_memory(entry.clone()));
    }
    for (from, edges) in &json.edges {
        for edge in edges {
            if !repair_edges && !added.contains(from) && !added.contains(&edge.target) {
                continue;
            }
            if matches!(edge.kind, jcode_memory_types::EdgeKind::HasTag) {
                if repair_edges
                    && let Some(tag) = edge.target.strip_prefix("tag:")
                {
                    backend.tag_memory(from, tag);
                }
                continue;
            }
            if matches!(edge.kind, jcode_memory_types::EdgeKind::InCluster)
                && !backend.clusters.contains_key(&edge.target)
                && let Some(cluster) = json.clusters.get(&edge.target)
            {
                backend.clusters.insert(edge.target.clone(), cluster.clone());
            }
            if !node_exists(backend, from) || !node_exists(backend, &edge.target) {
                continue;
            }
            backend.add_edge(from, &edge.target, edge.kind.clone());
        }
    }
    if repair_edges {
        for (id, entry) in &json.memories {
            if !backend.memories.contains_key(id) {
                continue;
            }
            for tag in &entry.tags {
                backend.tag_memory(id, tag);
            }
        }
    }
    (
        added.len(),
        backend.edge_count().saturating_sub(edges_before),
        backend.tags.len().saturating_sub(tags_before),
    )
}

fn node_exists(graph: &MemoryGraph, id: &str) -> bool {
    graph.memories.contains_key(id) || graph.tags.contains_key(id) || graph.clusters.contains_key(id)
}

fn insert_link_suggestion(
    candidates: &mut std::collections::BTreeMap<(String, String), LinkSuggestion>,
    left: &str,
    right: &str,
    weight: f32,
    reason: &str,
) {
    let key = link_pair(left, right);
    match candidates.get(&key) {
        Some(existing) if existing.weight >= weight => {}
        _ => {
            candidates.insert(
                key.clone(),
                LinkSuggestion {
                    from: key.0,
                    to: key.1,
                    weight,
                    reason: reason.to_string(),
                },
            );
        }
    }
}

/// Convenience: derive a `StoreKey` from a project's directory hash
/// (matches `project_memory_path`).
pub fn project_store_key(project_dir: &std::path::Path) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    project_dir.hash(&mut hasher);
    format!("project:{:016x}", hasher.finish())
}

/// Convenience: the store key for the global memory scope.
pub fn global_store_key() -> &'static str {
    "global"
}

#[cfg(test)]
#[path = "memory_tests.rs"]
mod tests;
