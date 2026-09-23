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
fn apply_plan(graph: &mut MemoryGraph, new_id: &str, plan: &jcode_memory_types::rule_engine::RulePlan) {
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

/// Whether the sidecar is currently auto-disabled due to sustained judge
/// failures. Exposed for the TUI status badge so the user can see why memory
/// is silently running on the no-LLM path.
pub fn memory_sidecar_auto_disabled() -> bool {
    crate::memory_judge_metrics::sidecar_should_auto_disable()
}

/// Minimum meaningful content length (in chars) for an LLM-extracted memory.
///
/// Content shorter than this is almost always extraction noise (a bare
/// identifier, a truncated path fragment, a lone symbol) that pollutes vector
/// recall without adding semantic value. See the P0-3 noise audit.
pub const MIN_MEANINGFUL_MEMORY_LEN: usize = 12;

/// Classify a memory's content as low-value extraction noise.
///
/// These are the patterns observed polluting the recall pool (P0-3 audit):
/// - content shorter than [`MIN_MEANINGFUL_MEMORY_LEN`] chars (bare ids, paths)
/// - content with no alphabetic/digit substance (only symbols, e.g. "---", "::")
/// - pure git commit hashes, which the extraction prompt already says to skip
///
/// Returns true when the content looks like noise that should NOT be stored.
pub fn is_extraction_noise(content: &str) -> bool {
    let c = content.trim();
    if c.is_empty() {
        return true;
    }
    if c.chars().count() < MIN_MEANINGFUL_MEMORY_LEN {
        return true;
    }
    // Must have at least one "word-ish" character; content that is only
    // whitespace/symbols (e.g. "----", ":::", "**") is not a real memory.
    let has_word_chars = c
        .chars()
        .any(|ch| ch.is_alphabetic() || ch.is_ascii_digit());
    if !has_word_chars {
        return true;
    }
    // Pure commit-hash shaped strings (7-40 hex chars with no spaces) are
    // transient git noise, not durable knowledge.
    let trimmed_len = c.len();
    let is_hex_hash = (7..=40).contains(&trimmed_len)
        && !c.contains(char::is_whitespace)
        && c.chars().all(|ch| ch.is_ascii_hexdigit());
    if is_hex_hash {
        return true;
    }
    false
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

#[derive(Debug, Clone)]
pub struct MemoryManager {
    project_dir: Option<PathBuf>,
    /// When true, use isolated test storage instead of real memory
    test_mode: bool,
    include_skills: bool,
    /// Ontology registry consulted for rule dispatch + activity scheduling.
    /// Held by `Arc` so cloning the manager is cheap and tests can substitute
    /// a custom registry without touching global state.
    ontology_registry: OntologyRegistry,
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
        Self {
            project_dir: None,
            test_mode: false,
            include_skills: true,
            ontology_registry: OntologyRegistry::with_default(),
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

        let project_hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            project_dir.hash(&mut hasher);
            format!("{:016x}", hasher.finish())
        };

        let memory_dir = storage::jcode_dir()?.join("memory").join("projects");
        Ok(Some(memory_dir.join(format!("{}.json", project_hash))))
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

    /// Store without embedding inference. Exact duplicates reinforce an
    /// existing entry only within the requested scope, never mutate a
    /// different project. Inadmissible provenance (below the extraction
    /// method's admission threshold) is rejected instead of committed.
    pub fn remember_project(&self, entry: MemoryEntry) -> Result<String> {
        anyhow::ensure!(
            self.project_memory_path()?.is_some(),
            "Project memory requires a working directory; use global scope explicitly"
        );
        crate::memory_types::validate_new_entry(&entry)
            .map_err(|issues| anyhow::anyhow!("memory validation failed: {:?}", issues))?;
        let mut graph = self.load_project_graph()?;
        let id = Self::remember_in_graph(&mut graph, entry);
        self.save_project_graph(&graph)?;
        Ok(id)
    }

    pub fn remember_global(&self, entry: MemoryEntry) -> Result<String> {
        crate::memory_types::validate_new_entry(&entry)
            .map_err(|issues| anyhow::anyhow!("memory validation failed: {:?}", issues))?;
        let mut graph = self.load_global_graph()?;
        let id = Self::remember_in_graph(&mut graph, entry);
        self.save_global_graph(&graph)?;
        Ok(id)
    }

    fn remember_in_graph(graph: &mut MemoryGraph, entry: MemoryEntry) -> String {
        let normalized = entry.content.trim();
        let duplicate = graph
            .active_memories()
            .into_iter()
            .find(|existing| {
                existing.category == entry.category && existing.content.trim() == normalized
            })
            .map(|existing| existing.id.clone());
        if let Some(id) = duplicate {
            if let Some(existing) = graph.get_memory_mut(&id) {
                existing.reinforce(entry.source.as_deref().unwrap_or("dedup"), 0);
            }
            return id;
        }
        graph.add_memory(entry)
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
        let id = entry.id.clone();

        let Some(existing_snapshot) = graph.get_memory(&id).cloned() else {
            return graph.add_memory(entry);
        };

        let old_tags: std::collections::HashSet<String> =
            existing_snapshot.tags.iter().cloned().collect();
        let new_tags: std::collections::HashSet<String> = entry.tags.iter().cloned().collect();

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
            existing.updated_at = entry.updated_at;
            existing.source = entry.source;
            existing.trust = entry.trust;
            existing.active = entry.active;
            existing.superseded_by = entry.superseded_by;
            existing.confidence = entry.confidence;
            if content_changed {
                existing.set_embedding(None, None);
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
            fused
                .into_iter()
                .filter_map(|(idx, score)| {
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
    fn fts_search_scoped(
        &self,
        query: &str,
        scope: MemoryScope,
    ) -> Option<Vec<MemoryEntry>> {
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
                    crate::logging::warn(&format!(
                        "fts_search_scoped project: {e}"
                    ));
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
                    crate::logging::warn(&format!(
                        "fts_search_scoped global: {e}"
                    ));
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
        if !memory_llm_judge_available() {
            crate::logging::info("Memory transcript extraction skipped: LLM judge unavailable");
            return Ok(Vec::new());
        }

        let sidecar = Sidecar::new();
        let extracted = sidecar.extract_memories(transcript).await?;

        let mut ids = Vec::new();
        for memory in extracted {
            let category: MemoryCategory = memory.category.parse().unwrap_or(MemoryCategory::Fact);
            let trust = match memory.trust.as_str() {
                "high" => TrustLevel::High,
                "medium" => TrustLevel::Medium,
                _ => TrustLevel::Low,
            };

            let confidence = trust.provenance_confidence();
            let entry = MemoryEntry::new(category, memory.content)
                .with_source(session_id)
                .with_trust(trust)
                .with_provenance(
                    ProvenanceRecord::new(session_id, ExtractionMethod::LlmExtraction)
                        .with_confidence(confidence),
                );

            // Store in project scope by default
            let id = self.remember_project(entry)?;
            ids.push(id);
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
        Ok(
            crate::memory_jev::recall(self, context, max_candidates, MemoryScope::All)
                .await?
                .into_iter()
                .map(|(entry, _)| entry)
                .collect(),
        )
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

    /// Jev-only automatic recall. Storage and per-session dedup remain local;
    /// there is no embedding, conventional LLM, or unjudged fallback path.
    pub async fn get_relevant_parallel(
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
            .filter(|entry| entry.active && !is_memory_injected(session_id, &entry.id))
            .collect();
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
            crate::memory_jev::select(&client, &query, entries, 5).await
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

    /// Load project memories as a MemoryGraph without generating embeddings.
    pub fn load_project_graph(&self) -> Result<MemoryGraph> {
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
                        && let Some(legacy) =
                            self.load_legacy_project_json_for_migration()?
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
                    if !self.test_mode {
                        cache_graph_for_backend(backend.name(), &key, &graph, 0);
                    }
                    return Ok(graph);
                }
                Err(e) => {
                    crate::logging::warn(&format!(
                        "load_project_graph via {e}; falling back to JSON path"
                    ));
                    // fall through to legacy JSON handling
                }
            }
        }

        let Some(path) = self.project_memory_path()? else {
            return Ok(MemoryGraph::new());
        };

        if !self.test_mode
            && let Some(mut graph) = cached_graph(&path)
        {
            if Self::normalize_graph_search_text(&mut graph) {
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
                if self.import_legacy_notes_into_graph(&mut graph)? {
                    self.save_project_graph(&graph)?;
                } else if normalized {
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
            if self.import_legacy_notes_into_graph(&mut graph)? {
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
                        && let Some(legacy) =
                            self.load_legacy_global_json_for_migration()?
                    {
                        graph = legacy;
                        let _ = backend.save(&key, &graph);
                    }
                    if Self::normalize_graph_search_text(&mut graph) {
                        let _ = backend.save(&key, &graph);
                    }
                    if !self.test_mode {
                        cache_graph_for_backend(backend.name(), &key, &graph, 0);
                    }
                    return Ok(graph);
                }
                Err(e) => {
                    crate::logging::warn(&format!(
                        "load_global_graph via {e}; falling back to JSON path"
                    ));
                }
            }
        }

        let path = self.global_memory_path()?;
        if !self.test_mode
            && let Some(mut graph) = cached_graph(&path)
        {
            if Self::normalize_graph_search_text(&mut graph) {
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
                if Self::normalize_graph_search_text(&mut graph) {
                    storage::write_json(&path, &graph)?;
                }
                if !self.test_mode {
                    cache_graph(path, &graph);
                }
                return Ok(graph);
            }

            // Fall back to legacy MemoryStore and migrate
            let store: MemoryStore = storage::read_json(&path)?;
            let graph = MemoryGraph::from_legacy_store(store);

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
            let graph = MemoryGraph::new();
            if !self.test_mode {
                cache_graph(path, &graph);
            }
            Ok(graph)
        }
    }

   /// Save project memories as a MemoryGraph
   pub fn save_project_graph(&self, graph: &MemoryGraph) -> Result<()> {
       if crate::memory::active_backend_name() == "sqlite-gvec" {
           let backend = if self.test_mode { crate::memory::test_backend() } else { crate::memory::graph_backend() };
           let key = StoreKey::new(match self.get_project_dir() {
               Some(d) => crate::memory::project_store_key(&d),
               None => "project:none".to_string(),
           });
           if let Err(e) = backend.save(&key, graph) {
               crate::logging::warn(&format!(
                   "save_project_graph via {e}; falling back to JSON path"
               ));
               return self.save_project_graph_json(graph);
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

   /// Save global memories as a MemoryGraph
   pub fn save_global_graph(&self, graph: &MemoryGraph) -> Result<()> {
       if crate::memory::active_backend_name() == "sqlite-gvec" {
           let backend = if self.test_mode { crate::memory::test_backend() } else { crate::memory::graph_backend() };
           let key = StoreKey::new(crate::memory::global_store_key());
           if let Err(e) = backend.save(&key, graph) {
               crate::logging::warn(&format!(
                   "save_global_graph via {e}; falling back to JSON path"
               ));
               return self.save_global_graph_json(graph);
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

/// Rank memories by BM25 over their normalized search text.
///
/// Returns `(entry_index, score)` pairs sorted by score desc, truncated to
/// `limit`. Memories with zero query-term overlap are dropped.
fn bm25_rank(entries: &[MemoryEntry], query_text: &str, limit: usize) -> Vec<(usize, f32)> {
    const K1: f32 = 1.2;
    const B: f32 = 0.75;

    let q_terms: Vec<String> = normalize_search_text(query_text)
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    if q_terms.is_empty() {
        return Vec::new();
    }
    let q_set: std::collections::HashSet<&String> = q_terms.iter().collect();

    // Tokenize each doc once; compute df and doc lengths.
    let docs: Vec<Vec<String>> = entries
        .iter()
        .map(|e| {
            e.searchable_text()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect()
        })
        .collect();

    let n = docs.len().max(1) as f32;
    let avgdl = docs.iter().map(|d| d.len()).sum::<usize>() as f32 / n;
    let mut df: std::collections::HashMap<&str, f32> = std::collections::HashMap::new();
    for doc in &docs {
        let unique: std::collections::HashSet<&str> = doc.iter().map(|s| s.as_str()).collect();
        for t in unique {
            *df.entry(t).or_insert(0.0) += 1.0;
        }
    }

    let mut scored: Vec<(usize, f32)> = Vec::new();
    for (idx, doc) in docs.iter().enumerate() {
        if doc.is_empty() {
            continue;
        }
        let dl = doc.len() as f32;
        let mut tf: std::collections::HashMap<&str, f32> = std::collections::HashMap::new();
        for t in doc {
            *tf.entry(t.as_str()).or_insert(0.0) += 1.0;
        }
        let mut score = 0.0f32;
        for term in &q_set {
            let Some(&f) = tf.get(term.as_str()) else {
                continue;
            };
            let n_q = *df.get(term.as_str()).unwrap_or(&0.0);
            if n_q == 0.0 {
                continue;
            }
            let idf = (((n - n_q + 0.5) / (n_q + 0.5)) + 1.0).ln();
            let denom = f + K1 * (1.0 - B + B * dl / avgdl);
            score += idf * (f * (K1 + 1.0)) / denom;
        }
        if score > 0.0 {
            scored.push((idx, score));
        }
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(limit);
    scored
}

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
        let raw = std::env::var("JCODE_MEMORY_BACKEND").unwrap_or_default();
        match raw.to_lowercase().as_str() {
            "" => "json",
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
    let root = crate::memory::backend::JsonBackend::default_root()
        .unwrap_or_else(|_| PathBuf::from("."));
    Arc::new(crate::memory::backend::JsonBackend::new(root))
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
