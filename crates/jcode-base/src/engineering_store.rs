//! Persistent storage for deterministic engineering episode graphs.
//!
//! Each session is stored as one gzip-compressed JSON graph keyed by
//! `session_id`, with a separate entity index for anchor lookup. The schema keeps
//! the ontology identity alongside the graph so a later replay can detect schema
//! drift. This is deliberately a small, append/update store rather than a graph
//! database: the path engine runs in memory.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use jcode_memory_types::engineering_ontology::{
    CLASS_SESSION, EngineeringEntity, EngineeringGraph, EngineeringPath, EngineeringProvenance,
    EngineeringRelation, default_engineering_ontology,
};
use jcode_memory_types::{
    CompletionEdge, CompletionPromotionError, CompletionWriteReport, PathPlan, PathQuality,
    build_path_plan,
};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::storage;

/// One persisted session summary returned by [`EngineeringStore::list_sessions`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineeringSessionRecord {
    pub session_id: String,
    pub working_dir: Option<String>,
    pub ontology_id: String,
    pub ontology_hash: String,
    pub entity_count: usize,
    pub relation_count: usize,
    pub updated_at: String,
}

/// Maximum instance links returned per mention.
pub const MAX_ENTITY_LINKS: usize = 5;

/// One mention-to-instance link (deterministic lexical match).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineeringEntityLink {
    pub session_id: String,
    pub entity_id: String,
    pub name: String,
    pub class: String,
    pub score: f32,
    #[serde(default)]
    pub signals: Vec<String>,
}

fn link_tokens(text: &str) -> BTreeSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| token.len() >= 3)
        .map(str::to_lowercase)
        .collect()
}

fn score_entity(mention: &str, entity: &EngineeringEntity) -> Option<(f32, Vec<String>)> {
    let mention_lower = mention.trim().to_lowercase();
    if mention_lower.is_empty() {
        return None;
    }
    let name_lower = entity.name.to_lowercase();
    let id_lower = entity.id.to_lowercase();
    if name_lower == mention_lower {
        return Some((1.0, vec!["name exact".to_string()]));
    }
    let id_tail = id_lower.rsplit(':').next().unwrap_or(id_lower.as_str());
    if id_lower == mention_lower || id_tail == mention_lower {
        return Some((1.0, vec!["id exact".to_string()]));
    }
    if let Some(aliases) = entity.properties.get("aliases") {
        let alias_hit = match aliases {
            serde_json::Value::Array(items) => items
                .iter()
                .filter_map(serde_json::Value::as_str)
                .any(|alias| alias.to_lowercase() == mention_lower),
            serde_json::Value::String(alias) => alias.to_lowercase() == mention_lower,
            _ => false,
        };
        if alias_hit {
            return Some((0.95, vec!["alias match".to_string()]));
        }
    }
    let mention_tokens = link_tokens(&mention_lower);
    let name_tokens = link_tokens(&name_lower);
    if !mention_tokens.is_empty()
        && !name_tokens.is_empty()
        && (mention_tokens.is_subset(&name_tokens) || name_tokens.is_subset(&mention_tokens))
    {
        return Some((0.85, vec!["name token containment".to_string()]));
    }
    if mention_lower.len() >= 3 && id_tail.starts_with(&mention_lower) {
        return Some((0.8, vec!["id prefix".to_string()]));
    }
    if !mention_tokens.is_empty() && !name_tokens.is_empty() {
        let shared = mention_tokens.intersection(&name_tokens).count();
        let union = mention_tokens.union(&name_tokens).count();
        let overlap = shared as f32 / union as f32;
        if overlap >= 0.5 {
            return Some((
                (0.5 + 0.25 * overlap).min(0.75),
                vec![format!("name token overlap {overlap:.2}")],
            ));
        }
    }
    None
}

/// Maximum deterministic L4 bridge candidates returned for one anchor pair.
pub const MAX_BRIDGE_CANDIDATES: usize = 5;

/// A shared-neighbour bridge candidate (design 4.4.2 L4, deterministic part).
///
/// This is a *candidate*, not an edge: it must go through the completion ladder
/// and human review before anything is written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineeringBridgeCandidate {
    pub session_id: String,
    pub from_entity: String,
    pub to_entity: String,
    pub via_entity: String,
    pub relation: String,
    pub score: f32,
    pub evidence: Vec<String>,
}

fn neighbours_of(graph: &EngineeringGraph, id: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for relation in graph.relations.values() {
        if relation.from == id {
            out.push((
                relation.to.clone(),
                relation.id.clone(),
                relation.relation.clone(),
            ));
        } else if relation.to == id {
            out.push((
                relation.from.clone(),
                relation.id.clone(),
                relation.relation.clone(),
            ));
        }
    }
    out.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    out
}

/// One anchored subgraph returned by [`EngineeringStore::recall_anchor`].
#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringAnchorRecall {
    pub session_id: String,
    pub working_dir: Option<String>,
    pub graph: EngineeringGraph,
}

/// One scored path between two engineering anchors.
///
/// Labels are resolved at read time so callers can render the route without
/// re-opening the session graph.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringPathHit {
    pub session_id: String,
    pub working_dir: Option<String>,
    pub path: EngineeringPath,
    pub node_labels: Vec<String>,
    pub relation_labels: Vec<String>,
    pub plan: PathPlan,
    /// Runtime quality signals (4.6.9), including hypothesized completion edges.
    pub quality: PathQuality,
}

/// One cross-session alignment hint for an E09 gap.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringAlignmentCandidate {
    pub from_entity: String,
    pub to_entity: String,
    pub from_session: String,
    pub to_session: String,
    pub class: String,
    pub score: f32,
}

/// Structured diagnosis for a no-path engineering query (design 4.4.1/4.4.7).
#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringGapReport {
    pub from_anchor: String,
    pub to_anchor: String,
    pub depth: usize,
    /// Stable diagnosis codes: `E01`, `E03`, or `E09`.
    pub diagnosis: Vec<String>,
    pub from_sessions: Vec<String>,
    pub to_sessions: Vec<String>,
    pub shared_sessions: Vec<String>,
    /// Read-only alignment hints. Nothing is written to any graph.
    pub alignment_candidates: Vec<EngineeringAlignmentCandidate>,
    /// Relations of the diagnostic-only path found after relaxing depth or the
    /// relation allow-list. Empty when no relaxation recovers a path.
    pub path_relations: Vec<String>,
    pub next_actions: Vec<String>,
}

/// Completion debt for one engineering session (design 4.4.5/4.6).
#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringCompletionDebt {
    pub session_id: String,
    pub working_dir: Option<String>,
    pub edge_count: usize,
    pub hypothesis_edges: usize,
    pub debt: f32,
    /// Hypothesis edges expiring within the next 7 days.
    pub expiring_soon: usize,
}

/// Governance threshold from design 4.4.5: debt above this needs cleanup.
pub const MAX_ENGINEERING_COMPLETION_DEBT: f32 = 0.10;

/// One session selected by the engineering retention policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineeringRetentionCandidate {
    pub session_id: String,
    pub working_dir: Option<String>,
    pub entity_count: usize,
    pub relation_count: usize,
    pub updated_at: String,
    pub reason: String,
}

/// Retention summary for the engineering episode store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineeringRetentionReport {
    pub sessions: usize,
    pub entities: usize,
    pub relations: usize,
    pub storage_bytes: u64,
    pub empty_sessions: usize,
    pub candidates: Vec<EngineeringRetentionCandidate>,
    pub pruned: usize,
}

/// Result of rewriting the engineering store into compressed blobs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineeringCompactionReport {
    pub sessions: usize,
    pub before_bytes: u64,
    pub after_bytes: u64,
    pub compacted: usize,
}

/// Keep the newest two non-empty sessions and any session younger than 30 days.
/// Empty session-only graphs are always candidates for pruning.
pub const ENGINEERING_KEEP_LATEST: usize = 2;
pub const ENGINEERING_MAX_AGE_DAYS: i64 = 30;

const GRAPH_BLOB_MAGIC: &[u8] = b"JCEG1";

/// SQLite-backed store for engineering episode graphs.
pub struct EngineeringStore {
    path: PathBuf,
    connection: Connection,
}

impl std::fmt::Debug for EngineeringStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineeringStore")
            .field("path", &self.path)
            .finish()
    }
}

impl EngineeringStore {
    /// Open the default store at `<jcode_dir>/engineering/graphs.sqlite`.
    pub fn open_default() -> Result<Self> {
        let path = storage::jcode_dir()?
            .join("engineering")
            .join("graphs.sqlite");
        Self::open(path)
    }

    /// Open or create a store at `path`.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create engineering store dir {}", parent.display()))?;
        }
        let connection = Connection::open(&path)
            .with_context(|| format!("open engineering store {}", path.display()))?;
        configure_connection(&connection)?;
        let store = Self { path, connection };
        store.ensure_schema()?;
        store.ensure_entity_index()?;
        Ok(store)
    }

    /// Open an in-memory store for tests.
    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let connection =
            Connection::open_in_memory().context("open in-memory engineering store")?;
        configure_connection(&connection)?;
        let store = Self {
            path: PathBuf::from(":memory:"),
            connection,
        };
        store.ensure_schema()?;
        Ok(store)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn ensure_schema(&self) -> Result<()> {
        self.connection
            .execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS engineering_graphs (
                    session_id TEXT PRIMARY KEY,
                    working_dir TEXT,
                    ontology_id TEXT NOT NULL,
                    ontology_hash TEXT NOT NULL,
                    entity_count INTEGER NOT NULL,
                    relation_count INTEGER NOT NULL,
                    graph_json TEXT NOT NULL,
                    graph_blob BLOB,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS engineering_graphs_updated
                    ON engineering_graphs(updated_at DESC);
                CREATE TABLE IF NOT EXISTS engineering_entity_index (
                    session_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    class TEXT NOT NULL,
                    name TEXT NOT NULL,
                    PRIMARY KEY(session_id, entity_id)
                );
                CREATE INDEX IF NOT EXISTS engineering_entity_index_entity
                    ON engineering_entity_index(entity_id);
                "#,
            )
            .context("create engineering_graphs schema")?;
        let has_blob = self
            .connection
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('engineering_graphs') WHERE name = 'graph_blob'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .context("inspect engineering_graphs columns")?
            > 0;
        if !has_blob {
            self.connection
                .execute(
                    "ALTER TABLE engineering_graphs ADD COLUMN graph_blob BLOB",
                    [],
                )
                .context("add engineering_graphs.graph_blob")?;
        }
        Ok(())
    }

    /// Backfill the entity index for rows written before the index existed.
    fn ensure_entity_index(&self) -> Result<()> {
        let missing: Vec<String> = {
            let mut statement = self
                .connection
                .prepare(
                    r#"
                    SELECT session_id FROM engineering_graphs
                    WHERE session_id NOT IN (
                        SELECT DISTINCT session_id FROM engineering_entity_index
                    )
                    "#,
                )
                .context("prepare entity index backfill")?;
            let rows = statement
                .query_map([], |row| row.get(0))
                .context("query entity index backfill")?;
            let mut missing = Vec::new();
            for row in rows {
                missing.push(row.context("read entity index backfill row")?);
            }
            missing
        };
        for session_id in missing {
            let Some(graph) = self.load_graph(&session_id)? else {
                continue;
            };
            self.write_entity_index(&session_id, &graph)?;
        }
        Ok(())
    }

    fn write_entity_index(&self, session_id: &str, graph: &EngineeringGraph) -> Result<()> {
        let transaction = self
            .connection
            .unchecked_transaction()
            .context("begin entity index transaction")?;
        transaction
            .execute(
                "DELETE FROM engineering_entity_index WHERE session_id = ?1",
                params![session_id],
            )
            .context("clear entity index")?;
        {
            let mut statement = transaction
                .prepare(
                    "INSERT OR REPLACE INTO engineering_entity_index \
                     (session_id, entity_id, class, name) VALUES (?1, ?2, ?3, ?4)",
                )
                .context("prepare entity index insert")?;
            for entity in graph.entities.values() {
                statement
                    .execute(params![session_id, entity.id, entity.class, entity.name])
                    .context("insert entity index row")?;
            }
        }
        transaction.commit().context("commit entity index")?;
        Ok(())
    }

    /// Insert or replace one session graph.
    pub fn save_graph(
        &self,
        session_id: &str,
        working_dir: Option<&str>,
        graph: &EngineeringGraph,
    ) -> Result<()> {
        anyhow::ensure!(!session_id.trim().is_empty(), "session_id is empty");
        if graph.relations.is_empty()
            && graph
                .entities
                .values()
                .all(|entity| entity.class == CLASS_SESSION)
        {
            return Ok(());
        }
        let ontology = default_engineering_ontology();
        let graph_blob = encode_graph(graph)?;
        let now = Utc::now().to_rfc3339();
        let transaction = self
            .connection
            .unchecked_transaction()
            .context("begin engineering graph transaction")?;
        transaction
            .execute(
                r#"
                INSERT INTO engineering_graphs (
                    session_id, working_dir, ontology_id, ontology_hash,
                    entity_count, relation_count, graph_json, graph_blob, created_at, updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?9)
                ON CONFLICT(session_id) DO UPDATE SET
                    working_dir = excluded.working_dir,
                    ontology_id = excluded.ontology_id,
                    ontology_hash = excluded.ontology_hash,
                    entity_count = excluded.entity_count,
                    relation_count = excluded.relation_count,
                    graph_json = excluded.graph_json,
                    graph_blob = excluded.graph_blob,
                    updated_at = excluded.updated_at
                "#,
                params![
                    session_id,
                    working_dir,
                    ontology.id,
                    ontology.content_hash(),
                    graph.entities.len() as i64,
                    graph.relations.len() as i64,
                    "",
                    graph_blob,
                    now,
                ],
            )
            .context("upsert engineering graph")?;
        transaction
            .execute(
                "DELETE FROM engineering_entity_index WHERE session_id = ?1",
                params![session_id],
            )
            .context("clear engineering entity index")?;
        {
            let mut statement = transaction
                .prepare(
                    "INSERT OR REPLACE INTO engineering_entity_index \
                     (session_id, entity_id, class, name) VALUES (?1, ?2, ?3, ?4)",
                )
                .context("prepare engineering entity index")?;
            for entity in graph.entities.values() {
                statement
                    .execute(params![session_id, entity.id, entity.class, entity.name])
                    .context("insert engineering entity index row")?;
            }
        }
        transaction
            .commit()
            .context("commit engineering graph transaction")?;
        Ok(())
    }

    /// Load one session graph, or `None` if it has not been persisted.
    pub fn load_graph(&self, session_id: &str) -> Result<Option<EngineeringGraph>> {
        let row: Option<(String, Option<Vec<u8>>)> = self
            .connection
            .query_row(
                "SELECT graph_json, graph_blob FROM engineering_graphs WHERE session_id = ?1",
                params![session_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .context("load engineering graph")?;
        row.map(|(json, blob)| decode_graph(&json, blob.as_deref()))
            .transpose()
    }

    /// Persist candidate edges as hypothesis edges on one session graph.
    ///
    /// The graph is re-encoded and saved only when at least one candidate was
    /// accepted, so a rejected proposal never rewrites stored data.
    pub fn write_completion_edges(
        &self,
        session_id: &str,
        edges: &[CompletionEdge],
        source_tag: &str,
    ) -> Result<CompletionWriteReport> {
        let mut graph = self
            .load_graph(session_id)?
            .with_context(|| format!("no engineering graph stored for session {session_id}"))?;
        let report = graph.write_completion_edges(edges.iter().cloned(), source_tag);
        if !report.accepted.is_empty() {
            let working_dir = self.working_dir_of(session_id)?;
            self.save_graph(session_id, working_dir.as_deref(), &graph)?;
        }
        Ok(report)
    }

    /// Promote one hypothesis edge with non-LLM evidence and persist the graph.
    pub fn promote_completion_edge(
        &self,
        session_id: &str,
        edge_id: &str,
        evidence: EngineeringProvenance,
    ) -> Result<()> {
        let mut graph = self
            .load_graph(session_id)?
            .with_context(|| format!("no engineering graph stored for session {session_id}"))?;
        graph
            .promote_completion_edge(edge_id, evidence)
            .map_err(|error: CompletionPromotionError| anyhow::anyhow!(error))?;
        let working_dir = self.working_dir_of(session_id)?;
        self.save_graph(session_id, working_dir.as_deref(), &graph)
    }

    fn working_dir_of(&self, session_id: &str) -> Result<Option<String>> {
        self.connection
            .query_row(
                "SELECT working_dir FROM engineering_graphs WHERE session_id = ?1",
                params![session_id],
                |row| row.get(0),
            )
            .optional()
            .context("read engineering graph working_dir")
            .map(Option::flatten)
    }

    /// Soft-delete expired hypothesis edges on one session graph.
    pub fn purge_expired_hypotheses(&self, session_id: &str) -> Result<usize> {
        let Some(mut graph) = self.load_graph(session_id)? else {
            return Ok(0);
        };
        let removed = graph.purge_expired_hypotheses(Utc::now());
        if removed > 0 {
            let working_dir = self.working_dir_of(session_id)?;
            self.save_graph(session_id, working_dir.as_deref(), &graph)?;
        }
        Ok(removed)
    }

    /// Diagnose why an engineering path query returned nothing.
    ///
    /// This is read-only: it classifies the gap as an unresolved anchor (E01),
    /// a cross-session alignment gap (E09), or a same-session connectivity/depth
    /// gap (E03), and for E09 it ranks same-class cross-session entity pairs by
    /// lexical overlap as alignment hints. Hints are never written.
    pub fn diagnose_path_gap(
        &self,
        from_anchor: &str,
        to_anchor: &str,
        depth: usize,
        allowed_relations: &[&str],
    ) -> Result<EngineeringGapReport> {
        const MIN_ALIGNMENT_SCORE: f32 = 0.55;
        const MAX_ALIGNMENT_CANDIDATES: usize = 3;

        let from_rows = self.anchor_rows(from_anchor)?;
        let to_rows = self.anchor_rows(to_anchor)?;
        let from_sessions: Vec<String> = {
            let mut sessions: Vec<String> = from_rows.iter().map(|row| row.0.clone()).collect();
            sessions.sort();
            sessions.dedup();
            sessions
        };
        let to_sessions: Vec<String> = {
            let mut sessions: Vec<String> = to_rows.iter().map(|row| row.0.clone()).collect();
            sessions.sort();
            sessions.dedup();
            sessions
        };
        let shared_sessions: Vec<String> = from_sessions
            .iter()
            .filter(|session| to_sessions.contains(session))
            .cloned()
            .collect();

        let mut diagnosis = Vec::new();
        let mut next_actions = Vec::new();
        let concept_graph = if from_rows.is_empty() || to_rows.is_empty() {
            Some(EngineeringGraph::concept_graph(
                &default_engineering_ontology(),
            ))
        } else {
            None
        };
        if from_rows.is_empty() {
            let is_concept = concept_graph
                .as_ref()
                .is_some_and(|graph| graph.resolve_concept_anchor(from_anchor).is_some());
            let code = if is_concept { "E08" } else { "E01" };
            if !diagnosis.contains(&code.to_string()) {
                diagnosis.push(code.to_string());
            }
            if is_concept {
                next_actions.push(format!(
                    "E08: start anchor '{from_anchor}' is an ontology concept, not an engineering entity; re-run with graph=ontology"
                ));
            } else {
                next_actions.push(format!(
                    "E01: start anchor '{from_anchor}' is not in any engineering graph; check the id or recall it first"
                ));
            }
        }
        if to_rows.is_empty() {
            let is_concept = concept_graph
                .as_ref()
                .is_some_and(|graph| graph.resolve_concept_anchor(to_anchor).is_some());
            let code = if is_concept { "E08" } else { "E01" };
            if !diagnosis.contains(&code.to_string()) {
                diagnosis.push(code.to_string());
            }
            if is_concept {
                next_actions.push(format!(
                    "E08: goal anchor '{to_anchor}' is an ontology concept, not an engineering entity; re-run with graph=ontology"
                ));
            } else {
                next_actions.push(format!(
                    "E01: goal anchor '{to_anchor}' is not in any engineering graph; check the id or recall it first"
                ));
            }
        }

        let mut alignment_candidates = Vec::new();
        let mut path_relations = Vec::new();
        if diagnosis.is_empty() {
            if shared_sessions.is_empty() {
                diagnosis.push("E09".to_string());
                next_actions.push(format!(
                    "E09: anchors live in disjoint sessions ({} vs {}); propose a same_as alignment with action=completion completion_action=propose and evidence",
                    from_sessions.join(", "),
                    to_sessions.join(", ")
                ));
                let left = self.session_entities(&from_sessions)?;
                let right = self.session_entities(&to_sessions)?;
                let mut candidates = Vec::new();
                for (left_session, left_id, left_class, left_name) in &left {
                    for (right_session, right_id, right_class, right_name) in &right {
                        if left_id == right_id || left_class != right_class {
                            continue;
                        }
                        let score = lexical_overlap(left_name, right_name);
                        if score < MIN_ALIGNMENT_SCORE {
                            continue;
                        }
                        candidates.push(EngineeringAlignmentCandidate {
                            from_entity: left_id.clone(),
                            to_entity: right_id.clone(),
                            from_session: left_session.clone(),
                            to_session: right_session.clone(),
                            class: left_class.clone(),
                            score,
                        });
                    }
                }
                candidates.sort_by(|left, right| {
                    right
                        .score
                        .total_cmp(&left.score)
                        .then_with(|| left.from_entity.cmp(&right.from_entity))
                        .then_with(|| left.to_entity.cmp(&right.to_entity))
                });
                candidates.truncate(MAX_ALIGNMENT_CANDIDATES);
                alignment_candidates = candidates;
            } else {
                let probe_depth = (depth + 2).min(16);
                let mut deeper: Option<Vec<String>> = None;
                let mut unfiltered: Option<Vec<String>> = None;
                let mut inferred: Option<Vec<String>> = None;
                for session in &shared_sessions {
                    let Some(from_id) = from_rows
                        .iter()
                        .find(|row| &row.0 == session)
                        .map(|row| row.1.as_str())
                    else {
                        continue;
                    };
                    let Some(to_id) = to_rows
                        .iter()
                        .find(|row| &row.0 == session)
                        .map(|row| row.1.as_str())
                    else {
                        continue;
                    };
                    let Some(graph) = self.load_graph(session)? else {
                        continue;
                    };
                    if deeper.is_none()
                        && probe_depth > depth
                        && let Some(path) = graph.weighted_shortest_path(
                            from_id,
                            to_id,
                            probe_depth,
                            allowed_relations,
                        )
                    {
                        deeper = Some(path_relation_names(&graph, &path));
                    }
                    if unfiltered.is_none()
                        && !allowed_relations.is_empty()
                        && let Some(path) = graph.weighted_shortest_path(from_id, to_id, depth, &[])
                    {
                        unfiltered = Some(path_relation_names(&graph, &path));
                    }
                    if inferred.is_none() {
                        let inferred_graph = graph.infer_relations(&default_engineering_ontology());
                        if let Some(path) = inferred_graph.weighted_shortest_path(
                            from_id,
                            to_id,
                            probe_depth,
                            allowed_relations,
                        ) {
                            inferred = Some(path_relation_names(&inferred_graph, &path));
                        } else if let Some(path) = graph.weighted_shortest_path(
                            to_id,
                            from_id,
                            probe_depth,
                            allowed_relations,
                        ) {
                            let mut relations = path_relation_names(&graph, &path);
                            relations.push("reverse_direction".to_string());
                            inferred = Some(relations);
                        }
                    }
                }

                if let Some(relations) = unfiltered {
                    diagnosis.push("E04".to_string());
                    next_actions.push(format!(
                        "E04: a path exists when relation_allow is relaxed; the route uses [{}]; re-run without the filter or allow those relations",
                        relations.join(", ")
                    ));
                    path_relations = relations;
                }
                if let Some(relations) = deeper {
                    diagnosis.push("E05".to_string());
                    next_actions.push(format!(
                        "E05: a path exists at depth {probe_depth}; raise depth to at least {probe_depth}"
                    ));
                    if path_relations.is_empty() {
                        path_relations = relations;
                    }
                }
                if diagnosis.is_empty()
                    && let Some(relations) = inferred
                {
                    diagnosis.push("E10".to_string());
                    next_actions.push(format!(
                        "E10: a path exists only with inferred inverse/sub-property edges or in the reverse direction [{}]; enable inference, allow the parent relation, or verify the edge direction",
                        relations.join(", ")
                    ));
                    path_relations = relations;
                }
                if diagnosis.is_empty() {
                    diagnosis.push("E03".to_string());
                    next_actions.push(format!(
                        "E03: anchors exist in session(s) {} but no path exists within depth {probe_depth} even without a relation filter; propose a completion edge or check extraction",
                        shared_sessions.join(", ")
                    ));
                }
            }
        }

        Ok(EngineeringGapReport {
            from_anchor: from_anchor.to_string(),
            to_anchor: to_anchor.to_string(),
            depth,
            diagnosis,
            from_sessions,
            to_sessions,
            shared_sessions,
            alignment_candidates,
            path_relations,
            next_actions,
        })
    }

    /// Completion debt per session that still carries hypothesis edges.
    ///
    /// Sessions without pending candidates are omitted so the quality board
    /// stays short; the scan itself is bounded by `limit` recent sessions.
    pub fn completion_debt_by_session(
        &self,
        limit: usize,
    ) -> Result<Vec<EngineeringCompletionDebt>> {
        let soon = Utc::now() + Duration::days(7);
        let mut out = Vec::new();
        for record in self.list_sessions(limit)? {
            let Some(graph) = self.load_graph(&record.session_id)? else {
                continue;
            };
            let hypothesis: Vec<&EngineeringRelation> = graph.hypothesis_edges();
            if hypothesis.is_empty() {
                continue;
            }
            let expiring_soon = hypothesis
                .iter()
                .filter(|edge| edge.valid_to.is_some_and(|until| until <= soon))
                .count();
            out.push(EngineeringCompletionDebt {
                session_id: record.session_id,
                working_dir: record.working_dir,
                edge_count: graph.relations.len(),
                hypothesis_edges: hypothesis.len(),
                debt: graph.completion_debt(),
                expiring_soon,
            });
        }
        Ok(out)
    }

    /// `(session_id, entity_id, class, name)` rows matching an id or display name.
    fn anchor_rows(&self, anchor: &str) -> Result<Vec<(String, String, String, String)>> {
        let mut statement = self
            .connection
            .prepare(
                r#"
                SELECT session_id, entity_id, class, name
                FROM engineering_entity_index
                WHERE entity_id = ?1 OR name = ?1 COLLATE NOCASE
                ORDER BY session_id, entity_id
                "#,
            )
            .context("prepare anchor resolution")?;
        let rows = statement
            .query_map(params![anchor], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .context("query anchor resolution")?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.context("read anchor row")?);
        }
        Ok(out)
    }

    /// All indexed entities for the given sessions, capped per session.
    fn session_entities(
        &self,
        sessions: &[String],
    ) -> Result<Vec<(String, String, String, String)>> {
        const PER_SESSION_CAP: usize = 64;
        let mut out = Vec::new();
        let mut statement = self
            .connection
            .prepare(
                "SELECT entity_id, class, name FROM engineering_entity_index \
                 WHERE session_id = ?1 ORDER BY entity_id LIMIT ?2",
            )
            .context("prepare session entities")?;
        for session in sessions {
            let rows = statement
                .query_map(params![session, PER_SESSION_CAP as i64], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })
                .context("query session entities")?;
            for row in rows {
                let (id, class, name) = row.context("read session entity")?;
                out.push((session.clone(), id, class, name));
            }
        }
        Ok(out)
    }

    /// List stored sessions, newest first.
    pub fn list_sessions(&self, limit: usize) -> Result<Vec<EngineeringSessionRecord>> {
        let mut statement = self
            .connection
            .prepare(
                r#"
                SELECT session_id, working_dir, ontology_id, ontology_hash,
                       entity_count, relation_count, updated_at
                FROM engineering_graphs
                ORDER BY updated_at DESC
                LIMIT ?1
                "#,
            )
            .context("prepare session list")?;
        let rows = statement
            .query_map(params![limit as i64], |row| {
                Ok(EngineeringSessionRecord {
                    session_id: row.get(0)?,
                    working_dir: row.get(1)?,
                    ontology_id: row.get(2)?,
                    ontology_hash: row.get(3)?,
                    entity_count: row.get::<_, i64>(4)?.max(0) as usize,
                    relation_count: row.get::<_, i64>(5)?.max(0) as usize,
                    updated_at: row.get(6)?,
                })
            })
            .context("query session list")?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.context("read session row")?);
        }
        Ok(out)
    }

    /// Link a mention to concrete entities across stored session graphs.
    ///
    /// Deterministic lexical matching only: exact name/id, `aliases` property,
    /// token containment, id prefix and name-token overlap. No embedding or
    /// ANN lookup is applied, so results are ordered by lexical score and then
    /// by name/session/id for stability.
    pub fn link_entities(
        &self,
        mention: &str,
        class_filter: Option<&str>,
        limit: usize,
    ) -> Result<Vec<EngineeringEntityLink>> {
        if limit == 0 || mention.trim().is_empty() {
            return Ok(Vec::new());
        }
        let limit = limit.min(MAX_ENTITY_LINKS * 4);
        let sessions = self.list_sessions(1024)?;
        let mut links = Vec::new();
        for session in sessions {
            let Some(graph) = self.load_graph(&session.session_id)? else {
                continue;
            };
            for entity in graph.entities.values() {
                if entity.id == session.session_id && entity.class == CLASS_SESSION {
                    continue;
                }
                if let Some(class_filter) = class_filter
                    && entity.class != class_filter
                {
                    continue;
                }
                if let Some((score, signals)) = score_entity(mention, entity) {
                    links.push(EngineeringEntityLink {
                        session_id: session.session_id.clone(),
                        entity_id: entity.id.clone(),
                        name: entity.name.clone(),
                        class: entity.class.clone(),
                        score,
                        signals,
                    });
                }
            }
        }
        links.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.name.cmp(&right.name))
                .then_with(|| left.session_id.cmp(&right.session_id))
                .then_with(|| left.entity_id.cmp(&right.entity_id))
        });
        links.truncate(limit);
        Ok(links)
    }

    /// Propose bridge candidates between two entity ids that share a neighbour.
    ///
    /// Deterministic L4 shared-neighbour probe: for every common neighbour `w`
    /// of `from_entity` and `to_entity`, emit a candidate relation from the
    /// `from` side with the neighbour's relation name and both edge ids as
    /// evidence. Candidates are hypotheses for the completion ladder, never
    /// written edges.
    pub fn shared_neighbor_candidates(
        &self,
        from_entity: &str,
        to_entity: &str,
        limit: usize,
    ) -> Result<Vec<EngineeringBridgeCandidate>> {
        if limit == 0 || from_entity == to_entity {
            return Ok(Vec::new());
        }
        let mut candidates: Vec<EngineeringBridgeCandidate> = Vec::new();
        for session in self.list_sessions(1024)? {
            let Some(graph) = self.load_graph(&session.session_id)? else {
                continue;
            };
            let from_neighbours = neighbours_of(&graph, from_entity);
            let to_neighbours = neighbours_of(&graph, to_entity);
            let mut seen = BTreeSet::new();
            for (via, from_relation_id, relation) in &from_neighbours {
                if via == from_entity || via == to_entity {
                    continue;
                }
                let Some((_, to_relation_id, _)) = to_neighbours
                    .iter()
                    .find(|(neighbour, _, _)| neighbour == via)
                else {
                    continue;
                };
                if !seen.insert(via.clone()) {
                    continue;
                }
                let degree = neighbours_of(&graph, via).len() as f32;
                let score = 1.0 / (1.0 + degree);
                candidates.push(EngineeringBridgeCandidate {
                    session_id: session.session_id.clone(),
                    from_entity: from_entity.to_string(),
                    to_entity: to_entity.to_string(),
                    via_entity: via.clone(),
                    relation: relation.clone(),
                    score,
                    evidence: vec![from_relation_id.clone(), to_relation_id.clone()],
                });
            }
        }
        candidates.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.via_entity.cmp(&right.via_entity))
                .then_with(|| left.session_id.cmp(&right.session_id))
        });
        candidates.truncate(limit.min(MAX_BRIDGE_CANDIDATES));
        Ok(candidates)
    }

    /// Delete one session graph. Returns whether a row existed.
    pub fn delete_graph(&self, session_id: &str) -> Result<bool> {
        let transaction = self
            .connection
            .unchecked_transaction()
            .context("begin delete engineering graph")?;
        let removed = transaction
            .execute(
                "DELETE FROM engineering_graphs WHERE session_id = ?1",
                params![session_id],
            )
            .context("delete engineering graph")?;
        transaction
            .execute(
                "DELETE FROM engineering_entity_index WHERE session_id = ?1",
                params![session_id],
            )
            .context("delete engineering entity index")?;
        transaction
            .commit()
            .context("commit delete engineering graph")?;
        Ok(removed > 0)
    }

    /// Report which sessions fall outside the default retention policy.
    pub fn retention_report(&self) -> Result<EngineeringRetentionReport> {
        self.retention_pass(false)
    }

    /// Apply the default retention policy. `apply=false` is a dry run.
    pub fn prune_sessions(&self, apply: bool) -> Result<EngineeringRetentionReport> {
        self.retention_pass(apply)
    }

    /// Rewrite stored JSON graphs as compressed blobs. Existing blobs are only
    /// rewritten when the new blob is smaller.
    pub fn compact_storage(&self, apply: bool) -> Result<EngineeringCompactionReport> {
        let rows: Vec<(String, i64, i64)> = {
            let mut statement = self
                .connection
                .prepare(
                    "SELECT session_id, length(graph_json), coalesce(length(graph_blob), 0) \
                     FROM engineering_graphs",
                )
                .context("prepare engineering compaction")?;
            let rows = statement
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                })
                .context("query engineering compaction")?;
            let mut rows_out = Vec::new();
            for row in rows {
                rows_out.push(row.context("read engineering compaction row")?);
            }
            rows_out
        };
        let mut report = EngineeringCompactionReport {
            sessions: rows.len(),
            before_bytes: 0,
            after_bytes: 0,
            compacted: 0,
        };
        for (session_id, json_bytes, blob_bytes) in rows {
            let before = json_bytes.max(0) as u64 + blob_bytes.max(0) as u64;
            report.before_bytes += before;
            let Some(graph) = self.load_graph(&session_id)? else {
                report.after_bytes += before;
                continue;
            };
            let blob = encode_graph(&graph)?;
            let after = blob.len() as u64;
            let target = after.min(before);
            report.after_bytes += target;
            if apply && after < before {
                self.connection
                    .execute(
                        "UPDATE engineering_graphs SET graph_json = '', graph_blob = ?1 \
                         WHERE session_id = ?2",
                        params![blob, session_id],
                    )
                    .context("compact engineering graph")?;
                self.write_entity_index(&session_id, &graph)?;
                report.compacted += 1;
            }
        }
        Ok(report)
    }

    fn retention_pass(&self, apply: bool) -> Result<EngineeringRetentionReport> {
        let sessions = self.list_sessions(10_000)?;
        let now = Utc::now();
        let non_empty: Vec<&EngineeringSessionRecord> = sessions
            .iter()
            .filter(|session| session.entity_count > 1 || session.relation_count > 0)
            .collect();
        let newest_non_empty: std::collections::BTreeSet<&str> = non_empty
            .iter()
            .take(ENGINEERING_KEEP_LATEST)
            .map(|session| session.session_id.as_str())
            .collect();

        let mut candidates = Vec::new();
        for session in &sessions {
            let empty = session.entity_count <= 1 && session.relation_count == 0;
            let old = !newest_non_empty.contains(session.session_id.as_str())
                && session
                    .updated_at
                    .parse::<DateTime<Utc>>()
                    .ok()
                    .is_some_and(|updated| {
                        now.signed_duration_since(updated).num_days() > ENGINEERING_MAX_AGE_DAYS
                    });
            let reason = if empty {
                "empty session graph"
            } else if old {
                "older than 30 days and outside newest two sessions"
            } else {
                continue;
            };
            candidates.push(EngineeringRetentionCandidate {
                session_id: session.session_id.clone(),
                working_dir: session.working_dir.clone(),
                entity_count: session.entity_count,
                relation_count: session.relation_count,
                updated_at: session.updated_at.clone(),
                reason: reason.to_string(),
            });
        }

        let mut pruned = 0;
        if apply {
            for candidate in &candidates {
                let removed = self
                    .connection
                    .execute(
                        "DELETE FROM engineering_graphs WHERE session_id = ?1",
                        params![candidate.session_id],
                    )
                    .context("prune engineering session")?;
                self.connection
                    .execute(
                        "DELETE FROM engineering_entity_index WHERE session_id = ?1",
                        params![candidate.session_id],
                    )
                    .context("prune engineering entity index")?;
                pruned += removed;
            }
        }
        let storage_bytes = [
            &self.path,
            &self.path.with_extension("sqlite-wal"),
            &self.path.with_extension("sqlite-shm"),
        ]
        .into_iter()
        .filter_map(|path| std::fs::metadata(path).ok())
        .map(|metadata| metadata.len())
        .sum();
        Ok(EngineeringRetentionReport {
            sessions: sessions.len(),
            entities: sessions.iter().map(|session| session.entity_count).sum(),
            relations: sessions.iter().map(|session| session.relation_count).sum(),
            storage_bytes,
            empty_sessions: sessions
                .iter()
                .filter(|session| session.entity_count <= 1 && session.relation_count == 0)
                .count(),
            candidates,
            pruned,
        })
    }

    /// Find sessions whose graph contains `anchor_id`, then return the subgraph
    /// within `depth` hops of that anchor.
    pub fn recall_anchor(
        &self,
        anchor_id: &str,
        depth: usize,
        max_sessions: usize,
    ) -> Result<Vec<EngineeringAnchorRecall>> {
        if anchor_id.trim().is_empty() || max_sessions == 0 {
            return Ok(Vec::new());
        }
        let pattern = format!("%{}%", like_escape(anchor_id));
        let mut statement = self
            .connection
            .prepare(
                r#"
                SELECT e.session_id, e.working_dir, e.graph_json, e.graph_blob
                FROM engineering_graphs e
                WHERE EXISTS (
                    SELECT 1 FROM engineering_entity_index i
                    WHERE i.session_id = e.session_id
                      AND i.entity_id LIKE ?1 ESCAPE '\'
                )
                ORDER BY e.updated_at DESC
                "#,
            )
            .context("prepare anchor recall")?;
        let rows = statement
            .query_map(params![pattern], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<Vec<u8>>>(3)?,
                ))
            })
            .context("query anchor recall")?;

        let mut out = Vec::new();
        for row in rows {
            let (session_id, working_dir, json, blob) = row.context("read anchor row")?;
            let graph = match decode_graph(&json, blob.as_deref()) {
                Ok(graph) => graph,
                Err(error) => {
                    crate::logging::warn(&format!(
                        "engineering store skipped malformed graph for {session_id}: {error}"
                    ));
                    continue;
                }
            };
            if let Some(subgraph) = subgraph_around(&graph, anchor_id, depth) {
                out.push(EngineeringAnchorRecall {
                    session_id,
                    working_dir,
                    graph: subgraph,
                });
            }
            if out.len() >= max_sessions {
                break;
            }
        }
        Ok(out)
    }

    /// Find stored engineering paths between two stable anchors.
    ///
    /// Only sessions whose entity index contains both anchors are decoded. The
    /// result is sorted by path cost, then session id and node sequence so the
    /// same store always yields the same route order.
    pub fn path_between_anchors(
        &self,
        from_anchor: &str,
        to_anchor: &str,
        depth: usize,
        limit: usize,
        allowed_relations: &[&str],
    ) -> Result<Vec<EngineeringPathHit>> {
        self.path_between_anchors_impl(
            from_anchor,
            to_anchor,
            depth,
            limit,
            allowed_relations,
            false,
        )
    }

    /// Query engineering instances with the ontology concept layer joined at
    /// query time.
    ///
    /// Entities gain `instance_of` edges to their class nodes and classes expose
    /// `has_instance` back-edges, so a path can go
    /// `instance -> concept -> concept -> instance`. The persisted graph is not
    /// modified.
    pub fn path_between_anchors_with_ontology(
        &self,
        from_anchor: &str,
        to_anchor: &str,
        depth: usize,
        limit: usize,
        allowed_relations: &[&str],
    ) -> Result<Vec<EngineeringPathHit>> {
        self.path_between_anchors_impl(
            from_anchor,
            to_anchor,
            depth,
            limit,
            allowed_relations,
            true,
        )
    }

    fn path_between_anchors_impl(
        &self,
        from_anchor: &str,
        to_anchor: &str,
        depth: usize,
        limit: usize,
        allowed_relations: &[&str],
        include_ontology: bool,
    ) -> Result<Vec<EngineeringPathHit>> {
        let from_anchor = from_anchor.trim();
        let to_anchor = to_anchor.trim();
        if from_anchor.is_empty() || to_anchor.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let depth = depth.clamp(1, 16);
        let limit = limit.min(20);
        let mut statement = self
            .connection
            .prepare(
                r#"
                SELECT e.session_id, e.working_dir, e.graph_json, e.graph_blob
                FROM engineering_graphs e
                WHERE EXISTS (
                    SELECT 1 FROM engineering_entity_index i
                    WHERE i.session_id = e.session_id AND i.entity_id = ?1
                )
                  AND EXISTS (
                    SELECT 1 FROM engineering_entity_index i
                    WHERE i.session_id = e.session_id AND i.entity_id = ?2
                )
                ORDER BY e.updated_at DESC
                "#,
            )
            .context("prepare engineering path query")?;
        let rows = statement
            .query_map(params![from_anchor, to_anchor], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<Vec<u8>>>(3)?,
                ))
            })
            .context("query engineering paths")?;

        let mut hits = Vec::new();
        for row in rows {
            let (session_id, working_dir, json, blob) = row.context("read engineering path row")?;
            let graph = match decode_graph(&json, blob.as_deref()) {
                Ok(graph) => graph,
                Err(error) => {
                    crate::logging::warn(&format!(
                        "engineering store skipped malformed graph for {session_id}: {error}"
                    ));
                    continue;
                }
            };
            let ontology = default_engineering_ontology();
            let graph = if include_ontology {
                graph.with_ontology_layer(&ontology)
            } else {
                graph
            };
            let graph = graph.infer_relations(&ontology);
            for path in
                graph.k_shortest_paths(from_anchor, to_anchor, depth, allowed_relations, limit)
            {
                let node_labels = path
                    .nodes
                    .iter()
                    .map(|id| {
                        graph
                            .entity(id)
                            .map(|entity| entity.name.clone())
                            .unwrap_or_else(|| id.clone())
                    })
                    .collect();
                let relation_labels = path
                    .relations
                    .iter()
                    .map(|id| {
                        graph
                            .relation(id)
                            .map(|relation| relation.relation.clone())
                            .unwrap_or_else(|| id.clone())
                    })
                    .collect();
                let plan = build_path_plan(&graph, &path);
                let quality = graph.quality_of_path(&path);
                hits.push(EngineeringPathHit {
                    session_id: session_id.clone(),
                    working_dir: working_dir.clone(),
                    path,
                    node_labels,
                    relation_labels,
                    plan,
                    quality,
                });
            }
        }
        hits.sort_by(|left, right| {
            left.path
                .cost
                .total_cmp(&right.path.cost)
                .then_with(|| left.session_id.cmp(&right.session_id))
                .then_with(|| left.path.nodes.cmp(&right.path.nodes))
        });
        hits.truncate(limit);
        Ok(hits)
    }

    /// Query the ontology concept layer (`G_o`) without touching session stores.
    ///
    /// Concept anchors resolve by `concept:<class>`, bare class id, or display
    /// name. Returns the same [`EngineeringPathHit`] shape as instance paths so
    /// callers can render and plan both layers uniformly.
    pub fn ontology_path_between_anchors(
        from_anchor: &str,
        to_anchor: &str,
        depth: usize,
        limit: usize,
        allowed_relations: &[&str],
    ) -> Result<Vec<EngineeringPathHit>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let ontology = default_engineering_ontology();
        let graph = EngineeringGraph::concept_graph(&ontology);
        let Some(from) = graph.resolve_concept_anchor(from_anchor) else {
            return Ok(Vec::new());
        };
        let Some(to) = graph.resolve_concept_anchor(to_anchor) else {
            return Ok(Vec::new());
        };
        let depth = depth.clamp(1, 16);
        let limit = limit.min(20);
        let mut hits = Vec::new();
        for path in graph.k_shortest_paths(&from, &to, depth, allowed_relations, limit) {
            let node_labels = path
                .nodes
                .iter()
                .map(|id| {
                    graph
                        .entity(id)
                        .map(|entity| entity.name.clone())
                        .unwrap_or_else(|| id.clone())
                })
                .collect();
            let relation_labels = path
                .relations
                .iter()
                .map(|id| {
                    graph
                        .relation(id)
                        .map(|relation| relation.relation.clone())
                        .unwrap_or_else(|| id.clone())
                })
                .collect();
            let plan = build_path_plan(&graph, &path);
            let quality = graph.quality_of_path(&path);
            hits.push(EngineeringPathHit {
                session_id: "ontology".to_string(),
                working_dir: None,
                path,
                node_labels,
                relation_labels,
                plan,
                quality,
            });
        }
        hits.truncate(limit);
        Ok(hits)
    }
}

fn configure_connection(connection: &Connection) -> Result<()> {
    connection
        .execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA busy_timeout = 5000;
            "#,
        )
        .context("configure engineering store connection")?;
    Ok(())
}

fn encode_graph(graph: &EngineeringGraph) -> Result<Vec<u8>> {
    let json = serde_json::to_vec(graph).context("serialize engineering graph")?;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(&json)
        .context("compress engineering graph")?;
    let body = encoder.finish().context("finish engineering graph gzip")?;
    let mut blob = Vec::with_capacity(GRAPH_BLOB_MAGIC.len() + body.len());
    blob.extend_from_slice(GRAPH_BLOB_MAGIC);
    blob.extend_from_slice(&body);
    Ok(blob)
}

fn decode_graph(json: &str, blob: Option<&[u8]>) -> Result<EngineeringGraph> {
    if let Some(blob) = blob {
        anyhow::ensure!(
            blob.starts_with(GRAPH_BLOB_MAGIC),
            "unknown engineering graph blob encoding"
        );
        let mut decoder = GzDecoder::new(&blob[GRAPH_BLOB_MAGIC.len()..]);
        let mut json = Vec::new();
        decoder
            .read_to_end(&mut json)
            .context("decompress engineering graph")?;
        return serde_json::from_slice(&json).context("parse compressed engineering graph");
    }
    serde_json::from_str(json).context("parse engineering graph")
}

fn like_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// Token Jaccard overlap over alphanumeric words of length 3+.
fn lexical_overlap(left: &str, right: &str) -> f32 {
    let tokens = |text: &str| -> std::collections::HashSet<String> {
        text.split(|ch: char| !ch.is_alphanumeric())
            .filter(|token| token.len() >= 3)
            .map(str::to_lowercase)
            .collect()
    };
    let left = tokens(left);
    let right = tokens(right);
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let shared = left.intersection(&right).count();
    shared as f32 / left.union(&right).count() as f32
}

/// Deduplicated relation names along one path, falling back to the edge id.
fn path_relation_names(graph: &EngineeringGraph, path: &EngineeringPath) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for id in &path.relations {
        let name = graph
            .relation(id)
            .map(|relation| relation.relation.clone())
            .unwrap_or_else(|| id.clone());
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

fn subgraph_around(
    graph: &EngineeringGraph,
    anchor_id: &str,
    depth: usize,
) -> Option<EngineeringGraph> {
    if !graph.entities.contains_key(anchor_id) {
        return None;
    }
    let mut visited: BTreeSet<String> = BTreeSet::from([anchor_id.to_string()]);
    let mut frontier = VecDeque::from([(anchor_id.to_string(), 0usize)]);
    while let Some((node, current_depth)) = frontier.pop_front() {
        if current_depth >= depth {
            continue;
        }
        for relation in graph.relations.values() {
            let next = if relation.from == node {
                Some(&relation.to)
            } else if relation.to == node {
                Some(&relation.from)
            } else {
                None
            };
            let Some(next) = next else {
                continue;
            };
            if visited.insert(next.clone()) {
                frontier.push_back((next.clone(), current_depth + 1));
            }
        }
    }

    let entities: BTreeMap<String, EngineeringEntity> = visited
        .iter()
        .filter_map(|id| {
            graph
                .entities
                .get(id)
                .cloned()
                .map(|entity| (id.clone(), entity))
        })
        .collect();
    let relations: BTreeMap<String, EngineeringRelation> = graph
        .relations
        .iter()
        .filter(|(_, relation)| visited.contains(&relation.from) && visited.contains(&relation.to))
        .map(|(id, relation)| (id.clone(), relation.clone()))
        .collect();
    Some(EngineeringGraph {
        entities,
        relations,
        concept_layer: graph.concept_layer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use jcode_memory_types::engineering_ontology::{
        CLASS_COMMAND, CLASS_ERROR, CLASS_FILE, CLASS_SESSION, CLASS_TASK, EngineeringProvenance,
        REL_CAUSES, REL_CONTAINS, REL_DEFINES, REL_DEPENDS_ON, REL_MODIFIES, REL_PART_OF,
        REL_TESTS,
    };
    use serde_json::json;

    fn sample_graph() -> EngineeringGraph {
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(
            EngineeringEntity::new(CLASS_FILE, "file:src/lib.rs", "src/lib.rs")
                .with_property("path", json!("src/lib.rs"))
                .with_provenance(EngineeringProvenance::new("session:test")),
        );
        graph.insert_entity(
            EngineeringEntity::new(CLASS_COMMAND, "command:cargo-test", "cargo test")
                .with_property("command", json!("cargo test"))
                .with_provenance(EngineeringProvenance::new("session:test")),
        );
        graph.insert_entity(
            EngineeringEntity::new(CLASS_ERROR, "error:boom", "boom")
                .with_property("signature", json!("boom"))
                .with_provenance(EngineeringProvenance::new("session:test")),
        );
        graph.insert_relation(EngineeringRelation::new(
            "command:cargo-test",
            REL_CAUSES,
            "error:boom",
        ));
        graph.insert_relation(EngineeringRelation::new(
            "command:cargo-test",
            REL_MODIFIES,
            "file:src/lib.rs",
        ));
        graph
    }

    #[test]
    fn save_and_load_round_trip() {
        let store = EngineeringStore::open_in_memory().expect("store");
        let graph = sample_graph();
        store
            .save_graph("session-1", Some("/repo"), &graph)
            .expect("save");
        let loaded = store.load_graph("session-1").expect("load").expect("graph");
        assert_eq!(loaded, graph);
        assert_eq!(store.list_sessions(10).expect("list").len(), 1);
        assert_eq!(store.list_sessions(10).expect("list")[0].entity_count, 3);
    }

    #[test]
    fn recall_anchor_returns_neighbourhood() {
        let store = EngineeringStore::open_in_memory().expect("store");
        store
            .save_graph("session-1", None, &sample_graph())
            .expect("save");
        let recalls = store
            .recall_anchor("file:src/lib.rs", 2, 10)
            .expect("recall");
        assert_eq!(recalls.len(), 1);
        let graph = &recalls[0].graph;
        assert!(graph.entities.contains_key("file:src/lib.rs"));
        assert!(graph.entities.contains_key("command:cargo-test"));
        assert!(graph.entities.contains_key("error:boom"));
        assert_eq!(graph.relations.len(), 2);
    }

    #[test]
    fn path_between_anchors_returns_scored_typed_route() {
        let store = EngineeringStore::open_in_memory().expect("store");
        store
            .save_graph("session-1", Some("/repo"), &sample_graph())
            .expect("save");

        let hits = store
            .path_between_anchors("command:cargo-test", "file:src/lib.rs", 3, 5, &["modifies"])
            .expect("path");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].session_id, "session-1");
        assert_eq!(
            hits[0].path.nodes,
            vec![
                "command:cargo-test".to_string(),
                "file:src/lib.rs".to_string()
            ]
        );
        assert_eq!(hits[0].relation_labels, vec!["modifies"]);
        assert_eq!(
            hits[0].node_labels,
            vec!["cargo test".to_string(), "src/lib.rs".to_string()]
        );
        assert_eq!(hits[0].plan.tasks.len(), 1);
        assert_eq!(
            hits[0].plan.tasks[0].action_type,
            jcode_memory_types::PlanActionType::Execute
        );
        assert!(hits[0].plan.tasks[0].approval);
        assert!(hits[0].working_dir.as_deref() == Some("/repo"));

        let filtered = store
            .path_between_anchors("command:cargo-test", "file:src/lib.rs", 3, 5, &["causes"])
            .expect("filtered path");
        assert!(filtered.is_empty());
    }

    #[test]
    fn path_query_uses_inferred_inverse_relations() {
        use jcode_memory_types::engineering_ontology::{CLASS_SESSION, REL_CONTAINS, REL_PART_OF};

        let store = EngineeringStore::open_in_memory().expect("store");
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(EngineeringEntity::new(CLASS_FILE, "file:a", "a.rs"));
        graph.insert_entity(EngineeringEntity::new(
            CLASS_SESSION,
            "session:s",
            "session",
        ));
        graph.insert_relation(EngineeringRelation::new("file:a", REL_PART_OF, "session:s"));
        store
            .save_graph("session-inverse", None, &graph)
            .expect("save graph");

        let hits = store
            .path_between_anchors("session:s", "file:a", 2, 5, &[REL_CONTAINS])
            .expect("inferred path");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path.nodes, vec!["session:s", "file:a"]);
        assert_eq!(hits[0].relation_labels, vec![REL_CONTAINS]);
    }

    #[test]
    fn delete_removes_graph() {
        let store = EngineeringStore::open_in_memory().expect("store");
        store
            .save_graph("session-1", None, &sample_graph())
            .expect("save");
        assert!(store.delete_graph("session-1").expect("delete"));
        assert!(store.load_graph("session-1").expect("load").is_none());
        assert!(!store.delete_graph("session-1").expect("delete again"));
        let indexed: i64 = store
            .connection
            .query_row(
                "SELECT COUNT(*) FROM engineering_entity_index WHERE session_id = 'session-1'",
                [],
                |row| row.get(0),
            )
            .expect("count index");
        assert_eq!(indexed, 0);
    }

    #[test]
    fn save_writes_compressed_blob_and_index() {
        let store = EngineeringStore::open_in_memory().expect("store");
        let graph = sample_graph();
        store.save_graph("session-1", None, &graph).expect("save");
        let (json, blob_len, indexed): (String, i64, i64) = store
            .connection
            .query_row(
                r#"
                SELECT g.graph_json, coalesce(length(g.graph_blob), 0),
                       (SELECT COUNT(*) FROM engineering_entity_index i
                        WHERE i.session_id = g.session_id)
                FROM engineering_graphs g WHERE g.session_id = 'session-1'
                "#,
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("read stored row");
        assert!(json.is_empty());
        assert!(blob_len > 4);
        assert_eq!(indexed, 3);
        let loaded = store.load_graph("session-1").expect("load").expect("graph");
        assert_eq!(loaded, graph);
    }

    #[test]
    fn compact_storage_rewrites_legacy_json() {
        let store = EngineeringStore::open_in_memory().expect("store");
        let graph = sample_graph();
        let json = serde_json::to_string(&graph).expect("json");
        store
            .connection
            .execute(
                r#"
                INSERT INTO engineering_graphs (
                    session_id, working_dir, ontology_id, ontology_hash,
                    entity_count, relation_count, graph_json, created_at, updated_at
                ) VALUES ('legacy', NULL, ?1, ?2, 3, 2, ?3, ?4, ?4)
                "#,
                params![
                    default_engineering_ontology().id,
                    default_engineering_ontology().content_hash(),
                    json,
                    Utc::now().to_rfc3339(),
                ],
            )
            .expect("insert legacy row");
        let dry = store.compact_storage(false).expect("dry run");
        assert_eq!(dry.sessions, 1);
        assert_eq!(dry.compacted, 0);
        let (json_before, blob_before): (String, Option<Vec<u8>>) = store
            .connection
            .query_row(
                "SELECT graph_json, graph_blob FROM engineering_graphs WHERE session_id = 'legacy'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("read legacy row");
        assert!(!json_before.is_empty());
        assert!(blob_before.is_none());
        let applied = store.compact_storage(true).expect("compact");
        assert_eq!(applied.compacted, 1);
        let (json_after, blob_after): (String, Option<Vec<u8>>) = store
            .connection
            .query_row(
                "SELECT graph_json, graph_blob FROM engineering_graphs WHERE session_id = 'legacy'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("read compacted row");
        assert!(json_after.is_empty());
        assert!(blob_after.is_some());
        let loaded = store.load_graph("legacy").expect("load").expect("graph");
        assert_eq!(loaded, graph);
        let recalls = store
            .recall_anchor("file:src/lib.rs", 2, 10)
            .expect("recall after compaction");
        assert_eq!(recalls.len(), 1);
        assert_eq!(recalls[0].graph, graph);
    }

    #[test]
    fn skips_session_only_graph() {
        let store = EngineeringStore::open_in_memory().expect("store");
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(EngineeringEntity::new(
            CLASS_SESSION,
            "session-1",
            "session-1",
        ));
        store
            .save_graph("session-1", None, &graph)
            .expect("empty save");
        assert!(store.load_graph("session-1").expect("load").is_none());
    }

    #[test]
    fn retention_report_and_prune_are_safe_by_default() {
        let store = EngineeringStore::open_in_memory().expect("store");
        let graph = sample_graph();
        store.save_graph("new-a", None, &graph).expect("save a");
        store.save_graph("new-b", None, &graph).expect("save b");
        store.save_graph("old", None, &graph).expect("save old");
        let old_time = (Utc::now() - chrono::Duration::days(60)).to_rfc3339();
        store
            .connection
            .execute(
                "UPDATE engineering_graphs SET updated_at = ?1 WHERE session_id = 'old'",
                params![old_time],
            )
            .expect("age old session");
        store
            .connection
            .execute(
                r#"
                INSERT INTO engineering_graphs (
                    session_id, working_dir, ontology_id, ontology_hash,
                    entity_count, relation_count, graph_json, created_at, updated_at
                ) VALUES (?1, NULL, ?2, ?3, 1, 0, ?4, ?5, ?5)
                "#,
                params![
                    "empty",
                    default_engineering_ontology().id,
                    default_engineering_ontology().content_hash(),
                    serde_json::to_string(&EngineeringGraph::new()).expect("empty graph"),
                    Utc::now().to_rfc3339(),
                ],
            )
            .expect("insert empty session");

        let report = store.retention_report().expect("retention report");
        assert_eq!(report.sessions, 4);
        assert_eq!(report.empty_sessions, 1);
        assert!(
            report
                .candidates
                .iter()
                .any(|candidate| candidate.session_id == "empty")
        );
        assert!(
            report
                .candidates
                .iter()
                .any(|candidate| candidate.session_id == "old")
        );
        assert!(
            !report
                .candidates
                .iter()
                .any(|candidate| candidate.session_id == "new-a")
        );
        assert!(
            !report
                .candidates
                .iter()
                .any(|candidate| candidate.session_id == "new-b")
        );

        let dry = store.prune_sessions(false).expect("dry run");
        assert_eq!(dry.pruned, 0);
        assert!(store.load_graph("empty").expect("load empty").is_some());

        let applied = store.prune_sessions(true).expect("prune");
        assert_eq!(applied.pruned, 2);
        assert!(store.load_graph("empty").expect("load empty").is_none());
        assert!(store.load_graph("old").expect("load old").is_none());
        assert!(store.load_graph("new-a").expect("load new").is_some());
    }

    #[test]
    fn ontology_path_queries_concept_layer_without_session_data() {
        let hits =
            EngineeringStore::ontology_path_between_anchors("File", "API", 2, 3, &[REL_DEFINES])
                .expect("ontology path");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].session_id, "ontology");
        assert_eq!(hits[0].node_labels, vec!["File", "API"]);
        assert_eq!(hits[0].relation_labels, vec![REL_DEFINES]);
        assert_eq!(hits[0].plan.tasks.len(), 1);
    }

    #[test]
    fn engineering_path_can_join_instances_through_ontology_layer() {
        let store = EngineeringStore::open_in_memory().expect("store");
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(
            EngineeringEntity::new(CLASS_FILE, "file:joined", "joined.rs")
                .with_property("path", json!("joined.rs")),
        );
        graph.insert_entity(EngineeringEntity::new("api", "api:joined", "joined api"));
        store
            .save_graph("ontology-join", None, &graph)
            .expect("save graph");

        let direct = store
            .path_between_anchors("file:joined", "api:joined", 4, 3, &[])
            .expect("direct path");
        assert!(direct.is_empty(), "{direct:?}");

        let joined = store
            .path_between_anchors_with_ontology("file:joined", "api:joined", 4, 3, &[])
            .expect("ontology-joined path");
        assert_eq!(joined.len(), 1, "{joined:?}");
        assert_eq!(
            joined[0].path.nodes,
            vec!["file:joined", "concept:file", "concept:api", "api:joined"]
        );
        assert!(
            joined[0]
                .relation_labels
                .iter()
                .any(|label| label == "instance_of")
        );
        assert!(
            joined[0]
                .relation_labels
                .iter()
                .any(|label| label == "has_instance")
        );
    }

    #[test]
    fn completion_writeback_preserves_working_dir_and_supports_promotion() {
        use jcode_memory_types::CompletionEdge;

        let store = EngineeringStore::open_in_memory().expect("store");
        let graph = sample_graph();
        store
            .save_graph("session-completion", Some("/repo"), &graph)
            .expect("save graph");

        let candidate = CompletionEdge::new("error:boom", REL_DEFINES, "file:src/lib.rs", 0.8)
            .with_evidence("doc:runbook#p4")
            .with_source("llm:proposal");
        let report = store
            .write_completion_edges("session-completion", &[candidate.clone()], "tool")
            .expect("write completion");
        assert_eq!(report.accepted.len(), 1, "{report:?}");
        assert!(report.completion_debt > 0.0);

        let hits = store
            .path_between_anchors("error:boom", "file:src/lib.rs", 2, 3, &[REL_DEFINES])
            .expect("hypothesis path");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert!(
            hits[0]
                .quality
                .has(jcode_memory_types::PathQualityFlag::HypothesizedEdge),
            "{:?}",
            hits[0].quality
        );
        assert!(
            hits[0]
                .plan
                .warnings
                .iter()
                .any(|warning| warning.contains("hypothesized edge(s)")),
            "{:?}",
            hits[0].plan.warnings
        );
        assert_eq!(
            hits[0].plan.tasks[0].action_type,
            jcode_memory_types::PlanActionType::Verify
        );

        let debt_rows = store
            .completion_debt_by_session(10)
            .expect("completion debt");
        assert_eq!(debt_rows.len(), 1, "{debt_rows:?}");
        assert_eq!(debt_rows[0].session_id, "session-completion");
        assert_eq!(debt_rows[0].hypothesis_edges, 1);
        assert_eq!(debt_rows[0].edge_count, 3);
        assert_eq!(debt_rows[0].expiring_soon, 0);
        assert!(debt_rows[0].debt > MAX_ENGINEERING_COMPLETION_DEBT);

        let sessions = store.list_sessions(5).expect("sessions");
        assert_eq!(sessions[0].working_dir.as_deref(), Some("/repo"));

        let evidence = EngineeringProvenance::new("runbook_review")
            .with_locator("doc:runbook#p4")
            .with_confidence(0.95);
        store
            .promote_completion_edge("session-completion", &report.accepted[0], evidence)
            .expect("promote");
        let reloaded = store
            .load_graph("session-completion")
            .expect("load")
            .expect("graph");
        let edge = reloaded.relations.get(&report.accepted[0]).expect("edge");
        assert!(!edge.hypothesis);
        assert_eq!(reloaded.completion_debt(), 0.0);
        let asserted_hits = store
            .path_between_anchors("error:boom", "file:src/lib.rs", 2, 3, &[REL_DEFINES])
            .expect("asserted path");
        assert!(
            !asserted_hits[0]
                .quality
                .has(jcode_memory_types::PathQualityFlag::HypothesizedEdge),
            "{:?}",
            asserted_hits[0].quality
        );
        assert!(
            store
                .completion_debt_by_session(10)
                .expect("debt after promotion")
                .is_empty()
        );

        let missing = EngineeringProvenance::new("human:review");
        assert!(
            store
                .promote_completion_edge("session-completion", &report.accepted[0], missing)
                .is_err()
        );
    }

    #[test]
    fn gap_diagnosis_classifies_unresolved_cross_session_and_depth_gaps() {
        let store = EngineeringStore::open_in_memory().expect("store");

        let unresolved = store
            .diagnose_path_gap("missing:from", "missing:to", 4, &[])
            .expect("diagnose unresolved");
        assert_eq!(unresolved.diagnosis, vec!["E01".to_string()]);
        assert_eq!(unresolved.next_actions.len(), 2);
        assert!(unresolved.alignment_candidates.is_empty());

        let layer_gap = store
            .diagnose_path_gap("File", "API", 4, &[])
            .expect("diagnose wrong layer");
        assert_eq!(layer_gap.diagnosis, vec!["E08".to_string()]);
        assert_eq!(layer_gap.next_actions.len(), 2);
        assert!(
            layer_gap.next_actions[0].contains("graph=ontology"),
            "{:?}",
            layer_gap.next_actions
        );

        let mut chain = EngineeringGraph::new();
        for id in ["task:a", "task:b", "task:c"] {
            chain.insert_entity(EngineeringEntity::new(CLASS_TASK, id, id));
        }
        chain.insert_relation(EngineeringRelation::new("task:a", REL_DEPENDS_ON, "task:b"));
        chain.insert_relation(EngineeringRelation::new("task:b", REL_DEPENDS_ON, "task:c"));
        store
            .save_graph("session-chain", Some("/repo"), &chain)
            .expect("save chain");
        let depth_gap = store
            .diagnose_path_gap("task:a", "task:c", 1, &[])
            .expect("diagnose depth");
        assert_eq!(depth_gap.diagnosis, vec!["E05".to_string()]);
        assert_eq!(depth_gap.shared_sessions, vec!["session-chain".to_string()]);
        assert_eq!(depth_gap.path_relations, vec![REL_DEPENDS_ON.to_string()]);
        assert!(
            depth_gap.next_actions[0].contains("raise depth"),
            "{:?}",
            depth_gap.next_actions
        );

        let filtered_gap = store
            .diagnose_path_gap("task:a", "task:c", 4, &[REL_TESTS])
            .expect("diagnose filter");
        assert!(
            filtered_gap.diagnosis.contains(&"E04".to_string()),
            "{:?}",
            filtered_gap.diagnosis
        );
        assert_eq!(
            filtered_gap.path_relations,
            vec![REL_DEPENDS_ON.to_string()]
        );

        let mut split = EngineeringGraph::new();
        for id in ["task:x", "task:y", "task:z", "task:w"] {
            split.insert_entity(EngineeringEntity::new(CLASS_TASK, id, id));
        }
        split.insert_relation(EngineeringRelation::new("task:x", REL_DEPENDS_ON, "task:y"));
        split.insert_relation(EngineeringRelation::new("task:z", REL_DEPENDS_ON, "task:w"));
        store
            .save_graph("session-split", Some("/repo"), &split)
            .expect("save split");
        let disconnected = store
            .diagnose_path_gap("task:x", "task:w", 4, &[])
            .expect("diagnose disconnected");
        assert_eq!(disconnected.diagnosis, vec!["E03".to_string()]);
        assert!(disconnected.path_relations.is_empty());

        let reverse_gap = store
            .diagnose_path_gap("task:c", "task:a", 4, &[])
            .expect("diagnose reverse direction");
        assert_eq!(reverse_gap.diagnosis, vec!["E10".to_string()]);
        assert!(
            reverse_gap
                .path_relations
                .contains(&"reverse_direction".to_string()),
            "{:?}",
            reverse_gap.path_relations
        );
        assert!(
            reverse_gap.next_actions[0].contains("reverse direction"),
            "{:?}",
            reverse_gap.next_actions
        );

        let mut inverted = EngineeringGraph::new();
        inverted.insert_entity(EngineeringEntity::new(CLASS_FILE, "file:p1", "auth.rs"));
        inverted.insert_entity(EngineeringEntity::new(CLASS_SESSION, "session:s1", "s1"));
        inverted.insert_relation(EngineeringRelation::new(
            "file:p1",
            REL_PART_OF,
            "session:s1",
        ));
        store
            .save_graph("session-inverted", Some("/repo"), &inverted)
            .expect("save inverted");
        let inverse_gap = store
            .diagnose_path_gap("session:s1", "file:p1", 4, &[REL_CONTAINS])
            .expect("diagnose inverse");
        assert_eq!(inverse_gap.diagnosis, vec!["E10".to_string()]);
        assert_eq!(inverse_gap.path_relations, vec![REL_CONTAINS.to_string()]);
        assert!(
            inverse_gap.next_actions[0].contains("enable inference"),
            "{:?}",
            inverse_gap.next_actions
        );

        let mut left = EngineeringGraph::new();
        left.insert_entity(EngineeringEntity::new(CLASS_FILE, "file:a1", "auth.rs"));
        left.insert_entity(EngineeringEntity::new(CLASS_TASK, "task:t1", "login flow"));
        store
            .save_graph("session-left", Some("/repo"), &left)
            .expect("save left");
        let mut right = EngineeringGraph::new();
        right.insert_entity(EngineeringEntity::new(CLASS_FILE, "file:b1", "auth.rs"));
        right.insert_entity(EngineeringEntity::new(CLASS_TASK, "task:t2", "login flow"));
        store
            .save_graph("session-right", Some("/repo"), &right)
            .expect("save right");

        let alignment_gap = store
            .diagnose_path_gap("file:a1", "file:b1", 4, &[])
            .expect("diagnose alignment");
        assert_eq!(alignment_gap.diagnosis, vec!["E09".to_string()]);
        assert_eq!(
            alignment_gap.from_sessions,
            vec!["session-left".to_string()]
        );
        assert_eq!(alignment_gap.to_sessions, vec!["session-right".to_string()]);
        assert!(!alignment_gap.alignment_candidates.is_empty());
        assert_eq!(alignment_gap.alignment_candidates[0].class, CLASS_FILE);
        assert!((alignment_gap.alignment_candidates[0].score - 1.0).abs() < 1e-6);
        assert!(
            alignment_gap.next_actions[0].contains("completion_action=propose"),
            "{:?}",
            alignment_gap.next_actions
        );
    }

    #[test]
    fn links_mentions_to_entities_by_name_alias_and_class() {
        let store = EngineeringStore::open_in_memory().expect("store");
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(
            EngineeringEntity::new(CLASS_FILE, "file:src/lib.rs", "src/lib.rs")
                .with_property("aliases", json!(["the main library", "lib"])),
        );
        graph.insert_entity(EngineeringEntity::new(
            CLASS_COMMAND,
            "command:cargo-test",
            "cargo test",
        ));
        graph.insert_entity(EngineeringEntity::new(CLASS_ERROR, "error:boom", "boom"));
        store
            .save_graph("session-link", Some("/repo"), &graph)
            .expect("save graph");

        let exact = store
            .link_entities("src/lib.rs", None, 5)
            .expect("link exact");
        assert_eq!(exact.len(), 1);
        assert_eq!(exact[0].entity_id, "file:src/lib.rs");
        assert!((exact[0].score - 1.0).abs() < 1e-6);
        assert_eq!(exact[0].signals, vec!["name exact".to_string()]);

        let alias = store.link_entities("lib", None, 5).expect("link alias");
        assert_eq!(alias[0].entity_id, "file:src/lib.rs");
        assert!((alias[0].score - 0.95).abs() < 1e-6);
        assert_eq!(alias[0].signals, vec!["alias match".to_string()]);

        let token = store.link_entities("cargo", None, 5).expect("link token");
        assert_eq!(token[0].entity_id, "command:cargo-test");
        assert!((token[0].score - 0.85).abs() < 1e-6);

        let filtered = store
            .link_entities("boom", Some(CLASS_FILE), 5)
            .expect("link filtered");
        assert!(filtered.is_empty());
        let unfiltered = store
            .link_entities("boom", Some(CLASS_ERROR), 5)
            .expect("link error class");
        assert_eq!(unfiltered[0].entity_id, "error:boom");
    }

    #[test]
    fn link_entities_orders_by_score_then_identity_and_respects_limit() {
        let store = EngineeringStore::open_in_memory().expect("store");
        for session in ["session-1", "session-2"] {
            let mut graph = EngineeringGraph::new();
            graph.insert_entity(EngineeringEntity::new(
                CLASS_FILE,
                "file:same.rs",
                "same.rs",
            ));
            store
                .save_graph(session, Some("/repo"), &graph)
                .expect("save graph");
        }
        let links = store
            .link_entities("same.rs", None, 1)
            .expect("link limited");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].entity_id, "file:same.rs");

        let all = store.link_entities("same.rs", None, 10).expect("link all");
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].session_id, "session-1");
        assert_eq!(all[1].session_id, "session-2");
        assert!(all[0].score >= all[1].score);
    }

    #[test]
    fn shared_neighbor_candidates_bridge_two_entities() {
        use jcode_memory_types::engineering_ontology::CLASS_SYMBOL;

        let store = EngineeringStore::open_in_memory().expect("store");
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(EngineeringEntity::new(CLASS_FILE, "file:a", "a.rs"));
        graph.insert_entity(EngineeringEntity::new(CLASS_FILE, "file:b", "b.rs"));
        graph.insert_entity(EngineeringEntity::new(CLASS_SYMBOL, "symbol:c", "c"));
        graph.insert_relation(EngineeringRelation::new("symbol:c", REL_DEFINES, "file:a"));
        graph.insert_relation(EngineeringRelation::new("symbol:c", REL_DEFINES, "file:b"));
        store
            .save_graph("session-bridge", Some("/repo"), &graph)
            .expect("save graph");

        let candidates = store
            .shared_neighbor_candidates("file:a", "file:b", 5)
            .expect("bridge candidates");
        assert_eq!(candidates.len(), 1);
        let candidate = &candidates[0];
        assert_eq!(candidate.via_entity, "symbol:c");
        assert_eq!(candidate.relation, REL_DEFINES);
        assert_eq!(candidate.evidence.len(), 2);
        assert!((candidate.score - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(candidate.session_id, "session-bridge");

        assert!(
            store
                .shared_neighbor_candidates("file:a", "file:a", 5)
                .expect("same entity")
                .is_empty()
        );
        assert!(
            store
                .shared_neighbor_candidates("file:a", "file:b", 0)
                .expect("zero limit")
                .is_empty()
        );
    }

    #[test]
    fn shared_neighbor_candidates_rank_lower_degree_bridges_higher() {
        use jcode_memory_types::engineering_ontology::CLASS_SYMBOL;

        let store = EngineeringStore::open_in_memory().expect("store");
        let mut sparse = EngineeringGraph::new();
        sparse.insert_entity(EngineeringEntity::new(CLASS_FILE, "file:a", "a.rs"));
        sparse.insert_entity(EngineeringEntity::new(CLASS_FILE, "file:b", "b.rs"));
        sparse.insert_entity(EngineeringEntity::new(CLASS_SYMBOL, "symbol:c", "c"));
        sparse.insert_relation(EngineeringRelation::new("symbol:c", REL_DEFINES, "file:a"));
        sparse.insert_relation(EngineeringRelation::new("symbol:c", REL_DEFINES, "file:b"));
        store
            .save_graph("session-sparse", Some("/repo"), &sparse)
            .expect("save sparse");

        let mut crowded = EngineeringGraph::new();
        crowded.insert_entity(EngineeringEntity::new(CLASS_FILE, "file:a", "a.rs"));
        crowded.insert_entity(EngineeringEntity::new(CLASS_FILE, "file:b", "b.rs"));
        crowded.insert_entity(EngineeringEntity::new(CLASS_SYMBOL, "symbol:hub", "hub"));
        for extra in ["file:e1", "file:e2"] {
            crowded.insert_entity(EngineeringEntity::new(CLASS_FILE, extra, extra));
        }
        for target in ["file:a", "file:b", "file:e1", "file:e2"] {
            crowded.insert_relation(EngineeringRelation::new("symbol:hub", REL_CONTAINS, target));
        }
        store
            .save_graph("session-crowded", Some("/repo"), &crowded)
            .expect("save crowded");

        let candidates = store
            .shared_neighbor_candidates("file:a", "file:b", 5)
            .expect("bridge candidates");
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].session_id, "session-sparse");
        assert_eq!(candidates[1].session_id, "session-crowded");
        assert!(candidates[0].score > candidates[1].score);
    }

    #[test]
    fn link_entities_ignores_empty_mentions_and_zero_limits() {
        let store = EngineeringStore::open_in_memory().expect("store");
        store
            .save_graph("session-empty", None, &sample_graph())
            .expect("save graph");
        assert!(
            store
                .link_entities("  ", None, 5)
                .expect("blank")
                .is_empty()
        );
        assert!(
            store
                .link_entities("src/lib.rs", None, 0)
                .expect("zero")
                .is_empty()
        );
    }
}
