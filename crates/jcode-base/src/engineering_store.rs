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
use chrono::{DateTime, Utc};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use jcode_memory_types::engineering_ontology::{
    CLASS_SESSION, EngineeringEntity, EngineeringGraph, EngineeringPath, EngineeringRelation,
    default_engineering_ontology,
};
use jcode_memory_types::{PathPlan, build_path_plan};
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
}

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
                .execute("ALTER TABLE engineering_graphs ADD COLUMN graph_blob BLOB", [])
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
        transaction.commit().context("commit delete engineering graph")?;
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
            for path in graph.k_shortest_paths(
                from_anchor,
                to_anchor,
                depth,
                allowed_relations,
                limit,
            ) {
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
                hits.push(EngineeringPathHit {
                    session_id: session_id.clone(),
                    working_dir: working_dir.clone(),
                    path,
                    node_labels,
                    relation_labels,
                    plan,
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use jcode_memory_types::engineering_ontology::{
        CLASS_COMMAND, CLASS_ERROR, CLASS_FILE, EngineeringProvenance, REL_CAUSES, REL_MODIFIES,
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
            .path_between_anchors(
                "command:cargo-test",
                "file:src/lib.rs",
                3,
                5,
                &["modifies"],
            )
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
            .path_between_anchors(
                "command:cargo-test",
                "file:src/lib.rs",
                3,
                5,
                &["causes"],
            )
            .expect("filtered path");
        assert!(filtered.is_empty());
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
        store
            .save_graph("session-1", None, &graph)
            .expect("save");
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
}
