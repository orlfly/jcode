//! Persistent storage for deterministic engineering episode graphs.
//!
//! Each session is stored as one JSON graph keyed by `session_id`. The schema
//! keeps the ontology identity alongside the graph so a later replay can detect
//! schema drift. This is deliberately a small, append/update store rather than a
//! graph database: the current episode graphs are small and the path engine runs
//! in memory.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Utc;
use jcode_memory_types::engineering_ontology::{
    EngineeringEntity, EngineeringGraph, EngineeringRelation, default_engineering_ontology,
};
use rusqlite::{Connection, OptionalExtension, params};

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
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS engineering_graphs_updated
                    ON engineering_graphs(updated_at DESC);
                "#,
            )
            .context("create engineering_graphs schema")?;
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
        let ontology = default_engineering_ontology();
        let graph_json = serde_json::to_string(graph).context("serialize engineering graph")?;
        let now = Utc::now().to_rfc3339();
        self.connection
            .execute(
                r#"
                INSERT INTO engineering_graphs (
                    session_id, working_dir, ontology_id, ontology_hash,
                    entity_count, relation_count, graph_json, created_at, updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8)
                ON CONFLICT(session_id) DO UPDATE SET
                    working_dir = excluded.working_dir,
                    ontology_id = excluded.ontology_id,
                    ontology_hash = excluded.ontology_hash,
                    entity_count = excluded.entity_count,
                    relation_count = excluded.relation_count,
                    graph_json = excluded.graph_json,
                    updated_at = excluded.updated_at
                "#,
                params![
                    session_id,
                    working_dir,
                    ontology.id,
                    ontology.content_hash(),
                    graph.entities.len() as i64,
                    graph.relations.len() as i64,
                    graph_json,
                    now,
                ],
            )
            .context("upsert engineering graph")?;
        Ok(())
    }

    /// Load one session graph, or `None` if it has not been persisted.
    pub fn load_graph(&self, session_id: &str) -> Result<Option<EngineeringGraph>> {
        let json: Option<String> = self
            .connection
            .query_row(
                "SELECT graph_json FROM engineering_graphs WHERE session_id = ?1",
                params![session_id],
                |row| row.get(0),
            )
            .optional()
            .context("load engineering graph")?;
        json.map(|json| {
            serde_json::from_str(&json)
                .with_context(|| format!("parse engineering graph for session {session_id}"))
        })
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
        let removed = self
            .connection
            .execute(
                "DELETE FROM engineering_graphs WHERE session_id = ?1",
                params![session_id],
            )
            .context("delete engineering graph")?;
        Ok(removed > 0)
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
                SELECT session_id, working_dir, graph_json
                FROM engineering_graphs
                WHERE graph_json LIKE ?1 ESCAPE '\'
                ORDER BY updated_at DESC
                "#,
            )
            .context("prepare anchor recall")?;
        let rows = statement
            .query_map(params![pattern], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .context("query anchor recall")?;

        let mut out = Vec::new();
        for row in rows {
            let (session_id, working_dir, json) = row.context("read anchor row")?;
            let graph: EngineeringGraph = match serde_json::from_str(&json) {
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
    fn delete_removes_graph() {
        let store = EngineeringStore::open_in_memory().expect("store");
        store
            .save_graph("session-1", None, &sample_graph())
            .expect("save");
        assert!(store.delete_graph("session-1").expect("delete"));
        assert!(store.load_graph("session-1").expect("load").is_none());
        assert!(!store.delete_graph("session-1").expect("delete again"));
    }
}
