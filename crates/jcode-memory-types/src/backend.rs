//! Pluggable storage backend trait for [`MemoryGraph`].
//!
//! The default implementation is the JSON file backend
//! (`JsonGraphBackend`) which writes the entire `MemoryGraph` to a single
//! atomic JSON snapshot. This module defines the [`GraphBackend`] trait
//! that lets callers swap that backend for any other implementation,
//! including the SQLite + sqlite-gvec backend provided by `jcode-base`
//! when the `sqlite-gvec` feature is enabled.
//!
//! The trait is intentionally narrow: it models the *whole-graph*
//! read/replace pattern that the existing `memory.rs` call sites already
//! rely on, plus an optional incremental mutation log that backends can
//! use to track fine-grained changes without losing the snapshot
//! semantics that the rest of the code expects.
//!
//! All backends must be safe to use from multiple threads as long as the
//! caller owns a single `&mut` reference to the backend. Concrete
//! backends may add interior mutability for caches, but the trait
//! itself does not require `Send + Sync`; that constraint is added by
//! each implementation based on its internal state.

use crate::graph::MemoryGraph;
use std::fmt::Debug;

/// Identifier for a logical memory store (project or global scope).
///
/// The two canonical values are produced by `MemoryStore::project_key()`
/// and `MemoryStore::global_key()`. The trait does not require any
/// particular format, but implementations may use the key to derive a
/// file path or SQLite database name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StoreKey(pub String);

impl StoreKey {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A coarse-grained mutation log entry that backends may apply
/// incrementally instead of rewriting the whole graph.
///
/// `JsonGraphBackend` ignores these and always writes a full snapshot.
/// The SQLite backend writes them row-by-row inside a single
/// transaction.
#[derive(Debug, Clone)]
pub enum GraphMutation {
    /// Insert or replace a memory node by id.
    UpsertMemory {
        id: String,
        json: String,
    },
    /// Delete a memory node and all edges that reference it.
    DeleteMemory {
        id: String,
    },
    /// Insert or replace a tag node.
    UpsertTag {
        id: String,
        json: String,
    },
    /// Insert or replace a cluster node.
    UpsertCluster {
        id: String,
        json: String,
    },
    /// Insert or replace an edge from `from` -> `to`.
    /// `kind_json` is the serialized `EdgeKind` (e.g.
    /// `{"kind":"has_tag"}` or `{"kind":"relates_to","weight":0.7}`).
    UpsertEdge {
        from: String,
        to: String,
        kind_json: String,
    },
    /// Remove an edge from `from` -> `to` of the given kind.
    DeleteEdge {
        from: String,
        to: String,
        kind_json: String,
    },
    /// Replace metadata.
    ReplaceMetadata {
        json: String,
    },
}

/// The set of operations the existing `memory.rs` call sites actually
/// use, expressed in terms of whole-graph snapshots plus an optional
/// mutation log.
///
/// Implementations are expected to be cheap to clone when no mutation
/// log is pending and to commit pending mutations atomically on
/// [`save`](GraphBackend::save) and [`save_with`](GraphBackend::save_with).
pub trait GraphBackend: Debug + Send + Sync {
    /// Human-readable backend name, used in logs and the
    /// `memory.backend` diagnostic field.
    fn name(&self) -> &str;

    /// Load the graph identified by `key`.
    ///
    /// Returns `Ok(MemoryGraph::new())` when the store does not yet
    /// exist (this matches the legacy JSON path which returns an empty
    /// graph for missing files).
    fn load(&self, key: &StoreKey) -> anyhow::Result<MemoryGraph>;

    /// Persist the full graph snapshot identified by `key`.
    ///
    /// Implementations must perform an atomic replacement: either the
    /// previous state or the new state must be observable after this
    /// call returns, never a partial mix.
    fn save(&self, key: &StoreKey, graph: &MemoryGraph) -> anyhow::Result<()>;

    /// Apply a sequence of mutations to the graph identified by `key`,
    /// then return the resulting full graph.
    ///
    /// The default implementation is equivalent to `load`, followed by
    /// replaying the mutations in memory, followed by `save`. SQLite
    /// backends override this to apply the mutations row-by-row inside
    /// a single transaction, which is cheaper than rewriting the
    /// whole graph.
    fn apply_mutations(
        &self,
        key: &StoreKey,
        mutations: &[GraphMutation],
    ) -> anyhow::Result<MemoryGraph> {
        let mut graph = self.load(key)?;
        apply_mutations_in_place(&mut graph, mutations)?;
        self.save(key, &graph)?;
        Ok(graph)
    }

    /// Save with an explicit mutation log: implementations that track
    /// pending mutations (e.g. a write-behind cache) may flush them as
    /// part of this call.
    ///
    /// The default implementation just calls [`save`](GraphBackend::save).
    fn save_with(
        &self,
        key: &StoreKey,
        graph: &MemoryGraph,
        _mutations: &[GraphMutation],
    ) -> anyhow::Result<()> {
        self.save(key, graph)
    }

    /// Run a full-text search over the Memory nodes of `key`.
    ///
    /// Returns `(memory_id, score)` pairs ordered by relevance, where
    /// `score` is a non-negative float and larger means more relevant.
    ///
    /// The default implementation is a no-op: backends without a
    /// dedicated text index return `Ok(vec![])`. Backends with FTS5
    /// (or equivalent) override this to do an indexed search.
    fn text_search(
        &self,
        _key: &StoreKey,
        _query: &str,
        _k: usize,
    ) -> anyhow::Result<Vec<(String, f32)>> {
        Ok(vec![])
    }
}

/// Replay a mutation log onto an in-memory `MemoryGraph`.
///
/// Exposed publicly so backends that prefer to operate on the in-memory
/// graph and only persist at the end (the default `apply_mutations`
/// path) can share the same replay logic.
pub fn apply_mutations_in_place(
    graph: &mut MemoryGraph,
    mutations: &[GraphMutation],
) -> anyhow::Result<()> {
    use crate::{ClusterEntry, Edge, EdgeKind, MemoryEntry, TagEntry};
    for m in mutations {
        match m {
            GraphMutation::UpsertMemory { id, json } => {
                let entry: MemoryEntry = serde_json::from_str(json)
                    .map_err(|e| anyhow::anyhow!("UpsertMemory({id}): {e}"))?;
                graph.memories.insert(id.clone(), entry);
            }
            GraphMutation::DeleteMemory { id } => {
                graph.remove_memory(id);
            }
            GraphMutation::UpsertTag { id, json } => {
                let tag: TagEntry = serde_json::from_str(json)
                    .map_err(|e| anyhow::anyhow!("UpsertTag({id}): {e}"))?;
                graph.tags.insert(id.clone(), tag);
            }
            GraphMutation::UpsertCluster { id, json } => {
                let cluster: ClusterEntry = serde_json::from_str(json)
                    .map_err(|e| anyhow::anyhow!("UpsertCluster({id}): {e}"))?;
                graph.clusters.insert(id.clone(), cluster);
            }
            GraphMutation::UpsertEdge {
                from,
                to,
                kind_json,
            } => {
                let kind: EdgeKind = serde_json::from_str(kind_json)
                    .map_err(|e| anyhow::anyhow!("UpsertEdge({from}->{to}): {e}"))?;
                let edge = Edge::new(to.clone(), kind);
                let entry = graph.edges.entry(from.clone()).or_default();
                // Replace an existing edge of the same kind.
                if let Some(slot) = entry.iter_mut().find(|e| std::mem::discriminant(&e.kind) == std::mem::discriminant(&edge.kind)) {
                    *slot = edge;
                } else {
                    entry.push(edge);
                }
                let reverse = graph.reverse_edges.entry(to.clone()).or_default();
                if !reverse.iter().any(|s| s == from) {
                    reverse.push(from.clone());
                }
            }
            GraphMutation::DeleteEdge {
                from,
                to,
                kind_json,
            } => {
                let kind: EdgeKind = serde_json::from_str(kind_json)
                    .map_err(|e| anyhow::anyhow!("DeleteEdge({from}->{to}): {e}"))?;
                let kind_disc = std::mem::discriminant(&kind);
                if let Some(edges) = graph.edges.get_mut(from) {
                    edges.retain(|e| std::mem::discriminant(&e.kind) != kind_disc || e.target != *to);
                }
                if let Some(sources) = graph.reverse_edges.get_mut(to) {
                    sources.retain(|s| s != from);
                }
            }
            GraphMutation::ReplaceMetadata { json } => {
                graph.metadata = serde_json::from_str(json)
                    .map_err(|e| anyhow::anyhow!("ReplaceMetadata: {e}"))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EdgeKind, MemoryCategory, MemoryEntry, MemoryScope};

    fn sample_entry(id: &str) -> MemoryEntry {
        let mut e = MemoryEntry::new(MemoryCategory::Fact, "test content");
        e.id = id.to_string();
        e.tags = vec!["alpha".to_string(), "beta".to_string()];
        e
    }

    #[test]
    fn replay_upserts_and_deletes() {
        let mut g = MemoryGraph::new();
        let e1 = sample_entry("m1");
        let json = serde_json::to_string(&e1).unwrap();
        let m1 = GraphMutation::UpsertMemory {
            id: "m1".into(),
            json,
        };
        let tag_json = serde_json::to_string(&crate::graph::TagEntry::new("alpha")).unwrap();
        let mt = GraphMutation::UpsertTag {
            id: "tag:alpha".into(),
            json: tag_json,
        };
        let me = GraphMutation::UpsertEdge {
            from: "m1".into(),
            to: "tag:alpha".into(),
            kind_json: serde_json::to_string(&EdgeKind::HasTag).unwrap(),
        };
        apply_mutations_in_place(&mut g, &[m1, mt, me]).unwrap();
        assert_eq!(g.memories.len(), 1);
        assert_eq!(g.tags.len(), 1);
        assert_eq!(g.edges.get("m1").map(|v| v.len()), Some(1));

        let del = GraphMutation::DeleteMemory { id: "m1".into() };
        apply_mutations_in_place(&mut g, &[del]).unwrap();
        assert!(g.memories.is_empty());
    }
}