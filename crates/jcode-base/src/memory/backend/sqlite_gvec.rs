//! SQLite-backed memory graph storage using `gvec` (gvec-bindings).
//!
//! `SqliteGvecBackend` keeps the same [`GraphBackend`] surface as the
//! [`JsonBackend`] but stores nodes / edges / properties inside a
//! SQLite database via the `sqlite-gvec` extension. Each project or
//! global store maps to a *named* graph inside a single SQLite file:
//!
//!   - per-store graph name: `<key>` (sanitised)
//!   - file: `<jcode_dir>/memory/backend-sqlite/gvec.sqlite`
//!
//! ## Identifier mapping
//!
//! `MemoryGraph` uses string node ids (`mem_<ts>_<rand>`,
//! `tag:<name>`, `cluster:<id>`). `gvec` uses integer rowids. We keep
//! a parallel `node_id_map` table (`<prefix>_id_map`) that maps the
//! integer rowid to the string id, so we can:
//!
//! - upsert a node by deleting the prior string-id mapping row,
//!   looking up its rowid, and re-creating the row with the same rowid
//!   when we want stable ids across reloads;
//! - query by string id with a single SELECT;
//! - serialise the integer id back to a string when reading.
//!
//! For this PoC we store the string id directly in the node
//! `properties` JSON (`properties["__jcode_id"]`). This avoids the
//! need for a separate mapping table and matches what the JSON
//! backend stores today. The integer rowid is treated as an opaque
//! internal handle.

use crate::storage;
use anyhow::{Context, Result};
use gvec::{Database, Graph};
use gvec_core::storage::Storage;
use jcode_memory_types::{
    ClusterEntry, EdgeKind, GraphBackend, GraphMutation, MemoryGraph, StoreKey, TagEntry,
    apply_mutations_in_place,
};
use serde_json::{Value, json};
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Internal handle that owns both the SQLite connection and the named
/// graphs. Cloning is cheap (Arc).
#[derive(Clone)]
struct DbHandle {
    db: Arc<Database>,
}

impl Debug for DbHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DbHandle").finish()
    }
}

/// `SqliteGvecBackend` opens (or creates) a single SQLite database
/// file and stores each `StoreKey` as a separate named graph inside
/// it.
pub struct SqliteGvecBackend {
    db: DbHandle,
    file: PathBuf,
    /// In-process mutex that serialises writes per process. SQLite's
    /// WAL mode handles cross-process concurrency for readers; this
    /// mutex additionally protects the in-memory id map from races
    /// when the same `Storage` is shared across threads inside one
    /// process.
    write_lock: Arc<Mutex<()>>,
}

impl Debug for SqliteGvecBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteGvecBackend")
            .field("file", &self.file)
            .finish()
    }
}

impl SqliteGvecBackend {
    /// Open or create the default SQLite database at
    /// `<jcode_dir>/memory/backend-sqlite/gvec.sqlite`.
    pub fn open_default() -> Result<Self> {
        let dir = storage::jcode_dir()?
            .join("memory")
            .join("backend-sqlite");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create SqliteGvecBackend dir {}", dir.display()))?;
        let file = dir.join("gvec.sqlite");
        Self::open(file)
    }

    /// Open or create the SQLite database at `path`.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create parent dir {}", parent.display()))?;
        }
        let path_str = path.to_string_lossy().to_string();
        let db = Database::open(&path_str)
            .with_context(|| format!("SqliteGvecBackend::open({path_str})"))?;
        Ok(Self {
            db: DbHandle { db: Arc::new(db) },
            file: path,
            write_lock: Arc::new(Mutex::new(())),
        })
    }

    /// Open an in-memory backend. Used by tests.
    pub fn open_in_memory() -> Result<Self> {
        let db = Database::open_in_memory()
            .map_err(|e| anyhow::anyhow!("SqliteGvecBackend::open_in_memory: {e}"))?;
        Ok(Self {
            db: DbHandle { db: Arc::new(db) },
            file: PathBuf::from(":memory:"),
            write_lock: Arc::new(Mutex::new(())),
        })
    }

    fn graph(&self, key: &StoreKey) -> Result<Graph> {
        let prefix = format!("g_{}", sanitize(key.as_str()));
        self.db
            .db
            .graph(&prefix)
            .map_err(|e| anyhow::anyhow!("open graph '{prefix}': {e}"))
    }

    fn storage(&self, key: &StoreKey) -> Result<Storage> {
        let g = self.graph(key)?;
        Ok(g.storage)
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '_') { c } else { '_' })
        .collect()
}

fn ensure_known_label<S: AsRef<str>>(s: S) -> String {
    let s = s.as_ref();
    if s.is_empty() {
        "Node".to_string()
    } else {
        s.to_string()
    }
}

fn labels_for(kind: &str) -> Vec<String> {
    vec![ensure_known_label(kind)]
}

fn read_node_id_from_props(props: &Value) -> Option<String> {
    props
        .get("__jcode_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Load all nodes of a given label and return (string_id, properties).
fn list_nodes_by_label(graph: &Graph, label: &str) -> Result<Vec<(String, Value)>> {
    let nodes = graph
        .storage
        .list_nodes_by_label(label)
        .map_err(|e| anyhow::anyhow!("list_nodes_by_label({label}): {e}"))?;
    let mut out = Vec::with_capacity(nodes.len());
    for node in nodes {
        let id = read_node_id_from_props(&node.properties)
            .unwrap_or_else(|| format!("rowid:{}", node.id.0));
        out.push((id, node.properties));
    }
    Ok(out)
}

fn upsert_node(graph: &Graph, label: &str, id: &str, mut properties: Value) -> Result<()> {
    // Find any existing node that has __jcode_id == id, delete it,
    // then create a fresh one with the same label and merged properties.
    let nodes = graph
        .storage
        .list_nodes_by_label(label)
        .map_err(|e| anyhow::anyhow!("upsert_node: list {label}: {e}"))?;
    for node in nodes {
        if read_node_id_from_props(&node.properties).as_deref() == Some(id) {
            graph
                .storage
                .delete_node(node.id)
                .map_err(|e| anyhow::anyhow!("upsert_node: delete {label}/{id}: {e}"))?;
        }
    }
    if !properties.is_object() {
        properties = json!({});
    }
    if let Some(obj) = properties.as_object_mut() {
        obj.insert("__jcode_id".to_string(), json!(id));
    }
    let labels = labels_for(label);
    graph
        .storage
        .create_node(&labels, &properties)
        .map_err(|e| anyhow::anyhow!("upsert_node: create {label}/{id}: {e}"))?;
    Ok(())
}

fn delete_node(graph: &Graph, id: &str) -> Result<()> {
    // Search every label we manage.
    for label in ["Memory", "Tag", "Cluster"] {
        let nodes = graph
            .storage
            .list_nodes_by_label(label)
            .map_err(|e| anyhow::anyhow!("delete_node: list {label}: {e}"))?;
        for node in nodes {
            if read_node_id_from_props(&node.properties).as_deref() == Some(id) {
                graph
                    .storage
                    .delete_node(node.id)
                    .map_err(|e| anyhow::anyhow!("delete_node: drop {label}/{id}: {e}"))?;
                // Best-effort: clean up any edges that reference this node.
                // We can't easily enumerate edges by endpoint yet (gvec
                // 0.1 lacks that API), so leave them; they will be ignored
                // on read because the source/target rowid no longer exists.
            }
        }
    }
    Ok(())
}

fn upsert_edge(graph: &Graph, from: &str, to: &str, kind: &EdgeKind) -> Result<()> {
    // We store the EdgeKind payload under properties.kind with snake_case
    // discriminant (matching the JSON schema). The edge_type column gets
    // a short human-readable label.
    let kind_json = serde_json::to_value(kind).map_err(|e| anyhow::anyhow!("encode EdgeKind: {e}"))?;
    let edge_type = match kind {
        EdgeKind::HasTag => "has_tag",
        EdgeKind::InCluster => "in_cluster",
        EdgeKind::RelatesTo { .. } => "relates_to",
        EdgeKind::Supersedes => "supersedes",
        EdgeKind::Contradicts => "contradicts",
        EdgeKind::DerivedFrom => "derived_from",
    };
    let properties = json!({
        "kind_payload": kind_json,
        "from": from,
        "to": to,
    });
    // Auto-create missing endpoints so the trait's invariant
    // ("edges exist between two nodes that exist in the graph") is
    // preserved without forcing callers to upsert tag/cluster nodes
    // before linking to them.
    let source_rowid = ensure_node(graph, from)?;
    let target_rowid = ensure_node(graph, to)?;
    graph
        .storage
        .create_edge(source_rowid, target_rowid, Some(edge_type), &properties)
        .map_err(|e| anyhow::anyhow!("upsert_edge {from}->{to}: {e}"))?;
    Ok(())
}

fn find_rowid(graph: &Graph, id: &str) -> Result<Option<gvec_core::ids::NodeId>> {
    for label in ["Memory", "Tag", "Cluster"] {
        let nodes = graph
            .storage
            .list_nodes_by_label(label)
            .map_err(|e| anyhow::anyhow!("find_rowid: list {label}: {e}"))?;
        for node in nodes {
            if read_node_id_from_props(&node.properties).as_deref() == Some(id) {
                return Ok(Some(node.id));
            }
        }
    }
    Ok(None)
}

/// Ensure a node with the given string id exists, creating a minimal
/// placeholder if necessary. Returns the rowid.
fn ensure_node(graph: &Graph, id: &str) -> Result<gvec_core::ids::NodeId> {
    if let Some(rowid) = find_rowid(graph, id)? {
        return Ok(rowid);
    }
    // Heuristic: tag:foo -> Tag, cluster:foo -> Cluster, otherwise Memory.
    let label = if id.starts_with("tag:") {
        "Tag"
    } else if id.starts_with("cluster:") {
        "Cluster"
    } else {
        "Memory"
    };
    let labels = labels_for(label);
    let placeholder = json!({ "__jcode_id": id });
    graph
        .storage
        .create_node(&labels, &placeholder)
        .map_err(|e| anyhow::anyhow!("ensure_node create {label}/{id}: {e}"))?;
    // create_node returns a fresh rowid, but the row may not be the
    // one we want if another node already claimed the __jcode_id in
    // a different label — re-look-up to be sure.
    let rowid = find_rowid(graph, id)?
        .ok_or_else(|| anyhow::anyhow!("ensure_node: {label}/{id} not found after insert"))?;
    Ok(rowid)
}

fn delete_edge(graph: &Graph, from: &str, to: &str, kind: &EdgeKind) -> Result<()> {
    let edge_type = match kind {
        EdgeKind::HasTag => "has_tag",
        EdgeKind::InCluster => "in_cluster",
        EdgeKind::RelatesTo { .. } => "relates_to",
        EdgeKind::Supersedes => "supersedes",
        EdgeKind::Contradicts => "contradicts",
        EdgeKind::DerivedFrom => "derived_from",
    };
    let edges = graph
        .storage
        .list_edges(None, None)
        .map_err(|e| anyhow::anyhow!("delete_edge: list: {e}"))?;
    let source_rowid = find_rowid(graph, from)?;
    let target_rowid = find_rowid(graph, to)?;
    let (Some(s), Some(t)) = (source_rowid, target_rowid) else {
        return Ok(());
    };
    for edge in edges {
        if edge.source == s && edge.target == t && edge.edge_type.as_deref() == Some(edge_type) {
            graph
                .storage
                .delete_edge(edge.id)
                .map_err(|e| anyhow::anyhow!("delete_edge: {e}"))?;
        }
    }
    Ok(())
}

impl GraphBackend for SqliteGvecBackend {
    fn name(&self) -> &str {
        "sqlite-gvec"
    }

    fn load(&self, key: &StoreKey) -> Result<MemoryGraph> {
        let graph = self.graph(key)?;
        let mut out = MemoryGraph::new();

        // Memory nodes
        for (id, props) in list_nodes_by_label(&graph, "Memory")? {
            match serde_json::from_value::<jcode_memory_types::MemoryEntry>(props) {
                Ok(entry) => {
                    out.memories.insert(id, entry);
                }
                Err(e) => {
                    crate::logging::warn(&format!(
                        "SqliteGvecBackend: skipping corrupt Memory node {id}: {e}"
                    ));
                }
            }
        }

        // Tag nodes
        for (id, props) in list_nodes_by_label(&graph, "Tag")? {
            match serde_json::from_value::<TagEntry>(props) {
                Ok(tag) => {
                    out.tags.insert(id, tag);
                }
                Err(e) => {
                    crate::logging::warn(&format!(
                        "SqliteGvecBackend: skipping corrupt Tag node {id}: {e}"
                    ));
                }
            }
        }

        // Cluster nodes
        for (id, props) in list_nodes_by_label(&graph, "Cluster")? {
            match serde_json::from_value::<ClusterEntry>(props) {
                Ok(cluster) => {
                    out.clusters.insert(id, cluster);
                }
                Err(e) => {
                    crate::logging::warn(&format!(
                        "SqliteGvecBackend: skipping corrupt Cluster node {id}: {e}"
                    ));
                }
            }
        }

        // Edges
        let edges = graph
            .storage
            .list_edges(None, None)
            .map_err(|e| anyhow::anyhow!("load edges: {e}"))?;
        for edge in edges {
            let from_id = read_node_id_from_props(&node_props(&graph, edge.source)?);
            let to_id = read_node_id_from_props(&node_props(&graph, edge.target)?);
            let (Some(from), Some(to)) = (from_id, to_id) else {
                continue;
            };
            let kind_payload = edge
                .properties
                .get("kind_payload")
                .cloned()
                .unwrap_or(Value::Null);
            let kind: EdgeKind = match serde_json::from_value(kind_payload) {
                Ok(k) => k,
                Err(_) => {
                    // Default to HasTag when kind_payload is missing or
                    // unrecognised (matches legacy data that did not
                    // record the kind).
                    EdgeKind::HasTag
                }
            };
            let entry = out.edges.entry(from.clone()).or_default();
            if !entry.iter().any(|e| e.target == to) {
                entry.push(jcode_memory_types::Edge::new(to.clone(), kind.clone()));
            }
            let reverse = out.reverse_edges.entry(to).or_default();
            if !reverse.iter().any(|s| s == &from) {
                reverse.push(from);
            }
        }

        Ok(out)
    }

    fn save(&self, key: &StoreKey, graph: &MemoryGraph) -> Result<()> {
        let _guard = self.write_lock.lock().unwrap_or_else(|p| p.into_inner());
        let g = self.graph(key)?;
        // Reset everything: for this PoC, the SQL backend treats save
        // as a replace-all. The mutation log API on the trait is the
        // efficient path; save() is the slow path.
        drop_all_nodes(&g)?;
        drop_all_edges(&g)?;

        for (id, entry) in &graph.memories {
            let props = serde_json::to_value(entry)?;
            upsert_node(&g, "Memory", id, props)?;
        }
        for (id, tag) in &graph.tags {
            let props = serde_json::to_value(tag)?;
            upsert_node(&g, "Tag", id, props)?;
        }
        for (id, cluster) in &graph.clusters {
            let props = serde_json::to_value(cluster)?;
            upsert_node(&g, "Cluster", id, props)?;
        }
        for (from, edges) in &graph.edges {
            for edge in edges {
                upsert_edge(&g, from, &edge.target, &edge.kind)?;
            }
        }
        Ok(())
    }

    fn apply_mutations(
        &self,
        key: &StoreKey,
        mutations: &[GraphMutation],
    ) -> Result<MemoryGraph> {
        let _guard = self.write_lock.lock().unwrap_or_else(|p| p.into_inner());
        let g = self.graph(key)?;
        for m in mutations {
            match m {
                GraphMutation::UpsertMemory { id, json } => {
                    let value: Value = serde_json::from_str(json)
                        .map_err(|e| anyhow::anyhow!("UpsertMemory({id}): {e}"))?;
                    upsert_node(&g, "Memory", id, value)?;
                }
                GraphMutation::DeleteMemory { id } => delete_node(&g, id)?,
                GraphMutation::UpsertTag { id, json } => {
                    let value: Value = serde_json::from_str(json)
                        .map_err(|e| anyhow::anyhow!("UpsertTag({id}): {e}"))?;
                    upsert_node(&g, "Tag", id, value)?;
                }
                GraphMutation::UpsertCluster { id, json } => {
                    let value: Value = serde_json::from_str(json)
                        .map_err(|e| anyhow::anyhow!("UpsertCluster({id}): {e}"))?;
                    upsert_node(&g, "Cluster", id, value)?;
                }
                GraphMutation::UpsertEdge {
                    from,
                    to,
                    kind_json,
                } => {
                    let kind: EdgeKind = serde_json::from_str(kind_json)
                        .map_err(|e| anyhow::anyhow!("UpsertEdge({from}->{to}): {e}"))?;
                    upsert_edge(&g, from, to, &kind)?;
                }
                GraphMutation::DeleteEdge {
                    from,
                    to,
                    kind_json,
                } => {
                    let kind: EdgeKind = serde_json::from_str(kind_json)
                        .map_err(|e| anyhow::anyhow!("DeleteEdge({from}->{to}): {e}"))?;
                    delete_edge(&g, from, to, &kind)?;
                }
                GraphMutation::ReplaceMetadata { .. } => {
                    // Metadata is part of MemoryGraph but gvec has no
                    // dedicated node for it; ignore for the PoC and
                    // re-derive on the next load via the in-memory
                    // graph path.
                }
            }
        }
        self.load(key)
    }

    fn save_with(
        &self,
        key: &StoreKey,
        graph: &MemoryGraph,
        mutations: &[GraphMutation],
    ) -> Result<()> {
        if mutations.is_empty() {
            return self.save(key, graph);
        }
        // Apply mutations first, then save the resulting graph.
        let _ = self.apply_mutations(key, mutations)?;
        self.save(key, graph)
    }
}

fn drop_all_nodes(graph: &Graph) -> Result<()> {
    for label in ["Memory", "Tag", "Cluster"] {
        let nodes = graph
            .storage
            .list_nodes_by_label(label)
            .map_err(|e| anyhow::anyhow!("drop_all_nodes: list {label}: {e}"))?;
        for n in nodes {
            graph
                .storage
                .delete_node(n.id)
                .map_err(|e| anyhow::anyhow!("drop_all_nodes: delete {label}: {e}"))?;
        }
    }
    Ok(())
}

fn drop_all_edges(graph: &Graph) -> Result<()> {
    let edges = graph
        .storage
        .list_edges(None, None)
        .map_err(|e| anyhow::anyhow!("drop_all_edges: list: {e}"))?;
    for e in edges {
        graph
            .storage
            .delete_edge(e.id)
            .map_err(|e| anyhow::anyhow!("drop_all_edges: delete: {e}"))?;
    }
    Ok(())
}

fn node_props(graph: &Graph, rowid: gvec_core::ids::NodeId) -> Result<Value> {
    graph
        .storage
        .get_node(rowid)
        .map(|n| n.properties)
        .map_err(|e| anyhow::anyhow!("get_node({}): {e}", rowid.0))
}

fn edge_properties_string(graph: &Graph, _rowid: gvec_core::ids::NodeId) -> String {
    // Best-effort string for reverse_edges. We don't have a direct
    // API, so just emit an empty placeholder; the upstream
    // reverse_edges cache will be rebuilt on the next save() call.
    let _ = graph;
    String::new()
}

/// Re-export a tiny helper so callers don't need to depend on gvec-core
/// directly to enumerate node ids.
pub fn apply_in_memory(mutations: &[GraphMutation], graph: &mut MemoryGraph) -> Result<()> {
    apply_mutations_in_place(graph, mutations)
        .map_err(|e| anyhow::anyhow!("apply_in_memory: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jcode_memory_types::{Edge, MemoryCategory, MemoryEntry, MemoryScope};

    #[test]
    fn open_in_memory_round_trip() {
        let backend = SqliteGvecBackend::open_in_memory().unwrap();
        let key = StoreKey::new("test-1");

        // Empty load.
        let empty = backend.load(&key).unwrap();
        assert_eq!(empty.memory_count(), 0);

        let mut entry = MemoryEntry::new(MemoryCategory::Fact, "rust language guide");
        entry.id = "m1".into();
        entry.tags = vec!["alpha".into()];
        let json = serde_json::to_string(&entry).unwrap();

        backend
            .apply_mutations(
                &key,
                &[
                    GraphMutation::UpsertMemory {
                        id: "m1".into(),
                        json,
                    },
                    GraphMutation::UpsertEdge {
                        from: "m1".into(),
                        to: "tag:alpha".into(),
                        kind_json: serde_json::to_string(&EdgeKind::HasTag).unwrap(),
                    },
                ],
            )
            .unwrap();

        let loaded = backend.load(&key).unwrap();
        assert_eq!(loaded.memory_count(), 1);
        let m = loaded.get_memory("m1").unwrap();
        assert_eq!(m.content, "rust language guide");
        // upsert_edge auto-creates the tag:alpha placeholder node so
        // the edge between m1 and tag:alpha is persisted.
        assert_eq!(loaded.edges.get("m1").map(|v| v.len()), Some(1));
    }

    #[test]
    fn sanitize_keeps_alnum_and_underscore() {
        assert_eq!(sanitize("a-b/c d"), "a_b_c_d");
        assert_eq!(sanitize("abc_123"), "abc_123");
    }

    #[test]
    fn save_round_trip_preserves_graph() {
        let backend = SqliteGvecBackend::open_in_memory().unwrap();
        let key = StoreKey::new("save-test");

        let mut g = MemoryGraph::new();
        let mut e = MemoryEntry::new(MemoryCategory::Fact, "hello world");
        e.id = "m1".into();
        e.tags = vec!["alpha".into()];
        g.add_memory(e);

        backend.save(&key, &g).unwrap();

        let loaded = backend.load(&key).unwrap();
        assert_eq!(loaded.memory_count(), 1);
        assert!(loaded.edges.contains_key("m1"));
        // The save path also creates the Tag node, so reload should
        // have a Tag entry.
        assert_eq!(loaded.tags.len(), 1);
    }

    #[test]
    fn ensure_known_label_default() {
        assert_eq!(ensure_known_label("Memory"), "Memory");
        assert_eq!(ensure_known_label(""), "Node");
    }

    // Suppress unused-import warning when the binary is built without tests.
    #[allow(dead_code)]
    fn _unused() {
        let _ = Edge::new("x", EdgeKind::HasTag);
    }
}