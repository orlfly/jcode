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
use gvec::{Database, Distance, Graph, VectorValue};
use gvec_core::sql::SqlExecutor as _;
use gvec_core::storage::Storage;
use jcode_memory_types::{
    ClusterEntry, EdgeKind, GraphBackend, GraphMutation, MemoryGraph, StoreKey, TagEntry,
    apply_mutations_in_place,
};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(test)]
static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

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
    /// Bumped after every successful write through this backend.
    ///
    /// SQLite's `PRAGMA data_version` only changes for commits made by
    /// other connections, so the process-local write counter covers our
    /// own writes and the pragma covers cross-process writers.
    revision: Arc<AtomicU64>,
    /// Cache namespace unique to this physical database.
    cache_namespace: String,
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
        let dir = storage::jcode_dir()?.join("memory").join("backend-sqlite");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create SqliteGvecBackend dir {}", dir.display()))?;
        let file = dir.join("gvec.sqlite");
        Self::open(file)
    }

    /// Open an isolated SQLite database under
    /// `<jcode_dir>/memory/test/backend-sqlite/gvec.sqlite`. Used by
    /// `MemoryManager::new_test()` so the test suite does not see or
    /// pollute the real user memory store.
    pub fn open_test_dir() -> Result<Self> {
        let dir = storage::jcode_dir()?
            .join("memory")
            .join("test")
            .join("backend-sqlite");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create test SqliteGvecBackend dir {}", dir.display()))?;
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
        let cache_namespace = format!("sqlite-gvec:{}", path.display());
        Ok(Self {
            db: DbHandle { db: Arc::new(db) },
            file: path,
            write_lock: Arc::new(Mutex::new(())),
            revision: Arc::new(AtomicU64::new(0)),
            cache_namespace,
        })
    }

    /// Open an in-memory backend. Used by tests.
    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let db = Database::open_in_memory()
            .map_err(|e| anyhow::anyhow!("SqliteGvecBackend::open_in_memory: {e}"))?;
        let cache_namespace = format!(
            "sqlite-gvec:memory:{}",
            NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed)
        );
        Ok(Self {
            db: DbHandle { db: Arc::new(db) },
            file: PathBuf::from(":memory:"),
            write_lock: Arc::new(Mutex::new(())),
            revision: Arc::new(AtomicU64::new(0)),
            cache_namespace,
        })
    }

    fn graph(&self, key: &StoreKey) -> Result<Graph> {
        let prefix = format!("g_{}", sanitize(key.as_str()));
        self.db
            .db
            .graph(&prefix)
            .map_err(|e| anyhow::anyhow!("open graph '{prefix}': {e}"))
    }

    fn revision_for(&self, graph: &Graph) -> u64 {
        let external = graph
            .storage
            .exec_ref()
            .query_one_json("PRAGMA data_version", &[])
            .ok()
            .flatten()
            .and_then(|mut row| row.pop())
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        let local = self.revision.load(Ordering::SeqCst);
        local.wrapping_mul(1_000_003).wrapping_add(external)
    }

    // Unused now that load() reads through graph() directly; kept for
    // callers that only need the storage handle.
    #[allow(dead_code)]
    fn storage(&self, key: &StoreKey) -> Result<Storage> {
        let g = self.graph(key)?;
        Ok(g.storage)
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_') {
                c
            } else {
                '_'
            }
        })
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
    let kind_json =
        serde_json::to_value(kind).map_err(|e| anyhow::anyhow!("encode EdgeKind: {e}"))?;
    let edge_type = match kind {
        EdgeKind::HasTag => "has_tag",
        EdgeKind::InCluster => "in_cluster",
        EdgeKind::RelatesTo { .. } => "relates_to",
        EdgeKind::Supersedes => "supersedes",
        EdgeKind::Contradicts => "contradicts",
        EdgeKind::DerivedFrom => "derived_from",
        EdgeKind::Contains => "contains",
        EdgeKind::NextStep => "next_step",
        EdgeKind::Bridge => "bridge",
        EdgeKind::Hypothesized { .. } => "hypothesized",
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
        EdgeKind::Contains => "contains",
        EdgeKind::NextStep => "next_step",
        EdgeKind::Bridge => "bridge",
        EdgeKind::Hypothesized { .. } => "hypothesized",
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

    fn graph_revision(&self, key: &StoreKey) -> Result<u64> {
        let graph = self.graph(key)?;
        Ok(self.revision_for(&graph))
    }

    fn cache_namespace(&self) -> String {
        self.cache_namespace.clone()
    }

    fn load(&self, key: &StoreKey) -> Result<MemoryGraph> {
        let graph = self.graph(key)?;
        let revision = self.revision_for(&graph);
        if let Some(cached) =
            crate::memory::cache::cached_graph_for_backend(&self.cache_namespace, key, revision)
        {
            return Ok(cached);
        }
        let mut out = MemoryGraph::new();
        // Graph metadata (ontology binding, counters) is stored beside the node
        // tables. Before this existed `ReplaceMetadata` was a silent no-op, so
        // the ontology a store was authored against could never be recorded.
        if let Some(metadata) = load_metadata(&graph)? {
            out.metadata = metadata;
        }

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
            // A single unreadable endpoint skips one edge; failing the whole
            // load sent callers into the stale-JSON rollback path.
            let (from_id, to_id) = match (node_props(&graph, edge.source), node_props(&graph, edge.target)) {
                (Ok(src), Ok(tgt)) => {
                    (read_node_id_from_props(&src), read_node_id_from_props(&tgt))
                }
                (Err(e), _) | (_, Err(e)) => {
                    crate::logging::info(&format!(
                        "load edge skipped (endpoint vanished mid-read): {e}"
                    ));
                    continue;
                }
            };
            let (Some(from), Some(to)) = (from_id, to_id) else {
                continue;
            };
            // Drop edges whose memory endpoints have no MemoryEntry in this
            // graph. A node row that only carries `__jcode_id` is a
            // placeholder `ensure_node` created for an edge whose target was
            // missing (e.g. a supersede row purged in the same session); the
            // loader cannot reconstruct the entry, so the edge would become
            // a permanent phantom endpoint: 17/51 edges in one live store
            // touched such nodes, none of them traversable, and each save
            // re-materialised the placeholders. Tag/cluster endpoints are
            // never placeholders-with-content and stay.
            if from.starts_with("mem_") && !out.memories.contains_key(&from) {
                continue;
            }
            if to.starts_with("mem_") && !out.memories.contains_key(&to) {
                continue;
            }
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

        crate::memory::cache::cache_graph_for_backend(
            &self.cache_namespace,
            key,
            &out,
            revision,
        );
        Ok(out)
    }

    fn save(&self, key: &StoreKey, graph: &MemoryGraph) -> Result<()> {
        let _guard = self.write_lock.lock().unwrap_or_else(|p| p.into_inner());
        let g = self.graph(key)?;
        // Ensure the FTS5 shadow index exists and stays in sync BEFORE
        // rewriting the nodes table. `remember_project`/`save_project_graph`
        // route through `save()` (the slow replace-all path) rather than
        // `apply_mutations`, so without this the FTS5 index is only ever
        // created on the mutation-log path or on the first explicit
        // `search()`. That left most stores without a text index (P1-2: only
        // a couple of graphs had FTS5 tables). Running it here guarantees
        // every store that is written also gets a live keyword index.
        ensure_text_index(&g)?;
        // Capture semantic edges added by another process after this snapshot
        // was loaded; they are re-applied after the replace-all below.
        let concurrent_relates = concurrent_relates_to_edges(&g, graph)?;
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
        for (from, to, kind) in concurrent_relates {
            upsert_edge(&g, &from, &to, &kind)?;
        }
        save_metadata(&g, &graph.metadata)?;
        self.revision.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn apply_mutations(&self, key: &StoreKey, mutations: &[GraphMutation]) -> Result<MemoryGraph> {
        let _guard = self.write_lock.lock().unwrap_or_else(|p| p.into_inner());
        let g = self.graph(key)?;
        ensure_text_index(&g)?;
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
                GraphMutation::ReplaceMetadata { json } => {
                    // Metadata is part of `MemoryGraph` but gvec has no node for
                    // it, so it is persisted in the side table instead of being
                    // dropped (see `save_metadata`).
                    let metadata: jcode_memory_types::graph::GraphMetadata =
                        serde_json::from_str(json)
                            .map_err(|e| anyhow::anyhow!("ReplaceMetadata({json}): {e}"))?;
                    save_metadata(&g, &metadata)?;
                }
            }
        }
        self.revision.fetch_add(1, Ordering::SeqCst);
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

    /// Enumerate store keys from persisted state: every table of the form
    /// `<prefix>_nodes` maps back to the store key whose sanitised name is
    /// `<prefix>`. Undoes `sanitize("project:<hash>")` -> `g_project_<hash>`.
    fn list_keys(&self) -> Result<Vec<StoreKey>> {
        let rows = self
            .db
            .db
            .conn
            .query_all_json(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE '%\\_nodes' ESCAPE '\\'",
                &[],
            )
            .map_err(|e| anyhow::anyhow!("list_keys: {e}"))?;
        let mut keys = Vec::new();
        for row in rows {
            let Some(Value::String(name)) = row.first() else {
                continue;
            };
            // Strip the trailing "_nodes", then reverse the "g_" prefix
            // added by `graph()`. Characters that `sanitize` mapped to '_'
            // are ambiguous, but store keys we write are safe identifiers
            // already ("global", "project:<16 hex>"), so a single recovery
            // pass is exact for them.
            let prefix = name.strip_suffix("_nodes").unwrap_or(name.as_str());
            let raw = prefix.strip_prefix("g_").unwrap_or(prefix);
            if raw.is_empty() || raw == "global" {
                // The global store has its own key name; keep it as-is so
                // callers see the real key space.
                keys.push(StoreKey::new("global".to_string()));
            } else if let Some(hash) = raw.strip_prefix("project_") {
                keys.push(StoreKey::new(format!("project:{hash}")));
            } else {
                keys.push(StoreKey::new(raw.to_string()));
            }
        }
        Ok(keys)
    }

    /// Run an FTS5 search across the Memory nodes of a single store.
    ///
    /// `query` follows FTS5 query syntax (raw user query is best
    /// sanitised by the caller — words joined with `AND` is usually
    /// safest). Returns `(memory_id, score)` pairs ordered by BM25
    /// relevance; `score` is a non-negative float where larger means
    /// more relevant (we negate gvec's raw BM25 to make the order
    /// intuitive).
    fn text_search(&self, key: &StoreKey, query: &str, k: usize) -> Result<Vec<(String, f32)>> {
        let g = self.graph(key)?;
        ensure_text_index(&g)?;
        let storage = &g.storage;
        let prefix = storage.prefix.clone();
        if !prefix
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(anyhow::anyhow!(
                "text_search: refusing to query non-identifier prefix {prefix:?}"
            ));
        }
        let fts_tbl = format!("jcode_fts_{prefix}");
        let sql = format!(
            "SELECT rowid, -bm25({fts_tbl}) AS score \
             FROM {fts_tbl} \
             WHERE {fts_tbl} MATCH ?1 \
             ORDER BY score DESC \
             LIMIT ?2"
        );
        let rows = storage
            .exec_ref()
            .query_all_json(
                &sql,
                &[
                    gvec_core::sql::SqlValue::Text(query.to_string()),
                    gvec_core::sql::SqlValue::Integer(k as i64),
                ],
            )
            .map_err(|e| anyhow::anyhow!("text_search query: {e}"))?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            if row.len() < 2 {
                continue;
            }
            let rowid = match &row[0] {
                Value::Number(n) => n.as_i64().unwrap_or(0),
                Value::String(s) => s.parse().unwrap_or(0),
                _ => 0,
            };
            let score = match &row[1] {
                Value::Number(n) => n.as_f64().unwrap_or(0.0) as f32,
                _ => 0.0,
            };
            if let Some(id) = rowid_to_memory_id(&g, rowid)? {
                out.push((id, score));
            }
        }
        Ok(out)
    }

    fn vector_search(&self, key: &StoreKey, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        if query.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let g = self.graph(key)?;
        let query = VectorValue::F32(query.to_vec());
        let indexed = ensure_vector_index(&g).and_then(|_| {
            g.vector_search("Memory", "embedding", &query, k)
                .map_err(|e| anyhow::anyhow!("gvec vector_search: {e}"))
        });
        let hits = match indexed {
            Ok(hits) => hits,
            Err(error) => {
                crate::logging::info(&format!(
                    "gvec vector index unavailable, using exact scan: {error}"
                ));
                g.exact_vector_search("Memory", "embedding", &query, k, Distance::Cosine)?
            }
        };
        let mut out = Vec::with_capacity(hits.len());
        for (rowid, distance) in hits {
            if let Some(id) = rowid_to_memory_id(&g, rowid)? {
                // The index uses cosine distance: 0 is identical, 2 is
                // opposite. Convert to a descending similarity score.
                out.push((id, (1.0 - distance).clamp(-1.0, 1.0)));
            }
        }
        Ok(out)
    }

    fn tag_search(&self, key: &StoreKey, tag: &str, k: usize) -> Result<Vec<(String, f32)>> {
        let tag = tag.trim();
        if tag.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let g = self.graph(key)?;
        ensure_tag_index(&g)?;
        let rowids = g.tag_search("Memory", "tags", tag, k)?;
        let mut out = Vec::with_capacity(rowids.len());
        for rowid in rowids {
            if let Some(id) = rowid_to_memory_id(&g, rowid)? {
                out.push((id, 1.0));
            }
        }
        Ok(out)
    }
}

/// Lazily create the cosine HNSW index over `Memory.embedding`.
///
/// The index is rebuilt by gvec when it is created. If the store contains
/// mixed embedding models with different dimensions, creation fails; callers
/// fall back to an exact scan that filters by query dimension.
fn ensure_vector_index(graph: &Graph) -> Result<()> {
    let exists = graph
        .list_vector_indexes()?
        .iter()
        .any(|meta| meta.label == "Memory" && meta.property == "embedding");
    if exists {
        return Ok(());
    }
    graph
        .create_vector_index(
            "Memory",
            "embedding",
            "hnsw",
            Some("cosine"),
            Some(json!({
                "m": 16,
                "ef_construction": 200,
                "ef_search": 64
            })),
        )
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("create Memory.embedding vector index: {e}"))
}

/// Lazily create the exact tag index over `Memory.tags`.
fn ensure_tag_index(graph: &Graph) -> Result<()> {
    let exists = graph
        .list_tag_indexes()?
        .iter()
        .any(|meta| meta.label == "Memory" && meta.property == "tags");
    if exists {
        return Ok(());
    }
    graph
        .create_tag_index("Memory", "tags")
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("create Memory.tags tag index: {e}"))
}

/// Lazily create a standalone-content FTS5 shadow index for
/// `Memory.content` and keep it in sync via triggers.
///
/// This deliberately mirrors `gvec_core::fts::create_text_index` in
/// shape (standalone-content FTS5 table + AI/AU/AD triggers) but is
/// inlined here so we own the schema and triggers outright. That gives
/// us two things gvec's helper doesn't:
///   1. We can name the table/triggers under a `jcode_fts_<prefix>`
///      namespace that doesn't collide with whatever the user types
///      into Cypher (e.g. `CREATE TEXT INDEX ON :Doc(body)` would
///      otherwise create a parallel table on top of ours).
///   2. We use `INSERT OR REPLACE` + `WHEN new.labels LIKE ...` so a
///      re-run of ensure_text_index on a save-rewritten nodes table
///      re-syncs cleanly without failing on duplicate rowids.
fn ensure_text_index(graph: &Graph) -> Result<()> {
    let storage = &graph.storage;
    let exec = storage.exec_ref();
    if !gvec_core::fts::fts5_available(exec) {
        crate::logging::warn("SQLite build has no FTS5; text_search disabled");
        return Ok(());
    }
    let prefix = storage.prefix.clone();
    // Identifiers are validated against gvec's prefix naming rules
    // when constructing the table via gvec (we only use plain
    // ASCII prefixes here).
    if !prefix
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err(anyhow::anyhow!(
            "ensure_text_index: refusing to create FTS5 for non-identifier prefix {prefix:?}"
        ));
    }
    let fts_tbl = format!("jcode_fts_{prefix}");
    let nodes_tbl = format!("{prefix}_nodes");
    let trig_ins = format!("jcode_fts_{prefix}_ai");
    let trig_upd = format!("jcode_fts_{prefix}_au");
    let trig_del = format!("jcode_fts_{prefix}_ad");
    let lbl_pat = format!("%Memory%");
    let prop_json = "$.content";

    // Standalone-content FTS5 table (no `content=` clause) so we can
    // insert directly with `(rowid, content)`. This matches what
    // gvec itself does and avoids the external-content 'insert'
    // command's stricter INSERT/VALUES-only rule.
    exec.execute(
        &format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {fts_tbl} USING fts5(
                content,
                tokenize='unicode61 remove_diacritics 2'
            )"
        ),
        &[],
    )
    .map_err(|e| anyhow::anyhow!("ensure_text_index create fts: {e}"))?;

    // Backfill: standalone-content tables accept direct
    // (rowid, content) inserts and `ON CONFLICT` lets us re-run the
    // backfill safely after `save` rewrites the nodes table.
    exec.execute(
        &format!(
            "INSERT OR REPLACE INTO {fts_tbl}(rowid, content)
                SELECT n.id, json_extract(n.properties, '{prop_json}')
                  FROM {nodes_tbl} n
                 WHERE n.labels LIKE ?1
                   AND json_extract(n.properties, '{prop_json}') IS NOT NULL"
        ),
        &[gvec_core::sql::SqlValue::Text(lbl_pat.clone())],
    )
    .map_err(|e| anyhow::anyhow!("ensure_text_index backfill: {e}"))?;

    // Sync triggers (one-time create). Standalone-content FTS5 tables
    // support `INSERT OR REPLACE` for both insert and update.
    let ai = format!(
        "CREATE TRIGGER IF NOT EXISTS {trig_ins}
            AFTER INSERT ON {nodes_tbl}
            WHEN new.labels LIKE '{lbl_pat}'
        BEGIN
            INSERT OR REPLACE INTO {fts_tbl}(rowid, content)
                VALUES(new.id, json_extract(new.properties, '{prop_json}'));
        END"
    );
    let ad = format!(
        "CREATE TRIGGER IF NOT EXISTS {trig_del}
            AFTER DELETE ON {nodes_tbl}
            WHEN old.labels LIKE '{lbl_pat}'
        BEGIN
            DELETE FROM {fts_tbl} WHERE rowid = old.id;
        END"
    );
    let au = format!(
        "CREATE TRIGGER IF NOT EXISTS {trig_upd}
            AFTER UPDATE ON {nodes_tbl}
            WHEN new.labels LIKE '{lbl_pat}'
        BEGIN
            INSERT OR REPLACE INTO {fts_tbl}(rowid, content)
                VALUES(new.id, json_extract(new.properties, '{prop_json}'));
        END"
    );
    exec.execute(&ai, &[])
        .map_err(|e| anyhow::anyhow!("ensure_text_index create trig_ins: {e}"))?;
    exec.execute(&ad, &[])
        .map_err(|e| anyhow::anyhow!("ensure_text_index create trig_del: {e}"))?;
    exec.execute(&au, &[])
        .map_err(|e| anyhow::anyhow!("ensure_text_index create trig_upd: {e}"))?;
    Ok(())
}

/// Side table holding one row of `GraphMetadata` per named graph.
///
/// Keyed by the gvec storage prefix (`g_project_<hash>`, `g_global`), which is
/// exactly the `StoreKey` the backend was opened with.
const META_TABLE: &str = "jcode_graph_meta";

fn ensure_meta_table(graph: &Graph) -> Result<()> {
    let sql = format!(
        "CREATE TABLE IF NOT EXISTS {META_TABLE} (store TEXT PRIMARY KEY, json TEXT NOT NULL)"
    );
    graph
        .storage
        .exec_ref()
        .execute(&sql, &[])
        .map_err(|e| anyhow::anyhow!("ensure_meta_table: {e}"))?;
    Ok(())
}

fn load_metadata(graph: &Graph) -> Result<Option<jcode_memory_types::graph::GraphMetadata>> {
    ensure_meta_table(graph)?;
    let sql = format!("SELECT json FROM {META_TABLE} WHERE store = ?1");
    let row = graph
        .storage
        .exec_ref()
        .query_one_json(&sql, &[gvec_core::sql::SqlValue::from(graph.storage.prefix.as_str())])
        .map_err(|e| anyhow::anyhow!("load_metadata: {e}"))?;
    let Some(mut row) = row else {
        return Ok(None);
    };
    let Some(value) = row.pop() else {
        return Ok(None);
    };
    let Some(raw) = value.as_str() else {
        return Ok(None);
    };
    match serde_json::from_str(raw) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(e) => {
            crate::logging::warn(&format!(
                "SqliteGvecBackend: ignoring corrupt graph metadata for {}: {e}",
                graph.storage.prefix
            ));
            Ok(None)
        }
    }
}

fn save_metadata(graph: &Graph, metadata: &jcode_memory_types::graph::GraphMetadata) -> Result<()> {
    ensure_meta_table(graph)?;
    let json = serde_json::to_string(metadata)?;
    let sql = format!("INSERT OR REPLACE INTO {META_TABLE} (store, json) VALUES (?1, ?2)");
    graph
        .storage
        .exec_ref()
        .execute(
            &sql,
            &[
                gvec_core::sql::SqlValue::from(graph.storage.prefix.as_str()),
                gvec_core::sql::SqlValue::from(json.as_str()),
            ],
        )
        .map_err(|e| anyhow::anyhow!("save_metadata: {e}"))?;
    Ok(())
}

/// Reverse-map a gvec rowid back to the jcode `Memory:<id>` string.
/// Returns `Ok(None)` if the row has been deleted between the FTS5 hit
/// and this lookup.
fn rowid_to_memory_id(graph: &Graph, rowid: i64) -> Result<Option<String>> {
    use gvec_core::ids::NodeId;
    let nid = NodeId(rowid);
    match graph.storage.get_node(nid) {
        Ok(node) => {
            let id = node
                .properties
                .get("__jcode_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            Ok(id)
        }
        Err(_) => Ok(None),
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

/// `save()` replaces the whole graph. A caller that loaded before another
/// process added a semantic edge would otherwise silently drop it on its next
/// write (last-writer-wins). Preserve `relates_to` edges that still connect
/// live memory nodes but are absent from the incoming snapshot.
///
/// Only `relates_to` is merged: it is written by discovery/review flows rather
/// than by whole-graph edits, and there is no user-facing "unlink" operation
/// that would depend on omission from a snapshot to delete it. Other kinds stay
/// authoritative in the snapshot, so tag/cluster rebuilds can still replace
/// them.
fn concurrent_relates_to_edges(
    graph: &Graph,
    incoming: &MemoryGraph,
) -> Result<Vec<(String, String, EdgeKind)>> {
    let mut present: HashSet<(String, String)> = HashSet::new();
    for (from, edges) in &incoming.edges {
        for edge in edges {
            if matches!(edge.kind, EdgeKind::RelatesTo { .. }) {
                present.insert(if from <= &edge.target {
                    (from.clone(), edge.target.clone())
                } else {
                    (edge.target.clone(), from.clone())
                });
            }
        }
    }

    let mut preserved = Vec::new();
    for edge in graph
        .storage
        .list_edges(None, None)
        .map_err(|e| anyhow::anyhow!("preserve relates_to: list: {e}"))?
    {
        if edge.edge_type.as_deref() != Some("relates_to") {
            continue;
        }
        let (Ok(src), Ok(tgt)) = (
            node_props(graph, edge.source),
            node_props(graph, edge.target),
        ) else {
            continue;
        };
        let (Some(from), Some(to)) = (read_node_id_from_props(&src), read_node_id_from_props(&tgt))
        else {
            continue;
        };
        if !incoming.memories.contains_key(&from) || !incoming.memories.contains_key(&to) {
            continue;
        }
        let pair = if from <= to {
            (from.clone(), to.clone())
        } else {
            (to.clone(), from.clone())
        };
        if !present.insert(pair) {
            continue;
        }
        let kind = serde_json::from_value(
            edge.properties
                .get("kind_payload")
                .cloned()
                .unwrap_or(Value::Null),
        )
        .unwrap_or(EdgeKind::RelatesTo { weight: 1.0 });
        let kind = match kind {
            EdgeKind::RelatesTo { weight } => EdgeKind::RelatesTo { weight },
            _ => EdgeKind::RelatesTo { weight: 1.0 },
        };
        preserved.push((from, to, kind));
    }
    Ok(preserved)
}

fn node_props(graph: &Graph, rowid: gvec_core::ids::NodeId) -> Result<Value> {
    match graph.storage.get_node(rowid) {
        Ok(n) => Ok(n.properties),
        // A concurrent replace-all save can retire the rowid between the
        // node listing above and this per-row read (`get_node(116): node
        // not found` raced three real loads). One refresh usually heals it;
        // if not, the caller skips the node instead of failing the whole
        // load (a whole-load failure is what used to route the caller into
        // the stale-JSON fallback and roll the store back).
        Err(_) => list_nodes_by_label(graph, "Memory")
            .and_then(|nodes| {
                nodes
                    .into_iter()
                    .find(|(id, _)| {
                        id.parse::<i64>()
                            .map(|candidate| candidate == rowid.0)
                            .unwrap_or(false)
                    })
                    .map(|(_, props)| props)
                    .ok_or_else(|| {
                        anyhow::anyhow!("get_node({}): gone after refresh", rowid.0)
                    })
            })
            .map_err(|e| anyhow::anyhow!("get_node({}): {e}", rowid.0)),
    }
}

// Placeholder for reverse_edges serialization; see comment below.
#[allow(dead_code)]
fn edge_properties_string(graph: &Graph, _rowid: gvec_core::ids::NodeId) -> String {
    // Best-effort string for reverse_edges. We don't have a direct
    // API, so just emit an empty placeholder; the upstream
    // reverse_edges cache will be rebuilt on the next save() call.
    let _ = graph;
    String::new()
}

/// Re-export a tiny helper so callers don't need to depend on gvec-core
/// directly to enumerate node ids.
// Dormant mutation helper kept alongside the backend trait's
// apply_mutations API for future bulk-edit call sites.
#[allow(dead_code)]
pub fn apply_in_memory(mutations: &[GraphMutation], graph: &mut MemoryGraph) -> Result<()> {
    apply_mutations_in_place(graph, mutations).map_err(|e| anyhow::anyhow!("apply_in_memory: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jcode_memory_types::{Edge, MemoryCategory, MemoryEntry};

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
    fn save_preserves_relates_edges_added_after_load() {
        let backend = SqliteGvecBackend::open_in_memory().unwrap();
        let key = StoreKey::new("concurrent-relates-test");

        let mut graph = MemoryGraph::new();
        for (id, content) in [
            ("mem_1", "first"),
            ("mem_2", "second"),
            ("mem_3", "third"),
        ] {
            let mut entry = MemoryEntry::new(MemoryCategory::Fact, content);
            entry.id = id.into();
            graph.add_memory(entry);
        }
        graph.add_edge("mem_1", "mem_2", EdgeKind::RelatesTo { weight: 0.8 });
        backend.save(&key, &graph).unwrap();

        // A second process loads, links mem_1 -> mem_3, and saves.
        let mut concurrent = backend.load(&key).unwrap();
        concurrent.add_edge("mem_1", "mem_3", EdgeKind::RelatesTo { weight: 0.7 });
        backend.save(&key, &concurrent).unwrap();

        // The first process still holds the pre-link snapshot. Its next
        // whole-graph save must not erase the concurrently added edge.
        backend.save(&key, &graph).unwrap();

        let loaded = backend.load(&key).unwrap();
        let rel = |from: &str, to: &str| {
            loaded.edges.get(from).is_some_and(|edges| {
                edges.iter().any(|edge| {
                    edge.target == to && matches!(edge.kind, EdgeKind::RelatesTo { .. })
                })
            })
        };
        assert!(rel("mem_1", "mem_2"), "snapshot edge must survive");
        assert!(rel("mem_1", "mem_3"), "concurrently added edge must survive");
    }

    #[test]
    fn save_drops_relates_edges_to_removed_memories() {
        let backend = SqliteGvecBackend::open_in_memory().unwrap();
        let key = StoreKey::new("removed-endpoint-test");

        let mut graph = MemoryGraph::new();
        for (id, content) in [("mem_1", "first"), ("mem_2", "second")] {
            let mut entry = MemoryEntry::new(MemoryCategory::Fact, content);
            entry.id = id.into();
            graph.add_memory(entry);
        }
        graph.add_edge("mem_1", "mem_2", EdgeKind::RelatesTo { weight: 0.8 });
        backend.save(&key, &graph).unwrap();

        // Removing the endpoint is authoritative: the edge must not be
        // resurrected by the concurrency merge.
        graph.memories.remove("mem_2");
        backend.save(&key, &graph).unwrap();

        let loaded = backend.load(&key).unwrap();
        assert!(loaded.memories.get("mem_2").is_none());
        assert!(
            loaded
                .edges
                .get("mem_1")
                .is_none_or(|edges| edges.iter().all(|edge| edge.target != "mem_2")),
            "edge to a removed memory must be dropped"
        );
    }

    #[test]
    fn ontology_metadata_round_trips_through_the_meta_table() {
        let backend = SqliteGvecBackend::open_in_memory().unwrap();
        let key = StoreKey::new("ontology-roundtrip");

        let ontology = jcode_memory_types::default_ontology();
        let hash = ontology.content_hash();
        let mut graph = MemoryGraph::new();
        graph.metadata.ontology_id = ontology.id.clone();
        graph.metadata.ontology_version = ontology.version;
        graph.metadata.ontology_hash = hash.clone();
        graph.metadata.ontology = Some(ontology.clone());
        graph.metadata.applied_migrations = vec!["v2->v3".to_string()];

        backend.save(&key, &graph).unwrap();
        let loaded = backend.load(&key).unwrap();

        let stored = loaded
            .metadata
            .ontology
            .expect("ontology must survive the meta table");
        assert_eq!(stored.id, ontology.id);
        assert_eq!(stored.version, ontology.version);
        assert_eq!(loaded.metadata.ontology_hash, hash);
        assert_eq!(loaded.metadata.applied_migrations, vec!["v2->v3".to_string()]);
        assert_eq!(stored.types.len(), ontology.types.len());
    }

    #[test]
    fn ensure_known_label_default() {
        assert_eq!(ensure_known_label("Memory"), "Memory");
        assert_eq!(ensure_known_label(""), "Node");
    }

    #[test]
    fn fts5_text_search_finds_upserted_memory() {
        let dir = tempfile::tempdir().unwrap();
        let backend = SqliteGvecBackend::open(dir.path().join("fts.sqlite")).unwrap();
        let key = StoreKey::new("fts-test".to_string());

        let mut graph = MemoryGraph::new();
        graph.add_memory(jcode_memory_types::MemoryEntry::new(
            jcode_memory_types::MemoryCategory::Fact,
            "rust ownership explained briefly",
        ));
        graph.add_memory(jcode_memory_types::MemoryEntry::new(
            jcode_memory_types::MemoryCategory::Fact,
            "python decorators and closures",
        ));
        backend.save(&key, &graph).unwrap();

        // Force the FTS5 shadow index to be created by applying an
        // idempotent mutation. (Real callers route through
        // apply_mutations.)
        backend
            .apply_mutations(
                &key,
                &[GraphMutation::UpsertMemory {
                    id: "noop".to_string(),
                    json: serde_json::to_string(&jcode_memory_types::MemoryEntry::new(
                        jcode_memory_types::MemoryCategory::Fact,
                        "noop",
                    ))
                    .unwrap(),
                }],
            )
            .unwrap();

        let hits = backend.text_search(&key, "rust", 5).unwrap();
        assert!(
            !hits.is_empty(),
            "expected FTS5 to find the rust memory, got empty"
        );
        let (id, score) = &hits[0];
        assert!(
            *score > 0.0,
            "expected positive BM25 score for 'rust' hit, got id={id:?} score={score}"
        );
        assert!(
            id.starts_with("mem_"),
            "id should be a jcode memory id: {id:?}"
        );

        // And the other memory must NOT appear for the unrelated query.
        let py_hits = backend.text_search(&key, "python", 5).unwrap();
        assert!(
            !py_hits.is_empty(),
            "expected FTS5 to find the python memory"
        );
        assert_ne!(
            hits[0].0, py_hits[0].0,
            "rust and python must hit distinct rows"
        );
    }

    #[test]
    fn vector_search_finds_nearest_memory_through_gvec_index() {
        let backend = SqliteGvecBackend::open_in_memory().unwrap();
        let key = StoreKey::new("vector-test");

        let mut graph = MemoryGraph::new();
        let mut near = MemoryEntry::new(MemoryCategory::Fact, "near vector");
        near.set_embedding(Some(vec![1.0, 0.0, 0.0]), Some("test-model".into()));
        let near_id = graph.add_memory(near);
        let mut far = MemoryEntry::new(MemoryCategory::Fact, "far vector");
        far.set_embedding(Some(vec![0.0, 1.0, 0.0]), Some("test-model".into()));
        let far_id = graph.add_memory(far);
        backend.save(&key, &graph).unwrap();

        let hits = backend.vector_search(&key, &[0.9, 0.1, 0.0], 2).unwrap();
        assert_eq!(
            hits.first().map(|(id, _)| id.as_str()),
            Some(near_id.as_str())
        );
        assert!(hits.iter().any(|(id, _)| id == &far_id));
    }

    #[test]
    fn tag_search_finds_memory_by_exact_tag() {
        let backend = SqliteGvecBackend::open_in_memory().unwrap();
        let key = StoreKey::new("tag-test");

        let mut graph = MemoryGraph::new();
        let rust = graph.add_memory(
            MemoryEntry::new(MemoryCategory::Fact, "rust memory").with_tags(vec!["rust".into()]),
        );
        let python = graph.add_memory(
            MemoryEntry::new(MemoryCategory::Fact, "python memory")
                .with_tags(vec!["python".into()]),
        );
        backend.save(&key, &graph).unwrap();

        let hits = backend.tag_search(&key, "rust", 10).unwrap();
        assert_eq!(hits, vec![(rust, 1.0)]);
        assert!(backend.tag_search(&key, "rust", 10).unwrap()[0].0 != python);
    }

    #[test]
    fn load_reuses_graph_until_write() {
        let backend = SqliteGvecBackend::open_in_memory().unwrap();
        let key = StoreKey::new("cache-test");

        let mut graph = MemoryGraph::new();
        graph.add_memory(MemoryEntry::new(MemoryCategory::Fact, "cached first"));
        backend.save(&key, &graph).unwrap();

        assert_eq!(backend.load(&key).unwrap().memory_count(), 1);
        assert_eq!(backend.load(&key).unwrap().memory_count(), 1);

        graph.add_memory(MemoryEntry::new(MemoryCategory::Fact, "cached second"));
        backend.save(&key, &graph).unwrap();
        assert_eq!(
            backend.load(&key).unwrap().memory_count(),
            2,
            "a backend write must invalidate the cached graph"
        );
    }

    #[test]
    fn graph_revision_sees_other_connection_writes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.sqlite");
        let reader = SqliteGvecBackend::open(&path).unwrap();
        let writer = SqliteGvecBackend::open(&path).unwrap();
        let key = StoreKey::new("shared-cache");

        let mut graph = MemoryGraph::new();
        graph.add_memory(MemoryEntry::new(MemoryCategory::Fact, "shared first"));
        reader.save(&key, &graph).unwrap();
        assert_eq!(reader.load(&key).unwrap().memory_count(), 1);
        let before = reader.graph_revision(&key).unwrap();

        graph.add_memory(MemoryEntry::new(MemoryCategory::Fact, "shared second"));
        writer.save(&key, &graph).unwrap();

        assert_ne!(
            before,
            reader.graph_revision(&key).unwrap(),
            "PRAGMA data_version must expose the other connection's commit"
        );
        assert_eq!(
            reader.load(&key).unwrap().memory_count(),
            2,
            "another connection's write must invalidate the cached graph"
        );
    }

    #[test]
    #[ignore = "benchmark; run with --ignored"]
    fn bench_load_reuses_backend_graph_cache() {
        let backend = SqliteGvecBackend::open_in_memory().unwrap();
        let key = StoreKey::new("cache-bench");
        let mut graph = MemoryGraph::new();
        for index in 0..2_000 {
            graph.add_memory(MemoryEntry::new(
                MemoryCategory::Fact,
                format!("cache benchmark memory {index}"),
            ));
        }
        backend.save(&key, &graph).unwrap();

        let baseline = std::time::Instant::now();
        for _ in 0..20 {
            backend.revision.fetch_add(1, Ordering::SeqCst);
            let _ = backend.load(&key).unwrap();
        }
        let baseline = baseline.elapsed();

        let cached = std::time::Instant::now();
        for _ in 0..20 {
            let _ = backend.load(&key).unwrap();
        }
        let cached = cached.elapsed();
        eprintln!("backend graph cache baseline={baseline:?} cached={cached:?}");
    }

    #[test]
    fn fts5_available_for_bundled_sqlite() {
        let dir = tempfile::tempdir().unwrap();
        let backend = SqliteGvecBackend::open(dir.path().join("probe.sqlite")).unwrap();
        let key = StoreKey::new("probe".to_string());
        let g = backend.graph(&key).unwrap();
        assert!(
            gvec_core::fts::fts5_available(g.storage.exec_ref()),
            "bundled rusqlite must expose FTS5"
        );
    }

    #[test]
    fn upstream_gvec_create_text_index_works_via_our_backend() {
        // Direct repro of gvec's own fts_create_and_search but
        // mounted through SqliteGvecBackend. This should pass —
        // otherwise the bug is upstream and our wrapper is innocent.
        let dir = tempfile::tempdir().unwrap();
        let backend = SqliteGvecBackend::open(dir.path().join("upstream.sqlite")).unwrap();
        let key = StoreKey::new("upstream-repro".to_string());
        let g = backend.graph(&key).unwrap();

        g.run(r#"CREATE (a:Doc {title: "Rust guide", body: "the rust programming language"})"#)
            .unwrap();
        g.run(
            r#"CREATE (b:Doc {title: "Python guide", body: "python is dynamic and high-level"})"#,
        )
        .unwrap();
        g.run(r#"CREATE (c:Doc {title: "Rust async", body: "rust async runtime and tokio"})"#)
            .unwrap();

        g.create_text_index("Doc", "body", None).unwrap();
        let hits = g.text_search("Doc", "body", "rust", 10).unwrap();
        assert_eq!(hits.len(), 2, "expected 2 rust hits, got {:?}", hits);
    }

    // Suppress unused-import warning when the binary is built without tests.
    #[allow(dead_code)]
    fn _unused() {
        let _ = Edge::new("x", EdgeKind::HasTag);
    }
}
