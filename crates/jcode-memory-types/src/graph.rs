//! Graph-based memory storage with tags, clusters, and semantic links
//!
//! This module provides a graph structure for organizing memories with:
//! - Tag nodes for explicit organization
//! - Cluster nodes for automatic grouping (future)
//! - Various edge types (HasTag, RelatesTo, Supersedes, etc.)
//! - BFS cascade retrieval through the graph

use crate::migration::{MigrationState, OntologySnapshot};
use crate::ontology::Ontology;
use crate::{MemoryEntry, MemoryStore, ValidationReport, validate_graph};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};

/// Current graph format version for migration detection
pub const GRAPH_VERSION: u32 = 2;

#[derive(Debug)]
struct TopKItem<T> {
    score: f32,
    ordinal: usize,
    value: T,
}

impl<T> PartialEq for TopKItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.ordinal == other.ordinal
    }
}

impl<T> Eq for TopKItem<T> {}

impl<T> PartialOrd for TopKItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for TopKItem<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.ordinal.cmp(&other.ordinal))
    }
}

fn top_k_scored<T, I>(items: I, limit: usize) -> Vec<(T, f32)>
where
    I: IntoIterator<Item = (T, f32)>,
{
    if limit == 0 {
        return Vec::new();
    }

    let mut heap: BinaryHeap<Reverse<TopKItem<T>>> = BinaryHeap::new();
    for (ordinal, (value, score)) in items.into_iter().enumerate() {
        let candidate = Reverse(TopKItem {
            score,
            ordinal,
            value,
        });

        if heap.len() < limit {
            heap.push(candidate);
            continue;
        }

        let replace = heap
            .peek()
            .map(|smallest| score > smallest.0.score)
            .unwrap_or(false);
        if replace {
            heap.pop();
            heap.push(candidate);
        }
    }

    let mut results: Vec<_> = heap
        .into_iter()
        .map(|Reverse(item)| (item.value, item.score, item.ordinal))
        .collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
    results
        .into_iter()
        .map(|(value, score, _)| (value, score))
        .collect()
}

/// Edge relationship types between nodes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EdgeKind {
    /// Memory has this explicit tag
    HasTag,
    /// Memory belongs to auto-discovered cluster
    InCluster,
    /// Semantic relationship with weight (0.0-1.0)
    RelatesTo {
        #[serde(default = "default_weight")]
        weight: f32,
    },
    /// Newer memory replaces older one
    Supersedes,
    /// Conflicting information (both kept, flagged)
    Contradicts,
    /// Procedural knowledge derived from facts
    DerivedFrom,
    /// Structural containment: an artifact/root contains an index part.
    Contains,
    /// Ordered step edge between consecutive procedural steps.
    NextStep,
    /// Explicit cross-store bridge between a project instance and a global one.
    ///
    /// This edge exists only in stitched path results; the bridge registry keeps
    /// the persisted pair, so project and global memory files stay independent.
    Bridge,
    /// Candidate edge from the shadow registry, used only inside a query.
    ///
    /// The main graph never persists this kind. Path results carrying it must be
    /// treated as hypotheses until the shadow edge is explicitly accepted.
    Hypothesized {
        relation: String,
        weight: f32,
        confidence: f32,
    },
    /// Verified entity alignment between two memory ids in the same store.
    ///
    /// Promotion requires an explicit evidence-bearing review. Cross-store
    /// alignment still uses the bridge registry so neither store is rewritten.
    SameAs,
}

/// One hop in a weighted memory-graph path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryPathEdge {
    pub from: String,
    pub to: String,
    pub kind: EdgeKind,
    pub cost: f32,
}

/// A loop-free path between two memory-graph nodes.
///
/// Paths may pass through `tag:` / `cluster:` hub nodes. Consumers that only
/// want memories should filter those node ids out for display, but keep them in
/// the path so the traversal remains auditable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryPath {
    pub nodes: Vec<String>,
    pub edges: Vec<MemoryPathEdge>,
    pub cost: f32,
    pub confidence: f32,
}

fn default_weight() -> f32 {
    1.0
}

fn scaled_memory_cost(cost: f32) -> u64 {
    (cost.max(0.0) * 1000.0).round() as u64
}

fn memory_edge_key(edge: &MemoryPathEdge) -> (String, String, String) {
    (
        edge.from.clone(),
        edge.to.clone(),
        serde_json::to_string(&edge.kind).unwrap_or_default(),
    )
}

fn relation_allowed(kind: &EdgeKind, allowed_relations: &[&str]) -> bool {
    allowed_relations.is_empty()
        || allowed_relations.iter().any(|allowed| {
            let allowed = allowed.trim();
            kind.as_str().eq_ignore_ascii_case(allowed)
                || matches!(
                    kind,
                    EdgeKind::Hypothesized { relation, .. }
                        if relation.eq_ignore_ascii_case(allowed)
                )
        })
}

impl EdgeKind {
    /// Canonical relation name without weight or confidence decorations.
    pub fn relation_name(&self) -> &str {
        match self {
            Self::HasTag => "has_tag",
            Self::InCluster => "in_cluster",
            Self::RelatesTo { .. } => "relates_to",
            Self::Supersedes => "supersedes",
            Self::Contradicts => "contradicts",
            Self::DerivedFrom => "derived_from",
            Self::Contains => "contains",
            Self::NextStep => "next_step",
            Self::Bridge => "bridge",
            Self::SameAs => "same_as",
            Self::Hypothesized { relation, .. } => relation,
        }
    }

    /// Canonical wire name used by relation filters and provenance.
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeKind::HasTag => "has_tag",
            EdgeKind::InCluster => "in_cluster",
            EdgeKind::RelatesTo { .. } => "relates_to",
            EdgeKind::Supersedes => "supersedes",
            EdgeKind::Contradicts => "contradicts",
            EdgeKind::DerivedFrom => "derived_from",
            EdgeKind::Contains => "contains",
            EdgeKind::NextStep => "next_step",
            EdgeKind::Bridge => "bridge",
            EdgeKind::SameAs => "same_as",
            EdgeKind::Hypothesized { .. } => "hypothesized",
        }
    }

    /// Parse a canonical relation name. Unknown names are rejected rather than
    /// silently widening a path query.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "has_tag" => EdgeKind::HasTag,
            "in_cluster" => EdgeKind::InCluster,
            "relates_to" => EdgeKind::RelatesTo { weight: 1.0 },
            "supersedes" => EdgeKind::Supersedes,
            "contradicts" => EdgeKind::Contradicts,
            "derived_from" => EdgeKind::DerivedFrom,
            "contains" => EdgeKind::Contains,
            "next_step" => EdgeKind::NextStep,
            "bridge" => EdgeKind::Bridge,
            "same_as" => EdgeKind::SameAs,
            "hypothesized" => EdgeKind::Hypothesized {
                relation: "relates_to".to_string(),
                weight: 0.5,
                confidence: 0.5,
            },
            _ => return None,
        })
    }

    /// Get the traversal weight for BFS scoring
    pub fn traversal_weight(&self) -> f32 {
        match self {
            EdgeKind::HasTag => 0.8,
            EdgeKind::InCluster => 0.6,
            EdgeKind::RelatesTo { weight } => *weight,
            EdgeKind::Supersedes => 0.9,
            EdgeKind::Contradicts => 0.3,
            EdgeKind::DerivedFrom => 0.7,
            EdgeKind::Contains => 0.9,
            EdgeKind::NextStep => 0.8,
            EdgeKind::Bridge => 0.9,
            EdgeKind::SameAs => 1.0,
            EdgeKind::Hypothesized { weight, .. } => *weight,
        }
    }
}

/// An edge in the memory graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Target node ID
    pub target: String,
    /// Type of relationship
    #[serde(flatten)]
    pub kind: EdgeKind,
}

impl Edge {
    pub fn new(target: impl Into<String>, kind: EdgeKind) -> Self {
        Self {
            target: target.into(),
            kind,
        }
    }
}

/// A tag node in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagEntry {
    /// Unique ID (format: "tag:{name}")
    pub id: String,
    /// Display name
    pub name: String,
    /// Optional description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Number of memories with this tag
    pub count: u32,
    /// When the tag was first created
    pub created_at: DateTime<Utc>,
}

impl TagEntry {
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            id: format!("tag:{}", name),
            name,
            description: None,
            count: 0,
            created_at: Utc::now(),
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// A cluster node (auto-discovered grouping via embeddings)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterEntry {
    /// Unique ID (format: "cluster:{id}")
    pub id: String,
    /// Optional human-readable name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Centroid embedding (average of member embeddings)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub centroid: Vec<f32>,
    /// Number of memories in this cluster
    pub member_count: u32,
    /// When the cluster was discovered
    pub created_at: DateTime<Utc>,
    /// When the cluster was last updated
    pub updated_at: DateTime<Utc>,
}

impl ClusterEntry {
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        let now = Utc::now();
        Self {
            id: format!("cluster:{}", id),
            name: None,
            centroid: Vec::new(),
            member_count: 0,
            created_at: now,
            updated_at: now,
        }
    }
}

/// Graph metadata for tracking statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphMetadata {
    /// When clusters were last updated
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_cluster_update: Option<DateTime<Utc>>,
    /// Total retrieval operations
    #[serde(default)]
    pub retrieval_count: u64,
    /// Total links discovered via co-relevance
    #[serde(default)]
    pub link_discovery_count: u64,
    /// Ontology id the graph is bound to (e.g. `jcode/default/v1`).
    /// Runtime resolution combines this reference with [`Self::ontology`]:
    /// when no snapshot is materialized, the compiled default for this id is
    /// used. Empty string means the store predates ontology binding.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub ontology_id: String,
    /// Ontology version that was bound.  Lets the runtime detect when the
    /// bound ontology has been upgraded and re-validate stored entries.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub ontology_version: u32,
    /// Materialized ontology override for this store.
    ///
    /// `None` means the store uses the compiled default selected by
    /// `ontology_id` / `ontology_version` / `ontology_hash`. A full snapshot is
    /// stored only when the effective ontology diverges from the compiled
    /// default, such as business concepts, custom rules or project overlays.
    /// This keeps default stores small and free of stale copies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ontology: Option<Ontology>,
    /// Deterministic content hash of the effective ontology. When
    /// [`Self::ontology`] is `None` this is the compiled default's hash.
    /// Used for drift detection and optimistic-concurrency checks.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub ontology_hash: String,
    /// Content hash of the parent ontology a project store composes on top of.
    /// Empty for global stores and stores without layered resolution.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub ontology_parent_hash: String,
    /// Deterministic hash of the project/domain layer stack.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub ontology_layer_hash: String,
    /// Human-readable layer ids that produced the materialized ontology.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ontology_layer_ids: Vec<String>,
    /// Bounded history of previously installed ontologies, newest last.
    /// Enables `memory ontology rollback` without external snapshot files.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ontology_history: Vec<OntologySnapshot>,
    /// Data migrations already applied to this store.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub applied_migrations: Vec<String>,
    /// Set while a migration is running. A store loaded with this set was
    /// interrupted and must be validated or rolled back before writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub migration_state: Option<MigrationState>,
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

/// The memory graph - HashMap-based for clean JSON serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGraph {
    /// Format version for migration detection
    pub graph_version: u32,

    /// Memory nodes by ID
    pub memories: HashMap<String, MemoryEntry>,

    /// Tag nodes by ID (format: "tag:{name}")
    pub tags: HashMap<String, TagEntry>,

    /// Cluster nodes by ID (format: "cluster:{id}")
    #[serde(default)]
    pub clusters: HashMap<String, ClusterEntry>,

    /// Forward edges: source_id -> Vec<Edge>
    #[serde(default)]
    pub edges: HashMap<String, Vec<Edge>>,

    /// Reverse edges for efficient BFS: target_id -> Vec<source_id>
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub reverse_edges: HashMap<String, Vec<String>>,

    /// Graph statistics and metadata
    #[serde(default)]
    pub metadata: GraphMetadata,
}

impl Default for MemoryGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryGraph {
    /// Create a new empty memory graph
    pub fn new() -> Self {
        Self {
            graph_version: GRAPH_VERSION,
            memories: HashMap::new(),
            tags: HashMap::new(),
            clusters: HashMap::new(),
            edges: HashMap::new(),
            reverse_edges: HashMap::new(),
            metadata: GraphMetadata::default(),
        }
    }

    /// Get the number of memories in the graph
    pub fn memory_count(&self) -> usize {
        self.memories.len()
    }

    // ==================== Memory Operations ====================

    /// Add a memory entry to the graph
    /// Also creates tag nodes and HasTag edges for any tags on the entry
    pub fn add_memory(&mut self, mut entry: MemoryEntry) -> String {
        // Link nodes own a richer match text than their bounded pointer (for
        // example a skill's full body). Only derive it for ordinary memories.
        if !crate::is_link_node(&entry) {
            entry.refresh_search_text();
        }
        let id = entry.id.clone();

        // Create tag nodes and edges for existing tags
        for tag_name in &entry.tags {
            self.ensure_tag(tag_name);
            let tag_id = format!("tag:{}", tag_name);
            self.add_edge_internal(&id, &tag_id, EdgeKind::HasTag);

            // Increment tag count
            if let Some(tag) = self.tags.get_mut(&tag_id) {
                tag.count += 1;
            }
        }

        // Handle superseded_by as a Supersedes edge (reverse direction)
        if let Some(ref superseded_by) = entry.superseded_by {
            // The newer memory supersedes this one
            self.add_edge_internal(superseded_by, &id, EdgeKind::Supersedes);
        }

        self.memories.insert(id.clone(), entry);
        id
    }

    /// Get a memory by ID
    pub fn get_memory(&self, id: &str) -> Option<&MemoryEntry> {
        self.memories.get(id)
    }

    /// Get a mutable memory by ID
    pub fn get_memory_mut(&mut self, id: &str) -> Option<&mut MemoryEntry> {
        self.memories.get_mut(id)
    }

    /// Remove a memory from the graph (also removes associated edges)
    pub fn remove_memory(&mut self, id: &str) -> Option<MemoryEntry> {
        // Remove all edges from this memory
        if let Some(edges) = self.edges.remove(id) {
            for edge in edges {
                // Update reverse edges
                if let Some(reverse) = self.reverse_edges.get_mut(&edge.target) {
                    reverse.retain(|src| src != id);
                }
                // Decrement tag count if HasTag
                if matches!(edge.kind, EdgeKind::HasTag)
                    && let Some(tag) = self.tags.get_mut(&edge.target)
                {
                    tag.count = tag.count.saturating_sub(1);
                }
            }
        }

        // Remove all edges pointing to this memory
        if let Some(sources) = self.reverse_edges.remove(id) {
            for source in sources {
                if let Some(edges) = self.edges.get_mut(&source) {
                    edges.retain(|e| e.target != id);
                }
            }
        }

        self.memories.remove(id)
    }

    /// Get all memories (for iteration)
    pub fn all_memories(&self) -> impl Iterator<Item = &MemoryEntry> {
        self.memories.values()
    }

    /// Get all active memories that are retrievable now (status + effective window).
    pub fn active_memories(&self) -> impl Iterator<Item = &MemoryEntry> {
        self.memories.values().filter(|m| m.is_retrievable_now())
    }

    /// Mark a memory as archived (soft delete).
    /// Returns `None` if the memory does not exist or the status transition is illegal.
    pub fn archive_memory(&mut self, id: &str, reason: impl Into<String>) -> Option<&MemoryEntry> {
        let reason = reason.into();
        if let Some(m) = self.memories.get_mut(id) {
            if !m.lifecycle.transition_to(crate::MemoryStatus::Archived) {
                return None;
            }
            m.active = false;
            m.lifecycle.deprecated_reason = Some(reason);
            m.updated_at = chrono::Utc::now();
            return Some(m);
        }
        None
    }

    /// Mark a memory as disputed until conflict is resolved.
    /// Returns `None` if the memory does not exist or the status transition is illegal.
    pub fn dispute_memory(&mut self, id: &str) -> Option<&MemoryEntry> {
        if let Some(m) = self.memories.get_mut(id) {
            if !m.lifecycle.transition_to(crate::MemoryStatus::Disputed) {
                return None;
            }
            m.updated_at = chrono::Utc::now();
            return Some(m);
        }
        None
    }

    /// Mark a memory as expired (past validity window).
    /// Returns `None` if the memory does not exist or the status transition is illegal.
    pub fn expire_memory(&mut self, id: &str) -> Option<&MemoryEntry> {
        if let Some(m) = self.memories.get_mut(id) {
            if !m.lifecycle.transition_to(crate::MemoryStatus::Expired) {
                return None;
            }
            m.lifecycle.effective_to = Some(chrono::Utc::now());
            m.updated_at = chrono::Utc::now();
            return Some(m);
        }
        None
    }

    /// Deprecate a memory, keeping it for history but hiding it from retrieval.
    /// Returns `None` if the memory does not exist.
    pub fn deprecate_memory(
        &mut self,
        id: &str,
        reason: impl Into<String>,
    ) -> Option<&MemoryEntry> {
        if let Some(m) = self.memories.get_mut(id) {
            m.lifecycle.deprecated = true;
            m.lifecycle.deprecated_reason = Some(reason.into());
            m.updated_at = chrono::Utc::now();
            return Some(m);
        }
        None
    }

    /// Resolve a memory back to active status.
    /// Returns `None` if the memory does not exist or the status transition is illegal.
    pub fn reactivate_memory(&mut self, id: &str) -> Option<&MemoryEntry> {
        if let Some(m) = self.memories.get_mut(id) {
            if !m.lifecycle.transition_to(crate::MemoryStatus::Active) {
                return None;
            }
            m.active = true;
            m.lifecycle.deprecated = false;
            m.updated_at = chrono::Utc::now();
            return Some(m);
        }
        None
    }

    /// Run the independent graph validator over this graph.
    pub fn validate(&self) -> ValidationReport {
        validate_graph(self)
    }

    /// Find memory IDs matching a query against identity keys or aliases.
    pub fn resolve_identity(&self, query: &str) -> Vec<&str> {
        self.memories
            .values()
            .filter(|m| m.identity.matches(query))
            .map(|m| m.id.as_str())
            .collect()
    }

    // ==================== Tag Operations ====================

    /// Ensure a tag exists, creating it if necessary
    pub fn ensure_tag(&mut self, name: &str) -> &TagEntry {
        let tag_id = format!("tag:{}", name);
        self.tags
            .entry(tag_id.clone())
            .or_insert_with(|| TagEntry::new(name))
    }

    /// Add a tag to a memory
    pub fn tag_memory(&mut self, memory_id: &str, tag_name: &str) {
        // Ensure tag exists
        self.ensure_tag(tag_name);
        let tag_id = format!("tag:{}", tag_name);

        // Check if edge already exists
        if let Some(edges) = self.edges.get(memory_id)
            && edges
                .iter()
                .any(|e| e.target == tag_id && matches!(e.kind, EdgeKind::HasTag))
        {
            return;
        }

        // Add edge
        self.add_edge_internal(memory_id, &tag_id, EdgeKind::HasTag);

        // Update tag count
        if let Some(tag) = self.tags.get_mut(&tag_id) {
            tag.count += 1;
        }

        // Update memory's tags list
        if let Some(memory) = self.memories.get_mut(memory_id)
            && !memory.tags.contains(&tag_name.to_string())
        {
            memory.tags.push(tag_name.to_string());
            memory.refresh_search_text();
        }
    }

    /// Remove a tag from a memory
    pub fn untag_memory(&mut self, memory_id: &str, tag_name: &str) {
        let tag_id = format!("tag:{}", tag_name);

        // Remove edge
        if let Some(edges) = self.edges.get_mut(memory_id) {
            edges.retain(|e| !(e.target == tag_id && matches!(e.kind, EdgeKind::HasTag)));
        }

        // Update reverse edges
        if let Some(sources) = self.reverse_edges.get_mut(&tag_id) {
            sources.retain(|s| s != memory_id);
        }

        // Update tag count
        if let Some(tag) = self.tags.get_mut(&tag_id) {
            tag.count = tag.count.saturating_sub(1);
        }

        // Update memory's tags list
        if let Some(memory) = self.memories.get_mut(memory_id) {
            memory.tags.retain(|t| t != tag_name);
            if !crate::is_link_node(memory) {
                memory.refresh_search_text();
            }
        }
    }

    /// Get all memories with a specific tag
    pub fn get_memories_by_tag(&self, tag_name: &str) -> Vec<&MemoryEntry> {
        let tag_id = format!("tag:{}", tag_name);

        // Find all sources pointing to this tag via HasTag
        self.reverse_edges
            .get(&tag_id)
            .map(|sources| {
                sources
                    .iter()
                    .filter_map(|id| self.memories.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all tags
    pub fn all_tags(&self) -> impl Iterator<Item = &TagEntry> {
        self.tags.values()
    }

    // ==================== Edge Operations ====================

    /// Add an edge between two nodes (internal, no validation)
    fn add_edge_internal(&mut self, from: &str, to: &str, kind: EdgeKind) {
        // Add forward edge
        self.edges
            .entry(from.to_string())
            .or_default()
            .push(Edge::new(to, kind));

        // Add reverse edge
        self.reverse_edges
            .entry(to.to_string())
            .or_default()
            .push(from.to_string());
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, from: &str, to: &str, kind: EdgeKind) {
        // Check if edge already exists
        if let Some(edges) = self.edges.get(from)
            && edges.iter().any(|e| e.target == to && e.kind == kind)
        {
            return;
        }

        self.add_edge_internal(from, to, kind);
    }

    /// Remove an edge between two nodes
    pub fn remove_edge(&mut self, from: &str, to: &str, kind: &EdgeKind) {
        if let Some(edges) = self.edges.get_mut(from) {
            edges.retain(|e| !(e.target == to && &e.kind == kind));
        }
        if let Some(sources) = self.reverse_edges.get_mut(to) {
            sources.retain(|s| s != from);
        }
    }

    /// Get all edges from a node
    pub fn get_edges(&self, node_id: &str) -> &[Edge] {
        self.edges.get(node_id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get all nodes pointing to this node
    pub fn get_incoming(&self, node_id: &str) -> Vec<&str> {
        self.reverse_edges
            .get(node_id)
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Outgoing plus incoming edge count for one node.
    pub fn node_degree(&self, node_id: &str) -> usize {
        self.get_edges(node_id).len() + self.get_incoming(node_id).len()
    }

    /// Link two memories with a RelatesTo edge
    pub fn link_memories(&mut self, from: &str, to: &str, weight: f32) {
        self.add_edge(from, to, EdgeKind::RelatesTo { weight });
        self.metadata.link_discovery_count += 1;
    }

    /// Mark a memory as superseding another
    pub fn supersede(&mut self, newer_id: &str, older_id: &str) {
        self.add_edge(newer_id, older_id, EdgeKind::Supersedes);
        // Mark older as inactive
        if let Some(older) = self.memories.get_mut(older_id) {
            older.active = false;
            older.superseded_by = Some(newer_id.to_string());
        }
    }

    /// Mark two memories as contradicting
    pub fn mark_contradiction(&mut self, id_a: &str, id_b: &str) {
        self.add_edge(id_a, id_b, EdgeKind::Contradicts);
        self.add_edge(id_b, id_a, EdgeKind::Contradicts);
    }

    // ==================== Graph Stats ====================

    /// Get total number of nodes (memories + tags + clusters)
    pub fn node_count(&self) -> usize {
        self.memories.len() + self.tags.len() + self.clusters.len()
    }

    /// Get total number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|v| v.len()).sum()
    }

    // ==================== Cascade Retrieval ====================

    /// Perform BFS cascade retrieval starting from seed memories
    ///
    /// Starting from embedding search hits (seeds), traverse through the graph
    /// via tags and other edges to find related memories.
    ///
    /// Returns (memory_id, score) pairs sorted by score descending.
    pub fn cascade_retrieve(
        &mut self,
        seed_ids: &[String],
        seed_scores: &[f32],
        max_depth: usize,
        max_results: usize,
    ) -> Vec<(String, f32)> {
        self.metadata.retrieval_count += 1;

        let mut visited: HashSet<String> = HashSet::new();
        let mut results: HashMap<String, f32> = HashMap::new();
        let mut queue: VecDeque<(String, f32, usize)> = VecDeque::new();

        // Initialize with seeds
        for (id, score) in seed_ids.iter().zip(seed_scores.iter()) {
            if self.memories.contains_key(id) {
                queue.push_back((id.clone(), *score, 0));
                results.insert(id.clone(), *score);
            }
        }

        // BFS traversal
        while let Some((node_id, score, depth)) = queue.pop_front() {
            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id.clone());

            if depth >= max_depth {
                continue;
            }

            // Traverse edges from this node
            for edge in self.get_edges(&node_id).to_vec() {
                let target = &edge.target;

                // Skip if already visited
                if visited.contains(target) {
                    continue;
                }

                // Calculate decayed score
                let edge_weight = edge.kind.traversal_weight();
                let decay = 0.7_f32.powi(depth as i32 + 1);
                let new_score = score * edge_weight * decay;

                // If target is a tag, find all memories with this tag
                if target.starts_with("tag:") {
                    for source_id in self.get_incoming(target).iter() {
                        let source_id = source_id.to_string();
                        if !visited.contains(&source_id) && self.memories.contains_key(&source_id) {
                            let existing = results.get(&source_id).copied().unwrap_or(0.0);
                            if new_score > existing {
                                results.insert(source_id.clone(), new_score);
                                queue.push_back((source_id, new_score, depth + 1));
                            }
                        }
                    }
                }
                // If target is a memory, add it only if retrievable
                else if self.memories.contains_key(target) {
                    if let Some(mem) = self.memories.get(target) {
                        if !mem.is_retrievable_now() {
                            continue;
                        }
                    }
                    let existing = results.get(target).copied().unwrap_or(0.0);
                    if new_score > existing {
                        results.insert(target.clone(), new_score);
                        queue.push_back((target.clone(), new_score, depth + 1));
                    }
                }
            }
        }

        // Keep only the top-scoring results
        top_k_scored(results, max_results)
    }

    // ==================== Weighted Path Search ====================

    /// Cheapest loop-free path between two graph nodes.
    ///
    /// The cost model mirrors the retrieval-layer design document:
    /// structural edge cost + hub penalty + low-confidence penalty. `HasTag`
    /// and `InCluster` edges are traversable in both directions so a tag can
    /// act as a bounded hub; typed directional edges (`DerivedFrom`,
    /// `Contains`, `NextStep`, `Supersedes`) only traverse forward.
    pub fn weighted_shortest_path(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
    ) -> Option<MemoryPath> {
        self.weighted_shortest_path_with_relations(from, to, max_depth, &[])
    }

    /// Cheapest loop-free path restricted to `allowed_relations`.
    ///
    /// An empty allow-list means "any relation", matching
    /// [`Self::weighted_shortest_path`]. Names use canonical `EdgeKind` wire
    /// names such as `relates_to`, `derived_from`, `contains`, `next_step`.
    pub fn weighted_shortest_path_with_relations(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
        allowed_relations: &[&str],
    ) -> Option<MemoryPath> {
        self.memory_dijkstra(
            from,
            to,
            max_depth,
            allowed_relations,
            &BTreeSet::new(),
            &BTreeSet::new(),
        )
    }

    /// Yen's k-best loop-free paths under [`Self::weighted_shortest_path`]'s
    /// cost model. Results are sorted by cost, then by node sequence for
    /// deterministic ties.
    pub fn k_shortest_paths(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
        k: usize,
    ) -> Vec<MemoryPath> {
        self.k_shortest_paths_with_relations(from, to, max_depth, k, &[])
    }

    /// Yen's k-best paths restricted to `allowed_relations`.
    pub fn k_shortest_paths_with_relations(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
        k: usize,
        allowed_relations: &[&str],
    ) -> Vec<MemoryPath> {
        if k == 0 {
            return Vec::new();
        }
        let Some(first) =
            self.weighted_shortest_path_with_relations(from, to, max_depth, allowed_relations)
        else {
            return Vec::new();
        };
        let mut accepted = vec![first];
        let mut candidates: Vec<MemoryPath> = Vec::new();

        while accepted.len() < k {
            let previous = accepted.last().expect("accepted path").clone();
            for spur_index in 0..previous.nodes.len().saturating_sub(1) {
                let spur_node = previous.nodes[spur_index].clone();
                let root_nodes = previous.nodes[..=spur_index].to_vec();
                let root_edges = previous.edges[..spur_index].to_vec();

                let mut removed_edges = BTreeSet::new();
                for path in &accepted {
                    if path.nodes.len() > spur_index
                        && path.nodes[..=spur_index] == root_nodes
                        && let Some(edge) = path.edges.get(spur_index)
                    {
                        removed_edges.insert(memory_edge_key(edge));
                    }
                }
                let blocked_nodes: BTreeSet<String> =
                    root_nodes[..spur_index].iter().cloned().collect();

                let Some(spur) = self.memory_dijkstra(
                    &spur_node,
                    to,
                    max_depth.saturating_sub(spur_index),
                    allowed_relations,
                    &removed_edges,
                    &blocked_nodes,
                ) else {
                    continue;
                };

                let mut nodes = root_nodes;
                nodes.extend(spur.nodes.into_iter().skip(1));
                let mut edges = root_edges;
                edges.extend(spur.edges);
                if accepted.iter().any(|path| path.nodes == nodes)
                    || candidates.iter().any(|path| path.nodes == nodes)
                {
                    continue;
                }
                candidates.push(self.memory_path_from(nodes, edges));
            }

            if candidates.is_empty() {
                break;
            }
            candidates.sort_by(|left, right| {
                left.cost
                    .total_cmp(&right.cost)
                    .then_with(|| left.nodes.cmp(&right.nodes))
            });
            accepted.push(candidates.remove(0));
        }
        accepted
    }

    fn memory_dijkstra(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
        allowed_relations: &[&str],
        removed_edges: &BTreeSet<(String, String, String)>,
        blocked_nodes: &BTreeSet<String>,
    ) -> Option<MemoryPath> {
        if from == to {
            return Some(self.memory_path_from(vec![from.to_string()], Vec::new()));
        }
        if !self.memories.contains_key(from) && !self.tags.contains_key(from) {
            return None;
        }
        let mut queue: BinaryHeap<Reverse<(u64, String, usize, Vec<String>)>> = BinaryHeap::new();
        queue.push(Reverse((0, from.to_string(), 0, vec![from.to_string()])));
        let mut best: HashMap<(String, usize), u64> = HashMap::new();
        best.insert((from.to_string(), 0), 0);

        while let Some(Reverse((cost, node, depth, nodes))) = queue.pop() {
            if node == to {
                return self.memory_path_from_nodes(nodes, allowed_relations);
            }
            if depth >= max_depth {
                continue;
            }
            for (next, edge) in self.memory_neighbors(&node, allowed_relations) {
                if removed_edges.contains(&memory_edge_key(&edge))
                    || blocked_nodes.contains(&next)
                    || nodes.contains(&next)
                {
                    continue;
                }
                let next_cost = cost.saturating_add(scaled_memory_cost(edge.cost));
                let key = (next.clone(), depth + 1);
                if best
                    .get(&key)
                    .is_some_and(|existing| *existing <= next_cost)
                {
                    continue;
                }
                best.insert(key, next_cost);
                let mut next_nodes = nodes.clone();
                next_nodes.push(next.clone());
                queue.push(Reverse((next_cost, next, depth + 1, next_nodes)));
            }
        }
        None
    }

    fn memory_neighbors(
        &self,
        node_id: &str,
        allowed_relations: &[&str],
    ) -> Vec<(String, MemoryPathEdge)> {
        let mut out = Vec::new();
        for edge in self.get_edges(node_id) {
            if !relation_allowed(&edge.kind, allowed_relations) {
                continue;
            }
            out.push((
                edge.target.clone(),
                MemoryPathEdge {
                    from: node_id.to_string(),
                    to: edge.target.clone(),
                    kind: edge.kind.clone(),
                    cost: self.memory_edge_cost(edge),
                },
            ));
        }
        for source in self.get_incoming(node_id) {
            let Some(edge) = self
                .get_edges(source)
                .iter()
                .find(|edge| edge.target == node_id)
            else {
                continue;
            };
            if matches!(
                edge.kind,
                EdgeKind::HasTag
                    | EdgeKind::InCluster
                    | EdgeKind::RelatesTo { .. }
                    | EdgeKind::Contradicts
                    | EdgeKind::Bridge
                    | EdgeKind::SameAs
                    | EdgeKind::Hypothesized { .. }
            ) && relation_allowed(&edge.kind, allowed_relations)
            {
                out.push((
                    source.to_string(),
                    MemoryPathEdge {
                        from: node_id.to_string(),
                        to: source.to_string(),
                        kind: edge.kind.clone(),
                        cost: self.memory_edge_cost(edge),
                    },
                ));
            }
        }
        out
    }

    fn memory_edge_cost(&self, edge: &Edge) -> f32 {
        let base = match &edge.kind {
            EdgeKind::HasTag => 0.45,
            EdgeKind::InCluster => 0.8,
            EdgeKind::RelatesTo { weight } => 1.0 - 0.8 * weight.clamp(0.0, 1.0),
            EdgeKind::Supersedes => 0.15,
            EdgeKind::Contradicts => 2.0,
            EdgeKind::DerivedFrom => 0.35,
            EdgeKind::Contains => 0.2,
            EdgeKind::NextStep => 0.2,
            EdgeKind::Bridge => 0.25,
            EdgeKind::SameAs => 0.05,
            EdgeKind::Hypothesized { confidence, .. } => {
                2.0 + 0.5 * (1.0 - confidence.clamp(0.0, 1.0))
            }
        };
        let degree = self
            .tags
            .get(&edge.target)
            .map(|tag| tag.count as usize)
            .or_else(|| {
                self.memories.contains_key(&edge.target).then(|| {
                    self.get_edges(&edge.target).len() + self.get_incoming(&edge.target).len()
                })
            })
            .unwrap_or(0);
        let hub_penalty = 0.10 * ((1 + degree) as f32).ln();
        let confidence = self
            .memories
            .get(&edge.target)
            .map(|entry| entry.confidence.clamp(0.0, 1.0))
            .unwrap_or(1.0);
        (base + hub_penalty + 0.5 * (1.0 - confidence)).max(0.01)
    }

    fn memory_path_from_nodes(
        &self,
        nodes: Vec<String>,
        allowed_relations: &[&str],
    ) -> Option<MemoryPath> {
        let mut edges = Vec::with_capacity(nodes.len().saturating_sub(1));
        for pair in nodes.windows(2) {
            let from = &pair[0];
            let to = &pair[1];
            let edge = self
                .memory_neighbors(from, allowed_relations)
                .into_iter()
                .filter(|(next, _)| next == to)
                .min_by(|left, right| left.1.cost.total_cmp(&right.1.cost))
                .map(|(_, edge)| edge)?;
            edges.push(edge);
        }
        Some(self.memory_path_from(nodes, edges))
    }

    fn memory_path_from(&self, nodes: Vec<String>, edges: Vec<MemoryPathEdge>) -> MemoryPath {
        let mut cost = 0.0;
        let mut confidence = 1.0f32;
        for edge in &edges {
            cost += edge.cost;
        }
        for node in &nodes {
            if let Some(entry) = self.memories.get(node) {
                confidence = confidence.min(entry.confidence.clamp(0.0, 1.0));
            }
        }
        MemoryPath {
            nodes,
            edges,
            cost,
            confidence,
        }
    }

    // ==================== Migration ====================

    /// Convert a legacy MemoryStore to a MemoryGraph
    ///
    /// This handles migration from the old flat JSON format to the graph format.
    pub fn from_legacy_store(store: MemoryStore) -> Self {
        let mut graph = MemoryGraph::new();

        for entry in store.entries {
            let memory_id = entry.id.clone();
            let tags = entry.tags.clone();
            let superseded_by = entry.superseded_by.clone();

            // Add memory (this will also create tag nodes and HasTag edges)
            graph.memories.insert(memory_id.clone(), entry);

            // Create tag nodes and edges
            for tag_name in &tags {
                graph.ensure_tag(tag_name);
                let tag_id = format!("tag:{}", tag_name);
                graph.add_edge_internal(&memory_id, &tag_id, EdgeKind::HasTag);

                // Update tag count
                if let Some(tag) = graph.tags.get_mut(&tag_id) {
                    tag.count += 1;
                }
            }

            // Create Supersedes edge if applicable
            if let Some(ref newer_id) = superseded_by {
                // newer_id supersedes memory_id
                graph.add_edge_internal(newer_id, &memory_id, EdgeKind::Supersedes);
            }
        }

        graph
    }

    /// Check if this graph was migrated from legacy format
    pub fn is_migrated(&self) -> bool {
        self.graph_version == GRAPH_VERSION
    }
}

#[cfg(test)]
mod graph_tests;
