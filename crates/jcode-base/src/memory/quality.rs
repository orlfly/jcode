//! Deterministic graph-quality metrics for the memory stores.
//!
//! These are the query-independent half of design section 4.6: structural
//! richness, connectivity, evidence coverage, alignment, hypotheses, and stale
//! nodes. Query-derived terms such as no-path rate remain in the query outcome
//! and are not invented here.

use std::collections::{HashMap, HashSet, VecDeque};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use super::{HIGH_DEGREE_HUB, STALE_EDGE_DAYS};
use crate::memory_graph::{EdgeKind, MemoryGraph};

/// Per-store quality counts. Ratios are computed by the caller so multiple
/// stores can be aggregated without losing the raw denominators.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct StoreQuality {
    pub store: String,
    pub memories: usize,
    pub memory_edges: usize,
    pub alignment_edges: usize,
    pub hypothesized_edges: usize,
    pub orphan_nodes: usize,
    pub hub_nodes: usize,
    pub provenance_nodes: usize,
    pub stale_nodes: usize,
    pub largest_component: usize,
}

/// Aggregated quality report for one or more memory stores.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GraphQualityReport {
    pub scope: String,
    pub memories: usize,
    pub memory_edges: usize,
    pub average_degree: f32,
    /// Largest-component ratio in the best-connected store.
    pub largest_component_ratio: f32,
    pub orphan_ratio: f32,
    pub hub_ratio: f32,
    pub provenance_coverage: f32,
    pub alignment_rate: f32,
    pub hypothesized_ratio: f32,
    pub stale_ratio: f32,
    pub pending_shadow_edges: usize,
    pub stores: Vec<StoreQuality>,
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

fn largest_component_size(
    nodes: &HashSet<&str>,
    adjacency: &HashMap<&str, Vec<&str>>,
) -> usize {
    let mut seen = HashSet::new();
    let mut largest = 0;
    for start in nodes {
        if !seen.insert(*start) {
            continue;
        }
        let mut queue = VecDeque::from([*start]);
        let mut size = 0;
        while let Some(node) = queue.pop_front() {
            size += 1;
            if let Some(neighbors) = adjacency.get(node) {
                for neighbor in neighbors {
                    if seen.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        largest = largest.max(size);
    }
    largest
}

/// Analyze one persisted memory graph.
pub fn analyze_store(store: &str, graph: &MemoryGraph) -> StoreQuality {
    let active: Vec<&crate::memory_types::MemoryEntry> = graph.active_memories().collect();
    let nodes: HashSet<&str> = active.iter().map(|entry| entry.id.as_str()).collect();
    let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut memory_edges = 0;
    let mut alignment_edges = 0;
    let mut hypothesized_edges = 0;

    for (from, edges) in &graph.edges {
        if !nodes.contains(from.as_str()) {
            continue;
        }
        for edge in edges {
            if !nodes.contains(edge.target.as_str()) {
                continue;
            }
            memory_edges += 1;
            match edge.kind {
                EdgeKind::Bridge | EdgeKind::SameAs => alignment_edges += 1,
                EdgeKind::Hypothesized { .. } => hypothesized_edges += 1,
                _ => {}
            }
            adjacency
                .entry(from.as_str())
                .or_default()
                .push(edge.target.as_str());
            adjacency
                .entry(edge.target.as_str())
                .or_default()
                .push(from.as_str());
        }
    }

    let now = Utc::now();
    let orphan_nodes = active
        .iter()
        .filter(|entry| adjacency.get(entry.id.as_str()).is_none_or(Vec::is_empty))
        .count();
    let hub_nodes = active
        .iter()
        .filter(|entry| graph.node_degree(&entry.id) >= HIGH_DEGREE_HUB)
        .count();
    let provenance_nodes = active
        .iter()
        .filter(|entry| {
            entry
                .provenance
                .as_ref()
                .and_then(|provenance| provenance.locator.as_ref())
                .is_some()
        })
        .count();
    let stale_nodes = active
        .iter()
        .filter(|entry| now.signed_duration_since(entry.updated_at).num_days() > STALE_EDGE_DAYS)
        .count();

    StoreQuality {
        store: store.to_string(),
        memories: active.len(),
        memory_edges,
        alignment_edges,
        hypothesized_edges,
        orphan_nodes,
        hub_nodes,
        provenance_nodes,
        stale_nodes,
        largest_component: largest_component_size(&nodes, &adjacency),
    }
}

/// Aggregate per-store counts into the report returned by the memory tool.
pub fn aggregate(
    scope: &str,
    stores: Vec<StoreQuality>,
    pending_shadow_edges: usize,
) -> GraphQualityReport {
    let memories = stores.iter().map(|store| store.memories).sum();
    let memory_edges = stores.iter().map(|store| store.memory_edges).sum();
    let alignment_edges = stores.iter().map(|store| store.alignment_edges).sum();
    let hypothesized_edges = stores.iter().map(|store| store.hypothesized_edges).sum();
    let orphan_nodes = stores.iter().map(|store| store.orphan_nodes).sum();
    let hub_nodes = stores.iter().map(|store| store.hub_nodes).sum();
    let provenance_nodes = stores.iter().map(|store| store.provenance_nodes).sum();
    let stale_nodes = stores.iter().map(|store| store.stale_nodes).sum();
    let largest_component = stores
        .iter()
        .map(|store| ratio(store.largest_component, store.memories))
        .fold(0.0f32, f32::max);

    GraphQualityReport {
        scope: scope.to_string(),
        memories,
        memory_edges,
        average_degree: if memories == 0 {
            0.0
        } else {
            2.0 * memory_edges as f32 / memories as f32
        },
        largest_component_ratio: largest_component,
        orphan_ratio: ratio(orphan_nodes, memories),
        hub_ratio: ratio(hub_nodes, memories),
        provenance_coverage: ratio(provenance_nodes, memories),
        alignment_rate: ratio(alignment_edges, memories),
        hypothesized_ratio: ratio(hypothesized_edges, memory_edges),
        stale_ratio: ratio(stale_nodes, memories),
        pending_shadow_edges,
        stores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_types::{MemoryCategory, MemoryEntry};

    #[test]
    fn detects_components_orphans_and_alignment() {
        let mut graph = MemoryGraph::new();
        let a = graph.add_memory(MemoryEntry::new(MemoryCategory::Fact, "alpha node"));
        let b = graph.add_memory(MemoryEntry::new(MemoryCategory::Fact, "beta node"));
        let c = graph.add_memory(MemoryEntry::new(MemoryCategory::Fact, "gamma node"));
        graph.add_edge(&a, &b, EdgeKind::SameAs);
        let quality = analyze_store("project", &graph);
        assert_eq!(quality.memories, 3);
        assert_eq!(quality.memory_edges, 1);
        assert_eq!(quality.alignment_edges, 1);
        assert_eq!(quality.orphan_nodes, 1);
        assert_eq!(quality.largest_component, 2);
        let report = aggregate("project", vec![quality], 0);
        assert!((report.alignment_rate - 1.0 / 3.0).abs() < 0.001);
        assert!((report.orphan_ratio - 1.0 / 3.0).abs() < 0.001);
        assert!(c != a);
    }

    #[test]
    fn empty_graph_has_zero_ratios() {
        let report = aggregate(
            "project",
            vec![analyze_store("project", &MemoryGraph::new())],
            0,
        );
        assert_eq!(report.memories, 0);
        assert_eq!(report.average_degree, 0.0);
        assert_eq!(report.largest_component_ratio, 0.0);
        assert_eq!(report.provenance_coverage, 0.0);
    }
}
