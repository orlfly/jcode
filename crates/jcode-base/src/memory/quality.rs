//! Deterministic graph-quality metrics for the memory stores.
//!
//! These cover design section 4.6's structural richness, connectivity, evidence
//! coverage, alignment, hypotheses, stale nodes, and audited query outcomes.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{HIGH_DEGREE_HUB, STALE_EDGE_DAYS};
use crate::memory_graph::{EdgeKind, MemoryGraph};

/// Maximum tolerated orphan ratio before a governance finding is emitted.
pub const MAX_ORPHAN_RATIO: f32 = 0.30;
/// Minimum tolerated provenance coverage.
pub const MIN_PROVENANCE_COVERAGE: f32 = 0.50;
/// Minimum tolerated largest-component ratio.
pub const MIN_LARGEST_COMPONENT_RATIO: f32 = 0.80;
/// Maximum tolerated pending-hypothesis edge ratio.
pub const MAX_HYPOTHESIZED_RATIO: f32 = 0.10;
/// Maximum tolerated stale-node ratio.
pub const MAX_STALE_RATIO: f32 = 0.25;
/// Maximum tolerated hub ratio.
pub const MAX_HUB_RATIO: f32 = 0.05;
/// Minimum tolerated path success rate (PSR).
pub const MIN_PATH_SUCCESS_RATE: f32 = 0.85;
/// Minimum tolerated cross-graph query success rate (XSR).
pub const MIN_CROSS_GRAPH_SUCCESS_RATE: f32 = 0.90;
/// Maximum tolerated completion debt from hypothesized edges.
pub const MAX_COMPLETION_DEBT: f32 = 0.10;
/// Minimum tolerated evidence coverage on returned paths.
pub const MIN_EVIDENCE_COVERAGE: f32 = 0.95;
/// Maximum tolerated share of paths crossing high-degree hubs.
pub const MAX_HUB_PATH_RATIO: f32 = 0.20;
/// Minimum tolerated PathQuery compile success rate (QCS).
pub const MIN_COMPILE_SUCCESS_RATE: f32 = 0.98;
/// Maximum tolerated share of path queries that needed relaxation.
pub const MAX_RELAXATION_RATE: f32 = 0.10;

/// Severity of one governance finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualitySeverity {
    Warning,
    Critical,
}

/// One actionable quality-threshold breach.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QualityFinding {
    pub code: String,
    pub severity: QualitySeverity,
    pub metric: String,
    pub value: f32,
    pub threshold: f32,
    pub action: String,
}

/// Lifecycle of one persisted quality-debt item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityDebtStatus {
    Open,
    Resolved,
}

/// One persisted governance finding with ownership and SLA metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QualityDebtItem {
    pub id: String,
    pub code: String,
    pub scope: String,
    pub severity: QualitySeverity,
    pub metric: String,
    pub value: f32,
    /// Value when the item was first opened, kept for regression comparison.
    pub baseline_value: f32,
    pub threshold: f32,
    pub action: String,
    pub status: QualityDebtStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    pub sla_days: u32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_at: Option<DateTime<Utc>>,
}

/// One path-query audit record used to derive online quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QueryAuditEntry {
    pub scope: String,
    pub kind: String,
    pub success: bool,
    #[serde(default)]
    pub path_count: usize,
    #[serde(default)]
    pub edge_count: usize,
    #[serde(default)]
    pub hypothesized_edges: usize,
    #[serde(default)]
    pub stores_searched: usize,
    #[serde(default)]
    pub depth: usize,
    #[serde(default)]
    pub diagnosis: Vec<String>,
    #[serde(default)]
    pub relaxations: Vec<String>,
    #[serde(default)]
    pub low_provenance_paths: usize,
    #[serde(default)]
    pub hub_paths: usize,
}

/// Query-derived quality metrics for one governance scope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryAuditMetrics {
    pub total_queries: u64,
    pub successful_queries: u64,
    pub no_path_queries: u64,
    pub path_success_rate: f32,
    pub total_edges: u64,
    #[serde(default)]
    pub total_paths: u64,
    pub hypothesized_edges: u64,
    pub completion_debt: f32,
    #[serde(default)]
    pub low_provenance_paths: u64,
    #[serde(default)]
    pub hub_paths: u64,
    #[serde(default = "default_rate")]
    pub evidence_coverage: f32,
    #[serde(default)]
    pub hub_path_ratio: f32,
    #[serde(default)]
    pub compile_attempts: u64,
    #[serde(default)]
    pub compile_successes: u64,
    #[serde(default = "default_rate")]
    pub compile_success_rate: f32,
    #[serde(default)]
    pub relaxed_queries: u64,
    #[serde(default)]
    pub relaxation_rate: f32,
    pub cross_graph_queries: u64,
    pub cross_graph_success: u64,
    pub cross_graph_success_rate: f32,
    #[serde(default)]
    pub diagnosis_counts: BTreeMap<String, u64>,
}

impl Default for QueryAuditMetrics {
    fn default() -> Self {
        Self {
            total_queries: 0,
            successful_queries: 0,
            no_path_queries: 0,
            path_success_rate: 1.0,
            total_edges: 0,
            total_paths: 0,
            hypothesized_edges: 0,
            completion_debt: 0.0,
            low_provenance_paths: 0,
            hub_paths: 0,
            evidence_coverage: 1.0,
            hub_path_ratio: 0.0,
            compile_attempts: 0,
            compile_successes: 0,
            compile_success_rate: 1.0,
            relaxed_queries: 0,
            relaxation_rate: 0.0,
            cross_graph_queries: 0,
            cross_graph_success: 0,
            cross_graph_success_rate: 1.0,
            diagnosis_counts: BTreeMap::new(),
        }
    }
}

/// Per-store quality counts. Ratios are computed by the caller so multiple
/// stores can be aggregated without losing the raw denominators.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct StoreQuality {
    pub store: String,
    pub memories: usize,
    pub memory_edges: usize,
    pub alignment_edges: usize,
    pub aligned_nodes: usize,
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
    #[serde(default)]
    pub alignment_coverage: f32,
    pub hypothesized_ratio: f32,
    pub stale_ratio: f32,
    pub pending_shadow_edges: usize,
    /// Threshold breaches with deterministic repair suggestions.
    #[serde(default)]
    pub findings: Vec<QualityFinding>,
    /// Persisted debt items for the requested scope.
    #[serde(default)]
    pub debt: Vec<QualityDebtItem>,
    /// Online path-query success, diagnosis and completion-debt metrics.
    #[serde(default)]
    pub query_metrics: QueryAuditMetrics,
    /// True when no threshold is breached.
    #[serde(default)]
    pub healthy: bool,
    pub stores: Vec<StoreQuality>,
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

fn default_rate() -> f32 {
    1.0
}

fn largest_component_size(nodes: &HashSet<&str>, adjacency: &HashMap<&str, Vec<&str>>) -> usize {
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
    let mut aligned_nodes = HashSet::new();
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
                EdgeKind::Bridge | EdgeKind::SameAs => {
                    alignment_edges += 1;
                    aligned_nodes.insert(from.as_str());
                    aligned_nodes.insert(edge.target.as_str());
                }
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
        aligned_nodes: aligned_nodes.len(),
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
    let aligned_nodes = stores.iter().map(|store| store.aligned_nodes).sum();
    let hypothesized_edges = stores.iter().map(|store| store.hypothesized_edges).sum();
    let orphan_nodes = stores.iter().map(|store| store.orphan_nodes).sum();
    let hub_nodes = stores.iter().map(|store| store.hub_nodes).sum();
    let provenance_nodes = stores.iter().map(|store| store.provenance_nodes).sum();
    let stale_nodes = stores.iter().map(|store| store.stale_nodes).sum();
    let largest_component = stores
        .iter()
        .map(|store| ratio(store.largest_component, store.memories))
        .fold(0.0f32, f32::max);

    let mut report = GraphQualityReport {
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
        alignment_coverage: ratio(aligned_nodes, memories),
        hypothesized_ratio: ratio(hypothesized_edges, memory_edges),
        stale_ratio: ratio(stale_nodes, memories),
        pending_shadow_edges,
        findings: Vec::new(),
        debt: Vec::new(),
        query_metrics: QueryAuditMetrics::default(),
        healthy: true,
        stores,
    };
    report.findings = diagnose(&report);
    report.healthy = report.findings.is_empty();
    report
}

fn finding(
    code: &str,
    severity: QualitySeverity,
    metric: &str,
    value: f32,
    threshold: f32,
    action: &str,
) -> QualityFinding {
    QualityFinding {
        code: code.to_string(),
        severity,
        metric: metric.to_string(),
        value,
        threshold,
        action: action.to_string(),
    }
}

/// Aggregate audited path queries for one governance scope.
pub fn summarize_queries(entries: &[QueryAuditEntry], scope: &str) -> QueryAuditMetrics {
    let mut metrics = QueryAuditMetrics::default();
    for entry in entries
        .iter()
        .filter(|entry| scope == "all" || entry.scope == scope)
    {
        if entry.kind == "compile" {
            metrics.compile_attempts += 1;
            if entry.success {
                metrics.compile_successes += 1;
            }
            for code in &entry.diagnosis {
                *metrics.diagnosis_counts.entry(code.clone()).or_default() += 1;
            }
            continue;
        }
        metrics.total_queries += 1;
        if entry.success {
            metrics.successful_queries += 1;
        }
        metrics.total_edges += entry.edge_count as u64;
        metrics.total_paths += entry.path_count as u64;
        metrics.hypothesized_edges += entry.hypothesized_edges as u64;
        metrics.low_provenance_paths += entry.low_provenance_paths as u64;
        metrics.hub_paths += entry.hub_paths as u64;
        if !entry.relaxations.is_empty() {
            metrics.relaxed_queries += 1;
        }
        if entry.stores_searched > 1 {
            metrics.cross_graph_queries += 1;
            if entry.success {
                metrics.cross_graph_success += 1;
            }
        }
        for code in &entry.diagnosis {
            *metrics.diagnosis_counts.entry(code.clone()).or_default() += 1;
        }
    }
    metrics.no_path_queries = metrics
        .total_queries
        .saturating_sub(metrics.successful_queries);
    metrics.path_success_rate = if metrics.total_queries == 0 {
        1.0
    } else {
        metrics.successful_queries as f32 / metrics.total_queries as f32
    };
    metrics.completion_debt = if metrics.total_edges == 0 {
        0.0
    } else {
        metrics.hypothesized_edges as f32 / metrics.total_edges as f32
    };
    metrics.evidence_coverage = if metrics.total_paths == 0 {
        1.0
    } else {
        1.0 - metrics.low_provenance_paths as f32 / metrics.total_paths as f32
    };
    metrics.hub_path_ratio = if metrics.total_paths == 0 {
        0.0
    } else {
        metrics.hub_paths as f32 / metrics.total_paths as f32
    };
    metrics.compile_success_rate = if metrics.compile_attempts == 0 {
        1.0
    } else {
        metrics.compile_successes as f32 / metrics.compile_attempts as f32
    };
    metrics.relaxation_rate = if metrics.total_queries == 0 {
        0.0
    } else {
        metrics.relaxed_queries as f32 / metrics.total_queries as f32
    };
    metrics.cross_graph_success_rate = if metrics.cross_graph_queries == 0 {
        1.0
    } else {
        metrics.cross_graph_success as f32 / metrics.cross_graph_queries as f32
    };
    metrics
}

/// Derive query-layer findings from audited outcomes.
pub fn query_diagnose(metrics: &QueryAuditMetrics) -> Vec<QualityFinding> {
    let mut findings = Vec::new();
    if metrics.total_queries > 0 && metrics.path_success_rate < MIN_PATH_SUCCESS_RATE {
        findings.push(finding(
            "Q08",
            QualitySeverity::Warning,
            "path_success_rate",
            metrics.path_success_rate,
            MIN_PATH_SUCCESS_RATE,
            "route no-path queries through diagnosis-specific repair",
        ));
    }
    if metrics.cross_graph_queries > 0
        && metrics.cross_graph_success_rate < MIN_CROSS_GRAPH_SUCCESS_RATE
    {
        findings.push(finding(
            "Q09",
            QualitySeverity::Warning,
            "cross_graph_success_rate",
            metrics.cross_graph_success_rate,
            MIN_CROSS_GRAPH_SUCCESS_RATE,
            "repair cross-store alignment or bridge coverage",
        ));
    }
    if metrics.completion_debt > MAX_COMPLETION_DEBT {
        findings.push(finding(
            "Q10",
            QualitySeverity::Critical,
            "completion_debt",
            metrics.completion_debt,
            MAX_COMPLETION_DEBT,
            "promote or reject hypothesized edges used by decisions",
        ));
    }
    if metrics.total_paths > 0 && metrics.evidence_coverage < MIN_EVIDENCE_COVERAGE {
        findings.push(finding(
            "Q11",
            QualitySeverity::Critical,
            "evidence_coverage",
            metrics.evidence_coverage,
            MIN_EVIDENCE_COVERAGE,
            "backfill provenance locators on paths used for decisions",
        ));
    }
    if metrics.hub_path_ratio > MAX_HUB_PATH_RATIO {
        findings.push(finding(
            "Q12",
            QualitySeverity::Warning,
            "hub_path_ratio",
            metrics.hub_path_ratio,
            MAX_HUB_PATH_RATIO,
            "add waypoints or split high-degree hubs used by paths",
        ));
    }
    if metrics.compile_attempts > 0 && metrics.compile_success_rate < MIN_COMPILE_SUCCESS_RATE {
        findings.push(finding(
            "Q13",
            QualitySeverity::Warning,
            "compile_success_rate",
            metrics.compile_success_rate,
            MIN_COMPILE_SUCCESS_RATE,
            "fix PathQuery IR templates or the compile whitelist",
        ));
    }
    if metrics.relaxation_rate > MAX_RELAXATION_RATE {
        findings.push(finding(
            "Q14",
            QualitySeverity::Warning,
            "relaxation_rate",
            metrics.relaxation_rate,
            MAX_RELAXATION_RATE,
            "loosen default filters or add missing graph coverage",
        ));
    }
    findings
}

/// Derive deterministic governance findings from one report.
pub fn diagnose(report: &GraphQualityReport) -> Vec<QualityFinding> {
    if report.memories == 0 {
        return Vec::new();
    }
    let mut findings = Vec::new();
    if report.largest_component_ratio < MIN_LARGEST_COMPONENT_RATIO {
        findings.push(finding(
            "Q01",
            QualitySeverity::Critical,
            "largest_component_ratio",
            report.largest_component_ratio,
            MIN_LARGEST_COMPONENT_RATIO,
            "connect or archive disconnected memory components",
        ));
    }
    if report.orphan_ratio > MAX_ORPHAN_RATIO {
        findings.push(finding(
            "Q02",
            QualitySeverity::Warning,
            "orphan_ratio",
            report.orphan_ratio,
            MAX_ORPHAN_RATIO,
            "link orphan memories to a subject or archive them",
        ));
    }
    if report.provenance_coverage < MIN_PROVENANCE_COVERAGE {
        findings.push(finding(
            "Q03",
            QualitySeverity::Critical,
            "provenance_coverage",
            report.provenance_coverage,
            MIN_PROVENANCE_COVERAGE,
            "backfill source locators before using these paths as evidence",
        ));
    }
    if report.hypothesized_ratio > MAX_HYPOTHESIZED_RATIO {
        findings.push(finding(
            "Q04",
            QualitySeverity::Critical,
            "hypothesized_ratio",
            report.hypothesized_ratio,
            MAX_HYPOTHESIZED_RATIO,
            "promote or reject pending hypothesis edges before planning",
        ));
    }
    if report.pending_shadow_edges > 0 {
        findings.push(finding(
            "Q05",
            QualitySeverity::Warning,
            "pending_shadow_edges",
            report.pending_shadow_edges as f32,
            0.0,
            "review the pending shadow-edge queue",
        ));
    }
    if report.stale_ratio > MAX_STALE_RATIO {
        findings.push(finding(
            "Q06",
            QualitySeverity::Warning,
            "stale_ratio",
            report.stale_ratio,
            MAX_STALE_RATIO,
            "refresh or expire stale memories",
        ));
    }
    if report.hub_ratio > MAX_HUB_RATIO {
        findings.push(finding(
            "Q07",
            QualitySeverity::Warning,
            "hub_ratio",
            report.hub_ratio,
            MAX_HUB_RATIO,
            "split high-degree hubs or add selective waypoints",
        ));
    }
    findings
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
        assert_eq!(quality.aligned_nodes, 2);
        assert_eq!(quality.orphan_nodes, 1);
        assert_eq!(quality.largest_component, 2);
        let report = aggregate("project", vec![quality], 0);
        assert!((report.alignment_rate - 1.0 / 3.0).abs() < 0.001);
        assert!((report.alignment_coverage - 2.0 / 3.0).abs() < 0.001);
        assert!((report.orphan_ratio - 1.0 / 3.0).abs() < 0.001);
        assert!(!report.healthy);
        assert!(report.findings.iter().any(|finding| finding.code == "Q01"));
        assert!(report.findings.iter().any(|finding| finding.code == "Q03"));
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
        assert_eq!(report.alignment_coverage, 0.0);
        assert!(report.healthy);
    }

    #[test]
    fn summarizes_query_audit_rates_and_completion_debt() {
        let entries = vec![
            QueryAuditEntry {
                scope: "project".to_string(),
                kind: "memory".to_string(),
                success: true,
                path_count: 1,
                edge_count: 4,
                hypothesized_edges: 1,
                stores_searched: 2,
                depth: 3,
                diagnosis: Vec::new(),
                relaxations: vec!["E04".to_string()],
                low_provenance_paths: 1,
                hub_paths: 0,
            },
            QueryAuditEntry {
                scope: "project".to_string(),
                kind: "memory".to_string(),
                success: false,
                path_count: 0,
                edge_count: 0,
                hypothesized_edges: 0,
                stores_searched: 2,
                depth: 3,
                diagnosis: vec!["E03".to_string()],
                relaxations: Vec::new(),
                low_provenance_paths: 0,
                hub_paths: 0,
            },
            QueryAuditEntry {
                scope: "project".to_string(),
                kind: "compile".to_string(),
                success: true,
                path_count: 0,
                edge_count: 0,
                hypothesized_edges: 0,
                stores_searched: 0,
                depth: 0,
                diagnosis: Vec::new(),
                relaxations: Vec::new(),
                low_provenance_paths: 0,
                hub_paths: 0,
            },
        ];
        let metrics = summarize_queries(&entries, "project");
        assert_eq!(metrics.total_queries, 2);
        assert_eq!(metrics.successful_queries, 1);
        assert_eq!(metrics.no_path_queries, 1);
        assert_eq!(metrics.path_success_rate, 0.5);
        assert_eq!(metrics.total_edges, 4);
        assert_eq!(metrics.hypothesized_edges, 1);
        assert_eq!(metrics.completion_debt, 0.25);
        assert_eq!(metrics.total_paths, 1);
        assert_eq!(metrics.low_provenance_paths, 1);
        assert_eq!(metrics.evidence_coverage, 0.0);
        assert_eq!(metrics.hub_paths, 0);
        assert_eq!(metrics.hub_path_ratio, 0.0);
        assert_eq!(metrics.cross_graph_queries, 2);
        assert_eq!(metrics.cross_graph_success, 1);
        assert_eq!(metrics.cross_graph_success_rate, 0.5);
        assert_eq!(metrics.diagnosis_counts.get("E03"), Some(&1));
        assert_eq!(metrics.compile_attempts, 1);
        assert_eq!(metrics.compile_successes, 1);
        assert_eq!(metrics.compile_success_rate, 1.0);
        assert_eq!(metrics.relaxed_queries, 1);
        assert_eq!(metrics.relaxation_rate, 0.5);
    }

    #[test]
    fn query_findings_flag_success_alignment_and_debt() {
        let metrics = QueryAuditMetrics {
            total_queries: 4,
            successful_queries: 2,
            no_path_queries: 2,
            path_success_rate: 0.5,
            total_edges: 8,
            total_paths: 8,
            hypothesized_edges: 2,
            completion_debt: 0.25,
            low_provenance_paths: 2,
            hub_paths: 2,
            evidence_coverage: 0.75,
            hub_path_ratio: 0.25,
            compile_attempts: 4,
            compile_successes: 3,
            compile_success_rate: 0.75,
            relaxed_queries: 1,
            relaxation_rate: 0.5,
            cross_graph_queries: 2,
            cross_graph_success: 1,
            cross_graph_success_rate: 0.5,
            diagnosis_counts: BTreeMap::new(),
        };
        let findings = query_diagnose(&metrics);
        assert!(findings.iter().any(|finding| finding.code == "Q08"));
        assert!(findings.iter().any(|finding| finding.code == "Q09"));
        assert!(findings.iter().any(|finding| finding.code == "Q10"));
        assert!(findings.iter().any(|finding| finding.code == "Q11"));
        assert!(findings.iter().any(|finding| finding.code == "Q12"));
        assert!(findings.iter().any(|finding| finding.code == "Q13"));
        assert!(findings.iter().any(|finding| finding.code == "Q14"));
    }
}
