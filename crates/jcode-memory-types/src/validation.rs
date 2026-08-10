//! Independent validation / audit for the memory graph.
//!
//! Modeled on `enterprise-ontology/scripts/validate_schema.py`: validates
//! structural invariants, instance-level consistency and action-level
//! preconditions. Can be run standalone (e.g. from a CLI or unit test) without
//! invoking the full memory agent.

use serde::{Deserialize, Serialize};

use crate::actions::{MemoryActionSpec, check_preconditions};
use crate::graph::EdgeKind;
use crate::instance::{MemoryStatus};
use crate::{MemoryEntry, MemoryGraph};

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Warning,
}

/// A single validation issue with a machine-readable code.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationIssue {
    pub code: String,
    pub severity: Severity,
    pub message: String,
    pub memory_id: Option<String>,
}

impl ValidationIssue {
    pub fn error(code: impl Into<String>, message: impl Into<String>, memory_id: Option<String>) -> Self {
        Self {
            code: code.into(),
            severity: Severity::Error,
            message: message.into(),
            memory_id,
        }
    }

    pub fn warning(code: impl Into<String>, message: impl Into<String>, memory_id: Option<String>) -> Self {
        Self {
            code: code.into(),
            severity: Severity::Warning,
            message: message.into(),
            memory_id,
        }
    }
}

/// Overall validation result with exit-code semantics.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ValidationReport {
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors().is_empty()
    }

    pub fn errors(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Error)
            .collect()
    }

    pub fn warnings(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Warning)
            .collect()
    }

    /// Exit code style: 0 = valid, 1 = validation errors, 2 = fatal.
    pub fn exit_code(&self) -> u8 {
        if self.issues.iter().any(|i| i.code.starts_with("F")) {
            2
        } else if self.is_valid() {
            0
        } else {
            1
        }
    }
}

/// Validate the graph. Returns a report with Error/Warning issues.
pub fn validate_graph(graph: &MemoryGraph) -> ValidationReport {
    let mut issues = Vec::new();

    validate_structure(graph, &mut issues);
    validate_instances(graph, &mut issues);
    validate_edges(graph, &mut issues);
    validate_lifecycle(graph, &mut issues);

    ValidationReport { issues }
}

fn validate_structure(graph: &MemoryGraph, issues: &mut Vec<ValidationIssue>) {
    // E01: duplicate memory IDs within the graph storage (HashMap prevents this,
    // but we check for IDs that collide with tag or cluster namespaces).
    for id in graph.memories.keys() {
        if id.starts_with("tag:") || id.starts_with("cluster:") {
            issues.push(ValidationIssue::error(
                "E01",
                format!("Memory ID '{}' collides with reserved tag/cluster namespace", id),
                Some(id.clone()),
            ));
        }
        if id.trim().is_empty() {
            issues.push(ValidationIssue::error("E01", "Memory ID is empty".to_string(), Some(id.clone())));
        }
    }

    // E02: graph version mismatch is not an error here; migrations handle it.
}

fn validate_instances(graph: &MemoryGraph, issues: &mut Vec<ValidationIssue>) {
    for (id, memory) in &graph.memories {
        // E20: required fields missing.
        if memory.content.trim().is_empty() {
            issues.push(ValidationIssue::error(
                "E20",
                "Memory content is empty".to_string(),
                Some(id.clone()),
            ));
        }

        // E21/E22: provenance admissibility. Missing provenance falls back to
        // the mem-plugin default heuristic (user_stated, 0.5) so legacy data is
        // not treated as invalid; only explicitly low-confidence extractions error.
        let prov = memory.effective_provenance();
        if !prov.is_admissible() {
            issues.push(ValidationIssue::error(
                "E22",
                format!(
                    "Provenance confidence {:.2} is below method admission threshold {:.2}",
                    prov.confidence,
                    prov.method.admission_threshold()
                ),
                Some(id.clone()),
            ));
        }
        if prov.confidence < 0.0 || prov.confidence > 1.0 {
            issues.push(ValidationIssue::error(
                "E23",
                format!("Provenance confidence {:.2} is outside [0,1]", prov.confidence),
                Some(id.clone()),
            ));
        }

        // E24: critical memory must have explicit provenance (default heuristic
        // is acceptable, but callers should annotate critical data).
        if memory.critical && memory.provenance.is_none() {
            issues.push(ValidationIssue::warning(
                "E24",
                "Critical memory uses default heuristic provenance; prefer explicit source".to_string(),
                Some(id.clone()),
            ));
        }

        // E25: confidence field outside [0,1].
        if memory.confidence < 0.0 || memory.confidence > 1.0 {
            issues.push(ValidationIssue::error(
                "E25",
                format!("Memory confidence {:.2} is outside [0,1]", memory.confidence),
                Some(id.clone()),
            ));
        }
    }
}

fn validate_edges(graph: &MemoryGraph, issues: &mut Vec<ValidationIssue>) {
    // E30: dangling edges (source or target missing).
    for (source_id, edges) in &graph.edges {
        if !source_id.starts_with("tag:") && !source_id.starts_with("cluster:") && !graph.memories.contains_key(source_id) {
            issues.push(ValidationIssue::error(
                "E30",
                format!("Edge source '{}' does not exist", source_id),
                Some(source_id.clone()),
            ));
        }
        for edge in edges {
            let target_exists = edge.target.starts_with("tag:") && graph.tags.contains_key(&edge.target)
                || edge.target.starts_with("cluster:") && graph.clusters.contains_key(&edge.target)
                || graph.memories.contains_key(&edge.target);
            if !target_exists {
                issues.push(ValidationIssue::error(
                    "E31",
                    format!(
                        "Edge target '{}' from '{}' does not exist",
                        edge.target, source_id
                    ),
                    Some(source_id.clone()),
                ));
            }

            // E32: edge weight outside [0,1] for weighted edges.
            if let EdgeKind::RelatesTo { weight } = &edge.kind {
                if *weight < 0.0 || *weight > 1.0 {
                    issues.push(ValidationIssue::error(
                        "E32",
                        format!(
                            "RelatesTo weight {:.2} from '{}' is outside [0,1]",
                            weight, source_id
                        ),
                        Some(source_id.clone()),
                    ));
                }
            }
        }
    }

    // W01: reverse edges out of sync with forward edges.
    for (target_id, sources) in &graph.reverse_edges {
        for source_id in sources {
            let forward = graph.edges.get(source_id).map(|v| v.as_slice()).unwrap_or(&[]);
            if !forward.iter().any(|e| e.target == *target_id) {
                issues.push(ValidationIssue::warning(
                    "W01",
                    format!(
                        "Reverse edge {} -> {} has no matching forward edge",
                        source_id, target_id
                    ),
                    Some(source_id.clone()),
                ));
            }
        }
    }
}

fn validate_lifecycle(graph: &MemoryGraph, issues: &mut Vec<ValidationIssue>) {
    // E40: archived memory still active flag.
    for (id, memory) in &graph.memories {
        if memory.lifecycle.status == MemoryStatus::Archived && memory.active {
            issues.push(ValidationIssue::error(
                "E40",
                "Archived memory is still marked active=true".to_string(),
                Some(id.clone()),
            ));
        }
        if memory.lifecycle.status == MemoryStatus::Disputed && memory.critical {
            issues.push(ValidationIssue::warning(
                "W02",
                "Critical memory is disputed; requires resolution".to_string(),
                Some(id.clone()),
            ));
        }
        if memory.lifecycle.deprecated && memory.lifecycle.status == MemoryStatus::Active {
            issues.push(ValidationIssue::warning(
                "W03",
                "Memory is deprecated but still active".to_string(),
                Some(id.clone()),
            ));
        }
        if let (Some(from), Some(to)) = (memory.lifecycle.effective_from, memory.lifecycle.effective_to) {
            if to < from {
                issues.push(ValidationIssue::error(
                    "E41",
                    "effective_to is before effective_from".to_string(),
                    Some(id.clone()),
                ));
            }
        }
    }
}

/// Validate a single action against a graph without mutating it.
pub fn validate_action(
    graph: &MemoryGraph,
    action: &MemoryActionSpec,
    source_id: Option<&str>,
    target_id: Option<&str>,
) -> Result<(), Vec<ValidationIssue>> {
    match check_preconditions(graph, action, source_id, target_id) {
        Ok(()) => Ok(()),
        Err(messages) => Err(messages
            .into_iter()
            .map(|m| ValidationIssue::error("A01", m, source_id.map(String::from)))
            .collect()),
    }
}

/// Convenience: check that a newly constructed memory entry is admissible.
pub fn validate_new_entry(entry: &MemoryEntry) -> Result<(), Vec<ValidationIssue>> {
    let mut issues = Vec::new();
    if entry.content.trim().is_empty() {
        issues.push(ValidationIssue::error("E20", "Memory content is empty", Some(entry.id.clone())));
    }
    let prov = entry.effective_provenance();
    if !prov.is_admissible() {
        issues.push(ValidationIssue::error(
            "E22",
            format!(
                "Provenance confidence {:.2} below admission threshold {:.2}",
                prov.confidence,
                prov.method.admission_threshold()
            ),
            Some(entry.id.clone()),
        ));
    }
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::{MemoryActionSpec};
    use crate::instance::{ExtractionMethod, LifecycleMetadata, ProvenanceRecord};
    use crate::{MemoryCategory, MemoryEntry, MemoryGraph};

    fn make_entry(content: &str) -> MemoryEntry {
        MemoryEntry::new(MemoryCategory::Fact, content)
    }

    #[test]
    fn valid_graph_has_no_errors() {
        let mut graph = MemoryGraph::new();
        graph.add_memory(
            make_entry("valid").with_provenance(ProvenanceRecord::new(
                "test",
                ExtractionMethod::UserStated,
            )),
        );
        let report = validate_graph(&graph);
        assert!(report.is_valid());
        assert!(report.warnings().is_empty());
    }

    #[test]
    fn empty_content_is_error() {
        let mut graph = MemoryGraph::new();
        let mut entry = make_entry("");
        entry.id = "mem_empty".to_string();
        graph.add_memory(entry);
        let report = validate_graph(&graph);
        assert!(!report.is_valid());
        assert!(report.errors().iter().any(|i| i.code == "E20"));
    }

    #[test]
    fn inadmissible_provenance_is_error() {
        let mut graph = MemoryGraph::new();
        let entry = make_entry("low confidence")
            .with_provenance(ProvenanceRecord::new("extraction", ExtractionMethod::LlmExtraction).with_confidence(0.5));
        graph.add_memory(entry);
        let report = validate_graph(&graph);
        assert!(!report.is_valid());
        assert!(report.errors().iter().any(|i| i.code == "E22"));
    }

    #[test]
    fn archived_but_active_is_error() {
        let mut graph = MemoryGraph::new();
        let mut entry = make_entry("old");
        entry.lifecycle = LifecycleMetadata::archived("obsolete");
        entry.active = true; // inconsistent
        graph.add_memory(entry);
        let report = validate_graph(&graph);
        assert!(!report.is_valid());
        assert!(report.errors().iter().any(|i| i.code == "E40"));
    }

    #[test]
    fn invalid_time_window_is_error() {
        let mut graph = MemoryGraph::new();
        let mut entry = make_entry("windowed");
        entry.lifecycle = LifecycleMetadata::active()
            .with_effective_from(chrono::Utc::now() + chrono::Duration::days(5))
            .with_effective_to(chrono::Utc::now());
        graph.add_memory(entry);
        let report = validate_graph(&graph);
        assert!(!report.is_valid());
        assert!(report.errors().iter().any(|i| i.code == "E41"));
    }

    #[test]
    fn dangling_edge_is_error() {
        let mut graph = MemoryGraph::new();
        let id = graph.add_memory(make_entry("source"));
        graph.add_edge(&id, "missing", crate::graph::EdgeKind::RelatesTo { weight: 0.5 });
        let report = validate_graph(&graph);
        assert!(!report.is_valid());
        assert!(report.errors().iter().any(|i| i.code == "E31"));
    }

    #[test]
    fn action_precondition_validation() {
        let graph = MemoryGraph::new();
        let action = MemoryActionSpec::forget();
        let err = validate_action(&graph, &action, Some("missing"), None).unwrap_err();
        assert!(err.iter().any(|i| i.code == "A01"));
    }

    #[test]
    fn validate_new_entry_allows_missing_provenance_with_heuristic_default() {
        let entry = make_entry("no provenance");
        let res = validate_new_entry(&entry);
        assert!(res.is_ok(), "missing provenance should fall back to user_stated/0.5 heuristic");
    }

    #[test]
    fn validate_new_entry_rejects_low_confidence() {
        let entry = make_entry("low confidence")
            .with_provenance(ProvenanceRecord::new("extraction", ExtractionMethod::LlmExtraction).with_confidence(0.5));
        let res = validate_new_entry(&entry);
        assert!(res.is_err());
        assert!(res.unwrap_err().iter().any(|i| i.code == "E22"));
    }
}
