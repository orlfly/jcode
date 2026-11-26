//! Stage S6 observation normalization (design 4.3).
//!
//! Query results are converted into a single [`Observation`] shape before any
//! reward or belief-state consumer sees them: nodes, edges (with edge-level
//! provenance and hypothesis flags), cost breakdown, diagnostics and quality
//! flags. Consumers must not read graph structures directly.

use serde::{Deserialize, Serialize};

use crate::engineering_ontology::{
    EngineeringGraph, EngineeringPath, EngineeringProvenance, EngineeringRelation,
};
use crate::path_plan::{PathPlan, PlanGateSeverity, PlanQualityIssue, PlanQualityReport};
use crate::path_quality::{PathQuality, PathQualityFlag};

/// One node in an observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationNode {
    pub id: String,
    pub label: String,
    pub class: Option<String>,
}

/// One edge in an observation, with the provenance consumers need to audit it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationEdge {
    pub id: String,
    pub from: String,
    pub to: String,
    pub relation: String,
    pub confidence: f32,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hypothesis: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_to: Option<String>,
    pub provenance: EngineeringProvenance,
    pub cost: f64,
}

/// Deterministic decomposition of an engineering path cost.
///
/// `structural + confidence + semantic + hub == total`, and `total` equals the
/// engine's `relation_cost` sum for the same edges.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub structural: f64,
    pub confidence: f64,
    pub semantic: f64,
    pub hub: f64,
    pub total: f64,
}

/// One observed path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationPath {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub nodes: Vec<ObservationNode>,
    pub edges: Vec<ObservationEdge>,
    pub cost: f64,
    pub confidence: f32,
    pub cost_breakdown: CostBreakdown,
    pub quality: PathQuality,
    pub provenance: Vec<EngineeringProvenance>,
}

/// Structured diagnostic attached to an observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationDiagnostic {
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

/// Normalized query result consumed by reward and belief state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub source_graph: String,
    pub paths: Vec<ObservationPath>,
    pub diagnostics: Vec<ObservationDiagnostic>,
    pub quality: PathQuality,
}

impl Observation {
    pub fn new(source_graph: impl Into<String>) -> Self {
        Self {
            source_graph: source_graph.into(),
            paths: Vec::new(),
            diagnostics: Vec::new(),
            quality: PathQuality::default(),
        }
    }

    pub fn push_path(&mut self, path: ObservationPath) {
        for flag in &path.quality.flags {
            self.quality.flags.push(*flag);
        }
        self.quality.flags.dedup();
        self.quality.reasons.extend(path.quality.reasons.clone());
        self.paths.push(path);
    }

    pub fn has_quality(&self, flag: PathQualityFlag) -> bool {
        self.quality.flags.contains(&flag)
    }

    /// Merge another observation of the same query (one per session graph).
    pub fn merge(&mut self, other: Observation) {
        for flag in other.quality.flags {
            if !self.quality.flags.contains(&flag) {
                self.quality.flags.push(flag);
            }
        }
        self.quality.reasons.extend(other.quality.reasons);
        self.paths.extend(other.paths);
        self.diagnostics.extend(other.diagnostics);
    }
}

fn edge_cost(relation: &EngineeringRelation) -> (f64, f64, f64) {
    let confidence = if relation.confidence.is_finite() {
        f64::from(relation.confidence.clamp(0.0, 1.0))
    } else {
        0.5
    };
    let confidence_penalty = 2.0 * (1.0 - confidence);
    let weight = relation
        .properties
        .get("weight")
        .and_then(serde_json::Value::as_f64)
        .filter(|weight| weight.is_finite() && *weight > 0.0)
        .unwrap_or(1.0);
    let semantic = (weight - 1.0) * (1.0 + confidence_penalty);
    (
        confidence_penalty,
        semantic,
        1.0 + confidence_penalty + semantic,
    )
}

/// Convert one engineering path into an observation path.
///
/// Edges whose relation is missing from the graph are skipped and reported as
/// an `O01` diagnostic in [`observation_from_graph_paths`]; this function
/// returns the path only.
pub fn observe_path(graph: &EngineeringGraph, path: &EngineeringPath) -> ObservationPath {
    let nodes = path
        .nodes
        .iter()
        .map(|id| {
            let entity = graph.entity(id);
            ObservationNode {
                id: id.clone(),
                label: entity
                    .map(|entity| entity.name.clone())
                    .unwrap_or_else(|| id.clone()),
                class: entity.map(|entity| entity.class.clone()),
            }
        })
        .collect();

    let mut breakdown = CostBreakdown {
        structural: 0.0,
        confidence: 0.0,
        semantic: 0.0,
        hub: 0.0,
        total: 0.0,
    };
    let edges = path
        .relations
        .iter()
        .filter_map(|id| {
            let relation = graph.relation(id)?;
            let (confidence_penalty, semantic, total) = edge_cost(relation);
            breakdown.structural += 1.0;
            breakdown.confidence += confidence_penalty;
            breakdown.semantic += semantic;
            breakdown.total += total;
            Some(ObservationEdge {
                id: relation.id.clone(),
                from: relation.from.clone(),
                to: relation.to.clone(),
                relation: relation.relation.clone(),
                confidence: relation.confidence,
                hypothesis: relation.hypothesis,
                valid_to: relation.valid_to.map(|until| until.to_rfc3339()),
                provenance: relation.provenance.clone(),
                cost: total,
            })
        })
        .collect();

    ObservationPath {
        session_id: None,
        nodes,
        edges,
        cost: path.cost,
        confidence: path.confidence,
        cost_breakdown: breakdown,
        quality: graph.quality_of_path(path),
        provenance: path.provenance.clone(),
    }
}

/// Build one observation from several paths of the same graph.
pub fn observation_from_graph_paths(
    graph: &EngineeringGraph,
    source_graph: impl Into<String>,
    paths: &[EngineeringPath],
) -> Observation {
    let mut observation = Observation::new(source_graph);
    for path in paths {
        let observed = observe_path(graph, path);
        if observed.edges.len() < path.relations.len() {
            let missing = path.relations.len() - observed.edges.len();
            observation.diagnostics.push(ObservationDiagnostic {
                code: "O01".to_string(),
                message: format!(
                    "{missing} relation(s) on a {} hop path are missing from the graph; edges were dropped",
                    path.hop_count()
                ),
                suggestion: Some("reload the session graph or treat the path as low provenance".to_string()),
            });
        }
        observation.push_path(observed);
    }
    observation
}

/// Attach diagnostics that came from the query layer (E-codes, budget, etc.).
pub fn with_diagnostics(
    mut observation: Observation,
    diagnostics: Vec<ObservationDiagnostic>,
) -> Observation {
    observation.diagnostics.extend(diagnostics);
    observation
}

/// Audit a compiled plan against the observation it was built from.
///
/// The plan's own gates (Q00-Q12) still run first; observation findings are
/// appended so a scheduler never executes a plan whose evidence was dropped or
/// whose query was truncated:
///
/// - `Q19` blocking: the observation dropped edges (`O01`), so evidence is incomplete.
/// - `Q20` warning: the query hit its budget (`E07`); results may be partial.
/// - `Q21` warning: the observation carries low-provenance edges.
/// - `Q22` warning: the observation carries stale edges.
pub fn audit_observation_plan(plan: &PathPlan, observation: &Observation) -> PlanQualityReport {
    let mut report = plan.audit_quality();
    let mut push = |code: &str, severity: PlanGateSeverity, message: String| {
        let issue = PlanQualityIssue {
            code: code.to_string(),
            severity,
            task_id: None,
            message,
        };
        if !report
            .issues
            .iter()
            .any(|existing| existing.code == issue.code && existing.message == issue.message)
        {
            report.issues.push(issue);
        }
    };

    for diagnostic in &observation.diagnostics {
        match diagnostic.code.as_str() {
            "O01" => push(
                "Q19",
                PlanGateSeverity::Blocking,
                format!(
                    "observation dropped edges: {}; rebuild the plan from a complete graph",
                    diagnostic.message
                ),
            ),
            "E07" => push(
                "Q20",
                PlanGateSeverity::Warning,
                format!(
                    "query budget truncated the observation: {}",
                    diagnostic.message
                ),
            ),
            _ => {}
        }
    }
    if observation.has_quality(PathQualityFlag::LowProvenance) {
        push(
            "Q21",
            PlanGateSeverity::Warning,
            "observation has low-provenance edges; attach evidence before scheduling side effects"
                .to_string(),
        );
    }
    if observation.has_quality(PathQualityFlag::StaleEdge) {
        push(
            "Q22",
            PlanGateSeverity::Warning,
            "observation has stale edges; re-verify or refresh the graph".to_string(),
        );
    }

    report.executable = !report
        .issues
        .iter()
        .any(|issue| issue.severity == PlanGateSeverity::Blocking);
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engineering_ontology::{EngineeringEntity, EngineeringGraph, EngineeringRelation};

    fn graph() -> EngineeringGraph {
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(EngineeringEntity::new("File", "f1", "src/lib.rs"));
        graph.insert_entity(EngineeringEntity::new("API", "a1", "/task"));
        graph.insert_entity(EngineeringEntity::new("Symbol", "s1", "run"));
        let mut first = EngineeringRelation::new("f1", "defines", "s1").with_id("r1");
        first.confidence = 0.8;
        first
            .properties
            .insert("weight".to_string(), serde_json::json!(2.0));
        first.provenance = EngineeringProvenance::new("code-index");
        let mut second = EngineeringRelation::new("s1", "exposes", "a1").with_id("r2");
        second.confidence = 0.5;
        second.hypothesis = true;
        graph.insert_relation(first);
        graph.insert_relation(second);
        graph
    }

    fn path() -> EngineeringPath {
        let mut path = EngineeringPath {
            nodes: vec!["f1".into(), "s1".into(), "a1".into()],
            relations: vec!["r1".into(), "r2".into()],
            cost: 0.0,
            confidence: 0.65,
            provenance: vec![EngineeringProvenance::new("query")],
        };
        path.cost = graph()
            .k_shortest_paths("f1", "a1", 4, &[], 1)
            .first()
            .map(|found| found.cost)
            .unwrap_or(0.0);
        path.confidence = 0.65;
        path
    }

    #[test]
    fn normalizes_nodes_edges_and_edge_level_provenance() {
        let graph = graph();
        let observation = observation_from_graph_paths(&graph, "engineering", &[path()]);
        assert_eq!(observation.source_graph, "engineering");
        assert!(observation.diagnostics.is_empty());
        let observed = &observation.paths[0];
        assert_eq!(observed.nodes[0].label, "src/lib.rs");
        assert_eq!(observed.nodes[0].class.as_deref(), Some("File"));
        assert_eq!(observed.edges.len(), 2);
        assert_eq!(observed.edges[0].relation, "defines");
        assert_eq!(observed.edges[0].provenance.source, "code-index");
        assert!(observed.edges[1].hypothesis);
    }

    #[test]
    fn cost_breakdown_reconstructs_the_path_cost() {
        let graph = graph();
        let path = path();
        let observed = observe_path(&graph, &path);
        let breakdown = &observed.cost_breakdown;
        let sum = breakdown.structural + breakdown.confidence + breakdown.semantic + breakdown.hub;
        assert!((sum - breakdown.total).abs() < 1e-9);
        assert!(
            (breakdown.total - path.cost).abs() < 1e-9,
            "breakdown {} vs path cost {}",
            breakdown.total,
            path.cost
        );
        assert_eq!(breakdown.structural, 2.0);
        assert!(breakdown.semantic > 0.0);
    }

    #[test]
    fn propagates_hypothesis_quality_to_the_observation() {
        let graph = graph();
        let observation = observation_from_graph_paths(&graph, "engineering", &[path()]);
        assert!(observation.has_quality(PathQualityFlag::HypothesizedEdge));
        assert!(
            observation.paths[0]
                .quality
                .has(PathQualityFlag::HypothesizedEdge)
        );
    }

    #[test]
    fn missing_relations_become_an_explicit_diagnostic() {
        let graph = graph();
        let mut path = path();
        path.relations.push("ghost".to_string());
        let observation = observation_from_graph_paths(&graph, "engineering", &[path.clone()]);
        assert_eq!(observation.paths[0].edges.len(), 2);
        assert_eq!(observation.diagnostics[0].code, "O01");
        assert!(observation.diagnostics[0].message.contains("missing"));
        assert_eq!(observation.paths[0].nodes.len(), path.nodes.len());
    }

    fn plan_and_observation() -> (PathPlan, Observation) {
        let graph = graph();
        let path = path();
        let plan = crate::path_plan::build_path_plan(&graph, &path);
        let observation = observation_from_graph_paths(&graph, "engineering", &[path]);
        (plan, observation)
    }

    #[test]
    fn dropped_edges_block_the_plan() {
        let (plan, mut observation) = plan_and_observation();
        observation.diagnostics.push(ObservationDiagnostic {
            code: "O01".to_string(),
            message: "1 relation is missing".to_string(),
            suggestion: None,
        });
        let report = audit_observation_plan(&plan, &observation);
        assert!(report.has_code("Q19"), "{report:?}");
        assert!(!report.executable);
    }

    #[test]
    fn budget_truncation_is_a_warning_not_a_blocker() {
        let (plan, mut observation) = plan_and_observation();
        let base = plan.audit_quality();
        observation.diagnostics.push(ObservationDiagnostic {
            code: "E07".to_string(),
            message: "budget exhausted".to_string(),
            suggestion: None,
        });
        let report = audit_observation_plan(&plan, &observation);
        assert!(report.has_code("Q20"), "{report:?}");
        assert_eq!(report.executable, base.executable);
    }

    #[test]
    fn observation_quality_flags_become_plan_warnings() {
        let (plan, mut observation) = plan_and_observation();
        observation
            .quality
            .flags
            .push(PathQualityFlag::LowProvenance);
        observation.quality.flags.push(PathQualityFlag::StaleEdge);
        let report = audit_observation_plan(&plan, &observation);
        assert!(report.has_code("Q21"), "{report:?}");
        assert!(report.has_code("Q22"), "{report:?}");
        assert!(
            !report
                .issues
                .iter()
                .any(|issue| issue.code == "Q21" && issue.severity == PlanGateSeverity::Blocking)
        );
    }

    #[test]
    fn clean_observation_keeps_the_plan_audit_unchanged() {
        let (plan, observation) = plan_and_observation();
        let base = plan.audit_quality();
        let report = audit_observation_plan(&plan, &observation);
        assert_eq!(report, base);
    }

    #[test]
    fn query_diagnostics_are_appended_not_replaced() {
        let graph = graph();
        let observation = observation_from_graph_paths(&graph, "engineering", &[path()]);
        let observation = with_diagnostics(
            observation,
            vec![ObservationDiagnostic {
                code: "E07".to_string(),
                message: "budget exhausted".to_string(),
                suggestion: None,
            }],
        );
        assert_eq!(observation.diagnostics.len(), 1);
        assert_eq!(observation.diagnostics[0].code, "E07");
    }
}
