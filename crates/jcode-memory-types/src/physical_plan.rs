//! Stage S5 physical plan IR (design 4.3.3).
//!
//! This compiles a validated [`PathQuery`] into ordered stages with parameter
//! maps. It deliberately does not render Cypher/nGQL: relation names and values
//! stay structured so the executor can bind them, and stages that need entity
//! ids are marked `deferred` instead of pretending grounding already happened.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::path_query::PathQuery;

/// One compiled stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    OntologyProbe,
    InstanceSearch,
    ProjectEnrichment,
    Fusion,
}

impl StageKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OntologyProbe => "ontology_probe",
            Self::InstanceSearch => "instance_search",
            Self::ProjectEnrichment => "project_enrichment",
            Self::Fusion => "fusion",
        }
    }
}

/// One stage with structured parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanStage {
    pub kind: StageKind,
    #[serde(default)]
    pub parameters: BTreeMap<String, Value>,
    /// True when the stage still needs information the compiler cannot supply,
    /// such as grounded entity ids.
    #[serde(default)]
    pub deferred: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl PlanStage {
    fn new(kind: StageKind) -> Self {
        Self {
            kind,
            parameters: BTreeMap::new(),
            deferred: false,
            note: None,
        }
    }

    fn with(mut self, key: &str, value: Value) -> Self {
        self.parameters.insert(key.to_string(), value);
        self
    }

    fn deferred(mut self, note: &str) -> Self {
        self.deferred = true;
        self.note = Some(note.to_string());
        self
    }
}

/// Ordered physical plan for one PathQuery.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicalPlan {
    pub stages: Vec<PlanStage>,
}

impl PhysicalPlan {
    pub fn stage(&self, kind: StageKind) -> Option<&PlanStage> {
        self.stages.iter().find(|stage| stage.kind == kind)
    }

    pub fn deferred_count(&self) -> usize {
        self.stages.iter().filter(|stage| stage.deferred).count()
    }
}

/// Compile a validated PathQuery into staged physical operations.
///
/// `concept_path` may come from S4; when empty the ontology probe is skipped.
/// Instance segments use anchor mentions as-is and stay deferred: grounding to
/// entity ids happens before 4.1 executes them.
pub fn compile_physical_plan(query: &PathQuery, concept_path: &[String]) -> PhysicalPlan {
    let mut stages = Vec::new();
    if !concept_path.is_empty() {
        stages.push(
            PlanStage::new(StageKind::OntologyProbe)
                .with("concepts", serde_json::json!(concept_path)),
        );
    }

    if !query.start.is_empty() && !query.goal.is_empty() {
        let start = query.start[0].clone();
        let goal = query.goal[0].clone();
        let mut anchors = vec![start];
        anchors.extend(query.waypoints.iter().cloned());
        anchors.push(goal);
        let depth = query.constraints.depth.unwrap_or(4);
        let relations = query.constraints.relations.clone();
        let mut stage = PlanStage::new(StageKind::InstanceSearch)
            .with("anchors", serde_json::json!(anchors))
            .with("depth", serde_json::json!(depth))
            .with("relations", serde_json::json!(relations))
            .deferred("anchors are mentions; ground them to entity ids before execution");
        let mut notes = Vec::new();
        if query.start.len() > 1 || query.goal.len() > 1 {
            notes.push(format!(
                "{} start and {} goal anchors present; this plan covers the first pair only",
                query.start.len(),
                query.goal.len()
            ));
        }
        if query.start.len() + query.goal.len() > 2 {
            stage = stage.with("anchor_fanout", serde_json::json!(true));
        }
        if !notes.is_empty() {
            stage.note = Some(format!(
                "{}; {}",
                stage.note.unwrap_or_default(),
                notes.join("; ")
            ));
        }
        stages.push(stage);
    }

    let scope = query
        .constraints
        .scope
        .map(|scope| scope.as_str())
        .unwrap_or("project");
    stages.push(
        PlanStage::new(StageKind::ProjectEnrichment)
            .with("scope", serde_json::json!(scope))
            .with("entities", serde_json::json!([]))
            .deferred("entity ids come from instance search"),
    );

    let objective = query
        .constraints
        .objective
        .map(|objective| objective.as_str())
        .unwrap_or("min_cost");
    stages.push(
        PlanStage::new(StageKind::Fusion)
            .with("objective", serde_json::json!(objective))
            .with(
                "limit",
                serde_json::json!(query.constraints.limit.unwrap_or(3)),
            )
            .with(
                "include_hypotheses",
                serde_json::json!(query.constraints.include_hypotheses),
            )
            .with(
                "min_confidence",
                serde_json::json!(query.constraints.min_confidence),
            ),
    );

    PhysicalPlan { stages }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path_query::{PathObjective, PathQuery, PathQueryConstraints};

    fn query() -> PathQuery {
        PathQuery {
            start: vec!["File".to_string()],
            goal: vec!["API".to_string()],
            waypoints: vec!["Symbol".to_string()],
            graph: None,
            constraints: PathQueryConstraints {
                relations: vec!["defines".to_string()],
                depth: Some(3),
                limit: Some(5),
                objective: Some(PathObjective::MaxConfidence),
                ..PathQueryConstraints::default()
            },
        }
    }

    #[test]
    fn compiles_ordered_stages_with_structured_parameters() {
        let plan = compile_physical_plan(
            &query(),
            &["File".to_string(), "Symbol".to_string(), "API".to_string()],
        );
        let kinds: Vec<StageKind> = plan.stages.iter().map(|stage| stage.kind).collect();
        assert_eq!(
            kinds,
            vec![
                StageKind::OntologyProbe,
                StageKind::InstanceSearch,
                StageKind::ProjectEnrichment,
                StageKind::Fusion
            ]
        );
        let probe = plan.stage(StageKind::OntologyProbe).expect("probe");
        assert_eq!(
            probe.parameters["concepts"],
            serde_json::json!(["File", "Symbol", "API"])
        );

        let search = plan.stage(StageKind::InstanceSearch).expect("search");
        assert!(search.deferred);
        assert_eq!(
            search.parameters["anchors"],
            serde_json::json!(["File", "Symbol", "API"])
        );
        assert_eq!(search.parameters["depth"], serde_json::json!(3));
        assert_eq!(
            search.parameters["relations"],
            serde_json::json!(["defines"])
        );
        assert!(
            search
                .note
                .as_deref()
                .unwrap_or_default()
                .contains("mentions")
        );

        let fusion = plan.stage(StageKind::Fusion).expect("fusion");
        assert_eq!(
            fusion.parameters["objective"],
            serde_json::json!("max_confidence")
        );
        assert_eq!(fusion.parameters["limit"], serde_json::json!(5));
        assert_eq!(plan.deferred_count(), 2);
    }

    #[test]
    fn skips_ontology_probe_without_a_concept_path() {
        let plan = compile_physical_plan(&query(), &[]);
        assert!(plan.stage(StageKind::OntologyProbe).is_none());
        assert_eq!(plan.stages.len(), 3);
    }

    #[test]
    fn defaults_fill_in_when_constraints_are_absent() {
        let mut query = query();
        query.constraints = PathQueryConstraints::default();
        let plan = compile_physical_plan(&query, &[]);
        let search = plan.stage(StageKind::InstanceSearch).expect("search");
        assert_eq!(search.parameters["depth"], serde_json::json!(4));
        let enrichment = plan
            .stage(StageKind::ProjectEnrichment)
            .expect("enrichment");
        assert_eq!(enrichment.parameters["scope"], serde_json::json!("project"));
        let fusion = plan.stage(StageKind::Fusion).expect("fusion");
        assert_eq!(
            fusion.parameters["objective"],
            serde_json::json!("min_cost")
        );
        assert_eq!(fusion.parameters["limit"], serde_json::json!(3));
    }

    #[test]
    fn notes_anchor_fanout_instead_of_silently_dropping_pairs() {
        let mut query = query();
        query.start = vec!["A".to_string(), "B".to_string()];
        query.goal = vec!["C".to_string(), "D".to_string()];
        let plan = compile_physical_plan(&query, &[]);
        let search = plan.stage(StageKind::InstanceSearch).expect("search");
        assert_eq!(search.parameters["anchor_fanout"], serde_json::json!(true));
        let note = search.note.clone().unwrap_or_default();
        assert!(note.contains("first pair only"), "{note}");
        assert!(note.contains("2 start and 2 goal"), "{note}");
    }
}
