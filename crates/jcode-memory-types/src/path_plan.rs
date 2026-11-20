//! Deterministic conversion from an engineering path to a task plan.
//!
//! This is the mechanical half of the design document's Path -> Plan step: each
//! relation is bound to one of the eight allowed action types, every task gets a
//! success check and an evidence reference, and side-effect tasks inherit an
//! approval gate. It does not invent missing preconditions or execute anything.

use serde::{Deserialize, Serialize};

use crate::engineering_ontology::{
    EngineeringGraph, EngineeringPath, REL_BLOCKED_BY, REL_CALLS, REL_CAUSES, REL_DEPENDS_ON,
    REL_EVIDENCED_BY, REL_FIXES, REL_IMPLEMENTS, REL_MODIFIES, REL_PRODUCES, REL_SUPERSEDES,
    REL_TESTS, REL_VERIFIES,
};

/// Action types allowed in a PathPlan.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanActionType {
    Query,
    Verify,
    Acquire,
    Execute,
    Notify,
    Decide,
    AwaitHuman,
    Monitor,
}

impl PlanActionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            PlanActionType::Query => "query",
            PlanActionType::Verify => "verify",
            PlanActionType::Acquire => "acquire",
            PlanActionType::Execute => "execute",
            PlanActionType::Notify => "notify",
            PlanActionType::Decide => "decide",
            PlanActionType::AwaitHuman => "await_human",
            PlanActionType::Monitor => "monitor",
        }
    }

    pub fn has_side_effects(&self) -> bool {
        matches!(
            self,
            PlanActionType::Acquire
                | PlanActionType::Execute
                | PlanActionType::Notify
                | PlanActionType::Decide
        )
    }
}

/// Coarse irreversibility class used for approval gating.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum PlanRisk {
    Low,
    Medium,
    High,
}

impl PlanRisk {
    pub fn as_str(&self) -> &'static str {
        match self {
            PlanRisk::Low => "low",
            PlanRisk::Medium => "medium",
            PlanRisk::High => "high",
        }
    }
}

/// One executable or verification task compiled from a path edge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PathPlanTask {
    pub id: String,
    pub action_type: PlanActionType,
    pub title: String,
    pub relation: String,
    pub from: String,
    pub to: String,
    pub depends_on: Vec<String>,
    pub success_check: String,
    pub evidence_ref: String,
    pub risk: PlanRisk,
    pub approval: bool,
    pub confidence: f32,
}

/// A deterministic plan for one engineering path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PathPlan {
    pub tasks: Vec<PathPlanTask>,
    pub confidence: f32,
    pub approval_gates: Vec<String>,
    pub warnings: Vec<String>,
}

impl PathPlan {
    pub fn task(&self, id: &str) -> Option<&PathPlanTask> {
        self.tasks.iter().find(|task| task.id == id)
    }
}

fn template(relation: &str) -> (PlanActionType, PlanRisk, &'static str) {
    match relation {
        REL_DEPENDS_ON => (
            PlanActionType::Execute,
            PlanRisk::Low,
            "predecessor.status == done",
        ),
        REL_BLOCKED_BY => (
            PlanActionType::Verify,
            PlanRisk::Low,
            "blocker.status == cleared",
        ),
        REL_CAUSES => (
            PlanActionType::Monitor,
            PlanRisk::Low,
            "monitored metric within expected range",
        ),
        REL_FIXES => (
            PlanActionType::Execute,
            PlanRisk::Medium,
            "error signature no longer reproduces",
        ),
        REL_MODIFIES => (
            PlanActionType::Execute,
            PlanRisk::Medium,
            "change applied and verified",
        ),
        REL_TESTS | REL_VERIFIES => {
            (PlanActionType::Verify, PlanRisk::Low, "check passes")
        }
        REL_PRODUCES => (
            PlanActionType::Execute,
            PlanRisk::Medium,
            "artifact exists",
        ),
        REL_CALLS => (
            PlanActionType::Execute,
            PlanRisk::Medium,
            "call completed",
        ),
        "requires" => (
            PlanActionType::Acquire,
            PlanRisk::Medium,
            "resource available >= required",
        ),
        REL_IMPLEMENTS => (
            PlanActionType::Execute,
            PlanRisk::Medium,
            "implementation compiled and linked",
        ),
        REL_SUPERSEDES => (
            PlanActionType::Verify,
            PlanRisk::Medium,
            "superseded artifact is no longer active",
        ),
        REL_EVIDENCED_BY => (
            PlanActionType::Query,
            PlanRisk::Low,
            "evidence is resolvable at its locator",
        ),
        _ => (
            PlanActionType::Verify,
            PlanRisk::Low,
            "relation evidence confirmed",
        ),
    }
}

fn task_title(
    action: PlanActionType,
    relation: &str,
    from_label: &str,
    to_label: &str,
) -> String {
    match action {
        PlanActionType::Monitor => format!("Monitor {to_label} after {relation}"),
        PlanActionType::Verify | PlanActionType::Query => {
            format!("Verify {to_label} via {relation} from {from_label}")
        }
        PlanActionType::Acquire => format!("Acquire {to_label} required by {from_label}"),
        _ => format!("{relation}: {from_label} -> {to_label}"),
    }
}

/// Compile one engineering path into a deterministic task DAG.
///
/// Missing relation templates degrade to `verify`, never to an unconstrained
/// side effect. The plan is linear in path order, which is the safe subset of
/// the document's DAG requirement; callers can later relax independent tasks.
pub fn build_path_plan(graph: &EngineeringGraph, path: &EngineeringPath) -> PathPlan {
    let mut tasks = Vec::with_capacity(path.relations.len());
    let mut confidence = path.confidence;
    let mut previous: Option<String> = None;

    for (index, relation_id) in path.relations.iter().enumerate() {
        let Some(relation) = graph.relation(relation_id) else {
            continue;
        };
        let (action, risk, success_check) = template(&relation.relation);
        let task_confidence = if relation.confidence.is_finite() {
            relation.confidence.clamp(0.0, 1.0)
        } else {
            path.confidence
        };
        confidence = confidence.min(task_confidence);
        let id = format!("T{}", index + 1);
        let from_label = path
            .nodes
            .get(index)
            .and_then(|id| graph.entity(id))
            .map(|entity| entity.name.clone())
            .unwrap_or_else(|| path.nodes.get(index).cloned().unwrap_or_default());
        let to_label = path
            .nodes
            .get(index + 1)
            .and_then(|id| graph.entity(id))
            .map(|entity| entity.name.clone())
            .unwrap_or_else(|| path.nodes.get(index + 1).cloned().unwrap_or_default());
        let depends_on = previous.iter().cloned().collect();
        let approval = action.has_side_effects() && risk != PlanRisk::Low;
        tasks.push(PathPlanTask {
            id: id.clone(),
            action_type: action,
            title: task_title(action, &relation.relation, &from_label, &to_label),
            relation: relation.relation.clone(),
            from: path.nodes.get(index).cloned().unwrap_or_default(),
            to: path.nodes.get(index + 1).cloned().unwrap_or_default(),
            depends_on,
            success_check: success_check.to_string(),
            evidence_ref: format!("path/edge/{index}"),
            risk,
            approval,
            confidence: task_confidence,
        });
        previous = Some(id);
    }

    let mut warnings = Vec::new();
    if tasks.len() > 20 {
        warnings.push(format!(
            "plan has {} tasks; split before execution (limit 20)",
            tasks.len()
        ));
    }
    if confidence < 0.6 {
        warnings.push(
            "path confidence below 0.6; side-effect tasks require explicit approval".to_string(),
        );
        for task in &mut tasks {
            if task.action_type.has_side_effects() {
                task.approval = true;
            }
        }
    }
    let approval_gates: Vec<String> = tasks
        .iter()
        .filter(|task| task.approval)
        .map(|task| task.id.clone())
        .collect();
    PathPlan {
        tasks,
        confidence,
        approval_gates,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engineering_ontology::{
        CLASS_CHANGE, CLASS_ERROR, CLASS_TASK, EngineeringEntity, EngineeringRelation,
        REL_BLOCKED_BY, REL_FIXES,
    };

    fn chain() -> (EngineeringGraph, EngineeringPath) {
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(EngineeringEntity::new(CLASS_TASK, "task:t1", "Fix build"));
        graph.insert_entity(EngineeringEntity::new(CLASS_ERROR, "error:e1", "E0425"));
        graph.insert_entity(EngineeringEntity::new(CLASS_CHANGE, "change:c1", "Patch"));
        graph.insert_relation(EngineeringRelation::new("task:t1", REL_BLOCKED_BY, "error:e1"));
        graph.insert_relation(EngineeringRelation::new("error:e1", REL_FIXES, "change:c1"));
        let path = graph
            .weighted_shortest_path("task:t1", "change:c1", 3, &[])
            .expect("path");
        (graph, path)
    }

    #[test]
    fn maps_relations_to_actions_and_dependencies() {
        let (graph, path) = chain();
        let plan = build_path_plan(&graph, &path);
        assert_eq!(plan.tasks.len(), 2);
        assert_eq!(plan.tasks[0].action_type, PlanActionType::Verify);
        assert_eq!(plan.tasks[0].relation, REL_BLOCKED_BY);
        assert!(plan.tasks[0].depends_on.is_empty());
        assert_eq!(plan.tasks[1].action_type, PlanActionType::Execute);
        assert_eq!(plan.tasks[1].depends_on, vec!["T1"]);
        assert!(plan.tasks[1].approval);
        assert_eq!(plan.approval_gates, vec!["T2"]);
        assert!(plan.tasks.iter().all(|task| !task.success_check.is_empty()));
        assert!(plan.tasks.iter().all(|task| !task.evidence_ref.is_empty()));
    }

    #[test]
    fn unknown_relation_degrades_to_verification() {
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(EngineeringEntity::new(CLASS_TASK, "task:t1", "Task"));
        graph.insert_entity(EngineeringEntity::new(CLASS_CHANGE, "change:c1", "Change"));
        graph.insert_relation(EngineeringRelation::new("task:t1", "custom_relation", "change:c1"));
        let path = graph
            .weighted_shortest_path("task:t1", "change:c1", 2, &[])
            .expect("path");
        let plan = build_path_plan(&graph, &path);
        assert_eq!(plan.tasks[0].action_type, PlanActionType::Verify);
        assert_eq!(
            plan.tasks[0].success_check,
            "relation evidence confirmed"
        );
        assert!(!plan.tasks[0].approval);
    }

    #[test]
    fn low_confidence_forces_side_effect_approval() {
        let (mut graph, path) = chain();
        for relation in graph.relations.values_mut() {
            relation.confidence = 0.4;
        }
        let plan = build_path_plan(&graph, &path);
        assert!(plan.confidence <= 0.4);
        assert!(
            plan.warnings
                .iter()
                .any(|warning| warning.contains("confidence below 0.6"))
        );
        assert!(
            plan.tasks
                .iter()
                .filter(|task| task.action_type.has_side_effects())
                .all(|task| task.approval)
        );
    }
}
