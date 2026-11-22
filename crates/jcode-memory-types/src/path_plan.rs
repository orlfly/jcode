//! Deterministic conversion from an engineering path to a task plan.
//!
//! This is the mechanical half of the design document's Path -> Plan step: each
//! relation is bound to one of the eight allowed action types, every task gets a
//! success check and an evidence reference, and side-effect tasks inherit an
//! approval gate. It derives existence preconditions and verification queries
//! from the path itself, never invents missing world-state preconditions, and
//! never executes anything.

use std::collections::{BTreeMap, BTreeSet, HashSet};

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
    #[serde(default)]
    pub preconditions: Vec<String>,
    #[serde(default)]
    pub postconditions: Vec<String>,
}

/// One executable verification query attached to a plan task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanVerificationQuery {
    pub id: String,
    pub task_id: String,
    pub query_ref: String,
    pub expr: String,
    pub evidence_ref: String,
}

/// Observed outcome for one plan task.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanTaskOutcome {
    Succeeded,
    Failed,
    Blocked,
    Skipped,
}

/// Current execution state for one plan task.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanTaskState {
    Pending,
    Succeeded,
    Failed,
    Blocked,
    Skipped,
}

impl PlanTaskState {
    pub fn is_terminal(self) -> bool {
        !matches!(self, PlanTaskState::Pending)
    }
}

/// One execution observation fed back into a plan.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanTaskObservation {
    pub task_id: String,
    pub outcome: PlanTaskOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Deterministic replay of observations against a plan DAG.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanExecutionReport {
    pub plan_id: String,
    pub states: BTreeMap<String, PlanTaskState>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub notes: BTreeMap<String, String>,
    #[serde(default)]
    pub ready_groups: Vec<Vec<String>>,
    #[serde(default)]
    pub recovery_tasks: Vec<PathPlanTask>,
    #[serde(default)]
    pub rollback: Vec<String>,
    #[serde(default)]
    pub recommendations: Vec<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
    pub done: bool,
    pub blocked: bool,
}

/// A deterministic plan for one engineering path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PathPlan {
    #[serde(default)]
    pub plan_id: String,
    #[serde(default)]
    pub source_path_id: String,
    #[serde(default)]
    pub goal: String,
    pub tasks: Vec<PathPlanTask>,
    #[serde(default)]
    pub verification_queries: Vec<PlanVerificationQuery>,
    /// Topological execution stages. Tasks in one stage have no dependency
    /// between them and may run in parallel.
    #[serde(default)]
    pub parallel_groups: Vec<Vec<String>>,
    #[serde(default)]
    pub rollback: Vec<String>,
    pub confidence: f32,
    pub approval_gates: Vec<String>,
    pub warnings: Vec<String>,
}

impl PathPlan {
    pub fn task(&self, id: &str) -> Option<&PathPlanTask> {
        self.tasks.iter().find(|task| task.id == id)
    }

    pub fn stage_count(&self) -> usize {
        self.parallel_groups.len()
    }

    pub fn apply_feedback(&self, observations: &[PlanTaskObservation]) -> PlanExecutionReport {
        apply_plan_feedback(self, observations)
    }
}

fn stable_plan_hash(parts: impl IntoIterator<Item = String>) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for byte in part.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn parallel_groups(tasks: &[PathPlanTask]) -> (Vec<Vec<String>>, bool) {
    let mut remaining: BTreeSet<String> = tasks.iter().map(|task| task.id.clone()).collect();
    let known: HashSet<String> = remaining.iter().cloned().collect();
    let mut completed = HashSet::new();
    let mut groups = Vec::new();
    while !remaining.is_empty() {
        let ready: Vec<String> = remaining
            .iter()
            .filter(|id| {
                tasks
                    .iter()
                    .find(|task| &task.id == *id)
                    .is_some_and(|task| {
                        task.depends_on.iter().all(|dependency| {
                            completed.contains(dependency) || !known.contains(dependency)
                        })
                    })
            })
            .cloned()
            .collect();
        if ready.is_empty() {
            return (groups, false);
        }
        for id in &ready {
            remaining.remove(id);
            completed.insert(id.clone());
        }
        groups.push(ready);
    }
    (groups, true)
}

/// Replay observed task outcomes against a plan DAG.
///
/// This is deterministic feedback, not autonomous execution: it blocks
/// downstream tasks, exposes the next runnable stages, and emits recovery
/// tasks for the caller to approve or dispatch.
pub fn apply_plan_feedback(
    plan: &PathPlan,
    observations: &[PlanTaskObservation],
) -> PlanExecutionReport {
    let tasks: BTreeMap<&str, &PathPlanTask> = plan
        .tasks
        .iter()
        .map(|task| (task.id.as_str(), task))
        .collect();
    let mut states: BTreeMap<String, PlanTaskState> = plan
        .tasks
        .iter()
        .map(|task| (task.id.clone(), PlanTaskState::Pending))
        .collect();
    let mut notes = BTreeMap::new();
    let mut warnings = Vec::new();
    for observation in observations {
        if !tasks.contains_key(observation.task_id.as_str()) {
            warnings.push(format!("unknown plan task '{}'", observation.task_id));
            continue;
        }
        let state = match observation.outcome {
            PlanTaskOutcome::Succeeded => PlanTaskState::Succeeded,
            PlanTaskOutcome::Failed => PlanTaskState::Failed,
            PlanTaskOutcome::Blocked => PlanTaskState::Blocked,
            PlanTaskOutcome::Skipped => PlanTaskState::Skipped,
        };
        states.insert(observation.task_id.clone(), state);
        if let Some(note) = &observation.note {
            notes.insert(observation.task_id.clone(), note.clone());
        }
    }

    loop {
        let mut changed = false;
        for task in &plan.tasks {
            if states.get(&task.id) != Some(&PlanTaskState::Pending) {
                continue;
            }
            let blocker = task.depends_on.iter().find(|dependency| {
                matches!(
                    states.get(*dependency),
                    Some(PlanTaskState::Failed | PlanTaskState::Blocked | PlanTaskState::Skipped)
                )
            });
            if let Some(blocker) = blocker {
                states.insert(task.id.clone(), PlanTaskState::Blocked);
                notes.insert(
                    task.id.clone(),
                    format!("blocked by unfinished dependency {blocker}"),
                );
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut schedulable: HashSet<String> = states
        .iter()
        .filter(|(_, state)| matches!(state, PlanTaskState::Succeeded | PlanTaskState::Skipped))
        .map(|(id, _)| id.clone())
        .collect();
    let mut ready_groups = Vec::new();
    loop {
        let ready: Vec<String> = plan
            .tasks
            .iter()
            .filter(|task| {
                states.get(&task.id) == Some(&PlanTaskState::Pending)
                    && !schedulable.contains(&task.id)
                    && task.depends_on.iter().all(|dependency| {
                        schedulable.contains(dependency) || !states.contains_key(dependency)
                    })
            })
            .map(|task| task.id.clone())
            .collect();
        if ready.is_empty() {
            break;
        }
        for id in &ready {
            schedulable.insert(id.clone());
        }
        ready_groups.push(ready);
    }
    if plan
        .tasks
        .iter()
        .any(|task| states.get(&task.id) == Some(&PlanTaskState::Pending))
        && ready_groups.is_empty()
    {
        warnings.push("no runnable tasks; plan dependency cycle or all tasks blocked".to_string());
    }

    let mut recovery_tasks = Vec::new();
    let mut recommendations = Vec::new();
    for task in &plan.tasks {
        if states.get(&task.id) != Some(&PlanTaskState::Failed) {
            continue;
        }
        let mut recovery = task.clone();
        recovery.id = format!("R{}", task.id);
        recovery.depends_on = task.depends_on.clone();
        recovery.confidence = (task.confidence * 0.9).clamp(0.0, 1.0);
        if task.action_type == PlanActionType::Verify {
            recovery.title = format!("Retry: {}", task.title);
            recovery.action_type = PlanActionType::Verify;
            recovery.risk = PlanRisk::Low;
            recovery.approval = false;
            recommendations.push(format!(
                "retry verification {} with a fresh observation",
                task.id
            ));
        } else {
            recovery.title = format!("Resolve failed task: {}", task.title);
            recovery.action_type = PlanActionType::AwaitHuman;
            recovery.risk = PlanRisk::High;
            recovery.approval = false;
            recommendations.push(format!(
                "request human decision before retrying {}",
                task.id
            ));
        }
        recovery_tasks.push(recovery);
    }

    let rollback: Vec<String> = plan
        .rollback
        .iter()
        .filter(|entry| {
            entry
                .split_once(':')
                .map(|(id, _)| id.trim())
                .is_some_and(|id| {
                    matches!(
                        states.get(id),
                        Some(PlanTaskState::Failed | PlanTaskState::Blocked)
                    )
                })
        })
        .cloned()
        .collect();
    let blocked = states.values().any(|state| {
        matches!(
            state,
            PlanTaskState::Failed | PlanTaskState::Blocked
        )
    });
    let done = !plan.tasks.is_empty()
        && states
            .values()
            .all(|state| matches!(state, PlanTaskState::Succeeded | PlanTaskState::Skipped));
    PlanExecutionReport {
        plan_id: plan.plan_id.clone(),
        states,
        notes,
        ready_groups,
        recovery_tasks,
        rollback,
        recommendations,
        warnings,
        done,
        blocked,
    }
}

const MAX_PLAN_TASKS: usize = 20;
const MAX_PRECONDITION_DEPTH: usize = 2;

fn is_precondition_relation(relation: &str) -> bool {
    matches!(relation, REL_DEPENDS_ON | REL_BLOCKED_BY | "requires")
}

#[allow(clippy::too_many_arguments)]
fn collect_precondition_tasks(
    graph: &EngineeringGraph,
    path_edges: &HashSet<&str>,
    node: &str,
    remaining_depth: usize,
    main_task_count: usize,
    tasks: &mut Vec<PathPlanTask>,
    created: &mut BTreeMap<String, String>,
    verification_queries: &mut Vec<PlanVerificationQuery>,
    truncated: &mut bool,
) -> Vec<String> {
    if remaining_depth == 0 {
        return Vec::new();
    }
    let mut dependencies = Vec::new();
    for relation in graph.incoming(node) {
        if path_edges.contains(relation.id.as_str())
            || !is_precondition_relation(&relation.relation)
        {
            continue;
        }
        if let Some(id) = created.get(&relation.id) {
            dependencies.push(id.clone());
            continue;
        }
        if main_task_count + tasks.len() >= MAX_PLAN_TASKS {
            *truncated = true;
            break;
        }
        let depends_on = collect_precondition_tasks(
            graph,
            path_edges,
            &relation.from,
            remaining_depth - 1,
            main_task_count,
            tasks,
            created,
            verification_queries,
            truncated,
        );
        if main_task_count + tasks.len() >= MAX_PLAN_TASKS {
            *truncated = true;
            break;
        }
        let id = format!("P{}", tasks.len() + 1);
        let (_, _, success_check) = template(&relation.relation);
        let from_label = graph
            .entity(&relation.from)
            .map(|entity| entity.name.clone())
            .unwrap_or_else(|| relation.from.clone());
        let to_label = graph
            .entity(&relation.to)
            .map(|entity| entity.name.clone())
            .unwrap_or_else(|| relation.to.clone());
        let confidence = if relation.confidence.is_finite() {
            relation.confidence.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let evidence_ref = format!("path/precondition/{}", relation.id);
        tasks.push(PathPlanTask {
            id: id.clone(),
            action_type: PlanActionType::Verify,
            title: format!(
                "Verify prerequisite {to_label} via {} from {from_label}",
                relation.relation
            ),
            relation: relation.relation.clone(),
            from: relation.from.clone(),
            to: relation.to.clone(),
            depends_on,
            success_check: success_check.to_string(),
            evidence_ref: evidence_ref.clone(),
            risk: PlanRisk::Low,
            approval: false,
            confidence,
            preconditions: vec![format!("{from_label} exists")],
            postconditions: vec![format!("{to_label} satisfies '{success_check}'")],
        });
        verification_queries.push(PlanVerificationQuery {
            id: format!("VQ-{id}"),
            task_id: id.clone(),
            query_ref: format!("path/precondition/{}/verify", relation.id),
            expr: success_check.to_string(),
            evidence_ref,
        });
        created.insert(relation.id.clone(), id.clone());
        dependencies.push(id);
    }
    dependencies
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
/// side effect. Path order defines the safe data dependencies; `parallel_groups`
/// exposes independent stages so callers can schedule them concurrently.
pub fn build_path_plan(graph: &EngineeringGraph, path: &EngineeringPath) -> PathPlan {
    let mut tasks = Vec::with_capacity(path.relations.len());
    let mut precondition_tasks = Vec::new();
    let mut verification_queries = Vec::with_capacity(path.relations.len());
    let mut rollback = Vec::new();
    let mut confidence = path.confidence;
    let mut previous: Option<String> = None;
    let path_edges: HashSet<&str> = path.relations.iter().map(String::as_str).collect();
    let mut created_preconditions = BTreeMap::new();
    let mut closure_truncated = false;
    let source_path_id = format!(
        "PATH-{}",
        stable_plan_hash(
            path.nodes
                .iter()
                .cloned()
                .chain(path.relations.iter().cloned())
        )
    );
    let plan_id = format!("PLAN-{}", stable_plan_hash([source_path_id.clone()]));

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
        let mut depends_on: Vec<String> = previous.iter().cloned().collect();
        depends_on.extend(collect_precondition_tasks(
            graph,
            &path_edges,
            &path.nodes[index],
            MAX_PRECONDITION_DEPTH,
            path.relations.len(),
            &mut precondition_tasks,
            &mut created_preconditions,
            &mut verification_queries,
            &mut closure_truncated,
        ));
        let approval = action.has_side_effects() && risk != PlanRisk::Low;
        let from_class = graph
            .entity(&path.nodes[index])
            .map(|entity| entity.class.clone())
            .unwrap_or_default();
        let precondition = if from_class.is_empty() {
            format!("{from_label} exists")
        } else {
            format!("{from_class}({from_label}) exists")
        };
        let postcondition = format!("{to_label} satisfies '{success_check}'");
        let evidence_ref = format!("path/edge/{index}");
        tasks.push(PathPlanTask {
            id: id.clone(),
            action_type: action,
            title: task_title(action, &relation.relation, &from_label, &to_label),
            relation: relation.relation.clone(),
            from: path.nodes.get(index).cloned().unwrap_or_default(),
            to: path.nodes.get(index + 1).cloned().unwrap_or_default(),
            depends_on,
            success_check: success_check.to_string(),
            evidence_ref: evidence_ref.clone(),
            risk,
            approval,
            confidence: task_confidence,
            preconditions: vec![precondition],
            postconditions: vec![postcondition],
        });
        verification_queries.push(PlanVerificationQuery {
            id: format!("VQ-{}", index + 1),
            task_id: id.clone(),
            query_ref: format!("path/edge/{index}/verify"),
            expr: success_check.to_string(),
            evidence_ref,
        });
        if action.has_side_effects() {
            rollback.push(format!("{id}: revert {} on {to_label}", relation.relation));
        }
        previous = Some(id);
    }
    tasks.extend(precondition_tasks);
    confidence = tasks
        .iter()
        .fold(confidence, |current, task| current.min(task.confidence));

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
    if closure_truncated {
        warnings.push(format!(
            "precondition closure truncated at {MAX_PLAN_TASKS} tasks; split before execution"
        ));
    }
    let (groups, acyclic) = parallel_groups(&tasks);
    if !acyclic {
        warnings.push("plan dependency cycle detected; split before execution".to_string());
    }
    let approval_gates: Vec<String> = tasks
        .iter()
        .filter(|task| task.approval)
        .map(|task| task.id.clone())
        .collect();
    PathPlan {
        plan_id,
        source_path_id,
        goal: path
            .nodes
            .last()
            .and_then(|id| graph.entity(id))
            .map(|entity| format!("Reach {}", entity.name))
            .unwrap_or_else(|| "Reach goal".to_string()),
        tasks,
        verification_queries,
        parallel_groups: groups,
        rollback,
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
        assert!(plan.plan_id.starts_with("PLAN-"));
        assert!(plan.source_path_id.starts_with("PATH-"));
        assert_eq!(plan.goal, "Reach Patch");
        assert_eq!(plan.verification_queries.len(), 2);
        assert_eq!(plan.verification_queries[0].task_id, "T1");
        assert_eq!(
            plan.parallel_groups,
            vec![vec!["T1".to_string()], vec!["T2".to_string()]]
        );
        assert_eq!(plan.stage_count(), 2);
        assert_eq!(plan.rollback.len(), 1);
        assert!(!plan.tasks[0].preconditions.is_empty());
        assert!(!plan.tasks[0].postconditions.is_empty());
    }

    #[test]
    fn precondition_closure_adds_parallel_verify_tasks() {
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(EngineeringEntity::new(CLASS_TASK, "task:t0", "Prepare"));
        graph.insert_entity(EngineeringEntity::new(CLASS_TASK, "task:t1", "Fix build"));
        graph.insert_entity(EngineeringEntity::new(CLASS_ERROR, "error:e1", "E0425"));
        graph.insert_entity(EngineeringEntity::new(CLASS_CHANGE, "change:c1", "Patch"));
        graph.insert_relation(EngineeringRelation::new("task:t0", REL_DEPENDS_ON, "task:t1"));
        graph.insert_relation(EngineeringRelation::new("task:t1", REL_BLOCKED_BY, "error:e1"));
        graph.insert_relation(EngineeringRelation::new("error:e1", REL_FIXES, "change:c1"));
        let path = graph
            .weighted_shortest_path("task:t1", "change:c1", 3, &[])
            .expect("path");
        let plan = build_path_plan(&graph, &path);

        assert_eq!(plan.tasks.len(), 3, "{plan:?}");
        let precondition = plan
            .tasks
            .iter()
            .find(|task| task.id.starts_with('P'))
            .expect("precondition task");
        assert_eq!(precondition.action_type, PlanActionType::Verify);
        assert_eq!(precondition.relation, REL_DEPENDS_ON);
        assert_eq!(plan.verification_queries.len(), 3);
        assert!(
            plan.task("T1")
                .expect("T1")
                .depends_on
                .contains(&precondition.id)
        );
        assert!(plan.parallel_groups[0].contains(&precondition.id));
    }

    #[test]
    fn precondition_closure_stops_at_task_budget() {
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(EngineeringEntity::new(CLASS_TASK, "task:t1", "Fix build"));
        graph.insert_entity(EngineeringEntity::new(CLASS_ERROR, "error:e1", "E0425"));
        graph.insert_entity(EngineeringEntity::new(CLASS_CHANGE, "change:c1", "Patch"));
        for index in 0..25 {
            let id = format!("task:p{index}");
            graph.insert_entity(EngineeringEntity::new(CLASS_TASK, &id, &id));
            graph.insert_relation(EngineeringRelation::new(&id, REL_DEPENDS_ON, "task:t1"));
        }
        graph.insert_relation(EngineeringRelation::new("task:t1", REL_BLOCKED_BY, "error:e1"));
        graph.insert_relation(EngineeringRelation::new("error:e1", REL_FIXES, "change:c1"));
        let path = graph
            .weighted_shortest_path("task:t1", "change:c1", 3, &[])
            .expect("path");
        let plan = build_path_plan(&graph, &path);
        assert_eq!(plan.tasks.len(), MAX_PLAN_TASKS);
        assert!(
            plan.warnings
                .iter()
                .any(|warning| warning.contains("precondition closure truncated"))
        );
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

    #[test]
    fn parallel_groups_expose_independent_tasks_and_detect_cycles() {
        let task = |id: &str, depends_on: &[&str]| PathPlanTask {
            id: id.to_string(),
            action_type: PlanActionType::Verify,
            title: id.to_string(),
            relation: REL_BLOCKED_BY.to_string(),
            from: "a".to_string(),
            to: "b".to_string(),
            depends_on: depends_on.iter().map(|id| (*id).to_string()).collect(),
            success_check: "ok".to_string(),
            evidence_ref: "path/edge/0".to_string(),
            risk: PlanRisk::Low,
            approval: false,
            confidence: 1.0,
            preconditions: Vec::new(),
            postconditions: Vec::new(),
        };
        let (groups, acyclic) = parallel_groups(&[
            task("T1", &[]),
            task("T2", &[]),
            task("T3", &["T1", "T2"]),
        ]);
        assert!(acyclic);
        assert_eq!(
            groups,
            vec![
                vec!["T1".to_string(), "T2".to_string()],
                vec!["T3".to_string()]
            ]
        );
        let (_, acyclic) = parallel_groups(&[task("T1", &["T2"]), task("T2", &["T1"])]);
        assert!(!acyclic);
    }

    #[test]
    fn feedback_blocks_downstream_and_emits_recovery_task() {
        let (graph, path) = chain();
        let plan = build_path_plan(&graph, &path);
        let report = plan.apply_feedback(&[PlanTaskObservation {
            task_id: "T1".to_string(),
            outcome: PlanTaskOutcome::Failed,
            note: Some("blocker still open".to_string()),
        }]);
        assert_eq!(report.states["T1"], PlanTaskState::Failed);
        assert_eq!(report.states["T2"], PlanTaskState::Blocked);
        assert!(report.blocked);
        assert!(!report.done);
        assert!(report.ready_groups.is_empty());
        assert_eq!(report.recovery_tasks.len(), 1);
        assert_eq!(report.recovery_tasks[0].id, "RT1");
        assert_eq!(
            report.recovery_tasks[0].action_type,
            PlanActionType::Verify
        );
        assert!(
            report
                .recommendations
                .iter()
                .any(|action| action.contains("retry verification T1"))
        );
        assert_eq!(
            report.notes.get("T1").map(String::as_str),
            Some("blocker still open")
        );
    }

    #[test]
    fn feedback_requests_human_and_rollback_for_failed_side_effect() {
        let (graph, path) = chain();
        let plan = build_path_plan(&graph, &path);
        let report = plan.apply_feedback(&[
            PlanTaskObservation {
                task_id: "T1".to_string(),
                outcome: PlanTaskOutcome::Succeeded,
                note: None,
            },
            PlanTaskObservation {
                task_id: "T2".to_string(),
                outcome: PlanTaskOutcome::Failed,
                note: None,
            },
        ]);
        assert_eq!(report.recovery_tasks[0].action_type, PlanActionType::AwaitHuman);
        assert_eq!(report.rollback.len(), 1);
        assert!(report.rollback[0].starts_with("T2:"));
        assert!(
            report
                .recommendations
                .iter()
                .any(|action| action.contains("human decision"))
        );
    }

    #[test]
    fn feedback_reports_ready_stage_and_completion() {
        let (graph, path) = chain();
        let plan = build_path_plan(&graph, &path);
        let partial = plan.apply_feedback(&[PlanTaskObservation {
            task_id: "T1".to_string(),
            outcome: PlanTaskOutcome::Succeeded,
            note: None,
        }]);
        assert_eq!(partial.ready_groups, vec![vec!["T2".to_string()]]);
        assert!(!partial.done);

        let finished = plan.apply_feedback(&[
            PlanTaskObservation {
                task_id: "T1".to_string(),
                outcome: PlanTaskOutcome::Succeeded,
                note: None,
            },
            PlanTaskObservation {
                task_id: "T2".to_string(),
                outcome: PlanTaskOutcome::Skipped,
                note: None,
            },
            PlanTaskObservation {
                task_id: "missing".to_string(),
                outcome: PlanTaskOutcome::Succeeded,
                note: None,
            },
        ]);
        assert!(finished.done);
        assert!(!finished.blocked);
        assert!(finished.ready_groups.is_empty());
        assert!(
            finished
                .warnings
                .iter()
                .any(|warning| warning.contains("unknown plan task"))
        );
    }
}
