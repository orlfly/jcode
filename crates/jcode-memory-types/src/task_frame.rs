//! TaskFrame IR for stage S1 of the NL -> PathQuery pipeline (design 4.3.2).
//!
//! S1 is an LLM extraction step with a JSON schema. This module owns the
//! deterministic half: validate the model-filled frame, drop unknown fields
//! loudly, and map anchors/scope/intent onto a [`PathQuery`]. It never guesses
//! entity ids and never executes a query.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::path_query::{PathObjective, PathQuery, PathQueryConstraints, PathQueryScope};

/// Task intents from design table 4-2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskIntent {
    Attribution,
    Impact,
    Plan,
    Compliance,
    Alternative,
}

impl TaskIntent {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Attribution => "attribution",
            Self::Impact => "impact",
            Self::Plan => "plan",
            Self::Compliance => "compliance",
            Self::Alternative => "alternative",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "attribution" | "cause" | "why" => Some(Self::Attribution),
            "impact" | "downstream" => Some(Self::Impact),
            "plan" | "how" => Some(Self::Plan),
            "compliance" | "comply" | "boolean" => Some(Self::Compliance),
            "alternative" | "optimal" | "best" => Some(Self::Alternative),
            _ => None,
        }
    }
}

/// Role an anchor mention plays in the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskAnchorRole {
    Start,
    Target,
    Waypoint,
}

impl TaskAnchorRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Target => "target",
            Self::Waypoint => "waypoint",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "start" | "from" | "source" => Some(Self::Start),
            "target" | "to" | "goal" => Some(Self::Target),
            "waypoint" | "via" => Some(Self::Waypoint),
            _ => None,
        }
    }
}

/// One extracted anchor mention with its optional source span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskAnchor {
    pub role: TaskAnchorRole,
    pub mention: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<(usize, usize)>,
}

/// One hard or soft constraint expression.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskConstraint {
    pub expr: String,
    #[serde(default)]
    pub hard: bool,
}

/// S1 output: what the task wants, not how to query it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskFrame {
    pub task_id: String,
    pub intent: TaskIntent,
    pub anchors: Vec<TaskAnchor>,
    #[serde(default)]
    pub constraints: Vec<TaskConstraint>,
    #[serde(default)]
    pub success_criteria: Vec<String>,
    #[serde(default)]
    pub scope_hint: Vec<String>,
    #[serde(default = "default_output_type")]
    pub output_type: String,
}

fn default_output_type() -> String {
    "decision_support".to_string()
}

/// Validated frame plus the PathQuery it maps onto.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskFrameCompileReport {
    pub frame: TaskFrame,
    pub path_query: PathQuery,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub dropped_fields: Vec<String>,
}

const FRAME_FIELDS: [&str; 7] = [
    "task_id",
    "intent",
    "anchors",
    "constraints",
    "success_criteria",
    "scope_hint",
    "output_type",
];
const ANCHOR_FIELDS: [&str; 3] = ["role", "mention", "span"];
const CONSTRAINT_FIELDS: [&str; 2] = ["expr", "hard"];
const OUTPUT_TYPES: [&str; 4] = ["decision_support", "path_list", "boolean", "ranking"];

fn non_empty_str(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

/// Validate a model-filled TaskFrame and map it onto a PathQuery.
///
/// Unknown fields are reported, not silently ignored. Structural errors return
/// the full list so the caller can ask the model for a targeted repair instead
/// of guessing which field failed.
pub fn compile_task_frame_json(value: &Value) -> Result<TaskFrameCompileReport, Vec<String>> {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut dropped_fields = Vec::new();

    let Some(root) = value.as_object() else {
        return Err(vec!["task frame JSON must be an object".to_string()]);
    };
    let root = if root.get("intent").is_none() && root.get("task_frame").is_some() {
        match root.get("task_frame").and_then(Value::as_object) {
            Some(inner) => inner,
            None => return Err(vec!["task_frame wrapper must be an object".to_string()]),
        }
    } else {
        root
    };
    for key in root.keys() {
        if key != "task_frame" && !FRAME_FIELDS.contains(&key.as_str()) {
            dropped_fields.push(key.clone());
            warnings.push(format!("dropped unsupported field: {key}"));
        }
    }

    let task_id = root
        .get("task_id")
        .and_then(non_empty_str)
        .unwrap_or_else(|| {
            warnings.push("task_id missing; caller should keep a stable id".to_string());
            String::new()
        });

    let intent = match root.get("intent").and_then(Value::as_str) {
        Some(raw) => match TaskIntent::parse(raw) {
            Some(intent) => intent,
            None => {
                errors.push(format!(
                    "intent must be attribution, impact, plan, compliance, or alternative (got {raw})"
                ));
                TaskIntent::Plan
            }
        },
        None => {
            errors.push("intent is required".to_string());
            TaskIntent::Plan
        }
    };

    let output_type = match root.get("output_type") {
        Some(value) => {
            let text = non_empty_str(value).unwrap_or_default();
            if OUTPUT_TYPES.contains(&text.as_str()) {
                text
            } else {
                errors.push(format!(
                    "output_type must be one of {} (got {text})",
                    OUTPUT_TYPES.join(", ")
                ));
                default_output_type()
            }
        }
        None => default_output_type(),
    };

    let mut anchors = Vec::new();
    match root.get("anchors") {
        Some(Value::Array(items)) => {
            for (index, item) in items.iter().enumerate() {
                let Some(object) = item.as_object() else {
                    errors.push(format!("anchors[{index}] must be an object"));
                    continue;
                };
                for key in object.keys() {
                    if !ANCHOR_FIELDS.contains(&key.as_str()) {
                        dropped_fields.push(format!("anchors[{index}].{key}"));
                        warnings.push(format!(
                            "dropped unsupported anchor field: anchors[{index}].{key}"
                        ));
                    }
                }
                let role = match object.get("role").and_then(Value::as_str) {
                    Some(raw) => match TaskAnchorRole::parse(raw) {
                        Some(role) => role,
                        None => {
                            errors.push(format!(
                                "anchors[{index}].role must be start, target, or waypoint (got {raw})"
                            ));
                            continue;
                        }
                    },
                    None => {
                        errors.push(format!("anchors[{index}].role is required"));
                        continue;
                    }
                };
                let mention = match object.get("mention").and_then(non_empty_str) {
                    Some(mention) => mention,
                    None => {
                        errors.push(format!(
                            "anchors[{index}].mention must be a non-empty string"
                        ));
                        continue;
                    }
                };
                let span = match object.get("span") {
                    Some(value) => match value.as_array() {
                        Some(parts) if parts.len() == 2 => {
                            let start = parts[0].as_u64().and_then(|v| usize::try_from(v).ok());
                            let end = parts[1].as_u64().and_then(|v| usize::try_from(v).ok());
                            match (start, end) {
                                (Some(start), Some(end)) if start <= end => Some((start, end)),
                                _ => {
                                    errors.push(format!(
                                        "anchors[{index}].span must be [start, end] with start <= end"
                                    ));
                                    None
                                }
                            }
                        }
                        _ => {
                            errors.push(format!("anchors[{index}].span must be [start, end]"));
                            None
                        }
                    },
                    None => None,
                };
                anchors.push(TaskAnchor {
                    role,
                    mention,
                    span,
                });
            }
        }
        Some(_) => errors.push("anchors must be an array".to_string()),
        None => errors.push("anchors is required".to_string()),
    }

    let mut constraints = Vec::new();
    match root.get("constraints") {
        Some(Value::Array(items)) => {
            for (index, item) in items.iter().enumerate() {
                let Some(object) = item.as_object() else {
                    errors.push(format!("constraints[{index}] must be an object"));
                    continue;
                };
                for key in object.keys() {
                    if !CONSTRAINT_FIELDS.contains(&key.as_str()) {
                        dropped_fields.push(format!("constraints[{index}].{key}"));
                        warnings.push(format!(
                            "dropped unsupported constraint field: constraints[{index}].{key}"
                        ));
                    }
                }
                let expr = match object.get("expr").and_then(non_empty_str) {
                    Some(expr) => expr,
                    None => {
                        errors.push(format!("constraints[{index}].expr must be non-empty"));
                        continue;
                    }
                };
                let hard = object.get("hard").and_then(Value::as_bool).unwrap_or(false);
                constraints.push(TaskConstraint { expr, hard });
            }
        }
        Some(_) => errors.push("constraints must be an array".to_string()),
        None => {}
    }

    let success_criteria = match root.get("success_criteria") {
        Some(Value::Array(items)) => items.iter().filter_map(non_empty_str).collect(),
        Some(_) => {
            errors.push("success_criteria must be an array of strings".to_string());
            Vec::new()
        }
        None => Vec::new(),
    };

    let mut scope_hint = Vec::new();
    if let Some(value) = root.get("scope_hint") {
        match value.as_array() {
            Some(items) => {
                for item in items {
                    match item.as_str().map(str::trim) {
                        Some("project") => scope_hint.push("project".to_string()),
                        Some("global") => scope_hint.push("global".to_string()),
                        _ => {
                            warnings.push(format!("dropped unsupported scope hint: {item}"));
                        }
                    }
                }
            }
            None => errors.push("scope_hint must be an array".to_string()),
        }
    }

    let start: Vec<String> = anchors
        .iter()
        .filter(|anchor| anchor.role == TaskAnchorRole::Start)
        .map(|anchor| anchor.mention.clone())
        .collect();
    let goal: Vec<String> = anchors
        .iter()
        .filter(|anchor| anchor.role == TaskAnchorRole::Target)
        .map(|anchor| anchor.mention.clone())
        .collect();
    let waypoints: Vec<String> = anchors
        .iter()
        .filter(|anchor| anchor.role == TaskAnchorRole::Waypoint)
        .map(|anchor| anchor.mention.clone())
        .collect();
    if start.is_empty() {
        errors.push("anchors must include at least one start mention".to_string());
    }
    if goal.is_empty() {
        errors.push("anchors must include at least one target mention".to_string());
    }

    for constraint in &constraints {
        warnings.push(format!(
            "constraint deferred to S4/S6 ({}): {}",
            if constraint.hard { "hard" } else { "soft" },
            constraint.expr
        ));
    }
    let scope = if scope_hint.iter().any(|hint| hint == "global") {
        if scope_hint.iter().any(|hint| hint == "project") {
            PathQueryScope::All
        } else {
            PathQueryScope::Global
        }
    } else {
        PathQueryScope::Project
    };
    let (objective, limit) = match intent {
        TaskIntent::Attribution => (Some(PathObjective::MaxConfidence), Some(5)),
        TaskIntent::Impact => (None, Some(10)),
        TaskIntent::Plan => (Some(PathObjective::MinCost), None),
        TaskIntent::Compliance => (None, Some(3)),
        TaskIntent::Alternative => (Some(PathObjective::MaxConfidence), Some(5)),
    };
    let frame = TaskFrame {
        task_id,
        intent,
        anchors,
        constraints,
        success_criteria,
        scope_hint,
        output_type,
    };
    let path_query = PathQuery {
        start,
        goal,
        waypoints,
        graph: None,
        constraints: PathQueryConstraints {
            objective,
            limit,
            scope: Some(scope),
            ..PathQueryConstraints::default()
        },
    };
    if errors.is_empty()
        && let Err(query_errors) = path_query.validate()
    {
        errors.extend(query_errors);
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(TaskFrameCompileReport {
        frame,
        path_query,
        warnings,
        dropped_fields,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn full_frame() -> Value {
        json!({
            "task_id": "TASK-1",
            "intent": "plan",
            "anchors": [
                {"role": "start", "mention": "README", "span": [0, 6]},
                {"role": "target", "mention": "CI", "span": [10, 12]},
                {"role": "waypoint", "mention": "build"}
            ],
            "constraints": [
                {"expr": "tests pass", "hard": true},
                {"expr": "lead time <= 30"}
            ],
            "success_criteria": ["list tasks"],
            "scope_hint": ["project", "global"],
            "output_type": "decision_support"
        })
    }

    #[test]
    fn maps_valid_frame_onto_path_query() {
        let report = compile_task_frame_json(&full_frame()).expect("compile");
        assert_eq!(report.frame.intent, TaskIntent::Plan);
        assert_eq!(report.path_query.start, vec!["README".to_string()]);
        assert_eq!(report.path_query.goal, vec!["CI".to_string()]);
        assert_eq!(report.path_query.waypoints, vec!["build".to_string()]);
        assert_eq!(
            report.path_query.constraints.objective,
            Some(PathObjective::MinCost)
        );
        assert_eq!(
            report.path_query.constraints.scope,
            Some(PathQueryScope::All)
        );
        assert_eq!(report.frame.anchors[0].span, Some((0, 6)));
        assert_eq!(
            report
                .warnings
                .iter()
                .filter(|warning| warning.contains("deferred to S4/S6"))
                .count(),
            2
        );
    }

    #[test]
    fn drops_unknown_fields_and_scope_hints_loudly() {
        let mut frame = full_frame();
        frame["graph"] = json!("engineering");
        frame["anchors"][0]["id"] = json!("file:readme");
        frame["scope_hint"] = json!(["project", "tenant"]);
        let report = compile_task_frame_json(&frame).expect("compile");
        assert!(report.dropped_fields.contains(&"graph".to_string()));
        assert!(report.dropped_fields.contains(&"anchors[0].id".to_string()));
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.contains("unsupported scope hint"))
        );
        assert_eq!(
            report.path_query.constraints.scope,
            Some(PathQueryScope::Project)
        );
    }

    #[test]
    fn rejects_invalid_intent_role_output_and_anchor_shapes() {
        let mut frame = full_frame();
        frame["intent"] = json!("teleport");
        frame["output_type"] = json!("poem");
        frame["anchors"][0]["role"] = json!("origin");
        frame["anchors"][1]["span"] = json!([9, 3]);
        let errors = compile_task_frame_json(&frame).expect_err("must fail");
        for needle in [
            "intent must be",
            "output_type must be",
            "anchors[0].role",
            "span must be",
        ] {
            assert!(
                errors.iter().any(|error| error.contains(needle)),
                "missing {needle}: {errors:?}"
            );
        }
    }

    #[test]
    fn requires_start_and_target_mentions() {
        let frame = json!({
            "task_id": "TASK-2",
            "intent": "impact",
            "anchors": [{"role": "start", "mention": "A"}]
        });
        let errors = compile_task_frame_json(&frame).expect_err("must fail");
        assert!(
            errors.iter().any(|error| error.contains("target mention")),
            "{errors:?}"
        );
    }

    #[test]
    fn accepts_wrapped_frame_and_alternative_intent() {
        let frame = json!({
            "task_frame": {
                "task_id": "TASK-3",
                "intent": "alternative",
                "anchors": [
                    {"role": "start", "mention": "CHIP-X"},
                    {"role": "target", "mention": "supplier"}
                ],
                "scope_hint": ["global"]
            }
        });
        let report = compile_task_frame_json(&frame).expect("compile");
        assert_eq!(report.frame.intent, TaskIntent::Alternative);
        assert_eq!(
            report.path_query.constraints.scope,
            Some(PathQueryScope::Global)
        );
        assert_eq!(report.path_query.constraints.limit, Some(5));
    }
}
