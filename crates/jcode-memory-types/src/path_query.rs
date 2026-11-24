//! Structured PathQuery intermediate representation.
//!
//! This is the machine-readable contract that an LLM or external client fills
//! before the deterministic graph engine runs. It intentionally contains no
//! natural-language parser: the compiler stages can emit this object and the
//! existing path engines validate and execute it.

use serde::{Deserialize, Serialize};

use crate::graph::EdgeKind;

/// Which persisted graph a query targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathGraphKind {
    Memory,
    Engineering,
    Ontology,
}

impl PathGraphKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Memory => "memory",
            Self::Engineering => "engineering",
            Self::Ontology => "ontology",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "memory" => Some(Self::Memory),
            "engineering" => Some(Self::Engineering),
            "ontology" => Some(Self::Ontology),
            _ => None,
        }
    }
}

/// Memory store scope for a PathQuery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathQueryScope {
    Project,
    Global,
    All,
}

impl PathQueryScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Project => "project",
            Self::Global => "global",
            Self::All => "all",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "project" => Some(Self::Project),
            "global" => Some(Self::Global),
            "all" => Some(Self::All),
            _ => None,
        }
    }
}

/// Cost-versus-confidence ordering after constraints are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathObjective {
    MinCost,
    MaxConfidence,
}

impl PathObjective {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MinCost => "min_cost",
            Self::MaxConfidence => "max_confidence",
        }
    }
}

/// Query constraints after entity linking.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PathQueryConstraints {
    #[serde(default)]
    pub relations: Vec<String>,
    #[serde(default)]
    pub deny_relations: Vec<String>,
    #[serde(default)]
    pub min_confidence: Option<f32>,
    #[serde(default)]
    pub objective: Option<PathObjective>,
    #[serde(default)]
    pub depth: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub scope: Option<PathQueryScope>,
    #[serde(default)]
    pub include_hypotheses: bool,
    /// For engineering paths: join the ontology concept layer at query time.
    #[serde(default)]
    pub include_ontology: bool,
    /// Wall-clock budget for the path query and its L0 retries.
    #[serde(default)]
    pub query_budget_ms: Option<u64>,
    /// Minimum anchor-grounding score. L0 lowers this to 0.5 and reports E06
    /// when the requested threshold rejects every useful candidate.
    #[serde(default)]
    pub semantic_min_score: Option<f32>,
    #[serde(default)]
    pub auto_ground: bool,
    /// Propose L4 stitched-path gaps as pending shadow edges. Pending edges
    /// never enter the stored graph until reviewed.
    #[serde(default)]
    pub auto_propose: bool,
    #[serde(default)]
    pub plan: bool,
}

/// One validated path query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathQuery {
    pub start: Vec<String>,
    pub goal: Vec<String>,
    #[serde(default)]
    pub waypoints: Vec<String>,
    #[serde(default)]
    pub graph: Option<PathGraphKind>,
    #[serde(default)]
    pub constraints: PathQueryConstraints,
}

impl PathQuery {
    pub fn graph_kind(&self) -> PathGraphKind {
        self.graph.unwrap_or(PathGraphKind::Memory)
    }

    pub fn scope(&self) -> PathQueryScope {
        self.constraints.scope.unwrap_or(PathQueryScope::All)
    }

    pub fn depth(&self, default: usize) -> usize {
        self.constraints.depth.unwrap_or(default)
    }

    pub fn limit(&self, default: usize) -> usize {
        self.constraints.limit.unwrap_or(default)
    }

    /// Validate shape and relation names before the engine runs.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        let start: Vec<&str> = self
            .start
            .iter()
            .map(String::as_str)
            .map(str::trim)
            .filter(|anchor| !anchor.is_empty())
            .collect();
        let goal: Vec<&str> = self
            .goal
            .iter()
            .map(String::as_str)
            .map(str::trim)
            .filter(|anchor| !anchor.is_empty())
            .collect();
        if start.is_empty() {
            errors.push("start must contain at least one non-empty anchor".to_string());
        }
        if goal.is_empty() {
            errors.push("goal must contain at least one non-empty anchor".to_string());
        }
        if start.len() > 16 || goal.len() > 16 {
            errors.push("start and goal accept at most 16 anchors each".to_string());
        }
        if self.waypoints.len() > 3 {
            errors.push("waypoints accept at most 3 anchors".to_string());
        }
        if self.waypoints.iter().any(|anchor| anchor.trim().is_empty()) {
            errors.push("waypoints must not contain empty anchors".to_string());
        }
        if let Some(depth) = self.constraints.depth
            && !(1..=8).contains(&depth)
        {
            errors.push("depth must be between 1 and 8".to_string());
        }
        if let Some(limit) = self.constraints.limit
            && limit > 20
        {
            errors.push("limit must be at most 20".to_string());
        }
        if self.graph_kind() == PathGraphKind::Memory {
            for relation in self
                .constraints
                .relations
                .iter()
                .chain(self.constraints.deny_relations.iter())
            {
                let relation = relation.trim();
                if relation.is_empty() || EdgeKind::parse(relation).is_some() {
                    continue;
                }
                errors.push(format!("unknown memory relation: {relation}"));
            }
        }
        if let Some(confidence) = self.constraints.min_confidence
            && !(0.0..=1.0).contains(&confidence)
        {
            errors.push("min_confidence must be between 0.0 and 1.0".to_string());
        }
        if let Some(score) = self.constraints.semantic_min_score
            && !(0.0..=1.0).contains(&score)
        {
            errors.push("semantic_min_score must be between 0.0 and 1.0".to_string());
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

const QUERY_KEYS: [&str; 12] = [
    "from",
    "to",
    "via",
    "graph",
    "relations",
    "deny_relations",
    "scope",
    "depth",
    "limit",
    "objective",
    "budget_ms",
    "semantic_min",
];

fn query_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' | '\'' | '`' => {
                while let Some(inner) = chars.next() {
                    if inner == ch {
                        break;
                    }
                    current.push(inner);
                }
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            '[' => {
                while let Some(inner) = chars.next() {
                    if inner == ']' {
                        break;
                    }
                    current.push(inner);
                }
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            ch if ch.is_whitespace() || ch == ',' => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn is_query_key(token: &str) -> bool {
    QUERY_KEYS
        .iter()
        .any(|key| token.trim_end_matches('=').eq_ignore_ascii_case(key))
}

fn normalize_key(token: &str) -> String {
    token
        .trim_end_matches('=')
        .trim()
        .to_ascii_lowercase()
}

/// Compile a compact text form into the structured [`PathQuery`] IR.
///
/// This is the deterministic template stage of NL -> PathQuery, not an LLM
/// parser. It accepts `key value` pairs such as:
///
/// ```text
/// from "file:src/lib.rs" to "api:/task" graph engineering \
///   relations defines,implements depth 3 limit 2
/// ```
///
/// Unknown prose is ignored, relation names are checked against the supplied
/// allow-list (or the memory [`EdgeKind`] registry for memory graphs), and the
/// final query still goes through [`PathQuery::validate`].
pub fn compile_path_query_text(
    text: &str,
    allowed_relations: &[&str],
) -> Result<PathQuery, Vec<String>> {
    let tokens = query_tokens(text);
    let mut start = Vec::new();
    let mut goal = Vec::new();
    let mut waypoints = Vec::new();
    let mut relations = Vec::new();
    let mut deny_relations = Vec::new();
    let mut graph = None;
    let mut scope = None;
    let mut depth = None;
    let mut limit = None;
    let mut objective = None;
    let mut query_budget_ms = None;
    let mut semantic_min_score = None;
    let mut errors = Vec::new();

    let mut index = 0;
    while index < tokens.len() {
        let token = &tokens[index];
        if !is_query_key(token) {
            index += 1;
            continue;
        }
        let key = normalize_key(token);
        index += 1;
        let mut values = Vec::new();
        while index < tokens.len() && !is_query_key(&tokens[index]) {
            let value = tokens[index].trim();
            if value.is_empty() {
                index += 1;
                continue;
            }
            if matches!(value, "and" | "with" | "using") {
                index += 1;
                continue;
            }
            values.push(value.to_string());
            index += 1;
            if !matches!(
                key.as_str(),
                "from" | "to" | "via" | "relations" | "deny_relations"
            ) {
                break;
            }
        }
        if values.is_empty() {
            errors.push(format!("{key} requires a value"));
            continue;
        }
        match key.as_str() {
            "from" => start.extend(values),
            "to" => goal.extend(values),
            "via" => waypoints.extend(values),
            "relations" => {
                relations.extend(values.into_iter().flat_map(|value| {
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|item| !item.is_empty())
                        .map(str::to_string)
                        .collect::<Vec<_>>()
                }));
            }
            "deny_relations" => {
                deny_relations.extend(values.into_iter().flat_map(|value| {
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|item| !item.is_empty())
                        .map(str::to_string)
                        .collect::<Vec<_>>()
                }));
            }
            "graph" => match PathGraphKind::parse(&values[0]) {
                Some(value) => graph = Some(value),
                None => errors.push(format!("invalid graph: {}", values[0])),
            },
            "scope" => match PathQueryScope::parse(&values[0]) {
                Some(value) => scope = Some(value),
                None => errors.push(format!("invalid scope: {}", values[0])),
            },
            "depth" => match values[0].parse::<usize>() {
                Ok(value) => depth = Some(value),
                Err(_) => errors.push(format!("invalid depth: {}", values[0])),
            },
            "limit" => match values[0].parse::<usize>() {
                Ok(value) => limit = Some(value),
                Err(_) => errors.push(format!("invalid limit: {}", values[0])),
            },
            "objective" => match values[0].to_ascii_lowercase().as_str() {
                "min_cost" | "cost" => objective = Some(PathObjective::MinCost),
                "max_confidence" | "confidence" => {
                    objective = Some(PathObjective::MaxConfidence)
                }
                other => errors.push(format!("invalid objective: {other}")),
            },
            "budget_ms" => match values[0].parse::<u64>() {
                Ok(value) => query_budget_ms = Some(value),
                Err(_) => errors.push(format!("invalid budget_ms: {}", values[0])),
            },
            "semantic_min" => match values[0].parse::<f32>() {
                Ok(value) => semantic_min_score = Some(value),
                Err(_) => errors.push(format!("invalid semantic_min: {}", values[0])),
            },
            _ => {}
        }
    }

    let graph_kind = graph.unwrap_or(PathGraphKind::Memory);
    for relation in relations.iter().chain(deny_relations.iter()) {
        let known = match graph_kind {
            PathGraphKind::Memory => EdgeKind::parse(relation).is_some(),
            PathGraphKind::Engineering | PathGraphKind::Ontology => {
                allowed_relations
                    .iter()
                    .any(|allowed| allowed.eq_ignore_ascii_case(relation))
            }
        };
        if !known {
            errors.push(format!("unknown relation: {relation}"));
        }
    }

    let query = PathQuery {
        start,
        goal,
        waypoints,
        graph,
        constraints: PathQueryConstraints {
            relations,
            deny_relations,
            depth,
            limit,
            scope,
            objective,
            query_budget_ms,
            semantic_min_score,
            ..PathQueryConstraints::default()
        },
    };
    if let Err(validation) = query.validate() {
        errors.extend(validation);
    }
    if errors.is_empty() {
        Ok(query)
    } else {
        Err(errors)
    }
}

/// Result of coercing model-filled JSON into a validated PathQuery.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathQueryCompileReport {
    pub query: PathQuery,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub dropped_fields: Vec<String>,
}

const JSON_ROOT_FIELDS: [&str; 8] = [
    "start",
    "from",
    "goal",
    "to",
    "waypoints",
    "via",
    "graph",
    "constraints",
];

const JSON_CONSTRAINT_FIELDS: [&str; 18] = [
    "relations",
    "allow_relations",
    "deny_relations",
    "min_confidence",
    "min_conf",
    "objective",
    "depth",
    "max_depth",
    "limit",
    "max_results",
    "scope",
    "include_hypotheses",
    "include_ontology",
    "query_budget_ms",
    "budget_ms",
    "semantic_min_score",
    "semantic_min",
    "auto_ground",
];

fn json_first<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    names: &[&str],
) -> Option<(&'a str, &'a serde_json::Value)> {
    names
        .iter()
        .find_map(|name| object.get_key_value(*name).map(|(key, value)| (key.as_str(), value)))
}

fn json_anchors(
    value: &serde_json::Value,
    field: &str,
    errors: &mut Vec<String>,
) -> Option<Vec<String>> {
    match value {
        serde_json::Value::String(text) => Some(vec![text.clone()]),
        serde_json::Value::Array(items) => {
            let mut anchors = Vec::new();
            for item in items {
                match item.as_str() {
                    Some(anchor) => anchors.push(anchor.to_string()),
                    None => errors.push(format!("{field} must contain only strings")),
                }
            }
            Some(anchors)
        }
        _ => {
            errors.push(format!("{field} must be a string or array of strings"));
            None
        }
    }
}

fn json_usize(
    value: &serde_json::Value,
    field: &str,
    errors: &mut Vec<String>,
) -> Option<usize> {
    match value.as_u64() {
        Some(number) => usize::try_from(number).ok().or_else(|| {
            errors.push(format!("{field} is too large"));
            None
        }),
        None => {
            errors.push(format!("{field} must be a non-negative integer"));
            None
        }
    }
}

fn json_u64(value: &serde_json::Value, field: &str, errors: &mut Vec<String>) -> Option<u64> {
    match value.as_u64() {
        Some(number) => Some(number),
        None => {
            errors.push(format!("{field} must be a non-negative integer"));
            None
        }
    }
}

fn json_f32(value: &serde_json::Value, field: &str, errors: &mut Vec<String>) -> Option<f32> {
    match value.as_f64() {
        Some(number) => Some(number as f32),
        None => {
            errors.push(format!("{field} must be a number"));
            None
        }
    }
}

fn json_bool(value: &serde_json::Value, field: &str, errors: &mut Vec<String>) -> Option<bool> {
    match value.as_bool() {
        Some(flag) => Some(flag),
        None => {
            errors.push(format!("{field} must be a boolean"));
            None
        }
    }
}

/// Coerce model-filled JSON into a validated PathQuery.
///
/// The model may use the documented aliases (`from`/`to`, `max_depth`,
/// `max_results`, `min_conf`, `budget_ms`, `semantic_min`). Unsupported fields
/// are dropped and reported. Unknown relation names are hard errors so a model
/// cannot smuggle arbitrary traversal tokens into the engine.
pub fn compile_path_query_json(
    value: &serde_json::Value,
    allowed_relations: &[&str],
) -> Result<PathQueryCompileReport, Vec<String>> {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut dropped_fields = Vec::new();
    let Some(root) = value.as_object() else {
        return Err(vec!["query JSON must be an object".to_string()]);
    };
    let root = if root.get("start").is_none()
        && root.get("from").is_none()
        && root.get("query").is_some()
    {
        match root.get("query").and_then(serde_json::Value::as_object) {
            Some(inner) => inner,
            None => return Err(vec!["query wrapper must be an object".to_string()]),
        }
    } else {
        root
    };
    for key in root.keys() {
        if key != "query" && !JSON_ROOT_FIELDS.contains(&key.as_str()) {
            dropped_fields.push(key.clone());
            warnings.push(format!("dropped unsupported field: {key}"));
        }
    }
    let start = json_first(root, &["start", "from"])
        .and_then(|(_, value)| json_anchors(value, "start", &mut errors))
        .unwrap_or_default();
    let goal = json_first(root, &["goal", "to"])
        .and_then(|(_, value)| json_anchors(value, "goal", &mut errors))
        .unwrap_or_default();
    let waypoints = json_first(root, &["waypoints", "via"])
        .and_then(|(_, value)| json_anchors(value, "waypoints", &mut errors))
        .unwrap_or_default();
    let graph = match json_first(root, &["graph"]) {
        Some((_, value)) => match value.as_str().and_then(PathGraphKind::parse) {
            Some(graph) => graph,
            None => {
                errors.push("graph must be memory, engineering, or ontology".to_string());
                PathGraphKind::Memory
            }
        },
        None => PathGraphKind::Memory,
    };
    let mut constraints = PathQueryConstraints::default();
    if let Some((_, value)) = json_first(root, &["constraints"]) {
        let Some(object) = value.as_object() else {
            return Err(vec!["constraints must be an object".to_string()]);
        };
        for key in object.keys() {
            if !JSON_CONSTRAINT_FIELDS.contains(&key.as_str()) {
                dropped_fields.push(key.clone());
                warnings.push(format!("dropped unsupported constraint field: {key}"));
            }
        }
        if let Some((_, value)) = json_first(object, &["relations", "allow_relations"]) {
            constraints.relations =
                json_anchors(value, "relations", &mut errors).unwrap_or_default();
        }
        if let Some((_, value)) = json_first(object, &["deny_relations"]) {
            constraints.deny_relations =
                json_anchors(value, "deny_relations", &mut errors).unwrap_or_default();
        }
        if let Some((key, value)) = json_first(object, &["min_confidence", "min_conf"]) {
            constraints.min_confidence = json_f32(value, key, &mut errors);
        }
        if let Some((key, value)) = json_first(object, &["depth", "max_depth"]) {
            constraints.depth = json_usize(value, key, &mut errors);
        }
        if let Some((key, value)) = json_first(object, &["limit", "max_results"]) {
            constraints.limit = json_usize(value, key, &mut errors);
        }
        if let Some((key, value)) = json_first(object, &["query_budget_ms", "budget_ms"]) {
            constraints.query_budget_ms = json_u64(value, key, &mut errors);
        }
        if let Some((key, value)) = json_first(object, &["semantic_min_score", "semantic_min"]) {
            constraints.semantic_min_score = json_f32(value, key, &mut errors);
        }
        if let Some((_, value)) = json_first(object, &["scope"]) {
            constraints.scope = value.as_str().and_then(PathQueryScope::parse);
            if constraints.scope.is_none() {
                errors.push("scope must be project, global, or all".to_string());
            }
        }
        if let Some((_, value)) = json_first(object, &["objective"]) {
            constraints.objective = match value.as_str().map(str::trim) {
                Some("min_cost" | "cost") => Some(PathObjective::MinCost),
                Some("max_confidence" | "confidence") => Some(PathObjective::MaxConfidence),
                _ => {
                    errors.push(
                        "objective must be min_cost or max_confidence".to_string(),
                    );
                    None
                }
            };
        }
        if let Some((key, value)) = json_first(object, &["include_hypotheses"]) {
            constraints.include_hypotheses = json_bool(value, key, &mut errors).unwrap_or(false);
        }
        if let Some((key, value)) = json_first(object, &["include_ontology"]) {
            constraints.include_ontology = json_bool(value, key, &mut errors).unwrap_or(false);
        }
        if let Some((key, value)) = json_first(object, &["auto_ground"]) {
            constraints.auto_ground = json_bool(value, key, &mut errors).unwrap_or(false);
        }
    }
    for relation in constraints
        .relations
        .iter()
        .chain(constraints.deny_relations.iter())
    {
        let relation = relation.trim();
        let known = EdgeKind::parse(relation).is_some()
            || allowed_relations
                .iter()
                .any(|allowed| allowed.eq_ignore_ascii_case(relation));
        if relation.is_empty() || !known {
            errors.push(format!("unknown relation: {relation}"));
        }
    }
    let query = PathQuery {
        start,
        goal,
        waypoints,
        graph: Some(graph),
        constraints,
    };
    if errors.is_empty()
        && let Err(validation) = query.validate()
    {
        errors.extend(validation);
    }
    if errors.is_empty() {
        Ok(PathQueryCompileReport {
            query,
            warnings,
            dropped_fields,
        })
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validates_shape_scope_and_memory_relations() {
        let query = PathQuery {
            start: vec!["a".to_string()],
            goal: vec!["b".to_string()],
            waypoints: vec!["w".to_string()],
            graph: Some(PathGraphKind::Memory),
            constraints: PathQueryConstraints {
                relations: vec!["relates_to".to_string()],
                deny_relations: vec!["contradicts".to_string()],
                min_confidence: Some(0.6),
                objective: Some(PathObjective::MaxConfidence),
                depth: Some(4),
                limit: Some(3),
                scope: Some(PathQueryScope::Project),
                ..PathQueryConstraints::default()
            },
        };
        assert!(query.validate().is_ok());
        assert_eq!(query.graph_kind(), PathGraphKind::Memory);
        assert_eq!(query.scope(), PathQueryScope::Project);
        assert_eq!(
            query.constraints.objective,
            Some(PathObjective::MaxConfidence)
        );
    }

    #[test]
    fn rejects_unknown_memory_relation_and_bad_ranges() {
        let query = PathQuery {
            start: vec!["a".to_string()],
            goal: vec!["b".to_string()],
            waypoints: Vec::new(),
            graph: None,
            constraints: PathQueryConstraints {
                relations: vec!["invented".to_string()],
                deny_relations: vec!["also_invented".to_string()],
                min_confidence: Some(1.5),
                semantic_min_score: Some(1.5),
                depth: Some(99),
                limit: Some(99),
                ..PathQueryConstraints::default()
            },
        };
        let errors = query.validate().expect_err("must reject");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("unknown memory relation"))
        );
        assert!(errors.iter().any(|error| error.contains("min_confidence")));
        assert!(
            errors
                .iter()
                .any(|error| error.contains("semantic_min_score"))
        );
        assert!(errors.iter().any(|error| error.contains("depth")));
        assert!(errors.iter().any(|error| error.contains("limit")));
    }

    #[test]
    fn engineering_relations_are_not_memory_validated() {
        let query = PathQuery {
            start: vec!["file:src/lib.rs".to_string()],
            goal: vec!["error:E0425".to_string()],
            waypoints: Vec::new(),
            graph: Some(PathGraphKind::Engineering),
            constraints: PathQueryConstraints {
                relations: vec!["modifies".to_string()],
                ..PathQueryConstraints::default()
            },
        };
        assert!(query.validate().is_ok());
    }

    #[test]
    fn ontology_graph_kind_round_trips() {
        assert_eq!(PathGraphKind::parse("ontology"), Some(PathGraphKind::Ontology));
        let query = PathQuery {
            start: vec!["File".to_string()],
            goal: vec!["API".to_string()],
            waypoints: Vec::new(),
            graph: Some(PathGraphKind::Ontology),
            constraints: PathQueryConstraints {
                relations: vec!["defines".to_string()],
                ..PathQueryConstraints::default()
            },
        };
        assert_eq!(query.graph_kind(), PathGraphKind::Ontology);
        assert!(query.validate().is_ok());
    }

    #[test]
    fn compiles_engineering_template_with_quoted_anchors() {
        let query = compile_path_query_text(
            "find from \"file:src/lib.rs\" to `api:/task` graph engineering \
             relations defines implements depth 3 limit 2 budget_ms 50 semantic_min 0.7",
            &["defines", "implements"],
        )
        .expect("compile");
        assert_eq!(query.graph_kind(), PathGraphKind::Engineering);
        assert_eq!(query.start, vec!["file:src/lib.rs"]);
        assert_eq!(query.goal, vec!["api:/task"]);
        assert_eq!(query.constraints.relations, vec!["defines", "implements"]);
        assert_eq!(query.constraints.depth, Some(3));
        assert_eq!(query.constraints.limit, Some(2));
        assert_eq!(query.constraints.query_budget_ms, Some(50));
        assert_eq!(query.constraints.semantic_min_score, Some(0.7));
    }

    #[test]
    fn compile_rejects_unknown_relations_and_missing_goal() {
        let errors = compile_path_query_text(
            "from a graph engineering relations teleports to b",
            &["defines"],
        )
        .expect_err("unknown relation");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("unknown relation: teleports")),
            "{errors:?}"
        );

        let errors = compile_path_query_text("from a depth 2", &[]).expect_err("missing goal");
        assert!(
            errors.iter().any(|error| error.contains("goal")),
            "{errors:?}"
        );
    }

    #[test]
    fn compiles_memory_template_with_known_edge_kind() {
        let query = compile_path_query_text(
            "from mem:one to mem:two relations relates_to depth 2",
            &[],
        )
        .expect("compile");
        assert_eq!(query.graph_kind(), PathGraphKind::Memory);
        assert_eq!(query.constraints.relations, vec!["relates_to"]);
    }

    #[test]
    fn compiles_model_json_with_aliases_and_reports_dropped_fields() {
        let report = compile_path_query_json(
            &json!({
                "from": ["file:src/lib.rs"],
                "to": "api:/task",
                "via": ["file:src/main.rs"],
                "graph": "engineering",
                "constraints": {
                    "relations": ["defines"],
                    "max_depth": 3,
                    "max_results": 2,
                    "min_conf": 0.6,
                    "budget_ms": 50,
                    "semantic_min": 0.7,
                    "objective": "max_confidence",
                    "scope": "project",
                    "auto_ground": true,
                    "mystery": 1
                },
                "unexpected": true
            }),
            &["defines"],
        )
        .expect("compile model JSON");
        assert_eq!(report.query.graph_kind(), PathGraphKind::Engineering);
        assert_eq!(report.query.start, vec!["file:src/lib.rs"]);
        assert_eq!(report.query.goal, vec!["api:/task"]);
        assert_eq!(report.query.waypoints, vec!["file:src/main.rs"]);
        assert_eq!(report.query.constraints.depth, Some(3));
        assert_eq!(report.query.constraints.limit, Some(2));
        assert_eq!(report.query.constraints.min_confidence, Some(0.6));
        assert_eq!(report.query.constraints.query_budget_ms, Some(50));
        assert_eq!(report.query.constraints.semantic_min_score, Some(0.7));
        assert_eq!(
            report.query.constraints.objective,
            Some(PathObjective::MaxConfidence)
        );
        assert_eq!(report.query.constraints.scope, Some(PathQueryScope::Project));
        assert!(report.query.constraints.auto_ground);
        assert_eq!(
            report.dropped_fields,
            vec!["unexpected".to_string(), "mystery".to_string()]
        );
        assert_eq!(report.warnings.len(), 2);
    }

    #[test]
    fn model_json_rejects_unknown_relations_and_missing_goal() {
        let errors = compile_path_query_json(
            &json!({
                "start": "a",
                "goal": "b",
                "constraints": {"relations": ["teleports"]}
            }),
            &["defines"],
        )
        .expect_err("unknown relation");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("unknown relation: teleports")),
            "{errors:?}"
        );

        let errors = compile_path_query_json(
            &json!({"start": ["a"], "graph": "memory"}),
            &[],
        )
        .expect_err("missing goal");
        assert!(
            errors.iter().any(|error| error.contains("goal")),
            "{errors:?}"
        );
    }
}
