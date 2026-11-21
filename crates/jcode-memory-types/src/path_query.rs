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
}

impl PathGraphKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Memory => "memory",
            Self::Engineering => "engineering",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "memory" => Some(Self::Memory),
            "engineering" => Some(Self::Engineering),
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

/// Query constraints after entity linking.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathQueryConstraints {
    #[serde(default)]
    pub relations: Vec<String>,
    #[serde(default)]
    pub depth: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub scope: Option<PathQueryScope>,
    #[serde(default)]
    pub include_hypotheses: bool,
    #[serde(default)]
    pub plan: bool,
}

/// One validated path query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathQuery {
    pub start: Vec<String>,
    pub goal: Vec<String>,
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
            for relation in &self.constraints.relations {
                let relation = relation.trim();
                if relation.is_empty() || EdgeKind::parse(relation).is_some() {
                    continue;
                }
                errors.push(format!("unknown memory relation: {relation}"));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_shape_scope_and_memory_relations() {
        let query = PathQuery {
            start: vec!["a".to_string()],
            goal: vec!["b".to_string()],
            graph: Some(PathGraphKind::Memory),
            constraints: PathQueryConstraints {
                relations: vec!["relates_to".to_string()],
                depth: Some(4),
                limit: Some(3),
                scope: Some(PathQueryScope::Project),
                ..PathQueryConstraints::default()
            },
        };
        assert!(query.validate().is_ok());
        assert_eq!(query.graph_kind(), PathGraphKind::Memory);
        assert_eq!(query.scope(), PathQueryScope::Project);
    }

    #[test]
    fn rejects_unknown_memory_relation_and_bad_ranges() {
        let query = PathQuery {
            start: vec!["a".to_string()],
            goal: vec!["b".to_string()],
            graph: None,
            constraints: PathQueryConstraints {
                relations: vec!["invented".to_string()],
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
        assert!(errors.iter().any(|error| error.contains("depth")));
        assert!(errors.iter().any(|error| error.contains("limit")));
    }

    #[test]
    fn engineering_relations_are_not_memory_validated() {
        let query = PathQuery {
            start: vec!["file:src/lib.rs".to_string()],
            goal: vec!["error:E0425".to_string()],
            graph: Some(PathGraphKind::Engineering),
            constraints: PathQueryConstraints {
                relations: vec!["modifies".to_string()],
                ..PathQueryConstraints::default()
            },
        };
        assert!(query.validate().is_ok());
    }
}
