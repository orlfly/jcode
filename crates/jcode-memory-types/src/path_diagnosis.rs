//! Deterministic path-missing taxonomy from the retrieval design.
//!
//! The query engine only emits codes it can prove with structural probes. This
//! module owns the stable code-to-class-to-first-repair mapping so callers do
//! not have to re-implement the ladder per surface.

use serde::{Deserialize, Serialize};

/// Whether a missing path is the graph's fault or the query's fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathMissingClass {
    OntologySilence,
    SemanticGap,
    QuerySide,
}

impl PathMissingClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OntologySilence => "ontology_silence",
            Self::SemanticGap => "semantic_gap",
            Self::QuerySide => "query_side",
        }
    }
}

/// Stable error codes E01-E10.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PathDiagnosisCode {
    E01,
    E02,
    E03,
    E04,
    E05,
    E06,
    E07,
    E08,
    E09,
    E10,
}

impl PathDiagnosisCode {
    pub const ALL: [Self; 10] = [
        Self::E01,
        Self::E02,
        Self::E03,
        Self::E04,
        Self::E05,
        Self::E06,
        Self::E07,
        Self::E08,
        Self::E09,
        Self::E10,
    ];

    pub fn parse(code: &str) -> Option<Self> {
        match code.trim().to_ascii_uppercase().as_str() {
            "E01" => Some(Self::E01),
            "E02" => Some(Self::E02),
            "E03" => Some(Self::E03),
            "E04" => Some(Self::E04),
            "E05" => Some(Self::E05),
            "E06" => Some(Self::E06),
            "E07" => Some(Self::E07),
            "E08" => Some(Self::E08),
            "E09" => Some(Self::E09),
            "E10" => Some(Self::E10),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::E01 => "E01",
            Self::E02 => "E02",
            Self::E03 => "E03",
            Self::E04 => "E04",
            Self::E05 => "E05",
            Self::E06 => "E06",
            Self::E07 => "E07",
            Self::E08 => "E08",
            Self::E09 => "E09",
            Self::E10 => "E10",
        }
    }

    pub fn missing_class(self) -> PathMissingClass {
        match self {
            Self::E01 | Self::E03 | Self::E09 => PathMissingClass::SemanticGap,
            Self::E02 | Self::E10 => PathMissingClass::OntologySilence,
            Self::E04 | Self::E05 | Self::E06 | Self::E07 | Self::E08 => {
                PathMissingClass::QuerySide
            }
        }
    }

    /// First repair from table 4-3, prefixed with its ladder level.
    pub fn first_repair(self) -> &'static str {
        match self {
            Self::E01 => "L2: broaden entity candidates and trigger alignment",
            Self::E02 => "L2: propose a concept relation, require ontology review",
            Self::E03 => "L2/L3: align components or extract missing edges",
            Self::E04 => "L0: relax hard constraints one at a time",
            Self::E05 => "L0: increase depth budget",
            Self::E06 => "L0: lower semantic threshold and re-embed",
            Self::E07 => "L0/L4: prune, parallelize, cache, or split the query",
            Self::E08 => "L0: union the project and global scopes",
            Self::E09 => "L2: create same_as/instance_of alignment or a bridge",
            Self::E10 => "L1: use the inverse/subproperty relation or fix the data",
        }
    }
}

/// Map diagnosis strings to deduplicated repair actions in input order.
pub fn next_actions_for(codes: &[String]) -> Vec<String> {
    let mut actions = Vec::new();
    for code in codes {
        let Some(parsed) = PathDiagnosisCode::parse(code) else {
            continue;
        };
        let action = format!("{} {}", parsed.as_str(), parsed.first_repair());
        if !actions.iter().any(|seen| seen == &action) {
            actions.push(action);
        }
    }
    actions
}

/// Map diagnosis strings to deduplicated missing classes in input order.
pub fn missing_classes_for(codes: &[String]) -> Vec<String> {
    let mut classes = Vec::new();
    for code in codes {
        let Some(parsed) = PathDiagnosisCode::parse(code) else {
            continue;
        };
        let class = parsed.missing_class().as_str().to_string();
        if !classes.iter().any(|seen| seen == &class) {
            classes.push(class);
        }
    }
    classes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_code_has_a_class_and_repair() {
        for code in PathDiagnosisCode::ALL {
            assert_eq!(PathDiagnosisCode::parse(code.as_str()), Some(code));
            assert!(!code.first_repair().is_empty());
        }
        assert_eq!(
            PathDiagnosisCode::E05.missing_class(),
            PathMissingClass::QuerySide
        );
        assert_eq!(
            PathDiagnosisCode::E10.missing_class(),
            PathMissingClass::OntologySilence
        );
    }

    #[test]
    fn actions_and_classes_are_deduplicated_in_input_order() {
        let codes = vec![
            "E05".to_string(),
            "E03".to_string(),
            "E05".to_string(),
            "E09".to_string(),
            "not-a-code".to_string(),
        ];
        let actions = next_actions_for(&codes);
        assert_eq!(actions.len(), 3, "{actions:?}");
        assert!(actions[0].starts_with("E05 "), "{actions:?}");
        assert_eq!(
            missing_classes_for(&codes),
            vec!["query_side".to_string(), "semantic_gap".to_string()]
        );
    }
}
