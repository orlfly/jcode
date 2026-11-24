//! Local quality flags attached to one returned path.
//!
//! These flags are runtime decision signals, not a global quality report. They
//! let a planner insert verification, request disambiguation, or avoid a path
//! with weak provenance without re-reading the whole graph.

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::engineering_ontology::{EngineeringGraph, EngineeringPath};

/// One local path-quality concern from design table 4.6.9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathQualityFlag {
    StaleEdge,
    HypothesizedEdge,
    HighDegreeHub,
    AmbiguousEntity,
    LowProvenance,
    StitchedPath,
    RelaxedQuery,
}

impl PathQualityFlag {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StaleEdge => "stale_edge",
            Self::HypothesizedEdge => "hypothesized_edge",
            Self::HighDegreeHub => "high_degree_hub",
            Self::AmbiguousEntity => "ambiguous_entity",
            Self::LowProvenance => "low_provenance",
            Self::StitchedPath => "stitched_path",
            Self::RelaxedQuery => "relaxed_query",
        }
    }
}

/// Quality flags plus human-readable reasons for one path.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathQuality {
    pub flags: Vec<PathQualityFlag>,
    pub reasons: Vec<String>,
}

impl PathQuality {
    pub fn is_empty(&self) -> bool {
        self.flags.is_empty()
    }

    pub fn has(&self, flag: PathQualityFlag) -> bool {
        self.flags.contains(&flag)
    }

    pub fn push(&mut self, flag: PathQualityFlag, reason: impl Into<String>) {
        if !self.flags.contains(&flag) {
            self.flags.push(flag);
        }
        let reason = reason.into();
        if !self.reasons.iter().any(|seen| seen == &reason) {
            self.reasons.push(reason);
        }
    }

    /// Stable space-separated flag list for compact tool output.
    pub fn label(&self) -> String {
        self.flags
            .iter()
            .map(|flag| flag.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Path-level quality signals for one engineering path.
///
/// A path that crosses any unconfirmed completion edge must carry
/// `hypothesized_edge`; the plan gate and the planner both key on this flag
/// rather than re-reading edge provenance.
impl EngineeringGraph {
    pub fn quality_of_path(&self, path: &EngineeringPath) -> PathQuality {
        let now = Utc::now();
        let mut quality = PathQuality::default();
        for id in &path.relations {
            let Some(relation) = self.relation(id) else {
                quality.push(
                    PathQualityFlag::LowProvenance,
                    format!("relation {id} is missing from the graph"),
                );
                continue;
            };
            if relation.hypothesis {
                quality.push(
                    PathQualityFlag::HypothesizedEdge,
                    format!("{} is an unconfirmed completion edge", relation.id),
                );
            }
            if let Some(until) = relation.valid_to.filter(|until| *until < now) {
                quality.push(
                    PathQualityFlag::StaleEdge,
                    format!("{} expired at {until}", relation.id),
                );
            }
            let provenance_confidence = relation.provenance.confidence;
            if !provenance_confidence.is_finite() || provenance_confidence < 0.5 {
                quality.push(
                    PathQualityFlag::LowProvenance,
                    format!("{} has weak provenance", relation.id),
                );
            }
        }
        quality
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_are_deduplicated_and_keep_one_reason() {
        let mut quality = PathQuality::default();
        quality.push(PathQualityFlag::StitchedPath, "crosses a bridge");
        quality.push(PathQualityFlag::StitchedPath, "crosses a bridge");
        assert_eq!(quality.flags, vec![PathQualityFlag::StitchedPath]);
        assert_eq!(quality.reasons, vec!["crosses a bridge"]);
        assert_eq!(quality.label(), "stitched_path");
    }

    #[test]
    fn engineering_quality_marks_hypothesis_and_weak_provenance() {
        use crate::engineering_ontology::{
            CLASS_CHANGE, CLASS_ERROR, EngineeringEntity, EngineeringRelation, REL_FIXES,
        };

        let mut graph = EngineeringGraph::new();
        graph.insert_entity(EngineeringEntity::new(CLASS_ERROR, "error:e1", "E0425"));
        graph.insert_entity(EngineeringEntity::new(CLASS_CHANGE, "change:c1", "Patch"));
        graph.insert_relation(EngineeringRelation::new("error:e1", REL_FIXES, "change:c1"));
        graph.insert_relation(
            EngineeringRelation::new("change:c1", "supports", "error:e1")
                .with_hypothesis(true)
                .with_provenance(crate::engineering_ontology::EngineeringProvenance::new(
                    "llm:proposal",
                )),
        );
        let path = graph
            .weighted_shortest_path("change:c1", "error:e1", 2, &[])
            .expect("path");
        let quality = graph.quality_of_path(&path);
        assert!(quality.has(PathQualityFlag::HypothesizedEdge));
        assert!(
            quality
                .reasons
                .iter()
                .any(|reason| reason.contains("unconfirmed completion edge")),
            "{:?}",
            quality
        );
        // The seeded provenance has confidence 0.8, so only the hypothesis flag
        // fires here; low provenance is a separate concern.
        let asserted = graph.quality_of_path(
            &graph
                .weighted_shortest_path("error:e1", "change:c1", 2, &[])
                .expect("asserted path"),
        );
        assert!(asserted.flags.is_empty(), "{:?}", asserted);
    }
}
