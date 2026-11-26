//! Stage S3 intent-to-path-pattern mapping (design table 4-2).
//!
//! The mapping is deterministic: intent decides path direction, objective,
//! output shape, whether reverse edges are needed and whether multi-path output
//! must be preserved. Suggested relations are filtered against the ontology so
//! callers never get a whitelist entry the compiler would reject.

use serde::{Deserialize, Serialize};

use crate::engineering_ontology::{
    EngineeringOntology, REL_BLOCKED_BY, REL_CAUSES, REL_CONTAINS, REL_DEFINES, REL_DEPENDS_ON,
    REL_EVIDENCED_BY, REL_FIXES, REL_IMPLEMENTS, REL_MODIFIES, REL_PART_OF, REL_PRODUCES,
    REL_VERIFIES,
};
use crate::path_query::PathObjective;
use crate::task_frame::TaskIntent;

/// Path traversal direction implied by the task intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathDirection {
    /// Phenomenon <- causes: query backwards from the symptom.
    Reverse,
    /// Change point -> affected entities.
    Forward,
    /// Subject -> rule, existence check.
    Existence,
    /// Resource constraints -> ranked candidates.
    KBest,
}

impl PathDirection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reverse => "reverse",
            Self::Forward => "forward",
            Self::Existence => "existence",
            Self::KBest => "k_best",
        }
    }
}

/// Output shape the caller should render.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathOutputKind {
    CausalPathSet,
    ImpactSubgraph,
    ActionPlan,
    ComplianceEvidence,
    RankedCandidates,
}

/// One resolved path pattern.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathPattern {
    pub intent: TaskIntent,
    pub direction: PathDirection,
    pub objective: PathObjective,
    pub output: PathOutputKind,
    /// True when the intent needs to traverse edges against their direction.
    pub allow_reverse_edges: bool,
    /// Keep several paths instead of collapsing to the best one.
    pub multi_path: bool,
    /// Include a comparison/contrast explanation.
    pub needs_comparison: bool,
    /// Relations that suit this intent, filtered to ontology-declared names.
    pub suggested_relations: Vec<String>,
    pub depth_hint: usize,
    #[serde(default)]
    pub notes: Vec<String>,
}

impl PathPattern {
    pub fn summary(&self) -> String {
        format!(
            "{} path, objective {}, output {:?}, {} relation(s), depth {}",
            self.direction.as_str(),
            self.objective.as_str(),
            self.output,
            self.suggested_relations.len(),
            self.depth_hint
        )
    }
}

fn keep_declared(ontology: &EngineeringOntology, candidates: &[&str]) -> Vec<String> {
    candidates
        .iter()
        .filter(|relation| ontology.relations.contains_key(**relation))
        .map(|relation| (*relation).to_string())
        .collect()
}

/// Resolve the path pattern for one task intent.
pub fn pattern_for_intent(intent: TaskIntent, ontology: &EngineeringOntology) -> PathPattern {
    let (direction, objective, output, allow_reverse, multi_path, comparison, candidates, depth) =
        match intent {
            TaskIntent::Attribution => (
                PathDirection::Reverse,
                PathObjective::MaxConfidence,
                PathOutputKind::CausalPathSet,
                true,
                true,
                true,
                &[
                    REL_CAUSES,
                    REL_FIXES,
                    REL_BLOCKED_BY,
                    REL_EVIDENCED_BY,
                    REL_PRODUCES,
                ][..],
                4,
            ),
            TaskIntent::Impact => (
                PathDirection::Forward,
                PathObjective::MaxConfidence,
                PathOutputKind::ImpactSubgraph,
                false,
                true,
                false,
                &[
                    REL_DEPENDS_ON,
                    REL_CONTAINS,
                    REL_MODIFIES,
                    REL_DEFINES,
                    REL_PART_OF,
                    REL_PRODUCES,
                ][..],
                5,
            ),
            TaskIntent::Plan => (
                PathDirection::Forward,
                PathObjective::MinCost,
                PathOutputKind::ActionPlan,
                false,
                false,
                false,
                &[REL_DEPENDS_ON, REL_IMPLEMENTS, REL_FIXES, REL_VERIFIES][..],
                4,
            ),
            TaskIntent::Compliance => (
                PathDirection::Existence,
                PathObjective::MinCost,
                PathOutputKind::ComplianceEvidence,
                false,
                false,
                false,
                &[REL_VERIFIES, REL_EVIDENCED_BY, REL_IMPLEMENTS][..],
                3,
            ),
            TaskIntent::Alternative => (
                PathDirection::KBest,
                PathObjective::MaxConfidence,
                PathOutputKind::RankedCandidates,
                false,
                true,
                true,
                &[REL_IMPLEMENTS, REL_DEPENDS_ON, REL_PRODUCES][..],
                4,
            ),
        };

    let suggested_relations = keep_declared(ontology, candidates);
    let mut notes = Vec::new();
    if suggested_relations.len() < candidates.len() {
        notes.push(format!(
            "{} candidate relation(s) are not declared in the ontology and were filtered",
            candidates.len() - suggested_relations.len()
        ));
    }
    if allow_reverse {
        notes.push(
            "reverse traversal is not executed by the graph engine; swap the anchors or rely on \
             inverse-relation inference"
                .to_string(),
        );
    }
    if multi_path {
        notes.push(
            "preserve multiple paths; do not collapse to the single cheapest path".to_string(),
        );
    }

    PathPattern {
        intent,
        direction,
        objective,
        output,
        allow_reverse_edges: allow_reverse,
        multi_path,
        needs_comparison: comparison,
        suggested_relations,
        depth_hint: depth,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engineering_ontology::default_engineering_ontology;

    #[test]
    fn attribution_keeps_multiple_reverse_causal_paths() {
        let pattern = pattern_for_intent(TaskIntent::Attribution, &default_engineering_ontology());
        assert_eq!(pattern.direction, PathDirection::Reverse);
        assert_eq!(pattern.objective, PathObjective::MaxConfidence);
        assert_eq!(pattern.output, PathOutputKind::CausalPathSet);
        assert!(pattern.multi_path);
        assert!(pattern.needs_comparison);
        assert!(pattern.allow_reverse_edges);
        assert!(
            pattern
                .suggested_relations
                .contains(&REL_CAUSES.to_string())
        );
        assert!(
            pattern
                .notes
                .iter()
                .any(|note| note.contains("reverse traversal")),
            "{:?}",
            pattern.notes
        );
    }

    #[test]
    fn impact_is_forward_and_preserves_the_affected_subgraph() {
        let pattern = pattern_for_intent(TaskIntent::Impact, &default_engineering_ontology());
        assert_eq!(pattern.direction, PathDirection::Forward);
        assert_eq!(pattern.output, PathOutputKind::ImpactSubgraph);
        assert!(pattern.multi_path);
        assert!(!pattern.allow_reverse_edges);
        assert!(!pattern.needs_comparison);
        assert_eq!(pattern.depth_hint, 5);
    }

    #[test]
    fn plan_minimizes_cost_and_compliance_is_an_existence_check() {
        let ontology = default_engineering_ontology();
        let plan = pattern_for_intent(TaskIntent::Plan, &ontology);
        assert_eq!(plan.objective, PathObjective::MinCost);
        assert_eq!(plan.output, PathOutputKind::ActionPlan);
        assert!(!plan.multi_path);

        let compliance = pattern_for_intent(TaskIntent::Compliance, &ontology);
        assert_eq!(compliance.direction, PathDirection::Existence);
        assert_eq!(compliance.output, PathOutputKind::ComplianceEvidence);
        assert!(!compliance.multi_path);
        assert_eq!(compliance.depth_hint, 3);
    }

    #[test]
    fn alternative_ranks_candidates_and_compares_them() {
        let pattern = pattern_for_intent(TaskIntent::Alternative, &default_engineering_ontology());
        assert_eq!(pattern.direction, PathDirection::KBest);
        assert_eq!(pattern.output, PathOutputKind::RankedCandidates);
        assert!(pattern.multi_path);
        assert!(pattern.needs_comparison);
    }

    #[test]
    fn filters_relations_absent_from_the_ontology() {
        use crate::engineering_ontology::{BfoCategory, EngineeringClassDef};

        let mut ontology = EngineeringOntology::new("test", 1, "pattern test");
        ontology.classes.insert(
            "artifact".to_string(),
            EngineeringClassDef::new("artifact", "Artifact", BfoCategory::IndependentContinuant),
        );
        ontology.relations.insert(
            REL_DEPENDS_ON.to_string(),
            crate::engineering_ontology::EngineeringRelationDef::new(
                REL_DEPENDS_ON,
                "depends on",
                &["artifact"],
                &["artifact"],
            ),
        );
        let pattern = pattern_for_intent(TaskIntent::Impact, &ontology);
        assert_eq!(
            pattern.suggested_relations,
            vec![REL_DEPENDS_ON.to_string()]
        );
        assert!(
            pattern
                .notes
                .iter()
                .any(|note| note.contains("not declared")),
            "{:?}",
            pattern.notes
        );
    }
}
