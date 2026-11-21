//! Deterministic action selection over observed path-query signals.
//!
//! This is the decision-layer slice that can be computed without a belief-state
//! update: diagnosis codes, path confidence, and local quality flags select the
//! next action from the POMDP action space. It deliberately does not claim to
//! solve the full POMDP; it makes the next best action auditable and testable.

use serde::{Deserialize, Serialize};

use crate::path_quality::PathQualityFlag;

/// Action space from design section 5.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionAction {
    Query,
    Reason,
    Complete,
    Plan,
    Terminate,
    AwaitHuman,
}

impl DecisionAction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Reason => "reason",
            Self::Complete => "complete",
            Self::Plan => "plan",
            Self::Terminate => "terminate",
            Self::AwaitHuman => "await_human",
        }
    }
}

/// Observable inputs available after one path-query step.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DecisionSignals {
    pub has_paths: bool,
    /// Confidence of the best returned path, in 0.0..=1.0.
    pub path_confidence: f32,
    pub diagnosis: Vec<String>,
    pub quality_flags: Vec<PathQualityFlag>,
    pub planning_requested: bool,
    /// Remaining query/repair attempts in the current turn.
    pub attempts_remaining: u32,
}

impl DecisionSignals {
    pub fn has_diagnosis(&self, code: &str) -> bool {
        self.diagnosis
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(code))
    }

    pub fn has_quality(&self, flag: PathQualityFlag) -> bool {
        self.quality_flags.contains(&flag)
    }
}

/// Selected action with a short audit trail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Decision {
    pub action: DecisionAction,
    pub reason: String,
    /// 1 = blocking/urgent, 2 = needs work, 3 = ready to finish.
    pub priority: u8,
}

fn decision(action: DecisionAction, reason: impl Into<String>, priority: u8) -> Decision {
    Decision {
        action,
        reason: reason.into(),
        priority,
    }
}

/// Select the next action. Ordering is intentional: ambiguity and hypotheses
/// block side effects before confidence or planning are considered.
pub fn select_next_action(signals: &DecisionSignals) -> Decision {
    if signals.attempts_remaining == 0 {
        return decision(
            DecisionAction::AwaitHuman,
            "decision budget exhausted without a safe action",
            1,
        );
    }

    if signals.has_paths {
        if signals.has_quality(PathQualityFlag::AmbiguousEntity) {
            return decision(
                DecisionAction::Query,
                "disambiguate the endpoint anchor before trusting the path",
                1,
            );
        }
        if signals.has_quality(PathQualityFlag::HypothesizedEdge) {
            return decision(
                DecisionAction::Reason,
                "verify the shadow hypothesis before any side-effecting plan",
                1,
            );
        }
        if signals.has_quality(PathQualityFlag::StitchedPath) {
            return decision(
                DecisionAction::Complete,
                "validate the bridge gap before executing the stitched path",
                1,
            );
        }
        if signals.has_quality(PathQualityFlag::LowProvenance)
            || signals.has_quality(PathQualityFlag::StaleEdge)
            || signals.has_quality(PathQualityFlag::HighDegreeHub)
        {
            return decision(
                DecisionAction::Reason,
                "gather corroborating evidence for weak or stale path nodes",
                2,
            );
        }
        if signals.has_quality(PathQualityFlag::RelaxedQuery) {
            return decision(
                DecisionAction::Reason,
                "confirm that the recorded query relaxation is acceptable",
                2,
            );
        }
        if signals.path_confidence < 0.6 {
            return decision(
                DecisionAction::Reason,
                "raise path confidence before planning or terminating",
                2,
            );
        }
        if signals.planning_requested {
            return decision(
                DecisionAction::Plan,
                "path is ready to compile into a plan",
                3,
            );
        }
        return decision(
            DecisionAction::Terminate,
            "path is sufficiently confident to answer",
            3,
        );
    }

    for (code, action, reason) in [
        (
            "E08",
            DecisionAction::Query,
            "union the project and global scopes",
        ),
        ("E05", DecisionAction::Query, "increase the depth budget"),
        (
            "E01",
            DecisionAction::Query,
            "broaden or re-link the anchors",
        ),
        (
            "E09",
            DecisionAction::Complete,
            "create same_as/instance_of alignment or a bridge",
        ),
        (
            "E03",
            DecisionAction::Complete,
            "extract or infer the missing connecting edges",
        ),
        (
            "E10",
            DecisionAction::Reason,
            "resolve the relation direction or type mismatch",
        ),
        (
            "E02",
            DecisionAction::Complete,
            "propose the missing concept relation for ontology review",
        ),
        (
            "E04",
            DecisionAction::Query,
            "relax one hard constraint and record it",
        ),
        (
            "E06",
            DecisionAction::Query,
            "lower the semantic threshold or refresh embeddings",
        ),
        ("E07", DecisionAction::Query, "prune or split the search"),
    ] {
        if signals.has_diagnosis(code) {
            return decision(action, reason, 2);
        }
    }

    decision(
        DecisionAction::AwaitHuman,
        "no structural repair path remains; report a knowledge gap",
        1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambiguity_and_hypotheses_block_side_effects() {
        let ambiguous = select_next_action(&DecisionSignals {
            has_paths: true,
            path_confidence: 0.9,
            quality_flags: vec![PathQualityFlag::AmbiguousEntity],
            planning_requested: true,
            attempts_remaining: 1,
            ..DecisionSignals::default()
        });
        assert_eq!(ambiguous.action, DecisionAction::Query);

        let hypothesized = select_next_action(&DecisionSignals {
            has_paths: true,
            path_confidence: 0.9,
            quality_flags: vec![PathQualityFlag::HypothesizedEdge],
            planning_requested: true,
            attempts_remaining: 1,
            ..DecisionSignals::default()
        });
        assert_eq!(hypothesized.action, DecisionAction::Reason);
    }

    #[test]
    fn no_path_uses_diagnosis_to_choose_repair() {
        let depth = select_next_action(&DecisionSignals {
            diagnosis: vec!["E03".to_string(), "E05".to_string()],
            attempts_remaining: 2,
            ..DecisionSignals::default()
        });
        assert_eq!(depth.action, DecisionAction::Query);
        assert!(depth.reason.contains("depth"));

        let alignment = select_next_action(&DecisionSignals {
            diagnosis: vec!["E09".to_string()],
            attempts_remaining: 2,
            ..DecisionSignals::default()
        });
        assert_eq!(alignment.action, DecisionAction::Complete);
    }

    #[test]
    fn clean_path_plans_or_terminates_and_budget_exhaustion_stops() {
        let plan = select_next_action(&DecisionSignals {
            has_paths: true,
            path_confidence: 0.9,
            planning_requested: true,
            attempts_remaining: 1,
            ..DecisionSignals::default()
        });
        assert_eq!(plan.action, DecisionAction::Plan);

        let terminate = select_next_action(&DecisionSignals {
            has_paths: true,
            path_confidence: 0.9,
            attempts_remaining: 1,
            ..DecisionSignals::default()
        });
        assert_eq!(terminate.action, DecisionAction::Terminate);

        let exhausted = select_next_action(&DecisionSignals {
            has_paths: true,
            path_confidence: 0.9,
            attempts_remaining: 0,
            ..DecisionSignals::default()
        });
        assert_eq!(exhausted.action, DecisionAction::AwaitHuman);
    }
}
