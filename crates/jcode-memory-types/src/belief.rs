//! Deterministic belief-state updates for the POMDP decision loop.
//!
//! This is the memory-side slice between an observation and the next action.
//! It does not learn a policy and does not claim calibrated probabilities: it
//! records how one path-query result changed hypothesis confidence, which
//! observations remain unresolved, and which action the decision layer selects.
//! The update rule is Bayes' rule over a supplied evidence strength, so the
//! same observation always produces the same posterior.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::decision::{DecisionAction, DecisionSignals, select_next_action};
use crate::path_quality::PathQualityFlag;
use crate::reward::quality_signal;

/// Default minimum confidence retained by [`BeliefState::prune`].
pub const DEFAULT_PRUNE_THRESHOLD: f32 = 0.05;

/// One candidate explanation for the task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hypothesis {
    pub id: String,
    pub statement: String,
    pub confidence: f32,
    pub support: u32,
    pub refutations: u32,
}

impl Hypothesis {
    pub fn new(id: impl Into<String>, statement: impl Into<String>, confidence: f32) -> Self {
        Self {
            id: id.into(),
            statement: statement.into(),
            confidence: confidence.clamp(0.001, 0.999),
            support: 0,
            refutations: 0,
        }
    }
}

/// Observation produced by one decision step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BeliefObservation {
    PathFound {
        from: String,
        to: String,
        path_confidence: f32,
        cost: f32,
        quality_flags: Vec<PathQualityFlag>,
    },
    PathMissing {
        from: String,
        to: String,
        diagnosis: Vec<String>,
        missing_class: Option<String>,
    },
    EndpointAmbiguous {
        anchor: String,
        candidates: Vec<String>,
    },
    AnchorResolved {
        anchor: String,
        resolved_id: String,
    },
    HumanConfirmation {
        confirmed: bool,
        note: Option<String>,
    },
    Reasoning {
        note: String,
        added_support: u32,
    },
}

impl BeliefObservation {
    fn label(&self) -> &'static str {
        match self {
            Self::PathFound { .. } => "path_found",
            Self::PathMissing { .. } => "path_missing",
            Self::EndpointAmbiguous { .. } => "endpoint_ambiguous",
            Self::AnchorResolved { .. } => "anchor_resolved",
            Self::HumanConfirmation { .. } => "human_confirmation",
            Self::Reasoning { .. } => "reasoning",
        }
    }
}

/// One auditable step in the belief history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefStep {
    pub step: u32,
    pub observation_kind: String,
    pub action: DecisionAction,
    pub prior_confidence: f32,
    pub posterior_confidence: f32,
    pub evidence_strength: f32,
}

/// Belief state carried across path queries in one decision loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefState {
    pub goal: String,
    pub step: u32,
    pub max_steps: u32,
    pub hypotheses: Vec<Hypothesis>,
    pub history: Vec<BeliefStep>,
    pub resolved_anchors: BTreeMap<String, String>,
    pub blocked_anchors: BTreeSet<String>,
    pub open_diagnoses: BTreeSet<String>,
    pub missing_classes: BTreeSet<String>,
}

impl BeliefState {
    /// Start a task with one fallback hypothesis at maximum uncertainty.
    pub fn new(goal: impl Into<String>, max_steps: u32) -> Self {
        let goal = goal.into();
        let hypothesis = Hypothesis::new("goal", goal.clone(), 0.5);
        Self {
            goal,
            step: 0,
            max_steps: max_steps.max(1),
            hypotheses: vec![hypothesis],
            history: Vec::new(),
            resolved_anchors: BTreeMap::new(),
            blocked_anchors: BTreeSet::new(),
            open_diagnoses: BTreeSet::new(),
            missing_classes: BTreeSet::new(),
        }
    }

    /// Add or replace a hypothesis before the loop starts.
    pub fn with_hypothesis(
        mut self,
        id: impl Into<String>,
        statement: impl Into<String>,
        confidence: f32,
    ) -> Self {
        let id = id.into();
        let hypothesis = Hypothesis::new(id.clone(), statement, confidence);
        if let Some(existing) = self.hypotheses.iter_mut().find(|item| item.id == id) {
            *existing = hypothesis;
        } else {
            self.hypotheses.push(hypothesis);
        }
        self
    }

    /// Clear step history and reopen the loop while preserving hypotheses.
    pub fn initialize(&mut self) {
        self.step = 0;
        self.history.clear();
        self.resolved_anchors.clear();
        self.blocked_anchors.clear();
        self.open_diagnoses.clear();
        self.missing_classes.clear();
    }

    pub fn last_observation_kind(&self) -> Option<&str> {
        self.history
            .last()
            .map(|step| step.observation_kind.as_str())
    }

    pub fn confidence(&self) -> f32 {
        self.best_hypothesis()
            .map(|hypothesis| hypothesis.confidence)
            .unwrap_or(0.5)
    }

    /// Highest-confidence hypothesis. Ties break on the lexicographically
    /// smaller id so persisted traces are reproducible.
    pub fn best_hypothesis(&self) -> Option<&Hypothesis> {
        self.hypotheses.iter().max_by(|left, right| {
            left.confidence
                .total_cmp(&right.confidence)
                .then_with(|| right.id.cmp(&left.id))
        })
    }

    /// Apply one observation, update every hypothesis, and select the next
    /// action from the decision layer.
    pub fn observe(&mut self, observation: BeliefObservation) -> BeliefStep {
        self.step = self.step.saturating_add(1);
        let prior = self.confidence();
        let strength = evidence_strength(&observation);
        let action = action_for(&observation, self.remaining_attempts());

        for hypothesis in &mut self.hypotheses {
            let previous = hypothesis.confidence;
            let posterior = match &observation {
                BeliefObservation::EndpointAmbiguous { .. } => shrink_toward_uncertainty(previous),
                _ => {
                    let (positive, strength) = evidence(&observation);
                    update_confidence(previous, positive, strength)
                }
            };
            hypothesis.confidence = posterior;
            match &observation {
                BeliefObservation::PathFound { .. }
                | BeliefObservation::AnchorResolved { .. }
                | BeliefObservation::Reasoning { .. } => {
                    hypothesis.support = hypothesis.support.saturating_add(1);
                }
                BeliefObservation::PathMissing { .. } => {
                    hypothesis.refutations = hypothesis.refutations.saturating_add(1);
                }
                BeliefObservation::HumanConfirmation { confirmed, .. } => {
                    if *confirmed {
                        hypothesis.support = hypothesis.support.saturating_add(1);
                    } else {
                        hypothesis.refutations = hypothesis.refutations.saturating_add(1);
                    }
                }
                BeliefObservation::EndpointAmbiguous { .. } => {}
            }
        }

        match &observation {
            BeliefObservation::PathFound { .. } => {
                self.open_diagnoses.clear();
                self.missing_classes.clear();
            }
            BeliefObservation::PathMissing {
                diagnosis,
                missing_class,
                ..
            } => {
                self.open_diagnoses.extend(diagnosis.iter().cloned());
                if let Some(missing_class) = missing_class {
                    self.missing_classes.insert(missing_class.clone());
                }
            }
            BeliefObservation::EndpointAmbiguous { anchor, .. } => {
                self.blocked_anchors.insert(anchor.clone());
            }
            BeliefObservation::AnchorResolved {
                anchor,
                resolved_id,
            } => {
                self.blocked_anchors.remove(anchor);
                self.resolved_anchors
                    .insert(anchor.clone(), resolved_id.clone());
            }
            BeliefObservation::HumanConfirmation { .. } | BeliefObservation::Reasoning { .. } => {}
        }

        let step = BeliefStep {
            step: self.step,
            observation_kind: observation.label().to_string(),
            action,
            prior_confidence: prior,
            posterior_confidence: self.confidence(),
            evidence_strength: strength,
        };
        self.history.push(step.clone());
        step
    }

    /// Deterministic reasoning action: add support without inventing a new
    /// path observation.
    pub fn apply_reasoning(&mut self, note: impl Into<String>, added_support: u32) -> BeliefStep {
        self.observe(BeliefObservation::Reasoning {
            note: note.into(),
            added_support,
        })
    }

    /// Drop hypotheses below `threshold`, but never leave the state empty.
    pub fn prune(&mut self, threshold: f32) -> usize {
        let best = self.best_hypothesis().cloned();
        let before = self.hypotheses.len();
        self.hypotheses
            .retain(|hypothesis| hypothesis.confidence >= threshold);
        if self.hypotheses.is_empty()
            && let Some(best) = best
        {
            self.hypotheses.push(best);
        }
        before - self.hypotheses.len()
    }

    pub fn remaining_attempts(&self) -> u32 {
        self.max_steps.saturating_sub(self.step)
    }

    pub fn is_exhausted(&self) -> bool {
        self.remaining_attempts() == 0
    }
}

/// Bayes update for one binary evidence observation.
///
/// `strength` is the reliability of the observation in `0.5..=0.99`. At 0.5 the
/// observation is uninformative and the prior is returned unchanged.
pub fn update_confidence(prior: f32, positive: bool, strength: f32) -> f32 {
    let prior = prior.clamp(0.001, 0.999);
    let strength = strength.clamp(0.5, 0.99);
    let likelihood_true = if positive { strength } else { 1.0 - strength };
    let likelihood_false = if positive { 1.0 - strength } else { strength };
    let odds = prior / (1.0 - prior) * likelihood_true / likelihood_false;
    (odds / (1.0 + odds)).clamp(0.001, 0.999)
}

fn shrink_toward_uncertainty(prior: f32) -> f32 {
    (prior + (0.5 - prior) * 0.25).clamp(0.001, 0.999)
}

fn evidence(observation: &BeliefObservation) -> (bool, f32) {
    match observation {
        BeliefObservation::PathFound {
            path_confidence,
            quality_flags,
            ..
        } => (
            true,
            (0.5 + 0.5 * path_confidence.clamp(0.0, 1.0) * quality_signal(quality_flags))
                .clamp(0.5, 0.99),
        ),
        BeliefObservation::PathMissing { diagnosis, .. } => {
            (false, (0.6 + 0.05 * diagnosis.len() as f32).clamp(0.5, 0.9))
        }
        BeliefObservation::AnchorResolved { .. } => (true, 0.6),
        BeliefObservation::HumanConfirmation { confirmed, .. } => (*confirmed, 0.98),
        BeliefObservation::Reasoning { added_support, .. } => {
            (true, (0.5 + 0.05 * *added_support as f32).clamp(0.5, 0.8))
        }
        BeliefObservation::EndpointAmbiguous { .. } => (true, 0.5),
    }
}

fn evidence_strength(observation: &BeliefObservation) -> f32 {
    evidence(observation).1
}

fn action_for(observation: &BeliefObservation, attempts_remaining: u32) -> DecisionAction {
    match observation {
        BeliefObservation::PathFound {
            path_confidence,
            quality_flags,
            ..
        } => {
            select_next_action(&DecisionSignals {
                has_paths: true,
                path_confidence: *path_confidence,
                quality_flags: quality_flags.clone(),
                attempts_remaining,
                ..DecisionSignals::default()
            })
            .action
        }
        BeliefObservation::PathMissing { diagnosis, .. } => {
            select_next_action(&DecisionSignals {
                diagnosis: diagnosis.clone(),
                attempts_remaining,
                ..DecisionSignals::default()
            })
            .action
        }
        BeliefObservation::EndpointAmbiguous { .. } => {
            select_next_action(&DecisionSignals {
                diagnosis: vec!["E01".to_string()],
                quality_flags: vec![PathQualityFlag::AmbiguousEntity],
                attempts_remaining,
                ..DecisionSignals::default()
            })
            .action
        }
        BeliefObservation::AnchorResolved { .. } => DecisionAction::Query,
        BeliefObservation::HumanConfirmation { confirmed, .. } => {
            if *confirmed {
                DecisionAction::Reason
            } else {
                DecisionAction::AwaitHuman
            }
        }
        BeliefObservation::Reasoning { .. } => DecisionAction::Reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_and_negative_evidence_move_confidence_in_both_directions() {
        let raised = update_confidence(0.5, true, 0.9);
        let lowered = update_confidence(0.5, false, 0.9);
        assert!(raised > 0.8, "{raised}");
        assert!(lowered < 0.2, "{lowered}");
        assert_eq!(update_confidence(0.5, true, 0.5), 0.5);
    }

    #[test]
    fn path_found_raises_confidence_and_clears_diagnoses() {
        let mut belief = BeliefState::new("root cause of the failing build", 5);
        belief.observe(BeliefObservation::PathMissing {
            from: "error:E0425".into(),
            to: "file:src/lib.rs".into(),
            diagnosis: vec!["E03".into()],
            missing_class: Some("change".into()),
        });
        assert!(belief.open_diagnoses.contains("E03"));
        let prior = belief.confidence();

        let step = belief.observe(BeliefObservation::PathFound {
            from: "error:E0425".into(),
            to: "file:src/lib.rs".into(),
            path_confidence: 0.9,
            cost: 1.2,
            quality_flags: Vec::new(),
        });
        assert!(
            belief.confidence() > prior,
            "{prior} -> {}",
            belief.confidence()
        );
        assert!(belief.open_diagnoses.is_empty());
        assert!(belief.missing_classes.is_empty());
        assert_eq!(step.action, DecisionAction::Terminate);
        assert_eq!(belief.history.len(), 2);
    }

    #[test]
    fn ambiguous_endpoint_shrinks_confidence_and_blocks_the_anchor() {
        let mut belief = BeliefState::new("connect the bug to the module", 4);
        belief.observe(BeliefObservation::PathFound {
            from: "bug".into(),
            to: "module".into(),
            path_confidence: 1.0,
            cost: 1.0,
            quality_flags: Vec::new(),
        });
        let prior = belief.confidence();
        let step = belief.observe(BeliefObservation::EndpointAmbiguous {
            anchor: "module".into(),
            candidates: vec!["module:parser".into(), "module:render".into()],
        });
        assert!(belief.confidence() < prior);
        assert!(belief.confidence() > 0.5);
        assert!(belief.blocked_anchors.contains("module"));
        assert_eq!(step.action, DecisionAction::Query);
    }

    #[test]
    fn anchor_resolution_unblocks_and_human_confirmation_dominates() {
        let mut belief = BeliefState::new("which parser module", 6);
        belief.observe(BeliefObservation::EndpointAmbiguous {
            anchor: "module".into(),
            candidates: vec!["module:a".into(), "module:b".into()],
        });
        belief.observe(BeliefObservation::AnchorResolved {
            anchor: "module".into(),
            resolved_id: "module:a".into(),
        });
        assert!(!belief.blocked_anchors.contains("module"));
        assert_eq!(
            belief.resolved_anchors.get("module").map(String::as_str),
            Some("module:a")
        );

        let confirmed = belief.observe(BeliefObservation::HumanConfirmation {
            confirmed: true,
            note: Some("yes, parser".into()),
        });
        assert!(belief.confidence() > 0.95, "{}", belief.confidence());
        assert_eq!(confirmed.action, DecisionAction::Reason);

        let refuted = belief.observe(BeliefObservation::HumanConfirmation {
            confirmed: false,
            note: None,
        });
        assert!(belief.confidence() < 0.96, "{}", belief.confidence());
        assert_eq!(refuted.action, DecisionAction::AwaitHuman);

        for _ in 0..3 {
            belief.observe(BeliefObservation::HumanConfirmation {
                confirmed: false,
                note: None,
            });
        }
        assert!(belief.confidence() < 0.05, "{}", belief.confidence());
    }

    #[test]
    fn reasoning_support_and_pruning_keep_at_least_one_hypothesis() {
        let mut belief = BeliefState::new("root cause", 5).with_hypothesis(
            "stale-cache",
            "the cache is stale",
            0.2,
        );
        assert_eq!(
            belief.best_hypothesis().map(|item| item.id.as_str()),
            Some("goal")
        );

        let step = belief.apply_reasoning("the cache timestamp predates the deploy", 3);
        assert!(step.posterior_confidence > step.prior_confidence);
        assert_eq!(step.action, DecisionAction::Reason);

        let removed = belief.prune(0.99);
        assert_eq!(removed, 1);
        assert_eq!(belief.hypotheses.len(), 1);
        assert_eq!(
            belief.best_hypothesis().map(|item| item.id.as_str()),
            Some("goal")
        );
    }

    #[test]
    fn budget_exhaustion_blocks_a_safe_action() {
        let mut belief = BeliefState::new("find the path", 2);
        let first = belief.observe(BeliefObservation::PathMissing {
            from: "a".into(),
            to: "b".into(),
            diagnosis: vec!["E01".into()],
            missing_class: None,
        });
        assert_eq!(first.action, DecisionAction::Query);
        assert!(!belief.is_exhausted());

        let second = belief.observe(BeliefObservation::PathMissing {
            from: "a".into(),
            to: "b".into(),
            diagnosis: vec!["E01".into()],
            missing_class: None,
        });
        assert_eq!(second.action, DecisionAction::AwaitHuman);
        assert!(belief.is_exhausted());
    }

    #[test]
    fn best_hypothesis_ties_break_on_smaller_id() {
        let belief = BeliefState::new("goal", 3)
            .with_hypothesis("zeta", "z", 0.5)
            .with_hypothesis("alpha", "a", 0.5);
        assert_eq!(
            belief.best_hypothesis().map(|item| item.id.as_str()),
            Some("alpha")
        );
    }
}
