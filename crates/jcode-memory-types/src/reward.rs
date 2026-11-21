//! Dense reward shaping for path-selection and decision traces.
//!
//! The design document defines accuracy, efficiency, semantic fit, constraint
//! satisfaction, completion, plan, and quality terms. The values here are
//! normalized proxies supplied by the caller; they are intentionally not
//! presented as calibrated rewards.

use serde::{Deserialize, Serialize};

use crate::path_quality::PathQualityFlag;

/// Weights for the four base terms plus the three dense terms.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RewardWeights {
    pub accuracy: f32,
    pub efficiency: f32,
    pub semantic: f32,
    pub constraint: f32,
    pub completion: f32,
    pub plan: f32,
    pub quality: f32,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            accuracy: 0.4,
            efficiency: 0.2,
            semantic: 0.2,
            constraint: 0.2,
            completion: 1.0,
            plan: 1.5,
            quality: 1.0,
        }
    }
}

/// Normalized signals in `0.0..=1.0` (higher is better).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct RewardSignals {
    pub accuracy: f32,
    pub efficiency: f32,
    pub semantic: f32,
    pub constraint: f32,
    pub completion: f32,
    pub plan: f32,
    pub quality: f32,
}

/// Weighted sum of the seven terms. The result is a comparative score, not a
/// probability and not a calibrated return.
pub fn shaped_reward(signals: &RewardSignals, weights: &RewardWeights) -> f32 {
    weights.accuracy * signals.accuracy.clamp(0.0, 1.0)
        + weights.efficiency * signals.efficiency.clamp(0.0, 1.0)
        + weights.semantic * signals.semantic.clamp(0.0, 1.0)
        + weights.constraint * signals.constraint.clamp(0.0, 1.0)
        + weights.completion * signals.completion.clamp(0.0, 1.0)
        + weights.plan * signals.plan.clamp(0.0, 1.0)
        + weights.quality * signals.quality.clamp(0.0, 1.0)
}

/// Penalty contributed by local quality flags. Capped at 1.0 so one path
/// cannot dominate the whole reward term.
pub fn quality_penalty(flags: &[PathQualityFlag]) -> f32 {
    let mut penalty: f32 = 0.0;
    for flag in flags {
        penalty += match flag {
            PathQualityFlag::StaleEdge => 0.25,
            PathQualityFlag::HypothesizedEdge => 0.5,
            PathQualityFlag::HighDegreeHub => 0.15,
            PathQualityFlag::AmbiguousEntity => 0.4,
            PathQualityFlag::LowProvenance => 0.3,
            PathQualityFlag::StitchedPath => 0.2,
            PathQualityFlag::RelaxedQuery => 0.1,
        };
    }
    penalty.min(1.0)
}

/// Convenience quality signal for a path.
pub fn quality_signal(flags: &[PathQualityFlag]) -> f32 {
    1.0 - quality_penalty(flags)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_and_signals_produce_a_comparable_score() {
        let weights = RewardWeights::default();
        let clean = shaped_reward(
            &RewardSignals {
                accuracy: 1.0,
                efficiency: 1.0,
                semantic: 1.0,
                constraint: 1.0,
                completion: 1.0,
                plan: 1.0,
                quality: 1.0,
            },
            &weights,
        );
        let weak = shaped_reward(
            &RewardSignals {
                accuracy: 0.5,
                efficiency: 0.5,
                semantic: 0.5,
                constraint: 0.5,
                completion: 0.0,
                plan: 0.0,
                quality: 0.5,
            },
            &weights,
        );
        assert!(clean > weak);
    }

    #[test]
    fn quality_penalty_is_bounded_and_hypotheses_cost_more() {
        let clean = quality_penalty(&[]);
        let hypothesis = quality_penalty(&[
            PathQualityFlag::HypothesizedEdge,
            PathQualityFlag::StaleEdge,
        ]);
        assert_eq!(clean, 0.0);
        assert!(hypothesis > 0.5);
        assert_eq!(
            quality_penalty(&[PathQualityFlag::HypothesizedEdge; 10]),
            1.0
        );
        assert_eq!(quality_signal(&[]), 1.0);
    }
}
