//! Verified decision-step traces and lexical retrieval.
//!
//! Design section 5.2.6 uses validated `(problem, start node, selected
//! relation, direction)` examples as few-shot candidates when re-ranking
//! relations. Only `verified` traces may enter the registry: an unverified
//! trajectory would teach the model to repeat the mistake.

use std::collections::BTreeSet;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Direction of the relation selected in a decision step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionDirection {
    Outgoing,
    Incoming,
    Undirected,
}

impl DecisionDirection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Outgoing => "outgoing",
            Self::Incoming => "incoming",
            Self::Undirected => "undirected",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "outgoing" | "out" | "forward" => Some(Self::Outgoing),
            "incoming" | "in" | "reverse" => Some(Self::Incoming),
            "undirected" | "both" => Some(Self::Undirected),
            _ => None,
        }
    }
}

/// One validated decision step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionStep {
    pub id: String,
    pub problem: String,
    pub start_node: String,
    pub relation: String,
    pub direction: DecisionDirection,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_node: Option<String>,
    /// Must be `verified` for ranking. Reserved values such as `candidate` or
    /// `rejected` stay visible for audit but never become examples.
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    pub provenance: String,
    pub created_at: DateTime<Utc>,
}

impl DecisionStep {
    #[allow(clippy::too_many_arguments)]
    pub fn verified(
        id: impl Into<String>,
        problem: impl Into<String>,
        start_node: impl Into<String>,
        relation: impl Into<String>,
        direction: DecisionDirection,
        provenance: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            problem: problem.into(),
            start_node: start_node.into(),
            relation: relation.into(),
            direction,
            goal_node: None,
            source: "verified".to_string(),
            model_version: None,
            outcome: None,
            provenance: provenance.into(),
            created_at: Utc::now(),
        }
    }

    pub fn with_goal(mut self, goal_node: impl Into<String>) -> Self {
        self.goal_node = Some(goal_node.into());
        self
    }

    pub fn with_model_version(mut self, version: impl Into<String>) -> Self {
        self.model_version = Some(version.into());
        self
    }

    pub fn with_outcome(mut self, outcome: impl Into<String>) -> Self {
        self.outcome = Some(outcome.into());
        self
    }

    pub fn is_verified(&self) -> bool {
        self.source.eq_ignore_ascii_case("verified")
    }
}

/// Retrieval request for decision-step examples.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DecisionStepQuery {
    pub problem: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_node: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relation: Option<String>,
    #[serde(default)]
    pub limit: usize,
}

impl DecisionStepQuery {
    pub fn new(problem: impl Into<String>) -> Self {
        Self {
            problem: problem.into(),
            start_node: None,
            relation: None,
            limit: 5,
        }
    }

    pub fn with_start_node(mut self, start_node: impl Into<String>) -> Self {
        self.start_node = Some(start_node.into());
        self
    }

    pub fn with_relation(mut self, relation: impl Into<String>) -> Self {
        self.relation = Some(relation.into());
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

/// One ranked retrieval hit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionStepMatch {
    pub step: DecisionStep,
    pub score: f32,
}

/// Rank verified traces by lexical problem similarity plus exact structural
/// matches. The structural bonuses keep a same-anchor example visible when the
/// problem wording differs, without letting an unrelated anchor rank first.
pub fn rank_decision_steps(
    query: &DecisionStepQuery,
    steps: &[DecisionStep],
) -> Vec<DecisionStepMatch> {
    if query.limit == 0 {
        return Vec::new();
    }
    let query_terms = terms(&query.problem);
    let mut matches: Vec<DecisionStepMatch> = steps
        .iter()
        .filter(|step| step.is_verified())
        .filter_map(|step| {
            let step_terms = terms(&step.problem);
            let lexical = jaccard(&query_terms, &step_terms);
            let mut score = lexical;
            if let Some(start_node) = query.start_node.as_deref()
                && step.start_node.eq_ignore_ascii_case(start_node)
            {
                score += 0.15;
            }
            if let Some(relation) = query.relation.as_deref()
                && step.relation.eq_ignore_ascii_case(relation)
            {
                score += 0.10;
            }
            if score <= 0.0 {
                return None;
            }
            Some(DecisionStepMatch {
                step: step.clone(),
                score: score.min(1.0),
            })
        })
        .collect();
    matches.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.step.id.cmp(&right.step.id))
    });
    matches.truncate(query.limit);
    matches
}

fn jaccard(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let intersection = left.intersection(right).count() as f32;
    let union = left.union(right).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn terms(text: &str) -> BTreeSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric() && ch != '_' && ch != '-')
        .filter_map(|raw| {
            let token = raw.trim_matches(|ch: char| ch == '-' || ch == '_');
            if token.chars().count() < 2 {
                return None;
            }
            Some(token.to_ascii_lowercase())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(id: &str, problem: &str, start: &str, relation: &str) -> DecisionStep {
        DecisionStep::verified(
            id,
            problem,
            start,
            relation,
            DecisionDirection::Outgoing,
            format!("eval:{id}"),
        )
    }

    #[test]
    fn only_verified_steps_are_ranked() {
        let mut unverified = step("u", "fix parser crash", "module:parser", "causes");
        unverified.source = "candidate".to_string();
        let steps = vec![
            step("v", "fix parser crash", "module:parser", "causes"),
            unverified,
        ];
        let hits = rank_decision_steps(&DecisionStepQuery::new("fix parser crash"), &steps);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].step.id, "v");
        assert_eq!(hits[0].step.source, "verified");
    }

    #[test]
    fn lexical_match_outranks_weaker_match() {
        let steps = vec![
            step("parser", "fix parser crash", "module:parser", "causes"),
            step("deploy", "deploy parser service", "service:api", "uses"),
        ];
        let hits = rank_decision_steps(&DecisionStepQuery::new("parser crash root cause"), &steps);
        assert_eq!(hits[0].step.id, "parser");
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn exact_anchor_and_relation_add_structural_score() {
        let steps = vec![
            step("a", "unrelated wording", "module:parser", "causes"),
            step("b", "unrelated wording", "module:render", "uses"),
        ];
        let query = DecisionStepQuery::new("different problem")
            .with_start_node("module:parser")
            .with_relation("causes");
        let hits = rank_decision_steps(&query, &steps);
        assert_eq!(hits[0].step.id, "a");
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn limit_and_ties_are_deterministic() {
        let steps = vec![
            step("b", "same problem", "x", "relates_to"),
            step("a", "same problem", "x", "relates_to"),
        ];
        let query = DecisionStepQuery::new("same problem").with_limit(1);
        let hits = rank_decision_steps(&query, &steps);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].step.id, "a");
        assert!(rank_decision_steps(&DecisionStepQuery::new("x").with_limit(0), &steps).is_empty());
    }

    #[test]
    fn direction_parses_common_spellings() {
        assert_eq!(
            DecisionDirection::parse("forward"),
            Some(DecisionDirection::Outgoing)
        );
        assert_eq!(
            DecisionDirection::parse("reverse"),
            Some(DecisionDirection::Incoming)
        );
        assert_eq!(
            DecisionDirection::parse("both"),
            Some(DecisionDirection::Undirected)
        );
        assert_eq!(DecisionDirection::parse("sideways"), None);
    }
}
