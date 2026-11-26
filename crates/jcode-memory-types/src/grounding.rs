//! Stage S2 grounding: mention -> ontology concept candidates (design 4.3.2).
//!
//! The design's score has five terms (embedding, lexical, type, context,
//! recency). This deterministic implementation only has lexical signal: exact
//! match, token containment, partial substring, and description token overlap.
//! It reports which signals fired so callers can see that embedding/type terms
//! were unavailable rather than silently treating the score as complete.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::engineering_ontology::{EngineeringClassDef, EngineeringOntology};
use crate::task_frame::{TaskAnchor, TaskFrame};

/// Acceptance threshold: at or above this the mention is grounded.
pub const GROUNDING_ACCEPT_SCORE: f32 = 0.75;
/// Lower bound of the ambiguous band. Below it the mention is unresolved.
pub const GROUNDING_AMBIGUOUS_SCORE: f32 = 0.50;
/// Maximum candidates retained per mention.
pub const GROUNDING_MAX_CANDIDATES: usize = 5;

/// Grounding verdict for one mention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundingStatus {
    Grounded,
    Ambiguous,
    Unresolved,
}

impl GroundingStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Grounded => "grounded",
            Self::Ambiguous => "ambiguous",
            Self::Unresolved => "unresolved",
        }
    }
}

/// One scored concept candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConceptCandidate {
    pub concept_id: String,
    pub display_name: String,
    pub score: f32,
    /// Which deterministic signals contributed, e.g. `exact match`.
    #[serde(default)]
    pub signals: Vec<String>,
}

/// Grounding result for one anchor mention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MentionGrounding {
    pub mention: String,
    pub status: GroundingStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted: Option<ConceptCandidate>,
    #[serde(default)]
    pub candidates: Vec<ConceptCandidate>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Grounding result for every anchor in a task frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameGrounding {
    pub anchors: Vec<MentionGrounding>,
    pub grounded: usize,
    pub ambiguous: usize,
    pub unresolved: usize,
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl FrameGrounding {
    pub fn resolved(&self) -> bool {
        self.ambiguous == 0 && self.unresolved == 0
    }
}

fn tokens(text: &str) -> HashSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| token.len() >= 3)
        .map(str::to_lowercase)
        .collect()
}

fn jaccard(left: &HashSet<String>, right: &HashSet<String>) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let shared = left.intersection(right).count();
    shared as f32 / left.union(right).count() as f32
}

fn score_class(mention: &str, class: &EngineeringClassDef) -> Option<(f32, Vec<String>)> {
    let mention_lower = mention.trim().to_lowercase();
    if mention_lower.is_empty() {
        return None;
    }
    let name_lower = class.display_name.to_lowercase();
    let id_lower = class.id.to_lowercase();
    if mention_lower == name_lower || mention_lower == id_lower {
        return Some((1.0, vec!["exact match".to_string()]));
    }

    let mention_tokens = tokens(&mention_lower);
    let name_tokens = tokens(&name_lower);
    if !mention_tokens.is_empty()
        && !name_tokens.is_empty()
        && (mention_tokens.is_subset(&name_tokens) || name_tokens.is_subset(&mention_tokens))
    {
        return Some((0.85, vec!["token containment".to_string()]));
    }

    let longer_contains = (name_lower.len() >= 3 && mention_lower.contains(&name_lower))
        || (mention_lower.len() >= 3 && name_lower.contains(&mention_lower));
    if longer_contains {
        return Some((0.55, vec!["partial substring".to_string()]));
    }

    let lexical = jaccard(&mention_tokens, &name_tokens);
    let description = class
        .description
        .as_deref()
        .map(|description| jaccard(&mention_tokens, &tokens(description)))
        .unwrap_or(0.0);
    let score = (0.6 * lexical + 0.2 * description).clamp(0.0, 0.74);
    if score <= 0.0 {
        return None;
    }
    let mut signals = Vec::new();
    if lexical > 0.0 {
        signals.push(format!("name token overlap {lexical:.2}"));
    }
    if description > 0.0 {
        signals.push(format!("description token overlap {description:.2}"));
    }
    Some((score, signals))
}

/// Ground one mention against the ontology concept layer.
pub fn ground_mention(ontology: &EngineeringOntology, mention: &str) -> MentionGrounding {
    let mut candidates: Vec<ConceptCandidate> = ontology
        .classes
        .values()
        .filter_map(|class| {
            score_class(mention, class).map(|(score, signals)| ConceptCandidate {
                concept_id: class.id.clone(),
                display_name: class.display_name.clone(),
                score,
                signals,
            })
        })
        .collect();
    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.concept_id.cmp(&right.concept_id))
    });
    candidates.truncate(GROUNDING_MAX_CANDIDATES);

    let mut warnings = Vec::new();
    let top = candidates
        .first()
        .map(|candidate| candidate.score)
        .unwrap_or(0.0);
    let status = if top >= GROUNDING_ACCEPT_SCORE {
        GroundingStatus::Grounded
    } else if top >= GROUNDING_AMBIGUOUS_SCORE {
        GroundingStatus::Ambiguous
    } else {
        GroundingStatus::Unresolved
    };
    if status != GroundingStatus::Grounded {
        warnings.push(
            "embedding/type/context terms are unavailable; score is lexical-only".to_string(),
        );
    }
    let accepted = if status == GroundingStatus::Grounded {
        candidates.first().cloned()
    } else {
        None
    };
    if status == GroundingStatus::Ambiguous {
        warnings.push(format!(
            "top candidate is below the accept threshold {GROUNDING_ACCEPT_SCORE}"
        ));
    }
    MentionGrounding {
        mention: mention.to_string(),
        status,
        accepted,
        candidates,
        warnings,
    }
}

/// Ground every anchor mention in a task frame.
pub fn ground_task_frame(ontology: &EngineeringOntology, frame: &TaskFrame) -> FrameGrounding {
    let anchors: Vec<MentionGrounding> = frame
        .anchors
        .iter()
        .map(|anchor: &TaskAnchor| ground_mention(ontology, &anchor.mention))
        .collect();
    let grounded = anchors
        .iter()
        .filter(|anchor| anchor.status == GroundingStatus::Grounded)
        .count();
    let ambiguous = anchors
        .iter()
        .filter(|anchor| anchor.status == GroundingStatus::Ambiguous)
        .count();
    let unresolved = anchors
        .iter()
        .filter(|anchor| anchor.status == GroundingStatus::Unresolved)
        .count();
    let mut warnings = Vec::new();
    if unresolved > 0 {
        warnings.push(format!(
            "{unresolved} mention(s) stay unresolved and should route to 4.4 E01"
        ));
    }
    if ambiguous > 0 {
        warnings.push(format!(
            "{ambiguous} mention(s) are ambiguous; keep top-k until path compatibility decides"
        ));
    }
    FrameGrounding {
        anchors,
        grounded,
        ambiguous,
        unresolved,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engineering_ontology::default_engineering_ontology;
    use crate::task_frame::compile_task_frame_json;
    use serde_json::json;

    #[test]
    fn grounds_exact_and_token_containment_mentions() {
        let ontology = default_engineering_ontology();
        let exact = ground_mention(&ontology, "File");
        assert_eq!(exact.status, GroundingStatus::Grounded);
        assert_eq!(
            exact.accepted.as_ref().map(|c| c.concept_id.as_str()),
            Some("file")
        );
        assert!((exact.accepted.as_ref().unwrap().score - 1.0).abs() < 1e-6);

        let contained = ground_mention(&ontology, "build failure");
        assert_eq!(contained.status, GroundingStatus::Grounded);
        assert_eq!(
            contained.accepted.as_ref().map(|c| c.concept_id.as_str()),
            Some("build")
        );
    }

    #[test]
    fn keeps_partial_substring_matches_ambiguous() {
        let ontology = default_engineering_ontology();
        let ambiguous = ground_mention(&ontology, "Files");
        assert_eq!(ambiguous.status, GroundingStatus::Ambiguous);
        assert!(ambiguous.accepted.is_none());
        assert!(!ambiguous.candidates.is_empty());
        assert!(
            ambiguous
                .warnings
                .iter()
                .any(|warning| warning.contains("below the accept threshold")),
            "{:?}",
            ambiguous.warnings
        );
    }

    #[test]
    fn reports_unresolved_mentions_and_lexical_only_caveat() {
        let ontology = default_engineering_ontology();
        let unresolved = ground_mention(&ontology, "quantum flux capacitor");
        assert_eq!(unresolved.status, GroundingStatus::Unresolved);
        assert!(unresolved.candidates.is_empty());
        assert!(
            unresolved
                .warnings
                .iter()
                .any(|warning| warning.contains("lexical-only")),
            "{:?}",
            unresolved.warnings
        );
    }

    #[test]
    fn grounds_all_frame_anchors_and_summarizes_counts() {
        let frame = json!({
            "task_id": "TASK-G",
            "intent": "plan",
            "anchors": [
                {"role": "start", "mention": "File"},
                {"role": "target", "mention": "API endpoint"}
            ]
        });
        let report = compile_task_frame_json(&frame).expect("compile");
        let grounding = ground_task_frame(&default_engineering_ontology(), &report.frame);
        assert_eq!(grounding.anchors.len(), 2);
        assert_eq!(grounding.grounded, 2, "{grounding:?}");
        assert_eq!(grounding.ambiguous, 0);
        assert_eq!(grounding.unresolved, 0);
        assert!(grounding.resolved());
        assert_eq!(
            grounding.anchors[1]
                .accepted
                .as_ref()
                .map(|candidate| candidate.concept_id.as_str()),
            Some("api")
        );
    }
}
