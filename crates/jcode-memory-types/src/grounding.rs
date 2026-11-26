//! Stage S2 grounding: mention -> ontology concept candidates (design 4.3.2).
//!
//! The design's score has five terms (embedding, lexical, type, context,
//! recency). This deterministic implementation only has lexical signal: exact
//! match, token containment, partial substring, and description token overlap.
//! It reports which signals fired so callers can see that embedding/type terms
//! were unavailable rather than silently treating the score as complete.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::engineering_ontology::{
    EngineeringClassDef, EngineeringGraph, EngineeringOntology, concept_node_id,
};
use crate::task_frame::{TaskAnchor, TaskAnchorRole, TaskFrame};

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

/// Why one ambiguity candidate was rejected during the second pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AmbiguityRejection {
    pub concept_id: String,
    pub reason: String,
}

/// Second-pass disambiguation: pick the ambiguous candidate that can reach the
/// already-grounded endpoint on the ontology concept layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AmbiguityResolution {
    pub mention: String,
    pub target_concept: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen: Option<ConceptCandidate>,
    #[serde(default)]
    pub rejected: Vec<AmbiguityRejection>,
}

/// Resolve one ambiguous mention by concept-path compatibility with `target`.
///
/// Candidates that cannot reach the target within `max_depth` are rejected with
/// the reason; the shortest path wins, then the higher lexical score, then the
/// lexicographically smaller concept id. Returns `chosen: None` when no
/// candidate is compatible, leaving the mention unresolved.
pub fn disambiguate_by_concept_path(
    ontology: &EngineeringOntology,
    grounding: &MentionGrounding,
    target_concept: &str,
    max_depth: usize,
) -> AmbiguityResolution {
    let mut resolution = AmbiguityResolution {
        mention: grounding.mention.clone(),
        target_concept: target_concept.to_string(),
        chosen: None,
        rejected: Vec::new(),
    };
    if !ontology.classes.contains_key(target_concept) {
        for candidate in &grounding.candidates {
            resolution.rejected.push(AmbiguityRejection {
                concept_id: candidate.concept_id.clone(),
                reason: format!("target concept `{target_concept}` is not declared"),
            });
        }
        return resolution;
    }
    let graph = EngineeringGraph::concept_graph(ontology);
    let target_node = concept_node_id(target_concept);
    let mut compatible: Vec<(usize, ConceptCandidate)> = Vec::new();
    for candidate in &grounding.candidates {
        let from = concept_node_id(&candidate.concept_id);
        match graph.shortest_path(&from, &target_node, max_depth, &[]) {
            Some(path) => {
                let hops = path.len().saturating_sub(1);
                compatible.push((hops, candidate.clone()));
            }
            None => resolution.rejected.push(AmbiguityRejection {
                concept_id: candidate.concept_id.clone(),
                reason: format!("no concept path to `{target_concept}` within {max_depth} hop(s)"),
            }),
        }
    }
    compatible.sort_by(|(left_hops, left), (right_hops, right)| {
        left_hops
            .cmp(right_hops)
            .then_with(|| right.score.total_cmp(&left.score))
            .then_with(|| left.concept_id.cmp(&right.concept_id))
    });
    if let Some((chosen_hops, chosen)) = compatible.first().cloned() {
        for (hops, candidate) in compatible.iter().skip(1) {
            resolution.rejected.push(AmbiguityRejection {
                concept_id: candidate.concept_id.clone(),
                reason: format!(
                    "longer or weaker match: {hops} hop(s), score {:.2} (chosen {} hop(s))",
                    candidate.score, chosen_hops
                ),
            });
        }
        resolution.chosen = Some(chosen);
    }
    resolution
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resolutions: Vec<AmbiguityResolution>,
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

/// Resolve ambiguous frame anchors by concept-path compatibility (S2 second
/// pass). Returns how many anchors were grounded. Resolutions, including the
/// failed ones, are appended to `grounding.resolutions` so callers can audit
/// why a mention stayed ambiguous.
pub fn resolve_frame_ambiguities(
    ontology: &EngineeringOntology,
    frame: &TaskFrame,
    grounding: &mut FrameGrounding,
    max_depth: usize,
) -> usize {
    let mut resolved = 0;
    for _ in 0..2 {
        let pairs: Vec<(usize, String)> = frame
            .anchors
            .iter()
            .enumerate()
            .filter_map(|(index, anchor)| {
                let current = grounding.anchors.get(index)?;
                if current.accepted.is_some() || current.status != GroundingStatus::Ambiguous {
                    return None;
                }
                let other_role = match anchor.role {
                    TaskAnchorRole::Start => TaskAnchorRole::Target,
                    TaskAnchorRole::Target => TaskAnchorRole::Start,
                    _ => return None,
                };
                let (other_index, _) =
                    frame
                        .anchors
                        .iter()
                        .enumerate()
                        .find(|(other_index, other)| {
                            *other_index != index && other.role == other_role
                        })?;
                let target = grounding
                    .anchors
                    .get(other_index)?
                    .accepted
                    .as_ref()?
                    .concept_id
                    .clone();
                Some((index, target))
            })
            .collect();
        let mut progress = 0;
        for (index, target) in pairs {
            let resolution = disambiguate_by_concept_path(
                ontology,
                &grounding.anchors[index],
                &target,
                max_depth,
            );
            if let Some(chosen) = resolution.chosen.clone() {
                let anchor = &mut grounding.anchors[index];
                anchor.status = GroundingStatus::Grounded;
                anchor.accepted = Some(chosen.clone());
                anchor.warnings.push(format!(
                    "ambiguity resolved by concept-path compatibility: {}",
                    chosen.concept_id
                ));
                progress += 1;
                resolved += 1;
            }
            grounding.resolutions.push(resolution);
        }
        if progress == 0 {
            break;
        }
    }
    if resolved > 0 {
        grounding.grounded = grounding
            .anchors
            .iter()
            .filter(|anchor| anchor.status == GroundingStatus::Grounded)
            .count();
        grounding.ambiguous = grounding
            .anchors
            .iter()
            .filter(|anchor| anchor.status == GroundingStatus::Ambiguous)
            .count();
    }
    resolved
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
        resolutions: Vec::new(),
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

    fn path_ontology() -> EngineeringOntology {
        use crate::engineering_ontology::{
            BfoCategory, EngineeringClassDef, EngineeringRelationDef,
        };
        let mut ontology = EngineeringOntology::new("test", 1, "path test ontology");
        ontology.classes.clear();
        for (id, name) in [
            ("goal", "Goal"),
            ("near", "Near"),
            ("near2", "Near Two"),
            ("far", "Far"),
        ] {
            ontology.classes.insert(
                id.to_string(),
                EngineeringClassDef::new(id, name, BfoCategory::IndependentContinuant),
            );
        }
        for (id, domain) in [("links", "near"), ("links2", "near2")] {
            ontology.relations.insert(
                id.to_string(),
                EngineeringRelationDef::new(id, id, &[domain], &["goal"]),
            );
        }
        ontology
    }

    fn candidate(concept_id: &str, score: f32) -> ConceptCandidate {
        ConceptCandidate {
            concept_id: concept_id.to_string(),
            display_name: concept_id.to_string(),
            score,
            signals: Vec::new(),
        }
    }

    fn ambiguous(candidates: Vec<ConceptCandidate>) -> MentionGrounding {
        MentionGrounding {
            mention: "thing".to_string(),
            status: GroundingStatus::Ambiguous,
            accepted: None,
            candidates,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn disambiguation_prefers_a_concept_path_over_a_higher_lexical_score() {
        let ontology = path_ontology();
        let grounding = ambiguous(vec![candidate("far", 0.72), candidate("near", 0.55)]);
        let resolution = disambiguate_by_concept_path(&ontology, &grounding, "goal", 2);
        assert_eq!(
            resolution.chosen.as_ref().map(|c| c.concept_id.as_str()),
            Some("near")
        );
        assert_eq!(resolution.rejected.len(), 1);
        assert_eq!(resolution.rejected[0].concept_id, "far");
        assert!(
            resolution.rejected[0].reason.contains("no concept path"),
            "{}",
            resolution.rejected[0].reason
        );
    }

    #[test]
    fn disambiguation_tie_breaks_on_score_then_id() {
        let ontology = path_ontology();
        let grounding = ambiguous(vec![candidate("near", 0.60), candidate("near2", 0.68)]);
        let resolution = disambiguate_by_concept_path(&ontology, &grounding, "goal", 2);
        assert_eq!(
            resolution.chosen.as_ref().map(|c| c.concept_id.as_str()),
            Some("near2")
        );
        assert_eq!(resolution.rejected[0].concept_id, "near");
        assert!(resolution.rejected[0].reason.contains("chosen 1 hop"));
    }

    #[test]
    fn disambiguation_reports_an_undeclared_target_instead_of_guessing() {
        let ontology = path_ontology();
        let grounding = ambiguous(vec![candidate("near", 0.9)]);
        let resolution = disambiguate_by_concept_path(&ontology, &grounding, "missing", 2);
        assert!(resolution.chosen.is_none());
        assert_eq!(resolution.rejected.len(), 1);
        assert!(
            resolution.rejected[0].reason.contains("not declared"),
            "{}",
            resolution.rejected[0].reason
        );
    }

    #[test]
    fn disambiguation_leaves_the_mention_unresolved_without_a_compatible_candidate() {
        let ontology = path_ontology();
        let grounding = ambiguous(vec![candidate("far", 0.9)]);
        let resolution = disambiguate_by_concept_path(&ontology, &grounding, "goal", 1);
        assert!(resolution.chosen.is_none());
        assert_eq!(resolution.rejected.len(), 1);
    }

    #[test]
    fn frame_ambiguity_resolution_grounds_the_anchor_and_records_the_basis() {
        let ontology = path_ontology();
        let frame = json!({
            "intent": "plan",
            "anchors": [
                {"role": "start", "mention": "thing"},
                {"role": "target", "mention": "Goal"}
            ]
        });
        let report = compile_task_frame_json(&frame).expect("compile");
        let mut grounding = FrameGrounding {
            anchors: vec![
                ambiguous(vec![candidate("far", 0.72), candidate("near", 0.55)]),
                MentionGrounding {
                    mention: "Goal".to_string(),
                    status: GroundingStatus::Grounded,
                    accepted: Some(candidate("goal", 1.0)),
                    candidates: vec![candidate("goal", 1.0)],
                    warnings: Vec::new(),
                },
            ],
            grounded: 1,
            ambiguous: 1,
            unresolved: 0,
            warnings: Vec::new(),
            resolutions: Vec::new(),
        };
        let resolved = resolve_frame_ambiguities(&ontology, &report.frame, &mut grounding, 2);
        assert_eq!(resolved, 1);
        assert_eq!(grounding.grounded, 2);
        assert_eq!(grounding.ambiguous, 0);
        assert_eq!(
            grounding.anchors[0]
                .accepted
                .as_ref()
                .map(|candidate| candidate.concept_id.as_str()),
            Some("near")
        );
        assert!(
            grounding.anchors[0]
                .warnings
                .iter()
                .any(|warning| warning.contains("concept-path compatibility")),
            "{:?}",
            grounding.anchors[0].warnings
        );
        assert_eq!(grounding.resolutions.len(), 1);
        assert_eq!(
            grounding.resolutions[0]
                .chosen
                .as_ref()
                .map(|candidate| candidate.concept_id.as_str()),
            Some("near")
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
