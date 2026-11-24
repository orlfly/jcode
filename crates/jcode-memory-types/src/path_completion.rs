//! Completion-ladder write-back for the engineering graph (design 4.4.5).
//!
//! Completion edges are always candidates: they are written with
//! `hypothesis = true`, a 30-day TTL, and the proposing source in provenance.
//! Promotion to an asserted edge requires non-LLM evidence, so a model proposal
//! can never confirm itself. Nothing here executes or plans; it only records
//! candidate edges for review and reports the resulting completion debt.

use chrono::{Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::engineering_ontology::{
    EngineeringGraph, EngineeringProvenance, EngineeringRelation, EntityStatus,
};

/// Minimum confidence a candidate edge needs before it may enter the graph.
pub const COMPLETION_MIN_CONFIDENCE: f32 = 0.7;
/// Maximum number of new candidate edges accepted per write call.
pub const COMPLETION_MAX_NEW_EDGES: usize = 5;
/// Days an unconfirmed candidate edge stays effective.
pub const COMPLETION_TTL_DAYS: i64 = 30;

/// One candidate edge proposed by the completion ladder.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompletionEdge {
    pub from: String,
    pub relation: String,
    pub to: String,
    pub confidence: f32,
    /// Verifiable evidence locator. Free-form model confidence is not evidence.
    pub evidence: String,
    /// Proposing source, e.g. `llm:proposal` or `doc:cert-list.pdf#p3`.
    pub source: String,
}

impl CompletionEdge {
    pub fn new(
        from: impl Into<String>,
        relation: impl Into<String>,
        to: impl Into<String>,
        confidence: f32,
    ) -> Self {
        Self {
            from: from.into(),
            relation: relation.into(),
            to: to.into(),
            confidence,
            evidence: String::new(),
            source: "llm:proposal".to_string(),
        }
    }

    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence = evidence.into();
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }
}

/// Why a candidate edge was not written.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompletionRejectionCode {
    MissingEndpoint,
    SelfLoop,
    InvalidRelation,
    LowConfidence,
    MissingEvidence,
    DuplicateAsserted,
    DuplicateHypothesis,
    BudgetExhausted,
}

impl CompletionRejectionCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MissingEndpoint => "missing_endpoint",
            Self::SelfLoop => "self_loop",
            Self::InvalidRelation => "invalid_relation",
            Self::LowConfidence => "low_confidence",
            Self::MissingEvidence => "missing_evidence",
            Self::DuplicateAsserted => "duplicate_asserted",
            Self::DuplicateHypothesis => "duplicate_hypothesis",
            Self::BudgetExhausted => "budget_exhausted",
        }
    }
}

/// One rejected candidate with its reason.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompletionRejection {
    pub code: CompletionRejectionCode,
    pub from: String,
    pub relation: String,
    pub to: String,
    pub message: String,
}

/// Outcome of one write-back call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompletionWriteReport {
    /// Relation ids that were written as hypothesized edges.
    pub accepted: Vec<String>,
    pub rejected: Vec<CompletionRejection>,
    /// Unconfirmed hypothesis edges / all edges after the write.
    pub completion_debt: f32,
}

/// Why a hypothesis edge could not be promoted to an asserted edge.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompletionPromotionError {
    NotFound,
    AlreadyAsserted,
    MissingEvidenceSource,
    LlmCannotPromote,
    WeakEvidence,
}

impl CompletionPromotionError {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotFound => "not_found",
            Self::AlreadyAsserted => "already_asserted",
            Self::MissingEvidenceSource => "missing_evidence_source",
            Self::LlmCannotPromote => "llm_cannot_promote",
            Self::WeakEvidence => "weak_evidence",
        }
    }
}

impl std::fmt::Display for CompletionPromotionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::error::Error for CompletionPromotionError {}

/// True when the provenance source is a model proposal.
///
/// Model output may propose edges and evidence locators, but it can never be the
/// confirming source: promotion must come from an external system, a document
/// locator review, or a human.
pub fn is_llm_source(source: &str) -> bool {
    let source = source.trim().to_ascii_lowercase();
    source.starts_with("llm") || source.starts_with("model") || source.starts_with("completion:llm")
}

impl EngineeringGraph {
    /// Write candidate edges as hypothesized, expiring edges.
    ///
    /// Never overwrites an asserted edge, never writes the same candidate twice,
    /// and never accepts a candidate without a verifiable evidence locator.
    pub fn write_completion_edges(
        &mut self,
        edges: impl IntoIterator<Item = CompletionEdge>,
        source_tag: &str,
    ) -> CompletionWriteReport {
        let now = Utc::now();
        let mut accepted = Vec::new();
        let mut rejected = Vec::new();
        for edge in edges {
            let rejection = |code: CompletionRejectionCode, message: &str| CompletionRejection {
                code,
                from: edge.from.clone(),
                relation: edge.relation.clone(),
                to: edge.to.clone(),
                message: message.to_string(),
            };
            if accepted.len() >= COMPLETION_MAX_NEW_EDGES {
                rejected.push(rejection(
                    CompletionRejectionCode::BudgetExhausted,
                    "per-call candidate budget exhausted",
                ));
                continue;
            }
            if edge.from.trim().is_empty()
                || edge.to.trim().is_empty()
                || self.entity(&edge.from).is_none()
                || self.entity(&edge.to).is_none()
            {
                rejected.push(rejection(
                    CompletionRejectionCode::MissingEndpoint,
                    "both endpoints must exist in the graph",
                ));
                continue;
            }
            if edge.from == edge.to {
                rejected.push(rejection(
                    CompletionRejectionCode::SelfLoop,
                    "candidate connects a node to itself",
                ));
                continue;
            }
            if edge.relation.trim().is_empty() {
                rejected.push(rejection(
                    CompletionRejectionCode::InvalidRelation,
                    "relation name is empty",
                ));
                continue;
            }
            if !edge.confidence.is_finite() || edge.confidence < COMPLETION_MIN_CONFIDENCE {
                rejected.push(rejection(
                    CompletionRejectionCode::LowConfidence,
                    "candidate confidence is below the write threshold of 0.7",
                ));
                continue;
            }
            if edge.evidence.trim().is_empty() {
                rejected.push(rejection(
                    CompletionRejectionCode::MissingEvidence,
                    "candidate has no evidence locator",
                ));
                continue;
            }
            if let Some(existing) = self
                .relations
                .values()
                .find(|relation| {
                    relation.from == edge.from
                        && relation.relation == edge.relation
                        && relation.to == edge.to
                })
                .cloned()
            {
                if existing.hypothesis {
                    rejected.push(rejection(
                        CompletionRejectionCode::DuplicateHypothesis,
                        "the same candidate edge is already pending",
                    ));
                } else {
                    rejected.push(rejection(
                        CompletionRejectionCode::DuplicateAsserted,
                        "an asserted edge already connects these nodes",
                    ));
                }
                continue;
            }

            let id = format!("rel:{}:{}:{}", edge.from, edge.relation, edge.to);
            let mut relation = EngineeringRelation::new(&edge.from, &edge.relation, &edge.to)
                .with_id(id.clone())
                .with_confidence(edge.confidence)
                .with_hypothesis(true);
            let mut provenance = EngineeringProvenance::new(edge.source.clone())
                .with_locator(edge.evidence.clone())
                .with_confidence(edge.confidence);
            provenance.activity = Some(format!("completion:{source_tag}"));
            relation.provenance = provenance;
            relation.valid_from = Some(now);
            relation.valid_to = Some(now + Duration::days(COMPLETION_TTL_DAYS));
            relation.status = EntityStatus::Active;
            self.relations.insert(id.clone(), relation);
            accepted.push(id);
        }
        CompletionWriteReport {
            accepted,
            rejected,
            completion_debt: self.completion_debt(),
        }
    }

    /// All pending hypothesis edges, oldest first.
    pub fn hypothesis_edges(&self) -> Vec<&EngineeringRelation> {
        let mut edges: Vec<&EngineeringRelation> = self
            .relations
            .values()
            .filter(|edge| edge.hypothesis)
            .collect();
        edges.sort_by(|left, right| {
            left.provenance
                .recorded_at
                .cmp(&right.provenance.recorded_at)
                .then_with(|| left.id.cmp(&right.id))
        });
        edges
    }

    /// Unconfirmed hypothesis edges divided by all edges. Zero when empty.
    pub fn completion_debt(&self) -> f32 {
        if self.relations.is_empty() {
            return 0.0;
        }
        let pending = self
            .relations
            .values()
            .filter(|edge| edge.hypothesis)
            .count();
        pending as f32 / self.relations.len() as f32
    }

    /// Promote one hypothesis edge using non-LLM evidence.
    ///
    /// The previous provenance chain is preserved in `derived_from`, the TTL is
    /// dropped, and the edge keeps the stronger of the two confidences.
    pub fn promote_completion_edge(
        &mut self,
        id: &str,
        evidence: EngineeringProvenance,
    ) -> Result<(), CompletionPromotionError> {
        let Some(relation) = self.relations.get_mut(id) else {
            return Err(CompletionPromotionError::NotFound);
        };
        if !relation.hypothesis {
            return Err(CompletionPromotionError::AlreadyAsserted);
        }
        if evidence.source.trim().is_empty() {
            return Err(CompletionPromotionError::MissingEvidenceSource);
        }
        if is_llm_source(&evidence.source) {
            return Err(CompletionPromotionError::LlmCannotPromote);
        }
        if !evidence.confidence.is_finite() || evidence.confidence < COMPLETION_MIN_CONFIDENCE {
            return Err(CompletionPromotionError::WeakEvidence);
        }
        let previous = relation.provenance.clone();
        relation.confidence = relation.confidence.max(evidence.confidence).min(1.0);
        relation.hypothesis = false;
        relation.valid_to = None;
        relation.provenance = evidence;
        relation.provenance.confidence = relation.confidence;
        relation
            .provenance
            .derived_from
            .push(previous.source.clone());
        relation.provenance.recorded_at = Utc::now();
        Ok(())
    }

    /// Soft-delete hypothesis edges whose TTL has passed. Returns the count.
    pub fn purge_expired_hypotheses(&mut self, now: chrono::DateTime<Utc>) -> usize {
        let expired: Vec<String> = self
            .relations
            .values()
            .filter(|edge| edge.hypothesis && edge.valid_to.is_some_and(|until| until < now))
            .map(|edge| edge.id.clone())
            .collect();
        for id in &expired {
            self.relations.remove(id);
        }
        expired.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engineering_ontology::{
        CLASS_CHANGE, CLASS_ERROR, CLASS_FILE, EngineeringEntity, REL_FIXES, REL_USES,
    };

    fn graph() -> EngineeringGraph {
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(EngineeringEntity::new(CLASS_ERROR, "error:e1", "E0425"));
        graph.insert_entity(EngineeringEntity::new(CLASS_CHANGE, "change:c1", "Patch"));
        graph.insert_relation(EngineeringRelation::new("error:e1", REL_FIXES, "change:c1"));
        graph
    }

    fn candidate() -> CompletionEdge {
        CompletionEdge::new("change:c1", REL_USES, "error:e1", 0.82)
            .with_evidence("doc:review#p2")
            .with_source("llm:proposal")
    }

    #[test]
    fn writes_hypothesis_edge_with_ttl_and_evidence() {
        let mut graph = graph();
        let report = graph.write_completion_edges([candidate()], "l4");
        assert_eq!(report.accepted.len(), 1);
        assert!(report.rejected.is_empty());
        let edge = graph
            .relations
            .get(&report.accepted[0])
            .expect("written edge");
        assert!(edge.hypothesis);
        assert_eq!(edge.provenance.source, "llm:proposal");
        assert_eq!(edge.provenance.locator.as_deref(), Some("doc:review#p2"));
        assert_eq!(edge.provenance.activity.as_deref(), Some("completion:l4"));
        assert!(edge.valid_to.is_some());
        assert!((report.completion_debt - 0.5).abs() < 1e-6);
    }

    #[test]
    fn rejects_low_confidence_missing_evidence_and_duplicates() {
        let mut graph = graph();
        let mut low = candidate();
        low.confidence = 0.5;
        let mut no_evidence = candidate();
        no_evidence.evidence = "  ".to_string();
        let return_to = candidate();
        let report = graph.write_completion_edges([low, no_evidence, return_to], "l4");
        assert_eq!(report.rejected.len(), 2);
        assert_eq!(
            report.rejected[0].code,
            CompletionRejectionCode::LowConfidence
        );
        assert_eq!(
            report.rejected[1].code,
            CompletionRejectionCode::MissingEvidence
        );
        assert_eq!(report.accepted.len(), 1);

        let again = graph.write_completion_edges([candidate()], "l4");
        assert_eq!(
            again.rejected[0].code,
            CompletionRejectionCode::DuplicateHypothesis
        );
    }

    #[test]
    fn refuses_to_overwrite_asserted_edges_and_self_loops() {
        let mut graph = graph();
        let asserted = CompletionEdge::new("error:e1", REL_FIXES, "change:c1", 0.9)
            .with_evidence("doc:x")
            .with_source("llm:proposal");
        let looped = CompletionEdge::new("change:c1", REL_USES, "change:c1", 0.9)
            .with_evidence("doc:y")
            .with_source("llm:proposal");
        let missing = CompletionEdge::new("change:c1", REL_USES, "change:missing", 0.9)
            .with_evidence("doc:z")
            .with_source("llm:proposal");
        let report = graph.write_completion_edges([asserted, looped, missing], "l4");
        assert_eq!(report.accepted.len(), 0);
        assert_eq!(
            report.rejected[0].code,
            CompletionRejectionCode::DuplicateAsserted
        );
        assert_eq!(report.rejected[1].code, CompletionRejectionCode::SelfLoop);
        assert_eq!(
            report.rejected[2].code,
            CompletionRejectionCode::MissingEndpoint
        );
    }

    #[test]
    fn caps_new_edges_per_call() {
        let mut graph = graph();
        for index in 0..8 {
            graph.insert_entity(EngineeringEntity::new(
                CLASS_FILE,
                format!("file:f{index}"),
                format!("f{index}.rs"),
            ));
        }
        let edges: Vec<CompletionEdge> = (0..8)
            .map(|index| {
                CompletionEdge::new("change:c1", REL_USES, format!("file:f{index}"), 0.9)
                    .with_evidence(format!("doc:{index}"))
                    .with_source("llm:proposal")
            })
            .collect();
        let report = graph.write_completion_edges(edges, "l4");
        assert_eq!(report.accepted.len(), COMPLETION_MAX_NEW_EDGES);
        assert!(
            report
                .rejected
                .iter()
                .all(|rejection| rejection.code == CompletionRejectionCode::BudgetExhausted)
        );
    }

    #[test]
    fn promotion_requires_non_llm_evidence_and_clears_hypothesis() {
        let mut graph = graph();
        let report = graph.write_completion_edges([candidate()], "l3");
        let id = report.accepted[0].clone();

        let llm = EngineeringProvenance::new("llm:proposal").with_confidence(0.95);
        assert_eq!(
            graph.promote_completion_edge(&id, llm),
            Err(CompletionPromotionError::LlmCannotPromote)
        );
        let weak = EngineeringProvenance::new("procurement_api").with_confidence(0.5);
        assert_eq!(
            graph.promote_completion_edge(&id, weak),
            Err(CompletionPromotionError::WeakEvidence)
        );

        let evidence = EngineeringProvenance::new("procurement_api")
            .with_locator("po:check#S-77")
            .with_confidence(0.98);
        graph
            .promote_completion_edge(&id, evidence)
            .expect("promote");
        let edge = graph.relations.get(&id).expect("edge");
        assert!(!edge.hypothesis);
        assert_eq!(edge.confidence, 0.98);
        assert!(edge.valid_to.is_none());
        assert_eq!(edge.provenance.source, "procurement_api");
        assert_eq!(
            edge.provenance.derived_from,
            vec!["llm:proposal".to_string()]
        );
        assert_eq!(graph.completion_debt(), 0.0);
        assert_eq!(
            graph.promote_completion_edge(&id, EngineeringProvenance::new("human:review")),
            Err(CompletionPromotionError::AlreadyAsserted)
        );
    }

    #[test]
    fn expired_hypotheses_purge_without_touching_asserted_edges() {
        let mut graph = graph();
        let report = graph.write_completion_edges([candidate()], "l4");
        assert_eq!(graph.purge_expired_hypotheses(Utc::now()), 0);
        let later = Utc::now() + Duration::days(COMPLETION_TTL_DAYS + 1);
        assert_eq!(graph.purge_expired_hypotheses(later), 1);
        assert!(!graph.relations.contains_key(&report.accepted[0]));
        assert_eq!(graph.relations.len(), 1);
    }

    #[test]
    fn llm_source_detection_covers_common_prefixes() {
        for source in ["llm:proposal", "LLM", "model:glm-5.3", "completion:llm"] {
            assert!(is_llm_source(source), "{source}");
        }
        for source in ["procurement_api", "human:review", "doc:cert#p3"] {
            assert!(!is_llm_source(source), "{source}");
        }
    }
}
