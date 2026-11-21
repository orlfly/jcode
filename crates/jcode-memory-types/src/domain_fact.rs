//! Typed domain facts and lexical retrieval.
//!
//! These are the L1 instance rows that can be extracted deterministically from
//! user statements before compaction rewrites the conversation. The schema is a
//! triple plus provenance, not a free-form memory note, so a later ontology
//! layer can map predicates without re-parsing prose.

use std::collections::BTreeSet;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// One explicit subject-predicate-object fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DomainFact {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// Verbatim sentence that produced the fact.
    pub statement: String,
    pub source_session: String,
    pub source_role: String,
    pub source_locator: String,
    /// Extraction confidence in `0.0..=1.0`.
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
}

/// Retrieval request for domain facts.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DomainFactQuery {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicate: Option<String>,
    #[serde(default)]
    pub limit: usize,
}

impl DomainFactQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            subject: None,
            predicate: None,
            limit: 8,
        }
    }

    pub fn with_subject(mut self, subject: impl Into<String>) -> Self {
        self.subject = Some(subject.into());
        self
    }

    pub fn with_predicate(mut self, predicate: impl Into<String>) -> Self {
        self.predicate = Some(predicate.into());
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

/// One ranked domain-fact hit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DomainFactMatch {
    pub fact: DomainFact,
    pub score: f32,
}

/// Rank facts by statement similarity, with exact subject/predicate bonuses.
pub fn rank_domain_facts(query: &DomainFactQuery, facts: &[DomainFact]) -> Vec<DomainFactMatch> {
    if query.limit == 0 {
        return Vec::new();
    }
    let query_terms = terms(&query.text);
    let mut matches: Vec<DomainFactMatch> = facts
        .iter()
        .filter_map(|fact| {
            let haystack = format!(
                "{} {} {} {}",
                fact.subject, fact.predicate, fact.object, fact.statement
            );
            let score_terms = terms(&haystack);
            let mut score = jaccard(&query_terms, &score_terms);
            if let Some(subject) = query.subject.as_deref()
                && fact.subject.eq_ignore_ascii_case(subject)
            {
                score += 0.20;
            }
            if let Some(predicate) = query.predicate.as_deref()
                && fact.predicate.eq_ignore_ascii_case(predicate)
            {
                score += 0.10;
            }
            if score <= 0.0 {
                return None;
            }
            Some(DomainFactMatch {
                fact: fact.clone(),
                score: score.min(1.0),
            })
        })
        .collect();
    matches.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.fact.id.cmp(&right.fact.id))
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

    fn fact(id: &str, subject: &str, predicate: &str, object: &str) -> DomainFact {
        DomainFact {
            id: id.to_string(),
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            statement: format!("{subject} {predicate} {object}"),
            source_session: "s".to_string(),
            source_role: "user".to_string(),
            source_locator: "message:1".to_string(),
            confidence: 0.75,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn lexical_match_outranks_weaker_match() {
        let facts = vec![
            fact("a", "billing", "uses", "stripe"),
            fact("b", "billing", "uses", "postgres"),
        ];
        let hits = rank_domain_facts(&DomainFactQuery::new("stripe billing"), &facts);
        assert_eq!(hits[0].fact.id, "a");
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn subject_and_predicate_add_structural_score() {
        let facts = vec![
            fact("a", "Billing", "depends_on", "Stripe"),
            fact("b", "Search", "depends_on", "Elastic"),
        ];
        let hits = rank_domain_facts(
            &DomainFactQuery::new("unrelated wording")
                .with_subject("billing")
                .with_predicate("depends_on"),
            &facts,
        );
        assert_eq!(hits[0].fact.id, "a");
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn limit_and_zero_are_honored() {
        let facts = vec![fact("a", "a", "uses", "b"), fact("b", "a", "uses", "c")];
        assert_eq!(
            rank_domain_facts(&DomainFactQuery::new("a uses").with_limit(1), &facts).len(),
            1
        );
        assert!(rank_domain_facts(&DomainFactQuery::new("a").with_limit(0), &facts).is_empty());
    }
}
