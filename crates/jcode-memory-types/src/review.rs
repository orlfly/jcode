//! Review-queue records and confidence admission policy.
//!
//! The queue is the durable half of the write gate: a candidate that needs a
//! human decision is stored here instead of mutating the graph. Queue storage
//! lives in `jcode-base`; these types are pure data so both the runtime and
//! offline tooling can share them.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::MemoryEntry;
use crate::actions::ConfirmationGate;
use crate::instance::ExtractionMethod;

/// Change level for a memory or skill update.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "lowercase")]
pub enum ChangeLevel {
    /// Instance-level data change: content, tags, relation, body wording.
    #[default]
    L1,
    /// Backward-compatible schema or contract change.
    L2,
    /// Structural or breaking change.
    L3,
}

impl ChangeLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            ChangeLevel::L1 => "l1",
            ChangeLevel::L2 => "l2",
            ChangeLevel::L3 => "l3",
        }
    }
}

/// Confidence admission tier for an incoming memory candidate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionTier {
    /// Admit immediately.
    Auto,
    /// Admit, but mark the instance as needing review.
    NeedsReview,
    /// Do not admit; park the candidate in the review queue.
    PendingOnly,
}

/// Confidence at or above this is admitted without a review badge.
pub const AUTO_ADMIT_CONFIDENCE: f32 = 0.9;
/// Confidence at or above this is admitted with `needs_review = true`.
pub const REVIEW_BADGE_CONFIDENCE: f32 = 0.7;

/// Classify a provenance record into an admission tier.
///
/// Authoritative methods (direct user statements, structured mappings) are
/// never downgraded by the confidence tier: their confidence describes
/// extraction certainty, not authority, and the destructive-change gates still
/// apply. Probabilistic methods use the 0.9 / 0.7 thresholds.
pub fn admission_tier(method: ExtractionMethod, confidence: f32) -> AdmissionTier {
    if matches!(
        method,
        ExtractionMethod::UserStated | ExtractionMethod::StructuredMapping
    ) {
        return AdmissionTier::Auto;
    }
    if confidence >= AUTO_ADMIT_CONFIDENCE {
        AdmissionTier::Auto
    } else if confidence >= REVIEW_BADGE_CONFIDENCE {
        AdmissionTier::NeedsReview
    } else {
        AdmissionTier::PendingOnly
    }
}

/// One queued change awaiting a decision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryReviewItem {
    pub id: String,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub level: ChangeLevel,
    /// Gates that caused the confirmation requirement. Empty for a pure
    /// low-confidence admission hold.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gates: Vec<ConfirmationGate>,
    /// `project` or `global`.
    pub scope: String,
    /// Project directory the write belongs to, for accepting from another cwd.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_dir: Option<String>,
    /// Existing memory being changed, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old: Option<MemoryEntry>,
    /// Proposed new memory.
    pub new: MemoryEntry,
    /// Reconcile intent: `insert`, `reinforce`, `supersede`, `contradict`,
    /// `relate`, or a skill action such as `skill_update`.
    #[serde(default = "default_action")]
    pub action: String,
    /// Human-readable changed fields.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diff: Vec<String>,
    /// Why the item needs review.
    pub reason: String,
    #[serde(default)]
    pub recommendation: ReviewRecommendation,
    #[serde(default)]
    pub status: ReviewStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirmed_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirmed_at: Option<DateTime<Utc>>,
}

impl MemoryReviewItem {
    pub fn new(
        scope: impl Into<String>,
        old: Option<MemoryEntry>,
        new: MemoryEntry,
        gates: Vec<ConfirmationGate>,
        reason: impl Into<String>,
    ) -> Self {
        let ts = Utc::now().timestamp_millis();
        let rand: u64 = rand::random();
        Self {
            id: format!("review_{ts}_{rand}"),
            created_at: Utc::now(),
            level: ChangeLevel::L1,
            gates,
            scope: scope.into(),
            project_dir: None,
            diff: diff_entries(old.as_ref(), &new),
            old,
            new,
            action: default_action(),
            reason: reason.into(),
            recommendation: ReviewRecommendation::NeedsHuman,
            status: ReviewStatus::Pending,
            note: None,
            confirmed_by: None,
            confirmed_at: None,
        }
    }

    pub fn with_project_dir(mut self, dir: impl Into<String>) -> Self {
        self.project_dir = Some(dir.into());
        self
    }

    pub fn with_level(mut self, level: ChangeLevel) -> Self {
        self.level = level;
        self
    }

    pub fn with_recommendation(mut self, recommendation: ReviewRecommendation) -> Self {
        self.recommendation = recommendation;
        self
    }

    pub fn with_action(mut self, action: impl Into<String>) -> Self {
        self.action = action.into();
        self
    }

    pub fn is_pending(&self) -> bool {
        self.status == ReviewStatus::Pending
    }

    /// Short one-line description for list output.
    pub fn summary(&self) -> String {
        format!(
            "{} [{}] {} gate(s)={} target={} reason={}",
            self.id,
            self.level.as_str(),
            self.new.category,
            self.gates.len(),
            self.old
                .as_ref()
                .map(|entry| entry.id.as_str())
                .unwrap_or("<new>"),
            self.reason
        )
    }
}

fn default_action() -> String {
    "insert".to_string()
}

/// Reviewer-facing recommendation.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ReviewRecommendation {
    Accept,
    KeepOld,
    #[default]
    NeedsHuman,
}

/// Queue lifecycle state.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ReviewStatus {
    #[default]
    Pending,
    Accepted,
    Rejected,
    Deferred,
}

impl ReviewStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ReviewStatus::Pending => "pending",
            ReviewStatus::Accepted => "accepted",
            ReviewStatus::Rejected => "rejected",
            ReviewStatus::Deferred => "deferred",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "pending" => Some(ReviewStatus::Pending),
            "accepted" => Some(ReviewStatus::Accepted),
            "rejected" => Some(ReviewStatus::Rejected),
            "deferred" => Some(ReviewStatus::Deferred),
            _ => None,
        }
    }
}

/// Compute the changed fields between two entries.
pub fn diff_entries(old: Option<&MemoryEntry>, new: &MemoryEntry) -> Vec<String> {
    let Some(old) = old else {
        return vec!["new".to_string()];
    };
    let mut diff = Vec::new();
    if old.content != new.content {
        diff.push("content".to_string());
    }
    if old.category != new.category {
        diff.push("category".to_string());
    }
    if old.tags != new.tags {
        diff.push("tags".to_string());
    }
    if old.lifecycle.status != new.lifecycle.status {
        diff.push(format!(
            "status:{}->{}",
            old.lifecycle.status.as_str(),
            new.lifecycle.status.as_str()
        ));
    }
    if old.critical != new.critical {
        diff.push("critical".to_string());
    }
    if old.properties != new.properties {
        diff.push("properties".to_string());
    }
    if old.subject_key != new.subject_key {
        diff.push("subject_key".to_string());
    }
    if old.active != new.active {
        diff.push("active".to_string());
    }
    if diff.is_empty() {
        diff.push("content-equal".to_string());
    }
    diff
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryCategory;
    use crate::instance::ProvenanceRecord;

    fn entry(content: &str) -> MemoryEntry {
        MemoryEntry::new(MemoryCategory::Observation, content)
    }

    #[test]
    fn admission_tiers_are_applied_to_probabilistic_methods() {
        assert_eq!(
            admission_tier(ExtractionMethod::LlmExtraction, 0.95),
            AdmissionTier::Auto
        );
        assert_eq!(
            admission_tier(ExtractionMethod::LlmExtraction, 0.8),
            AdmissionTier::NeedsReview
        );
        assert_eq!(
            admission_tier(ExtractionMethod::LlmExtraction, 0.4),
            AdmissionTier::PendingOnly
        );
    }

    #[test]
    fn authoritative_methods_are_never_held_by_confidence() {
        assert_eq!(
            admission_tier(ExtractionMethod::UserStated, 0.1),
            AdmissionTier::Auto
        );
        assert_eq!(
            admission_tier(ExtractionMethod::StructuredMapping, 0.5),
            AdmissionTier::Auto
        );
    }

    #[test]
    fn review_item_round_trips_and_reports_diff() {
        let old = entry("old fact").with_id("mem_old");
        let new = entry("new fact")
            .with_id("mem_old")
            .with_tags(vec!["tag".to_string()]);
        let item = MemoryReviewItem::new(
            "project",
            Some(old),
            new,
            vec![ConfirmationGate::CriticalData],
            "critical",
        )
        .with_project_dir("/tmp/project");
        assert!(item.is_pending());
        assert!(item.diff.contains(&"content".to_string()));
        assert_eq!(item.project_dir.as_deref(), Some("/tmp/project"));
        let json = serde_json::to_string(&item).expect("serialize");
        let back: MemoryReviewItem = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.id, item.id);
        assert_eq!(back.gates, item.gates);
    }

    #[test]
    fn confirmed_provenance_records_reviewer() {
        let now = Utc::now();
        let prov = ProvenanceRecord::new("chat", ExtractionMethod::UserStated)
            .with_confirmation("reviewer", now);
        assert_eq!(prov.confirmed_by.as_deref(), Some("reviewer"));
        assert_eq!(prov.confirmed_at, Some(now));
    }
}
