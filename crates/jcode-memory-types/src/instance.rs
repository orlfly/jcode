//! Instance-layer metadata for the memory graph.
//!
//! Inspired by the enterprise-ontology / mem-plugin design, every memory node is
//! treated as an *instance* with provenance, lifecycle status, identity keys and
//! aliases. This makes the graph auditable, governs destructive operations and
//! enables entity disambiguation during retrieval.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// How the memory was extracted / produced.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionMethod {
    /// User explicitly stated it (highest authority).
    #[default]
    UserStated,
    /// Mapped from a structured source such as a config file or database.
    StructuredMapping,
    /// Extracted by a deterministic rule / script.
    RuleExtraction,
    /// Extracted by an LLM from conversation context.
    LlmExtraction,
    /// Derived by rule inference from other memories.
    RuleInference,
}

impl ExtractionMethod {
    /// Authority ordering for conflict resolution.
    /// Higher value = more authoritative source.
    pub fn authority_rank(self) -> u32 {
        match self {
            ExtractionMethod::UserStated => 5,
            ExtractionMethod::StructuredMapping => 4,
            ExtractionMethod::RuleExtraction => 3,
            ExtractionMethod::LlmExtraction => 2,
            ExtractionMethod::RuleInference => 1,
        }
    }

    /// Minimum confidence that should be considered for direct ingestion.
    pub fn admission_threshold(self) -> f32 {
        match self {
            ExtractionMethod::UserStated => 0.0,
            ExtractionMethod::StructuredMapping => 0.0,
            ExtractionMethod::RuleExtraction => 0.7,
            ExtractionMethod::LlmExtraction => 0.7,
            ExtractionMethod::RuleInference => 0.8,
        }
    }
}

/// Provenance metadata describing where a memory came from and how much to trust it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ProvenanceRecord {
    /// Human-readable source (e.g. file name, conversation session id, tool name).
    pub source: String,
    /// Optional locator within the source (e.g. "sheet=客户表,row=15").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locator: Option<String>,
    /// Method used to extract the memory.
    #[serde(default)]
    pub method: ExtractionMethod,
    /// When the memory was extracted.
    #[serde(default = "Utc::now")]
    pub extracted_at: DateTime<Utc>,
    /// Confidence in [0.0, 1.0]. Values below the method's admission threshold
    /// should be routed to a pending-confirmation queue rather than committed.
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

impl ProvenanceRecord {
    pub fn new(source: impl Into<String>, method: ExtractionMethod) -> Self {
        Self {
            source: source.into(),
            locator: None,
            method,
            extracted_at: Utc::now(),
            confidence: 1.0,
        }
    }

    pub fn with_locator(mut self, locator: impl Into<String>) -> Self {
        self.locator = Some(locator.into());
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_extracted_at(mut self, at: DateTime<Utc>) -> Self {
        self.extracted_at = at;
        self
    }

    /// Default heuristic provenance for legacy or unannotated memories.
    /// Mirrors mem-plugin's DEFAULT_PROVENANCE (method=user_stated, confidence=0.5).
    pub fn default_heuristic() -> Self {
        Self {
            source: String::new(),
            locator: None,
            method: ExtractionMethod::UserStated,
            extracted_at: Utc::now(),
            confidence: 0.5,
        }
    }

    /// Whether this provenance meets its own method's quality bar.
    pub fn is_admissible(&self) -> bool {
        self.confidence >= self.method.admission_threshold()
    }
}

fn default_confidence() -> f32 {
    1.0
}

/// Lifecycle status of a memory instance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "lowercase")]
pub enum MemoryStatus {
    /// Usable in retrieval.
    #[default]
    Active,
    /// Past its effective window or manually marked stale; kept for history but
    /// should not be retrieved by default.
    Expired,
    /// Soft-deleted; may be physically removed after two major ontology bumps.
    Archived,
    /// Conflicting information exists; requires resolution before surfacing.
    Disputed,
}

impl MemoryStatus {
    pub fn is_retrievable(self) -> bool {
        matches!(self, MemoryStatus::Active)
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, MemoryStatus::Archived | MemoryStatus::Expired)
    }

    /// Allowed state-machine transitions per mem-plugin lifecycle-ops.md:
    /// ACTIVE -> EXPIRED/ARCHIVED/DISPUTED
    /// EXPIRED -> ACTIVE/ARCHIVED
    /// ARCHIVED -> ACTIVE
    /// DISPUTED -> ACTIVE/ARCHIVED
    pub fn can_transition_to(self, target: MemoryStatus) -> bool {
        if self == target {
            return true;
        }
        matches!(
            (self, target),
            (MemoryStatus::Active, MemoryStatus::Expired)
                | (MemoryStatus::Active, MemoryStatus::Archived)
                | (MemoryStatus::Active, MemoryStatus::Disputed)
                | (MemoryStatus::Expired, MemoryStatus::Active)
                | (MemoryStatus::Expired, MemoryStatus::Archived)
                | (MemoryStatus::Archived, MemoryStatus::Active)
                | (MemoryStatus::Disputed, MemoryStatus::Active)
                | (MemoryStatus::Disputed, MemoryStatus::Archived)
        )
    }
}

/// Return the canonical lifecycle transition table.  Used by the rule
/// engine as a fallback when an ontology type is not registered.  Mirrors
/// the state machine in [`MemoryStatus::can_transition_to`].
pub fn default_lifecycle_transition_table() -> Vec<(MemoryStatus, MemoryStatus)> {
    vec![
        (MemoryStatus::Active, MemoryStatus::Expired),
        (MemoryStatus::Active, MemoryStatus::Archived),
        (MemoryStatus::Active, MemoryStatus::Disputed),
        (MemoryStatus::Expired, MemoryStatus::Active),
        (MemoryStatus::Expired, MemoryStatus::Archived),
        (MemoryStatus::Archived, MemoryStatus::Active),
        (MemoryStatus::Disputed, MemoryStatus::Active),
        (MemoryStatus::Disputed, MemoryStatus::Archived),
    ]
}

/// Identity metadata for entity disambiguation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct IdentityMetadata {
    /// Keys that uniquely identify the entity this memory represents.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keys: Vec<String>,
    /// Alternate names / spellings that should be treated as the same entity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
}

impl IdentityMetadata {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.keys.push(key.into());
        self
    }

    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.aliases.push(alias.into());
        self
    }

    pub fn matches(&self, query: &str) -> bool {
        let q = query.to_lowercase();
        self.keys.iter().chain(&self.aliases).any(|s| s.to_lowercase() == q)
    }
}

/// Lifecycle metadata for memories with validity windows and deprecation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct LifecycleMetadata {
    /// Status of the instance.
    #[serde(default)]
    pub status: MemoryStatus,
    /// When the memory became effective.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_from: Option<DateTime<Utc>>,
    /// When the memory stops being effective (exclusive).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_to: Option<DateTime<Utc>>,
    /// True if the memory type/property is deprecated.
    #[serde(default, skip_serializing_if = "is_false")]
    pub deprecated: bool,
    /// Reason for deprecation / archival.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deprecated_reason: Option<String>,
}

impl LifecycleMetadata {
    pub fn active() -> Self {
        Self {
            status: MemoryStatus::Active,
            effective_from: None,
            effective_to: None,
            deprecated: false,
            deprecated_reason: None,
        }
    }

    pub fn archived(reason: impl Into<String>) -> Self {
        Self {
            status: MemoryStatus::Archived,
            effective_from: None,
            effective_to: None,
            deprecated: false,
            deprecated_reason: Some(reason.into()),
        }
    }

    pub fn expired() -> Self {
        Self {
            status: MemoryStatus::Expired,
            effective_from: None,
            effective_to: None,
            deprecated: false,
            deprecated_reason: None,
        }
    }

    pub fn with_status(mut self, status: MemoryStatus) -> Self {
        self.status = status;
        self
    }

    pub fn with_effective_from(mut self, from: DateTime<Utc>) -> Self {
        self.effective_from = Some(from);
        self
    }

    pub fn with_effective_to(mut self, to: DateTime<Utc>) -> Self {
        self.effective_to = Some(to);
        self
    }

    pub fn deprecated(reason: impl Into<String>) -> Self {
        Self {
            status: MemoryStatus::Active,
            effective_from: None,
            effective_to: None,
            deprecated: true,
            deprecated_reason: Some(reason.into()),
        }
    }

    /// Whether the memory is effective as of the given instant.
    pub fn is_effective_at(&self, now: DateTime<Utc>) -> bool {
        self.effective_from.map(|f| now >= f).unwrap_or(true)
            && self.effective_to.map(|t| now < t).unwrap_or(true)
    }

    /// Whether the memory should be considered for retrieval right now.
    pub fn is_retrievable_now(&self) -> bool {
        self.status.is_retrievable() && !self.deprecated && self.is_effective_at(Utc::now())
    }

    /// Attempt a state-machine-guarded status transition.
    /// Returns true when the transition is allowed or the status is unchanged.
    pub fn transition_to(&mut self, target: MemoryStatus) -> bool {
        if self.status.can_transition_to(target) {
            self.status = target;
            true
        } else {
            false
        }
    }
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_authority_ordering() {
        assert!(ExtractionMethod::UserStated.authority_rank() > ExtractionMethod::LlmExtraction.authority_rank());
        assert!(ExtractionMethod::StructuredMapping.authority_rank() > ExtractionMethod::RuleInference.authority_rank());
    }

    #[test]
    fn provenance_admission_threshold() {
        let user = ProvenanceRecord::new("chat", ExtractionMethod::UserStated).with_confidence(0.5);
        assert!(user.is_admissible());

        let llm = ProvenanceRecord::new("extraction", ExtractionMethod::LlmExtraction).with_confidence(0.6);
        assert!(!llm.is_admissible());

        let llm_ok = ProvenanceRecord::new("extraction", ExtractionMethod::LlmExtraction).with_confidence(0.75);
        assert!(llm_ok.is_admissible());
    }

    #[test]
    fn lifecycle_retrievability() {
        let active = LifecycleMetadata::active();
        assert!(active.is_retrievable_now());

        let archived = LifecycleMetadata::archived("old");
        assert!(!archived.is_retrievable_now());

        let window = LifecycleMetadata::active()
            .with_effective_from(Utc::now() - chrono::Duration::days(10))
            .with_effective_to(Utc::now() - chrono::Duration::days(1));
        assert!(!window.is_retrievable_now());

        let future = LifecycleMetadata::active().with_effective_from(Utc::now() + chrono::Duration::days(1));
        assert!(!future.is_retrievable_now());
    }

    #[test]
    fn status_state_machine_matches_mem_plugin() {
        use MemoryStatus::*;
        assert!(Active.can_transition_to(Expired));
        assert!(Active.can_transition_to(Archived));
        assert!(Active.can_transition_to(Disputed));
        assert!(Expired.can_transition_to(Active));
        assert!(Expired.can_transition_to(Archived));
        assert!(Archived.can_transition_to(Active));
        assert!(Disputed.can_transition_to(Active));
        assert!(Disputed.can_transition_to(Archived));

        // Illegal transitions
        assert!(!Archived.can_transition_to(Expired));
        assert!(!Archived.can_transition_to(Disputed));
        assert!(!Expired.can_transition_to(Disputed));
    }

    #[test]
    fn default_heuristic_provenance_is_admissible() {
        let prov = ProvenanceRecord::default_heuristic();
        assert_eq!(prov.method, ExtractionMethod::UserStated);
        assert_eq!(prov.confidence, 0.5);
        assert!(prov.is_admissible());
    }

    #[test]
    fn identity_matches_keys_and_aliases() {
        let id = IdentityMetadata::empty()
            .with_key("postgres")
            .with_alias("postgresql")
            .with_alias("psql");
        assert!(id.matches("postgres"));
        assert!(id.matches("postgresql"));
        assert!(!id.matches("mysql"));
    }
}
