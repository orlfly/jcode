//! Layered software-engineering ontology for jcode's memory and reasoning.
//!
//! This module is deliberately a *profile*, not a full import of BFO,
//! ISO/IEC 24744 or OMG Essence. It provides the stable vocabulary that the
//! memory and episode graphs need:
//!
//! - BFO-lite upper categories on every class.
//! - An ISO/IEC 24744-shaped endeavour/work-unit/work-product backbone.
//! - OMG Essence alpha states for progress and belief.
//! - SWEBOK concern tags as a SKOS-style controlled vocabulary.
//! - PROV-O-shaped provenance on every entity and relation.
//! - SHACL-shaped validation for classes, properties, domain/range and time.
//!
//! The runtime representation is a property graph ([`EngineeringGraph`]).
//! OWL DL reasoning is intentionally out of scope: schema validation here is
//! cheap, deterministic and suitable for every memory write.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::validation::Severity;

// ---------------------------------------------------------------------------
// Class ids
// ---------------------------------------------------------------------------

pub const CLASS_AGENT: &str = "agent";
pub const CLASS_ACTION: &str = "action";
pub const CLASS_API: &str = "api";
pub const CLASS_ARTIFACT: &str = "artifact";
pub const CLASS_BUILD: &str = "build";
pub const CLASS_CHANGE: &str = "change";
pub const CLASS_COMMIT: &str = "commit";
pub const CLASS_COMMAND: &str = "command";
pub const CLASS_CONVENTION: &str = "convention";
pub const CLASS_CRATE: &str = "crate";
pub const CLASS_DECISION: &str = "decision";
pub const CLASS_ENVIRONMENT: &str = "environment";
pub const CLASS_ERROR: &str = "error";
pub const CLASS_FILE: &str = "file";
pub const CLASS_OBSERVATION: &str = "observation";
pub const CLASS_REVIEW: &str = "review";
pub const CLASS_SESSION: &str = "session";
pub const CLASS_SYMBOL: &str = "symbol";
pub const CLASS_TASK: &str = "task";
pub const CLASS_TEST: &str = "test";
pub const CLASS_TURN: &str = "turn";
pub const CLASS_WORK_UNIT: &str = "work_unit";

// ---------------------------------------------------------------------------
// Concern ids (SWEBOK, used as SKOS-style tags, not classes)
// ---------------------------------------------------------------------------

pub const CONCERN_REQUIREMENTS: &str = "requirements";
pub const CONCERN_DESIGN: &str = "design";
pub const CONCERN_CONSTRUCTION: &str = "construction";
pub const CONCERN_TESTING: &str = "testing";
pub const CONCERN_MAINTENANCE: &str = "maintenance";
pub const CONCERN_CONFIGURATION_MANAGEMENT: &str = "configuration_management";
pub const CONCERN_ENGINEERING_MANAGEMENT: &str = "engineering_management";
pub const CONCERN_SOFTWARE_QUALITY: &str = "software_quality";
pub const CONCERN_ENGINEERING_ECONOMICS: &str = "engineering_economics";

// ---------------------------------------------------------------------------
// Relation ids
// ---------------------------------------------------------------------------

pub const REL_APPLIES_TO: &str = "applies_to";
pub const REL_BLOCKED_BY: &str = "blocked_by";
pub const REL_CALLS: &str = "calls";
pub const REL_CAUSES: &str = "causes";
pub const REL_CONTAINS: &str = "contains";
pub const REL_DEFINES: &str = "defines";
pub const REL_DEPENDS_ON: &str = "depends_on";
pub const REL_DOCUMENTS: &str = "documents";
pub const REL_EVIDENCED_BY: &str = "evidenced_by";
pub const REL_FIXES: &str = "fixes";
pub const REL_IMPLEMENTS: &str = "implements";
pub const REL_IMPORTS: &str = "imports";
pub const REL_MODIFIES: &str = "modifies";
pub const REL_PART_OF: &str = "part_of";
pub const REL_PRODUCES: &str = "produces";
pub const REL_REVIEWS: &str = "reviews";
pub const REL_SUPERSEDES: &str = "supersedes";
pub const REL_TESTS: &str = "tests";
pub const REL_USES: &str = "uses";
pub const REL_VERIFIES: &str = "verifies";
/// Concept-layer edge from an entity to its declared ontology class.
pub const REL_INSTANCE_OF: &str = "instance_of";
/// Query-time inverse of `instance_of`, from a class to its instances.
pub const REL_HAS_INSTANCE: &str = "has_instance";
/// Concept-layer edge from a subclass to its parent class.
pub const REL_SUBCLASS_OF: &str = "subclass_of";

/// Wildcard used by permissive relations such as `evidenced_by`.
pub const CLASS_WILDCARD: &str = "*";

/// Stable node id for an ontology class in a concept-query graph.
pub fn concept_node_id(class_id: &str) -> String {
    format!("concept:{}", class_id.trim())
}

// ---------------------------------------------------------------------------
// Schema value types
// ---------------------------------------------------------------------------

/// BFO-lite upper categories.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BfoCategory {
    /// Independent continuant: a file, crate, environment.
    IndependentContinuant,
    /// Specifically dependent continuant: an error quality or state.
    SpecificallyDependentContinuant,
    /// Generically dependent continuant: code, contracts, conventions.
    GenericallyDependentContinuant,
    /// Occurrent: a session, action, build or review.
    Occurrent,
    /// Quality or disposition.
    Quality,
    /// Role.
    Role,
}

/// Runtime status of an instance.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EntityStatus {
    #[default]
    Active,
    Archived,
    Disputed,
}

/// Value types checked by [`EngineeringGraph::validate`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PropertyValueKind {
    String,
    Identifier,
    Integer,
    Float,
    Bool,
    DateTime,
    Enum { values: Vec<String> },
    Reference { class: String },
    Json,
}

/// A property declaration on a class.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineeringPropertyDef {
    pub id: String,
    pub kind: PropertyValueKind,
    #[serde(default)]
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl EngineeringPropertyDef {
    pub fn new(id: impl Into<String>, kind: PropertyValueKind, required: bool) -> Self {
        Self {
            id: id.into(),
            kind,
            required,
            description: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// A class in the engineering profile.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineeringClassDef {
    pub id: String,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub bfo: BfoCategory,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub properties: Vec<EngineeringPropertyDef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub concerns: Vec<String>,
}

impl EngineeringClassDef {
    pub fn new(id: impl Into<String>, display_name: impl Into<String>, bfo: BfoCategory) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            description: None,
            bfo,
            parent: None,
            properties: Vec::new(),
            concerns: Vec::new(),
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    pub fn with_property(mut self, property: EngineeringPropertyDef) -> Self {
        self.properties.push(property);
        self
    }

    pub fn with_concern(mut self, concern: impl Into<String>) -> Self {
        self.concerns.push(concern.into());
        self
    }
}

/// A relation signature. `*` in a domain or range means any class.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineeringRelationDef {
    pub id: String,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub domain: Vec<String>,
    #[serde(default)]
    pub range: Vec<String>,
    #[serde(default = "default_true")]
    pub directed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_cardinality: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub concerns: Vec<String>,
    /// Relation whose direction is the inverse of this one. Inference creates
    /// only virtual edges; the stored graph is never rewritten.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inverse_of: Option<String>,
    /// Broader relation this one specializes. A query allowing the parent also
    /// admits this relation as a virtual parent edge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sub_property_of: Option<String>,
}

fn default_true() -> bool {
    true
}

impl EngineeringRelationDef {
    pub fn new(
        id: impl Into<String>,
        display_name: impl Into<String>,
        domain: &[&str],
        range: &[&str],
    ) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            description: None,
            domain: domain.iter().map(|s| (*s).to_string()).collect(),
            range: range.iter().map(|s| (*s).to_string()).collect(),
            directed: true,
            max_cardinality: None,
            concerns: Vec::new(),
            inverse_of: None,
            sub_property_of: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_max_cardinality(mut self, max: u32) -> Self {
        self.max_cardinality = Some(max);
        self
    }

    pub fn with_inverse_of(mut self, relation: impl Into<String>) -> Self {
        self.inverse_of = Some(relation.into());
        self
    }

    pub fn with_sub_property_of(mut self, relation: impl Into<String>) -> Self {
        self.sub_property_of = Some(relation.into());
        self
    }

    pub fn with_concern(mut self, concern: impl Into<String>) -> Self {
        self.concerns.push(concern.into());
        self
    }
}

/// A SKOS-style concern tag. Concerns are not classes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConcernDef {
    pub id: String,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl ConcernDef {
    pub fn new(id: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            parent: None,
            description: None,
        }
    }
}

/// One OMG Essence alpha state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AlphaStateDef {
    pub id: String,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checklist: Vec<String>,
}

impl AlphaStateDef {
    pub fn new(id: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            checklist: Vec::new(),
        }
    }
}

/// An OMG Essence alpha with its state sequence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AlphaDef {
    pub id: String,
    pub display_name: String,
    #[serde(default)]
    pub states: Vec<AlphaStateDef>,
}

impl AlphaDef {
    pub fn new(id: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            states: Vec::new(),
        }
    }

    pub fn with_state(mut self, state: AlphaStateDef) -> Self {
        self.states.push(state);
        self
    }
}

/// The complete engineering ontology profile.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineeringOntology {
    pub id: String,
    pub version: u32,
    pub description: String,
    #[serde(default)]
    pub classes: BTreeMap<String, EngineeringClassDef>,
    #[serde(default)]
    pub relations: BTreeMap<String, EngineeringRelationDef>,
    #[serde(default)]
    pub concerns: BTreeMap<String, ConcernDef>,
    #[serde(default)]
    pub alphas: BTreeMap<String, AlphaDef>,
}

impl EngineeringOntology {
    pub fn new(id: impl Into<String>, version: u32, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version,
            description: description.into(),
            classes: BTreeMap::new(),
            relations: BTreeMap::new(),
            concerns: BTreeMap::new(),
            alphas: BTreeMap::new(),
        }
    }

    pub fn with_class(mut self, class: EngineeringClassDef) -> Self {
        self.classes.insert(class.id.clone(), class);
        self
    }

    pub fn with_relation(mut self, relation: EngineeringRelationDef) -> Self {
        self.relations.insert(relation.id.clone(), relation);
        self
    }

    pub fn with_concern(mut self, concern: ConcernDef) -> Self {
        self.concerns.insert(concern.id.clone(), concern);
        self
    }

    pub fn with_alpha(mut self, alpha: AlphaDef) -> Self {
        self.alphas.insert(alpha.id.clone(), alpha);
        self
    }

    /// Deterministic content hash, same FNV-1a convention as the memory
    /// ontology hash.
    pub fn content_hash(&self) -> String {
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{hash:016x}")
    }

    /// Validate the schema itself: parents, signatures, concerns and alphas.
    pub fn validate(&self) -> EngineeringValidationReport {
        let mut issues = Vec::new();

        if self.id.trim().is_empty() {
            issues.push(EngineeringValidationIssue::error(
                "EO01",
                "ontology id is empty",
                None,
            ));
        }
        if self.version == 0 {
            issues.push(EngineeringValidationIssue::error(
                "EO02",
                "ontology version must be positive",
                None,
            ));
        }

        for class in self.classes.values() {
            if let Some(parent) = &class.parent {
                if !self.classes.contains_key(parent) {
                    issues.push(EngineeringValidationIssue::error(
                        "EO10",
                        format!("class '{}' has unknown parent '{parent}'", class.id),
                        Some(class.id.clone()),
                    ));
                }
            }
            let mut property_ids = BTreeSet::new();
            for property in &class.properties {
                if !property_ids.insert(property.id.clone()) {
                    issues.push(EngineeringValidationIssue::error(
                        "EO11",
                        format!(
                            "class '{}' declares property '{}' more than once",
                            class.id, property.id
                        ),
                        Some(class.id.clone()),
                    ));
                }
                if let PropertyValueKind::Reference { class: target } = &property.kind
                    && target != CLASS_WILDCARD
                    && !self.classes.contains_key(target)
                {
                    issues.push(EngineeringValidationIssue::error(
                        "EO12",
                        format!(
                            "property '{}' on class '{}' references unknown class '{target}'",
                            property.id, class.id
                        ),
                        Some(class.id.clone()),
                    ));
                }
            }
            for concern in &class.concerns {
                if !self.concerns.contains_key(concern) {
                    issues.push(EngineeringValidationIssue::error(
                        "EO13",
                        format!("class '{}' uses unknown concern '{concern}'", class.id),
                        Some(class.id.clone()),
                    ));
                }
            }
            if let Some(cycle) = self.class_cycle(class) {
                issues.push(EngineeringValidationIssue::error(
                    "EO14",
                    format!("class inheritance cycle: {cycle}"),
                    Some(class.id.clone()),
                ));
            }
        }

        for relation in self.relations.values() {
            for class in relation.domain.iter().chain(relation.range.iter()) {
                if class != CLASS_WILDCARD && !self.classes.contains_key(class) {
                    issues.push(EngineeringValidationIssue::error(
                        "EO20",
                        format!(
                            "relation '{}' references unknown class '{class}'",
                            relation.id
                        ),
                        Some(relation.id.clone()),
                    ));
                }
            }
            for concern in &relation.concerns {
                if !self.concerns.contains_key(concern) {
                    issues.push(EngineeringValidationIssue::error(
                        "EO21",
                        format!(
                            "relation '{}' uses unknown concern '{concern}'",
                            relation.id
                        ),
                        Some(relation.id.clone()),
                    ));
                }
            }
            if relation.domain.is_empty() || relation.range.is_empty() {
                issues.push(EngineeringValidationIssue::warning(
                    "EO22",
                    format!("relation '{}' has an empty domain or range", relation.id),
                    Some(relation.id.clone()),
                ));
            }
        }

        for alpha in self.alphas.values() {
            if alpha.states.is_empty() {
                issues.push(EngineeringValidationIssue::warning(
                    "EO30",
                    format!("alpha '{}' has no states", alpha.id),
                    Some(alpha.id.clone()),
                ));
            }
            let mut seen = BTreeSet::new();
            for state in &alpha.states {
                if !seen.insert(state.id.clone()) {
                    issues.push(EngineeringValidationIssue::error(
                        "EO31",
                        format!("alpha '{}' repeats state '{}'", alpha.id, state.id),
                        Some(alpha.id.clone()),
                    ));
                }
            }
        }

        EngineeringValidationReport { issues }
    }

    fn class_cycle(&self, start: &EngineeringClassDef) -> Option<String> {
        let mut seen = BTreeSet::new();
        let mut current = Some(start);
        let mut chain = Vec::new();
        while let Some(class) = current {
            if !seen.insert(class.id.clone()) {
                chain.push(class.id.clone());
                return Some(chain.join(" -> "));
            }
            chain.push(class.id.clone());
            current = class
                .parent
                .as_ref()
                .and_then(|parent| self.classes.get(parent));
        }
        None
    }

    /// All properties available on a class, including inherited ones.
    pub fn effective_properties(&self, class: &str) -> Vec<&EngineeringPropertyDef> {
        let mut out = Vec::new();
        let mut current = self.classes.get(class);
        let mut seen = BTreeSet::new();
        while let Some(def) = current {
            if !seen.insert(def.id.clone()) {
                break;
            }
            for property in &def.properties {
                if !out
                    .iter()
                    .any(|existing: &&EngineeringPropertyDef| existing.id == property.id)
                {
                    out.push(property);
                }
            }
            current = def
                .parent
                .as_ref()
                .and_then(|parent| self.classes.get(parent));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Instances
// ---------------------------------------------------------------------------

/// PROV-O-shaped provenance carried by every instance and relation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineeringProvenance {
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locator: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activity: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub derived_from: Vec<String>,
    pub confidence: f32,
    #[serde(default = "Utc::now")]
    pub recorded_at: DateTime<Utc>,
}

impl Default for EngineeringProvenance {
    fn default() -> Self {
        Self {
            source: "unknown".to_string(),
            locator: None,
            activity: None,
            derived_from: Vec::new(),
            confidence: 0.5,
            recorded_at: Utc::now(),
        }
    }
}

impl EngineeringProvenance {
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            confidence: 0.8,
            ..Self::default()
        }
    }

    pub fn with_locator(mut self, locator: impl Into<String>) -> Self {
        self.locator = Some(locator.into());
        self
    }

    pub fn with_activity(mut self, activity: impl Into<String>) -> Self {
        self.activity = Some(activity.into());
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// A typed entity in the property graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineeringEntity {
    pub id: String,
    pub class: String,
    pub name: String,
    #[serde(default)]
    pub properties: BTreeMap<String, Value>,
    #[serde(default)]
    pub provenance: EngineeringProvenance,
    pub confidence: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_to: Option<DateTime<Utc>>,
    #[serde(default)]
    pub status: EntityStatus,
}

impl EngineeringEntity {
    pub fn new(class: impl Into<String>, id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            class: class.into(),
            name: name.into(),
            properties: BTreeMap::new(),
            provenance: EngineeringProvenance::default(),
            confidence: 0.8,
            valid_from: None,
            valid_to: None,
            status: EntityStatus::Active,
        }
    }

    pub fn with_property(mut self, key: impl Into<String>, value: Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }

    pub fn with_provenance(mut self, provenance: EngineeringProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn is_effective_at(&self, now: DateTime<Utc>) -> bool {
        self.status == EntityStatus::Active
            && self.valid_from.is_none_or(|t| t <= now)
            && self.valid_to.is_none_or(|t| t >= now)
    }
}

/// A typed relation instance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineeringRelation {
    pub id: String,
    pub from: String,
    pub relation: String,
    pub to: String,
    #[serde(default)]
    pub properties: BTreeMap<String, Value>,
    #[serde(default)]
    pub provenance: EngineeringProvenance,
    pub confidence: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_to: Option<DateTime<Utc>>,
    #[serde(default)]
    pub status: EntityStatus,
}

impl EngineeringRelation {
    pub fn new(
        from: impl Into<String>,
        relation: impl Into<String>,
        to: impl Into<String>,
    ) -> Self {
        let from = from.into();
        let relation = relation.into();
        let to = to.into();
        Self {
            id: format!("rel:{from}:{relation}:{to}"),
            from,
            relation,
            to,
            properties: BTreeMap::new(),
            provenance: EngineeringProvenance::default(),
            confidence: 0.8,
            valid_from: None,
            valid_to: None,
            status: EntityStatus::Active,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_property(mut self, key: impl Into<String>, value: Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }

    pub fn with_provenance(mut self, provenance: EngineeringProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn is_effective_at(&self, now: DateTime<Utc>) -> bool {
        self.status == EntityStatus::Active
            && self.valid_from.is_none_or(|t| t <= now)
            && self.valid_to.is_none_or(|t| t >= now)
    }
}

/// A property graph plus its schema.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EngineeringGraph {
    #[serde(default)]
    pub entities: BTreeMap<String, EngineeringEntity>,
    #[serde(default)]
    pub relations: BTreeMap<String, EngineeringRelation>,
}

/// A scored path through an [`EngineeringGraph`].
///
/// `nodes` and `relations` are parallel: `relations[i]` is the relation id that
/// connects `nodes[i]` to `nodes[i + 1]`. `cost` is the sum of edge costs;
/// lower is better. `confidence` is the weakest edge confidence on the path.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringPath {
    pub nodes: Vec<String>,
    pub relations: Vec<String>,
    pub cost: f64,
    pub confidence: f32,
    pub provenance: Vec<EngineeringProvenance>,
}

impl EngineeringPath {
    pub fn hop_count(&self) -> usize {
        self.relations.len()
    }
}

fn relation_cost(relation: &EngineeringRelation) -> f64 {
    let confidence = if relation.confidence.is_finite() {
        f64::from(relation.confidence.clamp(0.0, 1.0))
    } else {
        0.5
    };
    let confidence_penalty = 1.0 + 2.0 * (1.0 - confidence);
    let weight = relation
        .properties
        .get("weight")
        .and_then(Value::as_f64)
        .filter(|weight| weight.is_finite() && *weight > 0.0)
        .unwrap_or(1.0);
    confidence_penalty * weight
}

fn scaled_cost(cost: f64) -> u64 {
    (cost * 1_000_000.0).round().clamp(1.0, u64::MAX as f64) as u64
}

impl EngineeringGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Project an ontology's classes and relation signatures into a queryable
    /// concept graph.
    ///
    /// Class nodes use `concept:<class-id>` ids. Every declared domain/range
    /// pair becomes one relation edge, so the result can be searched by the
    /// same Dijkstra/Yen engine as an instance graph. Wildcard signatures are
    /// skipped because they do not name a concrete pair of concepts.
    pub fn concept_graph(ontology: &EngineeringOntology) -> Self {
        let mut graph = Self::new();
        for class in ontology.classes.values() {
            let mut entity = EngineeringEntity::new(
                class.id.clone(),
                concept_node_id(&class.id),
                class.display_name.clone(),
            )
            .with_property("concept", Value::Bool(true))
            .with_property("class_id", Value::String(class.id.clone()))
            .with_provenance(EngineeringProvenance::new("ontology:concept"))
            .with_confidence(1.0);
            entity.status = EntityStatus::Active;
            graph.insert_entity(entity);
        }
        for class in ontology.classes.values() {
            let Some(parent) = class.parent.as_deref() else {
                continue;
            };
            if !ontology.classes.contains_key(parent) {
                continue;
            }
            let mut relation = EngineeringRelation::new(
                concept_node_id(&class.id),
                REL_SUBCLASS_OF,
                concept_node_id(parent),
            )
            .with_id(format!(
                "ontology-subclass:{}:{}",
                class.id, parent
            ))
            .with_provenance(EngineeringProvenance::new("ontology:subclass"))
            .with_confidence(1.0);
            relation.status = EntityStatus::Active;
            graph.insert_relation(relation);
        }
        for definition in ontology.relations.values() {
            for domain in &definition.domain {
                if domain == CLASS_WILDCARD || !ontology.classes.contains_key(domain) {
                    continue;
                }
                for range in &definition.range {
                    if range == CLASS_WILDCARD || !ontology.classes.contains_key(range) {
                        continue;
                    }
                    let from = concept_node_id(domain);
                    let to = concept_node_id(range);
                    let mut relation = EngineeringRelation::new(
                        from,
                        definition.id.clone(),
                        to,
                    )
                    .with_id(format!(
                        "ontology-signature:{}:{}:{}",
                        definition.id, domain, range
                    ))
                    .with_property("domain", Value::String(domain.clone()))
                    .with_property("range", Value::String(range.clone()))
                    .with_provenance(EngineeringProvenance::new(format!(
                        "ontology:signature:{}",
                        definition.id
                    )))
                    .with_confidence(1.0);
                    relation.status = EntityStatus::Active;
                    graph.insert_relation(relation);
                }
            }
        }
        graph
    }

    /// Resolve a concept anchor by node id, class id, or display name.
    pub fn resolve_concept_anchor(&self, anchor: &str) -> Option<String> {
        let anchor = anchor.trim();
        if anchor.is_empty() {
            return None;
        }
        if self.entities.contains_key(anchor) {
            return Some(anchor.to_string());
        }
        let prefixed = concept_node_id(anchor);
        if self.entities.contains_key(&prefixed) {
            return Some(prefixed);
        }
        self.entities
            .values()
            .find(|entity| entity.name.eq_ignore_ascii_case(anchor))
            .map(|entity| entity.id.clone())
    }

    /// Return a query-time graph that joins this instance graph to the
    /// ontology concept layer.
    ///
    /// The stored graph is not modified. Each entity gains
    /// `entity -instance_of-> concept:<class>` and the reverse
    /// `concept:<class> -has_instance-> entity` edge, so an instance-to-instance
    /// search can use concept signatures as an intermediate layer.
    pub fn with_ontology_layer(&self, ontology: &EngineeringOntology) -> Self {
        let mut graph = self.clone();
        graph.merge(&Self::concept_graph(ontology));
        for entity in self.entities.values() {
            if !ontology.classes.contains_key(&entity.class) {
                continue;
            }
            let concept = concept_node_id(&entity.class);
            if !graph.entities.contains_key(&concept) {
                continue;
            }
            let mut to_concept = EngineeringRelation::new(
                entity.id.clone(),
                REL_INSTANCE_OF,
                concept.clone(),
            )
            .with_id(format!("instance-of:{}:{}", entity.id, entity.class))
            .with_provenance(entity.provenance.clone())
            .with_confidence(entity.confidence);
            to_concept.status = entity.status;
            graph.insert_relation(to_concept);

            let mut from_concept = EngineeringRelation::new(
                concept,
                REL_HAS_INSTANCE,
                entity.id.clone(),
            )
            .with_id(format!("has-instance:{}:{}", entity.class, entity.id))
            .with_provenance(entity.provenance.clone())
            .with_confidence(entity.confidence);
            from_concept.status = entity.status;
            graph.insert_relation(from_concept);
        }
        graph
    }

    pub fn insert_entity(&mut self, entity: EngineeringEntity) -> Option<EngineeringEntity> {
        self.entities.insert(entity.id.clone(), entity)
    }

    pub fn insert_relation(
        &mut self,
        relation: EngineeringRelation,
    ) -> Option<EngineeringRelation> {
        self.relations.insert(relation.id.clone(), relation)
    }

    /// Union another graph into this one, keeping existing entities and
    /// relations when ids collide. Compaction extracts only the active segment,
    /// so merging preserves facts routed from earlier checkpoints.
    pub fn merge(&mut self, other: &Self) {
        for (id, entity) in &other.entities {
            self.entities
                .entry(id.clone())
                .or_insert_with(|| entity.clone());
        }
        for (id, relation) in &other.relations {
            self.relations
                .entry(id.clone())
                .or_insert_with(|| relation.clone());
        }
    }

    pub fn entity(&self, id: &str) -> Option<&EngineeringEntity> {
        self.entities.get(id)
    }

    pub fn relation(&self, id: &str) -> Option<&EngineeringRelation> {
        self.relations.get(id)
    }

    pub fn find_by_class(&self, class: &str) -> Vec<&EngineeringEntity> {
        self.entities
            .values()
            .filter(|entity| entity.class == class)
            .collect()
    }

    /// Add query-time inverse and sub-property edges from the ontology.
    ///
    /// The returned graph is a virtual view: stored relations are unchanged.
    /// Each edge keeps the source relation id in its locator and is discounted
    /// to 95% of the source confidence so inferred hops never look stronger
    /// than observed evidence.
    pub fn infer_relations(&self, ontology: &EngineeringOntology) -> EngineeringGraph {
        let mut inferred = self.clone();
        let now = Utc::now();
        let sources: Vec<EngineeringRelation> = self
            .relations
            .values()
            .filter(|relation| relation.is_effective_at(now))
            .cloned()
            .collect();
        for relation in sources {
            let Some(definition) = ontology.relations.get(&relation.relation) else {
                continue;
            };
            if let Some(inverse) = definition
                .inverse_of
                .as_deref()
                .filter(|inverse| !inverse.trim().is_empty())
            {
                let id = format!("inferred:inverse:{}:{}", relation.id, inverse);
                if !inferred.relations.contains_key(&id) {
                    let confidence = (relation.confidence * 0.95).clamp(0.0, 1.0);
                    let mut edge = EngineeringRelation::new(&relation.to, inverse, &relation.from);
                    edge.id = id;
                    edge.confidence = confidence;
                    edge.provenance =
                        EngineeringProvenance::new(format!("inferred:inverse_of:{}", relation.id))
                            .with_locator(relation.id.clone())
                            .with_confidence(confidence);
                    inferred.relations.insert(edge.id.clone(), edge);
                }
            }
            if let Some(parent) = definition
                .sub_property_of
                .as_deref()
                .filter(|parent| !parent.trim().is_empty())
            {
                let id = format!("inferred:sub:{}:{}", relation.id, parent);
                if !inferred.relations.contains_key(&id) {
                    let confidence = (relation.confidence * 0.95).clamp(0.0, 1.0);
                    let mut edge =
                        EngineeringRelation::new(&relation.from, parent, &relation.to);
                    edge.id = id;
                    edge.confidence = confidence;
                    edge.provenance =
                        EngineeringProvenance::new(format!("inferred:sub_property_of:{}", relation.id))
                            .with_locator(relation.id.clone())
                            .with_confidence(confidence);
                    inferred.relations.insert(edge.id.clone(), edge);
                }
            }
        }
        inferred
    }

    pub fn outgoing(&self, id: &str) -> Vec<&EngineeringRelation> {
        let now = Utc::now();
        self.relations
            .values()
            .filter(|relation| relation.from == id && relation.is_effective_at(now))
            .collect()
    }

    pub fn incoming(&self, id: &str) -> Vec<&EngineeringRelation> {
        let now = Utc::now();
        self.relations
            .values()
            .filter(|relation| relation.to == id && relation.is_effective_at(now))
            .collect()
    }

    /// Directed breadth-first path over effective relations.
    ///
    /// This is the deterministic primitive the higher-level k-best path query
    /// can build on. It deliberately does not score or rerank.
    pub fn shortest_path(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
        allowed_relations: &[&str],
    ) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![from.to_string()]);
        }
        let mut queue = VecDeque::from([(from.to_string(), vec![from.to_string()])]);
        let mut seen = HashSet::from([from.to_string()]);

        while let Some((node, path)) = queue.pop_front() {
            if path.len() > max_depth + 1 {
                continue;
            }
            for relation in self.outgoing(&node) {
                if !allowed_relations.is_empty()
                    && !allowed_relations.contains(&relation.relation.as_str())
                {
                    continue;
                }
                let next = &relation.to;
                if next == to {
                    let mut result = path.clone();
                    result.push(next.clone());
                    return Some(result);
                }
                if seen.insert(next.clone()) {
                    let mut next_path = path.clone();
                    next_path.push(next.clone());
                    queue.push_back((next.clone(), next_path));
                }
            }
        }
        None
    }

    /// Best path under the graph's edge weights.
    ///
    /// Edge cost is `(1 + 2 * (1 - confidence)) * property["weight"]`, where
    /// `weight` defaults to 1.0. Low-confidence and explicitly expensive
    /// relations are therefore avoided without requiring a separate weight
    /// table. This is Dijkstra over effective, non-negative edges.
    pub fn weighted_shortest_path(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
        allowed_relations: &[&str],
    ) -> Option<EngineeringPath> {
        self.dijkstra(
            from,
            to,
            max_depth,
            allowed_relations,
            &BTreeSet::new(),
            &BTreeSet::new(),
        )
    }

    /// Yen's k-best loop-free paths under the same edge weights as
    /// [`Self::weighted_shortest_path`].
    ///
    /// The first path is the weighted shortest path; each subsequent path is
    /// the cheapest deviation from the previously accepted paths. Results are
    /// sorted by cost and then by node sequence for deterministic ties. This is
    /// the primitive for "show me alternative explanations" queries without
    /// switching to an external graph database.
    pub fn k_shortest_paths(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
        allowed_relations: &[&str],
        k: usize,
    ) -> Vec<EngineeringPath> {
        if k == 0 {
            return Vec::new();
        }
        let Some(first) = self.weighted_shortest_path(from, to, max_depth, allowed_relations)
        else {
            return Vec::new();
        };
        let mut accepted = vec![first];
        let mut candidates: Vec<EngineeringPath> = Vec::new();

        while accepted.len() < k {
            let previous = accepted.last().expect("accepted path").clone();
            for spur_index in 0..previous.nodes.len().saturating_sub(1) {
                let spur_node = previous.nodes[spur_index].clone();
                let root_nodes = previous.nodes[..=spur_index].to_vec();
                let root_relations = previous.relations[..spur_index].to_vec();

                let mut removed_relations = BTreeSet::new();
                for path in &accepted {
                    if path.nodes.len() > spur_index
                        && path.nodes[..=spur_index] == root_nodes
                        && let Some(relation) = path.relations.get(spur_index)
                    {
                        removed_relations.insert(relation.clone());
                    }
                }
                let blocked_nodes: BTreeSet<String> =
                    root_nodes[..spur_index].iter().cloned().collect();

                let Some(spur) = self.dijkstra(
                    &spur_node,
                    to,
                    max_depth.saturating_sub(spur_index),
                    allowed_relations,
                    &removed_relations,
                    &blocked_nodes,
                ) else {
                    continue;
                };

                let mut nodes = root_nodes;
                nodes.extend(spur.nodes.into_iter().skip(1));
                let mut relations = root_relations;
                relations.extend(spur.relations);
                if accepted.iter().any(|path| path.nodes == nodes)
                    || candidates.iter().any(|path| path.nodes == nodes)
                {
                    continue;
                }
                candidates.push(self.path_from(nodes, relations));
            }

            if candidates.is_empty() {
                break;
            }
            candidates.sort_by(|left, right| {
                left.cost
                    .partial_cmp(&right.cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.nodes.cmp(&right.nodes))
            });
            accepted.push(candidates.remove(0));
        }
        accepted
    }

    fn dijkstra(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
        allowed_relations: &[&str],
        removed_relations: &BTreeSet<String>,
        blocked_nodes: &BTreeSet<String>,
    ) -> Option<EngineeringPath> {
        if from == to {
            return Some(self.path_from(vec![from.to_string()], Vec::new()));
        }
        let mut queue: BTreeSet<(u64, String, usize, Vec<String>, Vec<String>)> = BTreeSet::new();
        queue.insert((0, from.to_string(), 0, vec![from.to_string()], Vec::new()));
        let mut best: HashMap<(String, usize), u64> = HashMap::new();
        best.insert((from.to_string(), 0), 0);

        while let Some((cost, node, depth, nodes, relations)) = queue.pop_first() {
            if node == to {
                return Some(self.path_from(nodes, relations));
            }
            if depth >= max_depth {
                continue;
            }
            for relation in self.outgoing(&node) {
                if !allowed_relations.is_empty()
                    && !allowed_relations.contains(&relation.relation.as_str())
                {
                    continue;
                }
                if removed_relations.contains(&relation.id) || blocked_nodes.contains(&relation.to)
                {
                    continue;
                }
                let next_depth = depth + 1;
                let next_cost = cost.saturating_add(scaled_cost(relation_cost(relation)));
                let key = (relation.to.clone(), next_depth);
                if best
                    .get(&key)
                    .is_some_and(|existing| *existing <= next_cost)
                {
                    continue;
                }
                best.insert(key, next_cost);
                let mut next_nodes = nodes.clone();
                next_nodes.push(relation.to.clone());
                let mut next_relations = relations.clone();
                next_relations.push(relation.id.clone());
                queue.insert((
                    next_cost,
                    relation.to.clone(),
                    next_depth,
                    next_nodes,
                    next_relations,
                ));
            }
        }
        None
    }

    fn path_from(&self, nodes: Vec<String>, relations: Vec<String>) -> EngineeringPath {
        let mut cost = 0.0;
        let mut confidence = 1.0f32;
        let mut provenance = Vec::with_capacity(relations.len());
        for relation_id in &relations {
            let Some(relation) = self.relations.get(relation_id) else {
                continue;
            };
            cost += relation_cost(relation);
            confidence = confidence.min(relation.confidence.clamp(0.0, 1.0));
            provenance.push(relation.provenance.clone());
        }
        EngineeringPath {
            nodes,
            relations,
            cost,
            confidence,
            provenance,
        }
    }

    /// Validate every entity and relation against the ontology.
    pub fn validate(&self, ontology: &EngineeringOntology) -> EngineeringValidationReport {
        let mut issues = Vec::new();
        let now = Utc::now();

        for entity in self.entities.values() {
            if entity.id.trim().is_empty() {
                issues.push(EngineeringValidationIssue::error(
                    "EG01",
                    "entity id is empty",
                    None,
                ));
            }
            if entity.name.trim().is_empty() {
                issues.push(EngineeringValidationIssue::warning(
                    "EG02",
                    format!("entity '{}' has an empty name", entity.id),
                    Some(entity.id.clone()),
                ));
            }
            if !(0.0..=1.0).contains(&entity.confidence) {
                issues.push(EngineeringValidationIssue::error(
                    "EG03",
                    format!("entity '{}' confidence is outside [0,1]", entity.id),
                    Some(entity.id.clone()),
                ));
            }
            if !(0.0..=1.0).contains(&entity.provenance.confidence) {
                issues.push(EngineeringValidationIssue::error(
                    "EG04",
                    format!(
                        "entity '{}' provenance confidence is outside [0,1]",
                        entity.id
                    ),
                    Some(entity.id.clone()),
                ));
            }
            if let (Some(from), Some(to)) = (entity.valid_from, entity.valid_to)
                && to < from
            {
                issues.push(EngineeringValidationIssue::error(
                    "EG05",
                    format!("entity '{}' has valid_to before valid_from", entity.id),
                    Some(entity.id.clone()),
                ));
            }
            let Some(class) = ontology.classes.get(&entity.class) else {
                issues.push(EngineeringValidationIssue::error(
                    "EG10",
                    format!(
                        "entity '{}' uses unknown class '{}'",
                        entity.id, entity.class
                    ),
                    Some(entity.id.clone()),
                ));
                continue;
            };
            let _ = class;
            let properties = ontology.effective_properties(&entity.class);
            let known: HashSet<&str> = properties.iter().map(|p| p.id.as_str()).collect();
            for property in &properties {
                let present = entity
                    .properties
                    .get(&property.id)
                    .is_some_and(|value| !value.is_null());
                if property.required && !present {
                    issues.push(EngineeringValidationIssue::error(
                        "EG11",
                        format!(
                            "entity '{}' is missing required property '{}'",
                            entity.id, property.id
                        ),
                        Some(entity.id.clone()),
                    ));
                }
                if let Some(value) = entity.properties.get(&property.id)
                    && !value.is_null()
                    && !property_value_matches(&property.kind, value)
                {
                    issues.push(EngineeringValidationIssue::error(
                        "EG12",
                        format!(
                            "entity '{}' property '{}' does not match {:?}",
                            entity.id, property.id, property.kind
                        ),
                        Some(entity.id.clone()),
                    ));
                }
            }
            for key in entity.properties.keys() {
                if !known.contains(key.as_str()) {
                    issues.push(EngineeringValidationIssue::warning(
                        "EG13",
                        format!(
                            "entity '{}' has undeclared property '{}' on class '{}'",
                            entity.id, key, entity.class
                        ),
                        Some(entity.id.clone()),
                    ));
                }
            }
            if entity.status == EntityStatus::Active && !entity.is_effective_at(now) {
                issues.push(EngineeringValidationIssue::warning(
                    "EG14",
                    format!(
                        "entity '{}' is active but outside its validity window",
                        entity.id
                    ),
                    Some(entity.id.clone()),
                ));
            }
        }

        for relation in self.relations.values() {
            if relation.id.trim().is_empty() {
                issues.push(EngineeringValidationIssue::error(
                    "EG20",
                    "relation id is empty",
                    None,
                ));
            }
            if !(0.0..=1.0).contains(&relation.confidence) {
                issues.push(EngineeringValidationIssue::error(
                    "EG21",
                    format!("relation '{}' confidence is outside [0,1]", relation.id),
                    Some(relation.id.clone()),
                ));
            }
            if !(0.0..=1.0).contains(&relation.provenance.confidence) {
                issues.push(EngineeringValidationIssue::error(
                    "EG22",
                    format!(
                        "relation '{}' provenance confidence is outside [0,1]",
                        relation.id
                    ),
                    Some(relation.id.clone()),
                ));
            }
            if let (Some(from), Some(to)) = (relation.valid_from, relation.valid_to)
                && to < from
            {
                issues.push(EngineeringValidationIssue::error(
                    "EG23",
                    format!("relation '{}' has valid_to before valid_from", relation.id),
                    Some(relation.id.clone()),
                ));
            }
            let Some(source) = self.entities.get(&relation.from) else {
                issues.push(EngineeringValidationIssue::error(
                    "EG24",
                    format!(
                        "relation '{}' source '{}' does not exist",
                        relation.id, relation.from
                    ),
                    Some(relation.id.clone()),
                ));
                continue;
            };
            let Some(target) = self.entities.get(&relation.to) else {
                issues.push(EngineeringValidationIssue::error(
                    "EG25",
                    format!(
                        "relation '{}' target '{}' does not exist",
                        relation.id, relation.to
                    ),
                    Some(relation.id.clone()),
                ));
                continue;
            };
            let Some(def) = ontology.relations.get(&relation.relation) else {
                issues.push(EngineeringValidationIssue::error(
                    "EG26",
                    format!(
                        "relation '{}' uses unknown relation '{}'",
                        relation.id, relation.relation
                    ),
                    Some(relation.id.clone()),
                ));
                continue;
            };
            if !domain_accepts(def, &source.class) {
                issues.push(EngineeringValidationIssue::error(
                    "EG27",
                    format!(
                        "relation '{}' domain does not accept class '{}'",
                        relation.id, source.class
                    ),
                    Some(relation.id.clone()),
                ));
            }
            if !range_accepts(def, &target.class) {
                issues.push(EngineeringValidationIssue::error(
                    "EG28",
                    format!(
                        "relation '{}' range does not accept class '{}'",
                        relation.id, target.class
                    ),
                    Some(relation.id.clone()),
                ));
            }
        }

        let mut counts: HashMap<(String, String), usize> = HashMap::new();
        for relation in self.relations.values() {
            if relation.is_effective_at(now) {
                *counts
                    .entry((relation.from.clone(), relation.relation.clone()))
                    .or_default() += 1;
            }
        }
        for ((from, relation_id), count) in counts {
            if let Some(def) = ontology.relations.get(&relation_id)
                && let Some(max) = def.max_cardinality
                && count > max as usize
            {
                issues.push(EngineeringValidationIssue::error(
                    "EG30",
                    format!(
                        "relation '{relation_id}' from '{from}' has {count} active edges, max is {max}"
                    ),
                    Some(from),
                ));
            }
        }

        EngineeringValidationReport { issues }
    }
}

fn domain_accepts(def: &EngineeringRelationDef, class: &str) -> bool {
    def.domain.iter().any(|c| c == CLASS_WILDCARD || c == class)
}

fn range_accepts(def: &EngineeringRelationDef, class: &str) -> bool {
    def.range.iter().any(|c| c == CLASS_WILDCARD || c == class)
}

fn property_value_matches(kind: &PropertyValueKind, value: &Value) -> bool {
    match kind {
        PropertyValueKind::String | PropertyValueKind::Identifier => {
            value.as_str().is_some_and(|text| !text.trim().is_empty())
        }
        PropertyValueKind::Integer => value.as_i64().is_some() || value.as_u64().is_some(),
        PropertyValueKind::Float => value.as_f64().is_some(),
        PropertyValueKind::Bool => value.as_bool().is_some(),
        PropertyValueKind::DateTime => value
            .as_str()
            .and_then(|text| DateTime::parse_from_rfc3339(text).ok())
            .is_some(),
        PropertyValueKind::Enum { values } => value
            .as_str()
            .is_some_and(|text| values.iter().any(|candidate| candidate == text)),
        PropertyValueKind::Reference { .. } => {
            value.as_str().is_some_and(|text| !text.trim().is_empty())
        }
        PropertyValueKind::Json => true,
    }
}

// ---------------------------------------------------------------------------
// Validation report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineeringValidationIssue {
    pub code: String,
    pub severity: Severity,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject_id: Option<String>,
}

impl EngineeringValidationIssue {
    pub fn error(
        code: impl Into<String>,
        message: impl Into<String>,
        subject_id: Option<String>,
    ) -> Self {
        Self {
            code: code.into(),
            severity: Severity::Error,
            message: message.into(),
            subject_id,
        }
    }

    pub fn warning(
        code: impl Into<String>,
        message: impl Into<String>,
        subject_id: Option<String>,
    ) -> Self {
        Self {
            code: code.into(),
            severity: Severity::Warning,
            message: message.into(),
            subject_id,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EngineeringValidationReport {
    pub issues: Vec<EngineeringValidationIssue>,
}

impl EngineeringValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors().is_empty()
    }

    pub fn errors(&self) -> Vec<&EngineeringValidationIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == Severity::Error)
            .collect()
    }

    pub fn warnings(&self) -> Vec<&EngineeringValidationIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == Severity::Warning)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Default profile
// ---------------------------------------------------------------------------

/// Default jcode engineering ontology profile.
///
/// This is intentionally smaller than BFO + ISO/IEC 24744 + Essence + SWEBOK.
/// It keeps only the terms that jcode can populate from tool calls and code
/// artifacts today.
pub fn default_engineering_ontology() -> EngineeringOntology {
    let mut ontology = EngineeringOntology::new(
        "jcode/engineering/v1",
        1,
        "Layered software-engineering profile: BFO-lite upper categories, ISO/IEC 24744-shaped endeavour/action/work-product backbone, OMG Essence alpha states, SWEBOK/SKOS concerns, PROV-O-shaped provenance.",
    );

    for concern in default_concerns() {
        ontology = ontology.with_concern(concern);
    }
    for class in default_classes() {
        ontology = ontology.with_class(class);
    }
    for relation in default_relations() {
        ontology = ontology.with_relation(relation);
    }
    for alpha in default_alphas() {
        ontology = ontology.with_alpha(alpha);
    }

    ontology
}

fn default_concerns() -> Vec<ConcernDef> {
    [
        (CONCERN_REQUIREMENTS, "Requirements"),
        (CONCERN_DESIGN, "Design"),
        (CONCERN_CONSTRUCTION, "Construction"),
        (CONCERN_TESTING, "Testing"),
        (CONCERN_MAINTENANCE, "Maintenance"),
        (CONCERN_CONFIGURATION_MANAGEMENT, "Configuration Management"),
        (CONCERN_ENGINEERING_MANAGEMENT, "Engineering Management"),
        (CONCERN_SOFTWARE_QUALITY, "Software Quality"),
        (CONCERN_ENGINEERING_ECONOMICS, "Engineering Economics"),
    ]
    .into_iter()
    .map(|(id, name)| ConcernDef::new(id, name))
    .collect()
}

fn default_classes() -> Vec<EngineeringClassDef> {
    let mut classes = Vec::new();

    classes.push(
        EngineeringClassDef::new(CLASS_SESSION, "Session", BfoCategory::Occurrent)
            .with_property(EngineeringPropertyDef::new(
                "session_id",
                PropertyValueKind::Identifier,
                true,
            ))
            .with_property(EngineeringPropertyDef::new(
                "working_dir",
                PropertyValueKind::String,
                false,
            ))
            .with_concern(CONCERN_ENGINEERING_MANAGEMENT),
    );
    classes.push(
        EngineeringClassDef::new(CLASS_TURN, "Turn", BfoCategory::Occurrent)
            .with_property(EngineeringPropertyDef::new(
                "turn_id",
                PropertyValueKind::Identifier,
                true,
            ))
            .with_concern(CONCERN_ENGINEERING_MANAGEMENT),
    );
    classes.push(
        EngineeringClassDef::new(CLASS_TASK, "Task", BfoCategory::Occurrent)
            .with_property(EngineeringPropertyDef::new(
                "title",
                PropertyValueKind::String,
                true,
            ))
            .with_property(EngineeringPropertyDef::new(
                "status",
                PropertyValueKind::Enum {
                    values: [
                        "pending",
                        "in_progress",
                        "blocked",
                        "completed",
                        "cancelled",
                    ]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
                },
                false,
            ))
            .with_property(EngineeringPropertyDef::new(
                "alpha_state",
                PropertyValueKind::String,
                false,
            ))
            .with_concern(CONCERN_REQUIREMENTS)
            .with_concern(CONCERN_ENGINEERING_MANAGEMENT),
    );
    classes.push(
        EngineeringClassDef::new(CLASS_WORK_UNIT, "Work Unit", BfoCategory::Occurrent)
            .with_property(EngineeringPropertyDef::new(
                "title",
                PropertyValueKind::String,
                true,
            ))
            .with_property(EngineeringPropertyDef::new(
                "status",
                PropertyValueKind::String,
                false,
            ))
            .with_concern(CONCERN_ENGINEERING_MANAGEMENT),
    );
    classes.push(
        EngineeringClassDef::new(CLASS_ACTION, "Action", BfoCategory::Occurrent)
            .with_property(EngineeringPropertyDef::new(
                "tool",
                PropertyValueKind::String,
                true,
            ))
            .with_property(EngineeringPropertyDef::new(
                "input",
                PropertyValueKind::Json,
                false,
            ))
            .with_property(EngineeringPropertyDef::new(
                "outcome",
                PropertyValueKind::String,
                false,
            ))
            .with_concern(CONCERN_CONSTRUCTION),
    );
    classes.push(
        EngineeringClassDef::new(CLASS_OBSERVATION, "Observation", BfoCategory::Occurrent)
            .with_property(EngineeringPropertyDef::new(
                "summary",
                PropertyValueKind::String,
                true,
            ))
            .with_concern(CONCERN_TESTING),
    );
    classes.push(
        EngineeringClassDef::new(CLASS_AGENT, "Agent", BfoCategory::IndependentContinuant)
            .with_property(EngineeringPropertyDef::new(
                "name",
                PropertyValueKind::String,
                true,
            ))
            .with_property(EngineeringPropertyDef::new(
                "kind",
                PropertyValueKind::String,
                false,
            ))
            .with_concern(CONCERN_ENGINEERING_MANAGEMENT),
    );
    classes.push(
        EngineeringClassDef::new(
            CLASS_ARTIFACT,
            "Artifact",
            BfoCategory::GenericallyDependentContinuant,
        )
        .with_property(EngineeringPropertyDef::new(
            "name",
            PropertyValueKind::String,
            true,
        ))
        .with_concern(CONCERN_CONFIGURATION_MANAGEMENT),
    );
    classes.push(
        EngineeringClassDef::new(CLASS_FILE, "File", BfoCategory::IndependentContinuant)
            .with_property(EngineeringPropertyDef::new(
                "path",
                PropertyValueKind::String,
                true,
            ))
            .with_property(EngineeringPropertyDef::new(
                "language",
                PropertyValueKind::String,
                false,
            ))
            .with_property(EngineeringPropertyDef::new(
                "repository",
                PropertyValueKind::String,
                false,
            ))
            .with_concern(CONCERN_CONSTRUCTION)
            .with_concern(CONCERN_MAINTENANCE),
    );
    classes.push(
        EngineeringClassDef::new(
            CLASS_SYMBOL,
            "Symbol",
            BfoCategory::GenericallyDependentContinuant,
        )
        .with_property(EngineeringPropertyDef::new(
            "name",
            PropertyValueKind::String,
            true,
        ))
        .with_property(EngineeringPropertyDef::new(
            "kind",
            PropertyValueKind::String,
            false,
        ))
        .with_concern(CONCERN_DESIGN)
        .with_concern(CONCERN_CONSTRUCTION),
    );
    classes.push(
        EngineeringClassDef::new(CLASS_CRATE, "Crate", BfoCategory::IndependentContinuant)
            .with_property(EngineeringPropertyDef::new(
                "name",
                PropertyValueKind::String,
                true,
            ))
            .with_property(EngineeringPropertyDef::new(
                "version",
                PropertyValueKind::String,
                false,
            ))
            .with_concern(CONCERN_CONFIGURATION_MANAGEMENT),
    );
    classes.push(
        EngineeringClassDef::new(
            CLASS_API,
            "API",
            BfoCategory::GenericallyDependentContinuant,
        )
        .with_property(EngineeringPropertyDef::new(
            "name",
            PropertyValueKind::String,
            true,
        ))
        .with_property(EngineeringPropertyDef::new(
            "method",
            PropertyValueKind::String,
            false,
        ))
        .with_property(EngineeringPropertyDef::new(
            "contract",
            PropertyValueKind::String,
            false,
        ))
        .with_concern(CONCERN_DESIGN)
        .with_concern(CONCERN_MAINTENANCE),
    );
    classes.push(
        EngineeringClassDef::new(
            CLASS_TEST,
            "Test",
            BfoCategory::GenericallyDependentContinuant,
        )
        .with_property(EngineeringPropertyDef::new(
            "name",
            PropertyValueKind::String,
            true,
        ))
        .with_property(EngineeringPropertyDef::new(
            "command",
            PropertyValueKind::String,
            false,
        ))
        .with_property(EngineeringPropertyDef::new(
            "status",
            PropertyValueKind::String,
            false,
        ))
        .with_concern(CONCERN_TESTING)
        .with_concern(CONCERN_SOFTWARE_QUALITY),
    );
    classes.push(
        EngineeringClassDef::new(CLASS_BUILD, "Build", BfoCategory::Occurrent)
            .with_property(EngineeringPropertyDef::new(
                "command",
                PropertyValueKind::String,
                true,
            ))
            .with_property(EngineeringPropertyDef::new(
                "status",
                PropertyValueKind::String,
                false,
            ))
            .with_concern(CONCERN_CONSTRUCTION)
            .with_concern(CONCERN_CONFIGURATION_MANAGEMENT),
    );
    classes.push(
        EngineeringClassDef::new(CLASS_COMMAND, "Command", BfoCategory::Occurrent)
            .with_property(EngineeringPropertyDef::new(
                "command",
                PropertyValueKind::String,
                true,
            ))
            .with_property(EngineeringPropertyDef::new(
                "cwd",
                PropertyValueKind::String,
                false,
            ))
            .with_property(EngineeringPropertyDef::new(
                "exit_code",
                PropertyValueKind::Integer,
                false,
            ))
            .with_concern(CONCERN_CONSTRUCTION),
    );
    classes.push(
        EngineeringClassDef::new(
            CLASS_ERROR,
            "Error",
            BfoCategory::SpecificallyDependentContinuant,
        )
        .with_property(EngineeringPropertyDef::new(
            "signature",
            PropertyValueKind::String,
            true,
        ))
        .with_property(EngineeringPropertyDef::new(
            "message",
            PropertyValueKind::String,
            false,
        ))
        .with_concern(CONCERN_MAINTENANCE)
        .with_concern(CONCERN_SOFTWARE_QUALITY),
    );
    classes.push(
        EngineeringClassDef::new(CLASS_CHANGE, "Change", BfoCategory::Occurrent)
            .with_property(EngineeringPropertyDef::new(
                "summary",
                PropertyValueKind::String,
                true,
            ))
            .with_property(EngineeringPropertyDef::new(
                "files",
                PropertyValueKind::Json,
                false,
            ))
            .with_concern(CONCERN_MAINTENANCE)
            .with_concern(CONCERN_CONFIGURATION_MANAGEMENT),
    );
    classes.push(
        EngineeringClassDef::new(
            CLASS_COMMIT,
            "Commit",
            BfoCategory::GenericallyDependentContinuant,
        )
        .with_property(EngineeringPropertyDef::new(
            "hash",
            PropertyValueKind::Identifier,
            true,
        ))
        .with_property(EngineeringPropertyDef::new(
            "message",
            PropertyValueKind::String,
            false,
        ))
        .with_concern(CONCERN_CONFIGURATION_MANAGEMENT),
    );
    classes.push(
        EngineeringClassDef::new(CLASS_REVIEW, "Review", BfoCategory::Occurrent)
            .with_property(EngineeringPropertyDef::new(
                "summary",
                PropertyValueKind::String,
                true,
            ))
            .with_property(EngineeringPropertyDef::new(
                "status",
                PropertyValueKind::String,
                false,
            ))
            .with_concern(CONCERN_SOFTWARE_QUALITY)
            .with_concern(CONCERN_ENGINEERING_MANAGEMENT),
    );
    classes.push(
        EngineeringClassDef::new(
            CLASS_DECISION,
            "Decision",
            BfoCategory::GenericallyDependentContinuant,
        )
        .with_property(EngineeringPropertyDef::new(
            "summary",
            PropertyValueKind::String,
            true,
        ))
        .with_property(EngineeringPropertyDef::new(
            "rationale",
            PropertyValueKind::String,
            false,
        ))
        .with_concern(CONCERN_DESIGN)
        .with_concern(CONCERN_REQUIREMENTS),
    );
    classes.push(
        EngineeringClassDef::new(
            CLASS_CONVENTION,
            "Convention",
            BfoCategory::GenericallyDependentContinuant,
        )
        .with_property(EngineeringPropertyDef::new(
            "statement",
            PropertyValueKind::String,
            true,
        ))
        .with_property(EngineeringPropertyDef::new(
            "scope",
            PropertyValueKind::String,
            false,
        ))
        .with_concern(CONCERN_CONSTRUCTION)
        .with_concern(CONCERN_SOFTWARE_QUALITY),
    );
    classes.push(
        EngineeringClassDef::new(
            CLASS_ENVIRONMENT,
            "Environment",
            BfoCategory::IndependentContinuant,
        )
        .with_property(EngineeringPropertyDef::new(
            "name",
            PropertyValueKind::String,
            true,
        ))
        .with_property(EngineeringPropertyDef::new(
            "kind",
            PropertyValueKind::String,
            false,
        ))
        .with_concern(CONCERN_CONFIGURATION_MANAGEMENT),
    );

    classes
}

fn default_relations() -> Vec<EngineeringRelationDef> {
    vec![
        EngineeringRelationDef::new(
            REL_CONTAINS,
            "Contains",
            &[
                CLASS_SESSION,
                CLASS_TURN,
                CLASS_TASK,
                CLASS_WORK_UNIT,
                CLASS_CRATE,
                CLASS_ARTIFACT,
            ],
            &[
                CLASS_TURN,
                CLASS_ACTION,
                CLASS_WORK_UNIT,
                CLASS_SYMBOL,
                CLASS_FILE,
                CLASS_ARTIFACT,
            ],
        )
        .with_inverse_of(REL_PART_OF)
        .with_concern(CONCERN_ENGINEERING_MANAGEMENT),
        EngineeringRelationDef::new(
            REL_PART_OF,
            "Part Of",
            &[
                CLASS_TURN,
                CLASS_ACTION,
                CLASS_WORK_UNIT,
                CLASS_SYMBOL,
                CLASS_FILE,
                CLASS_ARTIFACT,
            ],
            &[
                CLASS_SESSION,
                CLASS_TURN,
                CLASS_TASK,
                CLASS_CRATE,
                CLASS_ARTIFACT,
            ],
        )
        .with_inverse_of(REL_CONTAINS)
        .with_concern(CONCERN_ENGINEERING_MANAGEMENT),
        EngineeringRelationDef::new(
            REL_DEPENDS_ON,
            "Depends On",
            &[
                CLASS_TASK,
                CLASS_WORK_UNIT,
                CLASS_CRATE,
                CLASS_FILE,
                CLASS_CHANGE,
            ],
            &[
                CLASS_TASK,
                CLASS_WORK_UNIT,
                CLASS_CRATE,
                CLASS_FILE,
                CLASS_CHANGE,
            ],
        )
        .with_concern(CONCERN_ENGINEERING_MANAGEMENT),
        EngineeringRelationDef::new(
            REL_IMPORTS,
            "Imports",
            &[CLASS_FILE, CLASS_CRATE],
            &[CLASS_FILE, CLASS_CRATE],
        )
        .with_sub_property_of(REL_DEPENDS_ON)
        .with_concern(CONCERN_CONSTRUCTION),
        EngineeringRelationDef::new(
            REL_DEFINES,
            "Defines",
            &[CLASS_FILE, CLASS_CRATE],
            &[CLASS_SYMBOL, CLASS_API, CLASS_TEST],
        )
        .with_concern(CONCERN_DESIGN)
        .with_concern(CONCERN_CONSTRUCTION),
        EngineeringRelationDef::new(
            REL_CALLS,
            "Calls",
            &[CLASS_SYMBOL, CLASS_ACTION, CLASS_COMMAND],
            &[CLASS_SYMBOL, CLASS_API, CLASS_COMMAND],
        )
        .with_concern(CONCERN_DESIGN),
        EngineeringRelationDef::new(
            REL_TESTS,
            "Tests",
            &[CLASS_TEST, CLASS_ACTION],
            &[CLASS_FILE, CLASS_SYMBOL, CLASS_API, CLASS_CHANGE],
        )
        .with_concern(CONCERN_TESTING),
        EngineeringRelationDef::new(
            REL_MODIFIES,
            "Modifies",
            &[CLASS_ACTION, CLASS_CHANGE, CLASS_COMMIT],
            &[CLASS_FILE, CLASS_SYMBOL, CLASS_ARTIFACT],
        )
        .with_concern(CONCERN_MAINTENANCE),
        EngineeringRelationDef::new(
            REL_PRODUCES,
            "Produces",
            &[CLASS_ACTION, CLASS_BUILD, CLASS_COMMAND, CLASS_TURN],
            &[
                CLASS_OBSERVATION,
                CLASS_ARTIFACT,
                CLASS_CHANGE,
                CLASS_ERROR,
                CLASS_COMMIT,
            ],
        )
        .with_concern(CONCERN_CONSTRUCTION),
        EngineeringRelationDef::new(
            REL_CAUSES,
            "Causes",
            &[CLASS_ACTION, CLASS_COMMAND, CLASS_CHANGE, CLASS_ERROR],
            &[CLASS_ERROR, CLASS_OBSERVATION, CLASS_CHANGE],
        )
        .with_concern(CONCERN_MAINTENANCE),
        EngineeringRelationDef::new(
            REL_FIXES,
            "Fixes",
            &[CLASS_CHANGE, CLASS_COMMIT, CLASS_ACTION],
            &[CLASS_ERROR],
        )
        .with_concern(CONCERN_MAINTENANCE),
        EngineeringRelationDef::new(
            REL_VERIFIES,
            "Verifies",
            &[CLASS_TEST, CLASS_ACTION, CLASS_OBSERVATION],
            &[CLASS_CHANGE, CLASS_COMMIT, CLASS_ERROR, CLASS_FILE],
        )
        .with_concern(CONCERN_TESTING)
        .with_concern(CONCERN_SOFTWARE_QUALITY),
        EngineeringRelationDef::new(
            REL_DOCUMENTS,
            "Documents",
            &[CLASS_ARTIFACT, CLASS_FILE, CLASS_DECISION, CLASS_CONVENTION],
            &[
                CLASS_FILE,
                CLASS_API,
                CLASS_DECISION,
                CLASS_CONVENTION,
                CLASS_TASK,
            ],
        )
        .with_concern(CONCERN_CONFIGURATION_MANAGEMENT),
        EngineeringRelationDef::new(
            REL_SUPERSEDES,
            "Supersedes",
            &[CLASS_DECISION, CLASS_CONVENTION, CLASS_CHANGE, CLASS_COMMIT],
            &[CLASS_DECISION, CLASS_CONVENTION, CLASS_CHANGE, CLASS_COMMIT],
        )
        .with_concern(CONCERN_CONFIGURATION_MANAGEMENT),
        EngineeringRelationDef::new(
            REL_BLOCKED_BY,
            "Blocked By",
            &[CLASS_TASK, CLASS_WORK_UNIT, CLASS_ACTION],
            &[CLASS_ERROR, CLASS_TASK, CLASS_WORK_UNIT],
        )
        .with_concern(CONCERN_ENGINEERING_MANAGEMENT),
        EngineeringRelationDef::new(
            REL_APPLIES_TO,
            "Applies To",
            &[CLASS_CONVENTION, CLASS_DECISION],
            &[CLASS_FILE, CLASS_CRATE, CLASS_API, CLASS_ENVIRONMENT],
        )
        .with_concern(CONCERN_SOFTWARE_QUALITY),
        EngineeringRelationDef::new(
            REL_IMPLEMENTS,
            "Implements",
            &[CLASS_CHANGE, CLASS_COMMIT, CLASS_ACTION],
            &[CLASS_TASK, CLASS_DECISION, CLASS_API],
        )
        .with_concern(CONCERN_CONSTRUCTION),
        EngineeringRelationDef::new(
            REL_REVIEWS,
            "Reviews",
            &[CLASS_REVIEW],
            &[CLASS_CHANGE, CLASS_COMMIT, CLASS_DECISION],
        )
        .with_concern(CONCERN_SOFTWARE_QUALITY),
        EngineeringRelationDef::new(
            REL_USES,
            "Uses",
            &[CLASS_ACTION, CLASS_COMMAND, CLASS_BUILD, CLASS_TEST],
            &[
                CLASS_FILE,
                CLASS_CRATE,
                CLASS_API,
                CLASS_ENVIRONMENT,
                CLASS_AGENT,
            ],
        )
        .with_concern(CONCERN_CONSTRUCTION),
        EngineeringRelationDef::new(
            REL_EVIDENCED_BY,
            "Evidenced By",
            &[CLASS_WILDCARD],
            &[
                CLASS_OBSERVATION,
                CLASS_ARTIFACT,
                CLASS_SESSION,
                CLASS_TURN,
                CLASS_ACTION,
            ],
        )
        .with_concern(CONCERN_SOFTWARE_QUALITY),
    ]
}

fn default_alphas() -> Vec<AlphaDef> {
    vec![
        AlphaDef::new("opportunity", "Opportunity")
            .with_state(AlphaStateDef::new("identified", "Identified"))
            .with_state(AlphaStateDef::new("solution_needed", "Solution Needed"))
            .with_state(AlphaStateDef::new("value_established", "Value Established"))
            .with_state(AlphaStateDef::new("viable", "Viable"))
            .with_state(AlphaStateDef::new("actionable", "Actionable")),
        AlphaDef::new("stakeholders", "Stakeholders")
            .with_state(AlphaStateDef::new("identified", "Identified"))
            .with_state(AlphaStateDef::new("represented", "Represented"))
            .with_state(AlphaStateDef::new("involved", "Involved"))
            .with_state(AlphaStateDef::new("in_agreement", "In Agreement"))
            .with_state(AlphaStateDef::new("satisfied", "Satisfied")),
        AlphaDef::new("requirements", "Requirements")
            .with_state(AlphaStateDef::new("conceived", "Conceived"))
            .with_state(AlphaStateDef::new("bounded", "Bounded"))
            .with_state(AlphaStateDef::new("coherent", "Coherent"))
            .with_state(AlphaStateDef::new("acceptable", "Acceptable"))
            .with_state(AlphaStateDef::new("addressed", "Addressed"))
            .with_state(AlphaStateDef::new("fulfilled", "Fulfilled")),
        AlphaDef::new("software_system", "Software System")
            .with_state(AlphaStateDef::new(
                "architecture_selected",
                "Architecture Selected",
            ))
            .with_state(AlphaStateDef::new("demonstrable", "Demonstrable"))
            .with_state(AlphaStateDef::new("usable", "Usable"))
            .with_state(AlphaStateDef::new("ready", "Ready"))
            .with_state(AlphaStateDef::new("operational", "Operational"))
            .with_state(AlphaStateDef::new("retired", "Retired")),
        AlphaDef::new("work", "Work")
            .with_state(AlphaStateDef::new("initiated", "Initiated"))
            .with_state(AlphaStateDef::new("prepared", "Prepared"))
            .with_state(AlphaStateDef::new("started", "Started"))
            .with_state(AlphaStateDef::new("under_control", "Under Control"))
            .with_state(AlphaStateDef::new("concluded", "Concluded"))
            .with_state(AlphaStateDef::new("closed", "Closed")),
        AlphaDef::new("team", "Team")
            .with_state(AlphaStateDef::new("assembled", "Assembled"))
            .with_state(AlphaStateDef::new("formed", "Formed"))
            .with_state(AlphaStateDef::new("collaborating", "Collaborating"))
            .with_state(AlphaStateDef::new("performing", "Performing"))
            .with_state(AlphaStateDef::new("adjourned", "Adjourned")),
        AlphaDef::new("way_of_working", "Way of Working")
            .with_state(AlphaStateDef::new(
                "principles_established",
                "Principles Established",
            ))
            .with_state(AlphaStateDef::new(
                "foundation_established",
                "Foundation Established",
            ))
            .with_state(AlphaStateDef::new("in_use", "In Use"))
            .with_state(AlphaStateDef::new("in_place", "In Place"))
            .with_state(AlphaStateDef::new("working_well", "Working Well"))
            .with_state(AlphaStateDef::new("retired", "Retired")),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn file(id: &str) -> EngineeringEntity {
        EngineeringEntity::new(CLASS_FILE, id, id)
            .with_property("path", json!("crates/jcode-base/src/lib.rs"))
    }

    fn test(id: &str) -> EngineeringEntity {
        EngineeringEntity::new(CLASS_TEST, id, id).with_property("name", json!(id))
    }

    fn task(id: &str) -> EngineeringEntity {
        EngineeringEntity::new(CLASS_TASK, id, id).with_property("title", json!(id))
    }

    #[test]
    fn default_ontology_is_valid() {
        let ontology = default_engineering_ontology();
        let report = ontology.validate();
        assert!(report.is_valid(), "{:?}", report.errors());
        assert!(ontology.classes.contains_key(CLASS_FILE));
        assert!(ontology.relations.contains_key(REL_MODIFIES));
        assert!(ontology.concerns.contains_key(CONCERN_TESTING));
        assert!(ontology.alphas.contains_key("requirements"));
    }

    #[test]
    fn content_hash_is_stable() {
        let a = default_engineering_ontology();
        let b = default_engineering_ontology();
        assert_eq!(a.content_hash(), b.content_hash());
        assert_eq!(a.content_hash().len(), 16);
    }

    #[test]
    fn rejects_missing_required_property() {
        let ontology = default_engineering_ontology();
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(EngineeringEntity::new(CLASS_FILE, "f1", "f1"));
        let report = graph.validate(&ontology);
        assert!(report.errors().iter().any(|issue| issue.code == "EG11"));
    }

    #[test]
    fn rejects_wrong_domain() {
        let ontology = default_engineering_ontology();
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(file("f1"));
        graph.insert_entity(task("t1"));
        graph.insert_relation(EngineeringRelation::new("f1", REL_MODIFIES, "t1"));
        let report = graph.validate(&ontology);
        assert!(report.errors().iter().any(|issue| issue.code == "EG28"));
    }

    #[test]
    fn rejects_dangling_relation() {
        let ontology = default_engineering_ontology();
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(file("f1"));
        graph.insert_relation(EngineeringRelation::new("f1", REL_MODIFIES, "missing"));
        let report = graph.validate(&ontology);
        assert!(report.errors().iter().any(|issue| issue.code == "EG25"));
    }

    #[test]
    fn rejects_bad_time_window() {
        let ontology = default_engineering_ontology();
        let mut graph = EngineeringGraph::new();
        let mut entity = file("f1");
        entity.valid_from = Some(Utc::now());
        entity.valid_to = Some(Utc::now() - chrono::Duration::days(1));
        graph.insert_entity(entity);
        let report = graph.validate(&ontology);
        assert!(report.errors().iter().any(|issue| issue.code == "EG05"));
    }

    #[test]
    fn inverse_relation_inference_traverses_reverse_part_of() {
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(EngineeringEntity::new(CLASS_FILE, "file:a", "a.rs"));
        graph.insert_entity(EngineeringEntity::new(CLASS_SESSION, "session:s", "session"));
        graph.insert_relation(EngineeringRelation::new("file:a", REL_PART_OF, "session:s"));

        let inferred = graph.infer_relations(&default_engineering_ontology());
        let path = inferred
            .shortest_path("session:s", "file:a", 2, &[REL_CONTAINS])
            .expect("inverse contains path");
        assert_eq!(path, vec!["session:s", "file:a"]);
        let relation = inferred
            .relation("inferred:inverse:rel:file:a:part_of:session:s:contains")
            .expect("virtual inverse relation");
        assert_eq!(relation.relation, REL_CONTAINS);
        assert!(relation.provenance.source.starts_with("inferred:inverse_of:"));
        assert!(relation.confidence < 0.8);
    }

    #[test]
    fn subproperty_inference_expands_depends_on() {
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(EngineeringEntity::new(CLASS_FILE, "file:a", "a.rs"));
        graph.insert_entity(EngineeringEntity::new(CLASS_FILE, "file:b", "b.rs"));
        graph.insert_relation(EngineeringRelation::new("file:a", REL_IMPORTS, "file:b"));

        let inferred = graph.infer_relations(&default_engineering_ontology());
        let path = inferred
            .shortest_path("file:a", "file:b", 1, &[REL_DEPENDS_ON])
            .expect("subproperty depends_on path");
        assert_eq!(path, vec!["file:a", "file:b"]);
        let relation_id = format!("inferred:sub:rel:file:a:imports:file:b:{REL_DEPENDS_ON}");
        let relation = inferred
            .relation(&relation_id)
            .expect("virtual subproperty relation");
        assert_eq!(relation.relation, REL_DEPENDS_ON);
    }

    #[test]
    fn shortest_path_uses_typed_relations() {
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(task("t1"));
        graph.insert_entity(task("t2"));
        graph.insert_entity(
            EngineeringEntity::new(CLASS_ERROR, "e1", "E1").with_property("signature", json!("E1")),
        );
        graph.insert_relation(EngineeringRelation::new("t1", REL_DEPENDS_ON, "t2"));
        graph.insert_relation(EngineeringRelation::new("t2", REL_BLOCKED_BY, "e1"));
        let path = graph
            .shortest_path("t1", "e1", 3, &[REL_DEPENDS_ON, REL_BLOCKED_BY])
            .expect("path");
        assert_eq!(path, vec!["t1", "t2", "e1"]);
    }

    #[test]
    fn weighted_paths_honor_weights_and_return_provenance() {
        let mut graph = EngineeringGraph::new();
        for id in ["t0", "t1", "t2", "t3"] {
            graph.insert_entity(task(id));
        }
        graph.insert_relation(
            EngineeringRelation::new("t0", REL_DEPENDS_ON, "t1")
                .with_confidence(0.95)
                .with_provenance(EngineeringProvenance::new("trace:high-confidence")),
        );
        graph.insert_relation(
            EngineeringRelation::new("t1", REL_DEPENDS_ON, "t3")
                .with_confidence(0.95)
                .with_provenance(EngineeringProvenance::new("trace:high-confidence")),
        );
        graph.insert_relation(
            EngineeringRelation::new("t0", REL_DEPENDS_ON, "t2")
                .with_confidence(0.5)
                .with_property("weight", json!(0.2))
                .with_provenance(EngineeringProvenance::new("trace:cheap")),
        );
        graph.insert_relation(
            EngineeringRelation::new("t2", REL_DEPENDS_ON, "t3")
                .with_confidence(0.5)
                .with_property("weight", json!(0.2))
                .with_provenance(EngineeringProvenance::new("trace:cheap")),
        );

        let paths = graph.k_shortest_paths("t0", "t3", 3, &[REL_DEPENDS_ON], 2);
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].nodes, vec!["t0", "t2", "t3"]);
        assert_eq!(paths[1].nodes, vec!["t0", "t1", "t3"]);
        assert!(paths[0].cost < paths[1].cost);
        assert_eq!(paths[0].hop_count(), 2);
        assert_eq!(paths[0].provenance[0].source, "trace:cheap");
        assert!(paths[0].confidence < paths[1].confidence);
    }

    #[test]
    fn serde_round_trip_preserves_graph() {
        let ontology = default_engineering_ontology();
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(file("f1"));
        graph.insert_entity(test("test1"));
        graph.insert_relation(EngineeringRelation::new("test1", REL_TESTS, "f1"));
        let encoded = serde_json::to_string(&graph).expect("serialize");
        let decoded: EngineeringGraph = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(graph, decoded);
        let report = decoded.validate(&ontology);
        assert!(report.is_valid(), "{:?}", report.errors());
    }

    #[test]
    fn property_value_kind_is_enforced() {
        let ontology = default_engineering_ontology();
        let mut graph = EngineeringGraph::new();
        let mut entity = file("f1");
        entity.properties.insert("path".to_string(), json!(42));
        graph.insert_entity(entity);
        let report = graph.validate(&ontology);
        assert!(report.errors().iter().any(|issue| issue.code == "EG12"));
    }

    #[test]
    fn merge_unions_entities_and_relations_without_overwriting() {
        let mut graph = EngineeringGraph::new();
        let mut first = file("f1");
        first.name = "kept".to_string();
        graph.insert_entity(first);

        let mut other = EngineeringGraph::new();
        let mut replacement = file("f1");
        replacement.name = "ignored".to_string();
        other.insert_entity(replacement);
        other.insert_entity(test("test1"));
        other.insert_relation(EngineeringRelation::new("test1", REL_TESTS, "f1"));

        graph.merge(&other);
        assert_eq!(graph.entities["f1"].name, "kept");
        assert!(graph.entities.contains_key("test1"));
        assert_eq!(graph.relations.len(), 1);
    }

    #[test]
    fn concept_graph_projects_relation_signatures_and_resolves_anchors() {
        let ontology = default_engineering_ontology();
        let graph = EngineeringGraph::concept_graph(&ontology);
        assert!(graph.entities.contains_key(&concept_node_id(CLASS_FILE)));
        assert!(graph.entities.contains_key(&concept_node_id(CLASS_API)));
        assert_eq!(
            graph.resolve_concept_anchor("File").as_deref(),
            Some(concept_node_id(CLASS_FILE).as_str())
        );
        assert_eq!(
            graph.resolve_concept_anchor(CLASS_API).as_deref(),
            Some(concept_node_id(CLASS_API).as_str())
        );

        let paths = graph.k_shortest_paths(
            &concept_node_id(CLASS_FILE),
            &concept_node_id(CLASS_API),
            2,
            &[REL_DEFINES],
            1,
        );
        assert_eq!(paths.len(), 1, "{paths:?}");
        assert_eq!(paths[0].hop_count(), 1);
        assert_eq!(
            graph.relation(&paths[0].relations[0]).map(|rel| rel.relation.as_str()),
            Some(REL_DEFINES)
        );
    }

    #[test]
    fn ontology_layer_joins_instances_through_concept_signatures() {
        let ontology = default_engineering_ontology();
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(file("f1"));
        graph.insert_entity(EngineeringEntity::new(CLASS_API, "api:1", "api:1"));

        let joined = graph.with_ontology_layer(&ontology);
        let paths = joined.k_shortest_paths(
            "f1",
            "api:1",
            4,
            &[REL_INSTANCE_OF, REL_DEFINES, REL_HAS_INSTANCE],
            1,
        );
        assert_eq!(paths.len(), 1, "{paths:?}");
        assert_eq!(
            paths[0].nodes,
            vec!["f1", "concept:file", "concept:api", "api:1"]
        );

        // The join is query-time only; the source graph must stay untouched.
        assert_eq!(graph.entities.len(), 2);
        assert!(graph.relations.is_empty());
        assert_eq!(joined.entities.len(), graph.entities.len() + ontology.classes.len());
    }
}
