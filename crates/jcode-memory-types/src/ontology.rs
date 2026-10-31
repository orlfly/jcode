//! Ontology definition for the memory graph.
//!
//! An ontology is a data structure that captures the *schema* of memory
//! instances: which types exist, what properties each type carries, the
//! lifecycle states it can move through, the kinds of relationships it may
//! participate in, and the rules that fire on write/change/maintain events.
//!
//! Run-time code (`MemoryManager`, `MemoryAgent`, the `memory` tool) consults
//! the active ontology instead of relying on hardcoded `match MemoryCategory`
//! branches. The hardcoded branches in `lib.rs` are kept as a fallback when
//! no ontology is registered, but new code paths should be driven by the
//! data in this file.
//!
//! Three layers:
//!
//! 1. **Schema** ([`Ontology`], [`OntologyType`], [`PropertyDef`],
//!    [`RelationshipDef`], [`LifecyclePolicy`]) — the blueprint.
//! 2. **Rules** ([`Rule`], [`Effect`], [`Condition`]) — declarative actions
//!    that fire on events.
//! 3. **Activities** ([`Activity`], [`ActivityTrigger`], [`ActivityStep`]) —
//!    scheduled or event-driven work that summarizes, extracts, or transitions
//!    instances.

use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::MemoryCategory;
use crate::instance::{ExtractionMethod, MemoryStatus};

// ---------------------------------------------------------------------------
// Type constants.  These are the canonical identifiers used by the default
// ontology and exposed to callers that want to extend or override it.
// ---------------------------------------------------------------------------

/// Schema version of the `default_ontology()` payload below.  Bump this when
/// the default ontology is changed in a way that may invalidate stored
/// graphs (e.g. type ids renamed, rule semantics altered).
///
/// v2 splits the overloaded `fact` bucket into `env_fact` / `contract` /
/// `process_rule` / `observation`, drops the never-instantiated `note` and
/// `skill` types (skills are an external artifact linked through
/// [`LINK_SKILL`]), drops the never-maintained `in_cluster` relationship, and
/// replaces the unused continuous decay with declarative TTLs.
pub const DEFAULT_ONTOLOGY_VERSION: u32 = 2;
pub const DEFAULT_ONTOLOGY_ID: &str = "jcode/default/v1";

// ---------------------------------------------------------------------------
// Types. Every type declared here must have (a) a producer and (b) a runtime
// consumer that enforces its policy. Types without either do not belong in the
// ontology - that was the core defect of v1.
// ---------------------------------------------------------------------------

/// Perishable environment fact: a host, port, path, task id or key that is
/// only true for one machine / workspace and rots quickly.
pub const TYPE_ENV_FACT: &str = "env_fact";
/// A durable interface contract: API route, field, enum, payload shape.
pub const TYPE_CONTRACT: &str = "contract";
/// A durable procedure: "must / never / the correct way" workflow knowledge.
pub const TYPE_PROCESS_RULE: &str = "process_rule";
/// Low-authority residual bucket for anything that cannot be classified.
pub const TYPE_OBSERVATION: &str = "observation";
pub const TYPE_PREFERENCE: &str = "preference";
pub const TYPE_ENTITY: &str = "entity";
pub const TYPE_CORRECTION: &str = "correction";

// ---------------------------------------------------------------------------
// Link kinds. External artifacts (goals, skills) are mirrored into the graph
// under a reserved id prefix. They are *not* extractable types: the owning
// subsystem writes them, they carry a bounded pointer body, and they are
// exempt from TTL collection.
// ---------------------------------------------------------------------------

pub const LINK_GOAL: &str = "goal";
pub const LINK_SKILL: &str = "skill";

/// Built-in reserved link prefixes. Custom artifact kinds are registered at
/// runtime; these remain the compatibility defaults.
pub const LINK_PREFIXES: [&str; 2] = ["goal:", "skill:"];

/// A configurable external-artifact kind mirrored into the memory graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactKind {
    /// Stable id, e.g. `skill` or `goal`.
    pub id: String,
    /// Reserved id prefix, e.g. `skill:`.
    pub prefix: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
}

/// Built-in artifact kinds. Runtime configuration may add more.
pub fn default_artifact_kinds() -> Vec<ArtifactKind> {
    vec![
        ArtifactKind {
            id: LINK_SKILL.to_string(),
            prefix: "skill:".to_string(),
            description: "Executable procedure installed as SKILL.md".to_string(),
        },
        ArtifactKind {
            id: LINK_GOAL.to_string(),
            prefix: "goal:".to_string(),
            description: "Long-lived goal mirrored from its owner subsystem".to_string(),
        },
    ]
}

fn artifact_kind_registry() -> &'static std::sync::RwLock<Vec<ArtifactKind>> {
    static REGISTRY: std::sync::OnceLock<std::sync::RwLock<Vec<ArtifactKind>>> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| std::sync::RwLock::new(default_artifact_kinds()))
}

/// Current artifact kinds, including runtime-registered ones.
pub fn artifact_kinds() -> Vec<ArtifactKind> {
    artifact_kind_registry()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// Replace the registry with defaults plus `kinds`.
///
/// Ids and prefixes are deduplicated; empty prefixes are ignored. This is what
/// a config file or an embedding application calls at startup.
pub fn configure_artifact_kinds(kinds: Vec<ArtifactKind>) {
    let mut merged = default_artifact_kinds();
    for kind in kinds {
        let prefix = kind.prefix.trim();
        if prefix.is_empty() {
            continue;
        }
        if let Some(existing) = merged.iter_mut().find(|existing| existing.id == kind.id) {
            *existing = ArtifactKind {
                prefix: prefix.to_string(),
                ..kind
            };
        } else if !merged.iter().any(|existing| existing.prefix == prefix) {
            merged.push(ArtifactKind {
                prefix: prefix.to_string(),
                ..kind
            });
        }
    }
    *artifact_kind_registry()
        .write()
        .unwrap_or_else(|e| e.into_inner()) = merged;
}

/// Register one additional artifact kind without replacing the registry.
pub fn register_artifact_kind(kind: ArtifactKind) {
    let mut kinds = artifact_kinds();
    kinds.push(kind);
    configure_artifact_kinds(kinds);
}

/// The artifact kind whose prefix matches `id`, if any.
pub fn artifact_kind_for_id(id: &str) -> Option<ArtifactKind> {
    artifact_kind_registry()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|kind| id.starts_with(&kind.prefix))
        .cloned()
}

/// Whether `id` is a link node (external artifact mirror) rather than a
/// classic memory instance.
pub fn is_link_id(id: &str) -> bool {
    artifact_kind_registry()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .any(|kind| id.starts_with(&kind.prefix))
}

pub const REL_HAS_TAG: &str = "has_tag";
pub const REL_RELATES_TO: &str = "relates_to";
pub const REL_SUPERSEDES: &str = "supersedes";
pub const REL_CONTRADICTS: &str = "contradicts";
pub const REL_DERIVED_FROM: &str = "derived_from";
/// Deprecated. Kept so already-stored `in_cluster` edges stay resolvable and
/// validate; no runtime producer creates new ones (v1 produced 11 in total and
/// nothing ever maintained the cluster nodes).
pub const REL_IN_CLUSTER: &str = "in_cluster";

pub const RULE_REMEMBER: &str = "rule.remember";
pub const RULE_REMEMBER_GLOBAL_GATE: &str = "rule.remember.global_gate";
pub const RULE_UPSERT: &str = "rule.upsert";
/// Write-time reconciliation: one FTS/lexical pass over the target store that
/// decides between reinforce / supersede / contradict / relate / insert.
pub const RULE_WRITE_RECONCILE: &str = "rule.write_reconcile";
/// Noise gate: reject tool payloads, bare identifiers and stubs.
pub const RULE_REJECT: &str = "rule.reject";
pub const RULE_DEDUP_REINFORCE: &str = "rule.dedup_reinforce";
pub const RULE_SUPERSEDE: &str = "rule.supersede";
pub const RULE_CONTRADICT: &str = "rule.contradict";
pub const RULE_TAG: &str = "rule.tag";
pub const RULE_UNTAG: &str = "rule.untag";
pub const RULE_LINK: &str = "rule.link";
/// A successful action execution validates the knowledge behind it.
pub const RULE_SKILL_VALIDATED: &str = "rule.skill.validated";
/// A failed or cancelled action marks the knowledge for review.
pub const RULE_SKILL_INVALIDATED: &str = "rule.skill.invalidated";

pub const ACTIVITY_FINAL_EXTRACT: &str = "activity.final_extract";
/// Synchronous write-time reconciliation (no LLM, no embedder).
pub const ACTIVITY_WRITE_RECONCILE: &str = "activity.write_reconcile";
/// TTL sweep: archive instances whose type TTL elapsed without access.
pub const ACTIVITY_PERIODIC_GC: &str = "activity.periodic_gc";
/// Aggregate evidence memories into skill proposals (memory -> skill).
pub const ACTIVITY_SKILL_DISTILL: &str = "activity.skill_distill";

pub const EVENT_REMEMBER: &str = "event.remember";
pub const EVENT_REMEMBER_GLOBAL: &str = "event.remember.global";
pub const EVENT_UPSERT: &str = "event.upsert";
pub const EVENT_DEDUP: &str = "event.dedup";
pub const EVENT_SUPERSEDE: &str = "event.supersede";
pub const EVENT_CONTRADICT: &str = "event.contradict";
pub const EVENT_TAG: &str = "event.tag";
pub const EVENT_UNTAG: &str = "event.untag";
pub const EVENT_LINK: &str = "event.link";
pub const EVENT_TOPIC_CHANGE: &str = "event.topic_change";
pub const EVENT_TURN_TICK: &str = "event.turn_tick";
pub const EVENT_FINALIZE: &str = "event.finalize";
/// An action/skill was invoked. Carries the execution outcome.
pub const EVENT_ACTION_OUTCOME: &str = "event.action_outcome";

// ---------------------------------------------------------------------------
// Schema structures
// ---------------------------------------------------------------------------

/// A complete ontology definition.  An `Ontology` is a value object: it can
/// be loaded from JSON, mutated in-memory, and serialized back out.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Ontology {
    pub id: String,
    pub version: u32,
    pub description: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub types: BTreeMap<String, OntologyType>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub relationships: BTreeMap<String, RelationshipDef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rules: Vec<Rule>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub activities: Vec<Activity>,
    /// Mapping from legacy `MemoryCategory` strings to ontology type ids.
    /// Lets the existing category-based call sites keep working while the
    /// runtime consults the ontology for behavior.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub category_aliases: HashMap<String, String>,
}

impl Ontology {
    pub fn new(id: impl Into<String>, version: u32, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version,
            description: description.into(),
            types: BTreeMap::new(),
            relationships: BTreeMap::new(),
            rules: Vec::new(),
            activities: Vec::new(),
            category_aliases: HashMap::new(),
        }
    }

    /// Register a type and a corresponding `category_aliases` mapping for the
    /// legacy `MemoryCategory` name.
    pub fn with_type(mut self, type_def: OntologyType) -> Self {
        let type_id = type_def.id.clone();
        if let Some(alias) = type_def.legacy_category.clone() {
            self.category_aliases.insert(alias, type_id.clone());
        }
        self.types.insert(type_id, type_def);
        self
    }

    pub fn with_relationship(mut self, rel: RelationshipDef) -> Self {
        self.relationships.insert(rel.id.clone(), rel);
        self
    }

    pub fn with_rule(mut self, rule: Rule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn with_activity(mut self, activity: Activity) -> Self {
        self.activities.push(activity);
        self
    }

    /// Look up a type by id.
    pub fn get_type(&self, id: &str) -> Option<&OntologyType> {
        self.types.get(id)
    }

    /// Translate a legacy `MemoryCategory` into the canonical ontology type id.
    pub fn type_from_category(&self, category: &MemoryCategory) -> Option<&str> {
        let key = category.to_string();
        if let Some(s) = self.category_aliases.get(&key) {
            return Some(s.as_str());
        }
        if self.types.contains_key(&key) {
            return Some(self.types.get_key_value(&key)?.0.as_str());
        }
        None
    }

    /// All rules that fire on `event`.
    pub fn rules_for_event(&self, event: &str) -> impl Iterator<Item = &Rule> {
        self.rules.iter().filter(move |r| r.event == event)
    }

    /// All activities that fire on `event`.
    pub fn activities_for_event(&self, event: &str) -> impl Iterator<Item = &Activity> {
        self.activities.iter().filter(move |a| match &a.trigger {
            ActivityTrigger::OnEvent { event: e } => e == event,
            _ => false,
        })
    }

    /// Periodic activities whose cadence has elapsed since `last_tick`.
    pub fn periodic_activities_due(&self, last_tick: DateTime<Utc>) -> Vec<&Activity> {
        self.activities
            .iter()
            .filter(|a| match &a.trigger {
                ActivityTrigger::Periodic { interval_turns, .. } => {
                    let now = Utc::now();
                    let elapsed = now - last_tick;
                    let secs = elapsed.num_seconds().max(0) as u64;
                    secs >= *interval_turns as u64 * 60
                }
                _ => false,
            })
            .collect()
    }
}

/// Definition of a memory instance type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OntologyType {
    pub id: String,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// Legacy `MemoryCategory` name (e.g. "fact", "preference") that this
    /// type maps to.  Used by `Ontology::type_from_category`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_category: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub properties: Vec<PropertyDef>,
    pub lifecycle: LifecyclePolicy,
    pub scoring: ScoringPolicy,
    pub decay: DecayPolicy,
    /// Hard time-to-live in days for this type. Instances of a type with a
    /// `ttl_days` are archived by `activity.periodic_gc` once they have gone
    /// unaccessed for that long, and physically dropped after a further grace
    /// period. `None` means the instance is retained indefinitely.
    ///
    /// This is deliberately a discrete TTL rather than continuous decay: the
    /// runtime can evaluate it without scoring every instance, and it stays
    /// meaningful on machines with no embedder and no LLM judge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_days: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extraction_methods: Vec<ExtractionMethod>,
    /// Whether new instances of this type are persisted into the graph.
    /// `false` for synthetic types (`skill`) that are produced at runtime but
    /// never written to disk.
    #[serde(default = "default_persistent")]
    pub persistent: bool,
    /// Whether embedding should be generated for new instances.
    #[serde(default = "default_embeddable")]
    pub embeddable: bool,
    /// Whether this type is normally surfaced to the user as a "category"
    /// section in the prompt formatter.
    #[serde(default = "default_present_in_prompt")]
    pub present_in_prompt: bool,
}

fn default_persistent() -> bool {
    true
}
fn default_embeddable() -> bool {
    true
}
fn default_present_in_prompt() -> bool {
    true
}

impl OntologyType {
    pub fn new(id: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            parent: None,
            legacy_category: None,
            properties: Vec::new(),
            lifecycle: LifecyclePolicy::default(),
            scoring: ScoringPolicy::default(),
            decay: DecayPolicy::default(),
            ttl_days: None,
            extraction_methods: vec![ExtractionMethod::UserStated],
            persistent: true,
            embeddable: true,
            present_in_prompt: true,
        }
    }

    /// Attach a hard TTL (days) for this type.
    pub fn with_ttl_days(mut self, days: f32) -> Self {
        self.ttl_days = Some(days);
        self
    }

    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    pub fn with_legacy_category(mut self, name: impl Into<String>) -> Self {
        self.legacy_category = Some(name.into());
        self
    }

    pub fn with_lifecycle(mut self, lifecycle: LifecyclePolicy) -> Self {
        self.lifecycle = lifecycle;
        self
    }

    pub fn with_scoring(mut self, scoring: ScoringPolicy) -> Self {
        self.scoring = scoring;
        self
    }

    pub fn with_decay(mut self, decay: DecayPolicy) -> Self {
        self.decay = decay;
        self
    }

    pub fn with_extraction_method(mut self, method: ExtractionMethod) -> Self {
        self.extraction_methods.push(method);
        self
    }

    pub fn with_persistent(mut self, persistent: bool) -> Self {
        self.persistent = persistent;
        self
    }

    pub fn with_embeddable(mut self, embeddable: bool) -> Self {
        self.embeddable = embeddable;
        self
    }

    pub fn with_present_in_prompt(mut self, present: bool) -> Self {
        self.present_in_prompt = present;
        self
    }

    pub fn with_property(mut self, property: PropertyDef) -> Self {
        self.properties.push(property);
        self
    }

    /// Returns the union of this type's properties and those inherited from
    /// `parent`.  When no parent is set, returns the local properties.
    pub fn effective_properties<'a>(
        &'a self,
        parent: Option<&'a OntologyType>,
    ) -> Vec<&'a PropertyDef> {
        let mut out: Vec<&PropertyDef> = self.properties.iter().collect();
        if let Some(p) = parent {
            for prop in &p.properties {
                if !out.iter().any(|existing| existing.id == prop.id) {
                    out.push(prop);
                }
            }
        }
        out
    }
}

/// A typed property that an `OntologyType` exposes on its instances.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PropertyDef {
    pub id: String,
    pub kind: PropertyKind,
    #[serde(default)]
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PropertyKind {
    String,
    Tags,
    Embedding,
    Trust,
    Float,
    Bool,
    Status,
    Provenance,
    Lifecycle,
    Identity,
    Datetime,
}

/// Lifecycle states and allowed transitions for a type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LifecyclePolicy {
    /// States this type can take.
    pub states: Vec<MemoryStatus>,
    /// Allowed transitions as (from, to) edges.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transitions: Vec<LifecycleTransition>,
    /// State a new instance starts in.
    pub initial_state: MemoryStatus,
    /// Whether `active = false` follows when status becomes Archived.
    #[serde(default = "default_true")]
    pub archive_deactivates: bool,
}

fn default_true() -> bool {
    true
}

impl Default for LifecyclePolicy {
    fn default() -> Self {
        Self {
            states: vec![
                MemoryStatus::Active,
                MemoryStatus::Expired,
                MemoryStatus::Archived,
                MemoryStatus::Disputed,
            ],
            transitions: vec![
                LifecycleTransition {
                    from: MemoryStatus::Active,
                    to: MemoryStatus::Expired,
                },
                LifecycleTransition {
                    from: MemoryStatus::Active,
                    to: MemoryStatus::Archived,
                },
                LifecycleTransition {
                    from: MemoryStatus::Active,
                    to: MemoryStatus::Disputed,
                },
                LifecycleTransition {
                    from: MemoryStatus::Expired,
                    to: MemoryStatus::Active,
                },
                LifecycleTransition {
                    from: MemoryStatus::Expired,
                    to: MemoryStatus::Archived,
                },
                LifecycleTransition {
                    from: MemoryStatus::Archived,
                    to: MemoryStatus::Active,
                },
                LifecycleTransition {
                    from: MemoryStatus::Disputed,
                    to: MemoryStatus::Active,
                },
                LifecycleTransition {
                    from: MemoryStatus::Disputed,
                    to: MemoryStatus::Archived,
                },
            ],
            initial_state: MemoryStatus::Active,
            archive_deactivates: true,
        }
    }
}

impl LifecyclePolicy {
    pub fn can_transition(&self, from: MemoryStatus, to: MemoryStatus) -> bool {
        if from == to {
            return true;
        }
        self.transitions
            .iter()
            .any(|t| t.from == from && t.to == to)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LifecycleTransition {
    pub from: MemoryStatus,
    pub to: MemoryStatus,
}

/// Scoring policy for a type: base score, recency weight, etc.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoringPolicy {
    /// Base additive score awarded to every instance of this type.
    pub base_score: f32,
    /// Multiplier applied to the trust-modulated score.
    pub trust_multiplier: TrustMultiplier,
    /// Multiplier applied to `strength` (log) contribution.
    #[serde(default = "default_strength_multiplier")]
    pub strength_multiplier: f32,
}

impl Default for ScoringPolicy {
    fn default() -> Self {
        Self {
            base_score: 20.0,
            trust_multiplier: TrustMultiplier::default(),
            strength_multiplier: 5.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrustMultiplier {
    #[serde(rename = "High")]
    pub high: f32,
    #[serde(rename = "Medium")]
    pub medium: f32,
    #[serde(rename = "Low")]
    pub low: f32,
}

impl Default for TrustMultiplier {
    fn default() -> Self {
        Self {
            high: 1.5,
            medium: 1.0,
            low: 0.7,
        }
    }
}

fn default_strength_multiplier() -> f32 {
    5.0
}

/// Decay policy for a type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecayPolicy {
    /// Half-life in days for the exponential decay applied to confidence.
    pub half_life_days: f32,
    /// Whether each retrieval boosts confidence.
    #[serde(default = "default_true")]
    pub access_boost: bool,
}

impl Default for DecayPolicy {
    fn default() -> Self {
        Self {
            half_life_days: 45.0,
            access_boost: true,
        }
    }
}

/// Definition of a relationship between two instances (or an instance and a
/// tag/cluster node).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RelationshipDef {
    pub id: String,
    pub display_name: String,
    /// "memory" or "tag" or "cluster" — the kinds of source nodes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_kinds: Vec<String>,
    /// "memory" or "tag" or "cluster" — the kinds of target nodes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_kinds: Vec<String>,
    /// Optional weight range (for weighted relationships).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<WeightRange>,
    /// Traversal weight for cascade retrieval.
    #[serde(default = "default_traversal_weight")]
    pub traversal_weight: f32,
}

fn default_traversal_weight() -> f32 {
    1.0
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct WeightRange {
    pub min: f32,
    pub max: f32,
}

// ---------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------

/// A rule that fires when a specific event is dispatched.  Each rule consists
/// of a list of conditions (preconditions) and a list of effects (actions).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Rule {
    pub id: String,
    pub event: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Conditions that must all hold for the rule to apply.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<Condition>,
    /// Effects applied in order.
    pub effects: Vec<Effect>,
    /// Restrict to one or more ontology types.  Empty = applies to all types.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub applies_to: Vec<String>,
}

impl Rule {
    pub fn new(id: impl Into<String>, event: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            event: event.into(),
            description: None,
            conditions: Vec::new(),
            effects: Vec::new(),
            applies_to: Vec::new(),
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_condition(mut self, cond: Condition) -> Self {
        self.conditions.push(cond);
        self
    }

    pub fn with_effect(mut self, effect: Effect) -> Self {
        self.effects.push(effect);
        self
    }

    pub fn with_applies_to(mut self, type_id: impl Into<String>) -> Self {
        self.applies_to.push(type_id.into());
        self
    }

    pub fn applies_to_type(&self, type_id: &str) -> bool {
        self.applies_to.is_empty() || self.applies_to.iter().any(|t| t == type_id)
    }
}

/// A boolean condition evaluated at runtime.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Condition {
    /// The candidate type id matches one of the listed values.
    TypeIs { values: Vec<String> },
    /// The candidate's id is set (i.e. we're updating an existing instance).
    HasExistingId,
    /// The candidate has no existing id (we're inserting a new instance).
    NoExistingId,
    /// The candidate's content is non-empty after trimming.
    HasContent,
    /// The candidate has an embedding.
    HasEmbedding,
    /// The candidate's provenance method is at least `min_rank`.
    ProvenanceAtLeast { min_rank: u32 },
    /// The candidate's confidence is at least `min`.
    ConfidenceAtLeast { min: f32 },
    /// A similarity with the existing instance is at least `threshold`.
    SimilarityAtLeast { threshold: f32 },
    /// A custom expression supported by the runtime (e.g. "tag_count<=5").
    /// Evaluated by a registered expression handler.  Currently a no-op; the
    /// runtime treats unknown expressions as `true` so it can be added
    /// incrementally.
    Custom { expression: String },
    /// The action outcome equals one of the listed values
    /// (`started`, `succeeded`, `failed`, `cancelled`).
    ActionOutcomeIs { values: Vec<String> },
}

/// A single, ordered effect applied to a memory mutation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Effect {
    /// Set a property on the instance.
    SetField {
        property: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        value: Option<serde_json::Value>,
    },
    /// Add a literal tag to the instance.
    AddTag { tag: String },
    /// Remove a literal tag from the instance.
    RemoveTag { tag: String },
    /// Apply a lifecycle transition.
    TransitionLifecycle {
        to: MemoryStatus,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// Mark the instance as derived from another (typically used for
    /// cross-batch extraction edges).
    DeriveFrom {
        source_id: String,
        relation_kind: String,
    },
    /// Reinforce the existing instance (strength++ + access_count++).
    Reinforce { source_label: String },
    /// Supersede the existing instance with the new one.
    Supersede { old_id: String, new_id: String },
    /// Mark two instances as contradicting each other.
    MarkContradiction { other_id: String },
    /// Update the provenance record on the instance.
    SetProvenance {
        source: String,
        method: ExtractionMethod,
        confidence: f32,
    },
    /// Touch the instance (updated_at = now, access_count++).
    Touch,
    /// Set the artifact mirror's health.
    SetArtifactHealth { health: String },
    /// Reinforce the evidence memories behind an artifact by `amount`.
    ReinforceEvidence { amount: f32 },
    /// Invalidate the evidence memories behind an artifact by `amount`.
    InvalidateEvidence { amount: f32 },
    /// Link two instances with a relationship.
    Link {
        other_id: String,
        relationship: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        weight: Option<f32>,
    },
    /// Abort the operation (used to express "skip this rule chain").
    Abort { reason: String },
}

// ---------------------------------------------------------------------------
// Activities
// ---------------------------------------------------------------------------

/// A trigger that fires an activity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActivityTrigger {
    /// Fire when an event is dispatched.
    OnEvent { event: String },
    /// Fire after `interval_turns` turns (or roughly that many minutes,
    /// since the runtime maps turns to timeboxes).
    Periodic {
        interval_turns: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min_turns_between: Option<u32>,
    },
    /// Fire when a topic change is detected (cosine drop between consecutive
    /// context embeddings).
    OnTopicChange {
        #[serde(default = "default_topic_threshold")]
        threshold: f32,
    },
}

fn default_topic_threshold() -> f32 {
    0.3
}

/// A scheduled or event-driven piece of work that the runtime performs.
///
/// Activities are the *driver* of summarization, extraction and lifecycle
/// transitions.  The runtime does not decide when to extract or summarize on
/// its own — it consults the active ontology to find the matching activity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Activity {
    pub id: String,
    pub description: String,
    pub trigger: ActivityTrigger,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<ActivityStep>,
    /// Restrict to specific session states (e.g. "fresh_user_turn",
    /// "session_finalize").  Empty = no restriction.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub session_states: Vec<String>,
}

impl Activity {
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            trigger: ActivityTrigger::OnEvent {
                event: EVENT_TURN_TICK.to_string(),
            },
            steps: Vec::new(),
            session_states: Vec::new(),
        }
    }

    pub fn on_event(mut self, event: impl Into<String>) -> Self {
        self.trigger = ActivityTrigger::OnEvent {
            event: event.into(),
        };
        self
    }

    pub fn periodic(mut self, interval_turns: u32) -> Self {
        self.trigger = ActivityTrigger::Periodic {
            interval_turns,
            min_turns_between: None,
        };
        self
    }

    pub fn on_topic_change(mut self, threshold: f32) -> Self {
        self.trigger = ActivityTrigger::OnTopicChange { threshold };
        self
    }

    pub fn with_step(mut self, step: ActivityStep) -> Self {
        self.steps.push(step);
        self
    }

    pub fn with_session_state(mut self, state: impl Into<String>) -> Self {
        self.session_states.push(state.into());
        self
    }
}

/// A single ordered step in an activity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActivityStep {
    /// Run relevance scoring and emit a pending memory prompt.
    ComputeRelevance {
        #[serde(default = "default_max_candidates")]
        max_candidates: usize,
        #[serde(default = "default_max_results")]
        max_results: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
    },
    /// Run LLM-extraction over the current session context.
    ExtractMemories {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// Summarize a list of recent memories into a compact form.  The
    /// runtime is responsible for the actual algorithm; the ontology only
    /// declares that summarization should happen and how often.
    Summarize {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        target_type: Option<String>,
    },
    /// Garbage-collect archived terminal memories older than `age_days`.
    GcArchived {
        #[serde(default = "default_gc_age_days")]
        age_days: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
    },
    /// Run a periodic contradiction check via sidecar (when enabled).
    CheckContradictions {
        #[serde(default = "default_contradiction_threshold")]
        similarity_threshold: f32,
    },
    /// Reconcile an incoming instance against its store at write time.
    ///
    /// Pure lexical/`subject_key` comparison over the target graph: no
    /// embedder, no LLM. The thresholds mirror `rule.write_reconcile`.
    ReconcileOnWrite {
        #[serde(default = "default_reinforce_at")]
        reinforce_at: f32,
        #[serde(default = "default_supersede_at")]
        supersede_at: f32,
        #[serde(default = "default_relate_at")]
        relate_at: f32,
    },
    /// Aggregate evidence memories sharing a `subject_key` into a skill
    /// proposal.
    ///
    /// This is the memory -> skill learning path: `process_rule` and
    /// `correction` instances accumulate as evidence; once `min_evidence`
    /// instances spread across at least `min_stores` stores agree, a draft
    /// skill is proposed for review (never written silently).
    DistillSkillEvidence {
        #[serde(default = "default_min_evidence")]
        min_evidence: usize,
        #[serde(default = "default_min_stores")]
        min_stores: usize,
    },
    /// Emit an event so other rules can chain on it.
    Emit { event: String },
    /// Touch memories surfaced inside the activity (used to mark them as
    /// recently used so ranking rewards them).
    TouchSurfaced,
}

fn default_max_candidates() -> usize {
    30
}
fn default_max_results() -> usize {
    10
}
fn default_gc_age_days() -> u32 {
    90
}
fn default_contradiction_threshold() -> f32 {
    0.5
}
fn default_reinforce_at() -> f32 {
    0.85
}
fn default_supersede_at() -> f32 {
    0.5
}
fn default_relate_at() -> f32 {
    0.3
}
fn default_min_evidence() -> usize {
    3
}
fn default_min_stores() -> usize {
    2
}

// ---------------------------------------------------------------------------
// Built-in ontology
// ---------------------------------------------------------------------------

/// Rules that turn an action/skill outcome into knowledge feedback.
///
/// Exposed separately from [`default_ontology`] so a store carrying an older
/// ontology snapshot can still receive the rules at runtime.
pub fn skill_outcome_rules() -> Vec<Rule> {
    vec![
        Rule::new(RULE_SKILL_VALIDATED, EVENT_ACTION_OUTCOME)
            .with_description(
                "A successful action strengthens its evidence and keeps the artifact healthy.",
            )
            .with_condition(Condition::ActionOutcomeIs {
                values: vec!["succeeded".to_string()],
            })
            .with_effect(Effect::SetArtifactHealth {
                health: "ok".to_string(),
            })
            .with_effect(Effect::ReinforceEvidence { amount: 0.05 })
            .with_effect(Effect::AddTag {
                tag: "skill-validated".to_string(),
            }),
        Rule::new(RULE_SKILL_INVALIDATED, EVENT_ACTION_OUTCOME)
            .with_description(
                "A failed or cancelled action weakens its evidence and marks the artifact for review.",
            )
            .with_condition(Condition::ActionOutcomeIs {
                values: vec!["failed".to_string(), "cancelled".to_string()],
            })
            .with_effect(Effect::SetArtifactHealth {
                health: "needs_review".to_string(),
            })
            .with_effect(Effect::InvalidateEvidence { amount: 0.10 })
            .with_effect(Effect::AddTag {
                tag: "skill-invalidated".to_string(),
            }),
    ]
}

/// The default ontology installed at startup.  This captures the behavior
/// that the existing hardcoded `MemoryCategory`/`MemoryStatus`/`EdgeKind`
/// branches express so that the runtime can run off the ontology data
/// without changing user-visible behavior.
pub fn default_ontology() -> Ontology {
    let mut onto = Ontology::new(
        DEFAULT_ONTOLOGY_ID,
        DEFAULT_ONTOLOGY_VERSION,
        "Default memory ontology for jcode: environment facts, contracts, process rules, observations, preferences, entities and corrections. Goals and skills are external artifacts linked with `goal:` / `skill:` ids.",
    );

    // ---- Types -----------------------------------------------------------
    //
    // Authority ordering (v2): correction > process_rule > contract >
    // observation > env_fact. `env_fact` and `observation` carry hard TTLs so
    // the store converges instead of growing monotonically.
    let env_fact = OntologyType::new(TYPE_ENV_FACT, "Environment Fact")
        .with_legacy_category("env_fact")
        .with_extraction_method(ExtractionMethod::UserStated)
        .with_extraction_method(ExtractionMethod::StructuredMapping)
        .with_extraction_method(ExtractionMethod::RuleExtraction)
        .with_extraction_method(ExtractionMethod::LlmExtraction)
        .with_scoring(ScoringPolicy {
            base_score: 15.0,
            ..Default::default()
        })
        .with_decay(DecayPolicy {
            half_life_days: 7.0,
            access_boost: true,
        })
        .with_ttl_days(30.0);

    let contract = OntologyType::new(TYPE_CONTRACT, "Contract")
        .with_legacy_category("contract")
        .with_extraction_method(ExtractionMethod::UserStated)
        .with_extraction_method(ExtractionMethod::StructuredMapping)
        .with_extraction_method(ExtractionMethod::LlmExtraction)
        .with_scoring(ScoringPolicy {
            base_score: 40.0,
            ..Default::default()
        })
        .with_decay(DecayPolicy {
            half_life_days: 180.0,
            access_boost: true,
        });

    let process_rule = OntologyType::new(TYPE_PROCESS_RULE, "Process Rule")
        .with_legacy_category("process_rule")
        .with_extraction_method(ExtractionMethod::UserStated)
        .with_extraction_method(ExtractionMethod::LlmExtraction)
        .with_extraction_method(ExtractionMethod::RuleInference)
        .with_scoring(ScoringPolicy {
            base_score: 45.0,
            ..Default::default()
        })
        .with_decay(DecayPolicy {
            half_life_days: 365.0,
            access_boost: true,
        });

    let observation = OntologyType::new(TYPE_OBSERVATION, "Observation")
        .with_legacy_category("observation")
        .with_extraction_method(ExtractionMethod::UserStated)
        .with_extraction_method(ExtractionMethod::StructuredMapping)
        .with_extraction_method(ExtractionMethod::RuleExtraction)
        .with_extraction_method(ExtractionMethod::LlmExtraction)
        .with_extraction_method(ExtractionMethod::RuleInference)
        .with_scoring(ScoringPolicy {
            base_score: 10.0,
            ..Default::default()
        })
        .with_decay(DecayPolicy {
            half_life_days: 45.0,
            access_boost: true,
        })
        .with_ttl_days(90.0);

    let preference = OntologyType::new(TYPE_PREFERENCE, "Preference")
        .with_legacy_category("preference")
        .with_extraction_method(ExtractionMethod::UserStated)
        .with_extraction_method(ExtractionMethod::LlmExtraction)
        .with_scoring(ScoringPolicy {
            base_score: 30.0,
            ..Default::default()
        })
        .with_decay(DecayPolicy {
            half_life_days: 90.0,
            access_boost: true,
        });

    let entity = OntologyType::new(TYPE_ENTITY, "Entity")
        .with_legacy_category("entity")
        .with_extraction_method(ExtractionMethod::UserStated)
        .with_extraction_method(ExtractionMethod::StructuredMapping)
        .with_extraction_method(ExtractionMethod::LlmExtraction)
        .with_scoring(ScoringPolicy {
            base_score: 25.0,
            ..Default::default()
        })
        .with_decay(DecayPolicy {
            half_life_days: 180.0,
            access_boost: true,
        });

    let correction = OntologyType::new(TYPE_CORRECTION, "Correction")
        .with_legacy_category("correction")
        .with_extraction_method(ExtractionMethod::UserStated)
        .with_extraction_method(ExtractionMethod::LlmExtraction)
        .with_scoring(ScoringPolicy {
            base_score: 50.0,
            ..Default::default()
        })
        .with_decay(DecayPolicy {
            half_life_days: 365.0,
            access_boost: true,
        });

    onto = onto
        .with_type(env_fact)
        .with_type(contract)
        .with_type(process_rule)
        .with_type(observation)
        .with_type(preference)
        .with_type(entity)
        .with_type(correction);

    // Legacy `MemoryCategory` names -> v2 type ids. Old graphs and any
    // not-yet-migrated producer keep working: the overloaded `fact` bucket
    // lands in `observation` (the residual type), `Skills` lands in
    // `process_rule` (a skill is a distilled procedure), and `note` also lands
    // in `observation`. `goal:` / `skill:` ids are link nodes, not types.
    onto.category_aliases.insert("fact".to_string(), TYPE_OBSERVATION.to_string());
    onto.category_aliases
        .insert("note".to_string(), TYPE_OBSERVATION.to_string());
    onto.category_aliases
        .insert("Skills".to_string(), TYPE_PROCESS_RULE.to_string());
    onto.category_aliases
        .insert("skill".to_string(), TYPE_PROCESS_RULE.to_string());
    for type_id in [
        TYPE_ENV_FACT,
        TYPE_CONTRACT,
        TYPE_PROCESS_RULE,
        TYPE_OBSERVATION,
        TYPE_PREFERENCE,
        TYPE_ENTITY,
        TYPE_CORRECTION,
    ] {
        onto.category_aliases
            .insert(type_id.to_string(), type_id.to_string());
    }

    // ---- Relationships ---------------------------------------------------
    let has_tag = RelationshipDef {
        id: REL_HAS_TAG.to_string(),
        display_name: "Has Tag".to_string(),
        source_kinds: vec!["memory".to_string()],
        target_kinds: vec!["tag".to_string()],
        weight: None,
        traversal_weight: 0.8,
    };
    let in_cluster = RelationshipDef {
        id: REL_IN_CLUSTER.to_string(),
        display_name: "In Cluster (deprecated, no producer)".to_string(),
        source_kinds: vec!["memory".to_string()],
        target_kinds: vec!["cluster".to_string()],
        weight: None,
        traversal_weight: 0.6,
    };
    let relates_to = RelationshipDef {
        id: REL_RELATES_TO.to_string(),
        display_name: "Relates To".to_string(),
        source_kinds: vec!["memory".to_string()],
        target_kinds: vec!["memory".to_string()],
        weight: Some(WeightRange { min: 0.0, max: 1.0 }),
        traversal_weight: 1.0,
    };
    let supersedes = RelationshipDef {
        id: REL_SUPERSEDES.to_string(),
        display_name: "Supersedes".to_string(),
        source_kinds: vec!["memory".to_string()],
        target_kinds: vec!["memory".to_string()],
        weight: None,
        traversal_weight: 0.9,
    };
    let contradicts = RelationshipDef {
        id: REL_CONTRADICTS.to_string(),
        display_name: "Contradicts".to_string(),
        source_kinds: vec!["memory".to_string()],
        target_kinds: vec!["memory".to_string()],
        weight: None,
        traversal_weight: 0.3,
    };
    let derived_from = RelationshipDef {
        id: REL_DERIVED_FROM.to_string(),
        display_name: "Derived From".to_string(),
        source_kinds: vec!["memory".to_string()],
        target_kinds: vec!["memory".to_string()],
        weight: None,
        traversal_weight: 0.7,
    };

    onto = onto
        .with_relationship(has_tag)
        .with_relationship(in_cluster)
        .with_relationship(relates_to)
        .with_relationship(supersedes)
        .with_relationship(contradicts)
        .with_relationship(derived_from);

    // ---- Rules -----------------------------------------------------------
    let remember_rule = Rule::new(RULE_REMEMBER, EVENT_REMEMBER)
        .with_description("Validate + write a new memory instance.")
        .with_condition(Condition::HasContent)
        .with_condition(Condition::ConfidenceAtLeast { min: 0.0 })
        .with_effect(Effect::SetProvenance {
            source: "agent".to_string(),
            method: ExtractionMethod::UserStated,
            confidence: 1.0,
        })
        .with_effect(Effect::Touch);

    // Global-scope write gate. Global memory is injected into every project's
    // sessions, so it must hold conceptual, generalized, cross-project
    // knowledge only. The gate aborts writes whose content looks like
    // project-specific environment detail (concrete hosts, ports, task ids,
    // repositories) unless it was produced by analysis (extraction /
    // inference) rather than stated during a conversation. See
    // `rule_engine::is_generalized_content`.
    let remember_global_gate_rule = Rule::new(RULE_REMEMBER_GLOBAL_GATE, EVENT_REMEMBER_GLOBAL)
        .with_description(
            "Admit generalized/cross-project content to global scope; abort environment-specific content.",
        )
        .with_condition(Condition::Custom {
            expression: "global_scope_generalized".to_string(),
        })
        .with_effect(Effect::Touch);

    let upsert_rule = Rule::new(RULE_UPSERT, EVENT_UPSERT)
        .with_description("Replace content/tags for an existing instance.")
        .with_condition(Condition::HasExistingId)
        .with_effect(Effect::SetField {
            property: "content".to_string(),
            value: None,
        })
        .with_effect(Effect::Touch);

    let dedup_rule = Rule::new(RULE_DEDUP_REINFORCE, EVENT_DEDUP)
        .with_description("Reinforce an existing instance when dedup hits.")
        .with_condition(Condition::SimilarityAtLeast { threshold: 0.85 })
        .with_effect(Effect::Reinforce {
            source_label: "dedup".to_string(),
        })
        .with_effect(Effect::Touch);

    // Write-time reconciliation. Runs synchronously on every admit decision and
    // is deliberately local: lexical/`subject_key` similarity over the target
    // store, no embedder and no LLM. This is what makes `supersedes` /
    // `contradicts` actually exist - v1 declared both relations but produced
    // zero edges in 4.2k stored instances while 1.6k similarity>=0.5 pairs sat
    // unlinked.
    //
    // The runtime implementation lives in `crate::reconcile`; the rule below is
    // the declarative contract it must satisfy.
    let write_reconcile_rule = Rule::new(RULE_WRITE_RECONCILE, EVENT_REMEMBER)
        .with_description(
            "Classify an incoming instance against its store: reinforce (>=0.85), supersede (>=0.5), contradict/relate (>=0.3) or insert.",
        )
        .with_condition(Condition::HasContent)
        .with_effect(Effect::SetField {
            property: "reconcile".to_string(),
            value: Some(serde_json::json!({
                "reinforce_at": 0.85,
                "supersede_at": 0.5,
                "relate_at": 0.3,
            })),
        });

    // Noise gate. Tool payloads, bare identifiers and stubs are rejected before
    // they reach the graph (v2 replaces the ad-hoc length check that let raw
    // `#Tool: bash input: {...}` rows land in the store).
    let reject_rule = Rule::new(RULE_REJECT, EVENT_REMEMBER)
        .with_description("Reject tool payloads, bare identifiers and sub-threshold stubs.")
        .with_condition(Condition::Custom {
            expression: "content_is_noise".to_string(),
        })
        .with_effect(Effect::Abort {
            reason: "content looks like extraction noise".to_string(),
        });

    let supersede_rule = Rule::new(RULE_SUPERSEDE, EVENT_SUPERSEDE)
        .with_description("Mark the older instance as superseded.")
        .with_condition(Condition::SimilarityAtLeast { threshold: 0.5 })
        .with_effect(Effect::MarkContradiction {
            other_id: "__self__".to_string(),
        })
        .with_effect(Effect::Supersede {
            old_id: "__self__".to_string(),
            new_id: "__self__".to_string(),
        });

    let contradict_rule = Rule::new(RULE_CONTRADICT, EVENT_CONTRADICT)
        .with_description("Mark two instances as contradicting each other.")
        .with_effect(Effect::MarkContradiction {
            other_id: "__self__".to_string(),
        });

    let tag_rule = Rule::new(RULE_TAG, EVENT_TAG)
        .with_description("Add a tag to an instance.")
        .with_effect(Effect::AddTag {
            tag: "__self__".to_string(),
        });

    let untag_rule = Rule::new(RULE_UNTAG, EVENT_UNTAG)
        .with_description("Remove a tag from an instance.")
        .with_effect(Effect::RemoveTag {
            tag: "__self__".to_string(),
        });

    let link_rule = Rule::new(RULE_LINK, EVENT_LINK)
        .with_description("Link two instances with a RelatesTo edge.")
        .with_effect(Effect::Link {
            other_id: "__self__".to_string(),
            relationship: REL_RELATES_TO.to_string(),
            weight: None,
        });

    onto = onto
        .with_rule(remember_rule)
        .with_rule(remember_global_gate_rule)
        .with_rule(upsert_rule)
        .with_rule(write_reconcile_rule)
        .with_rule(reject_rule)
        .with_rule(dedup_rule)
        .with_rule(supersede_rule)
        .with_rule(contradict_rule)
        .with_rule(tag_rule)
        .with_rule(untag_rule)
        .with_rule(link_rule);

    // Action feedback rules: the outcome of executing a skill validates or
    // invalidates the knowledge that justified it. This is the "action
    // verifies cognition" half of the unified model. Kept in a reusable
    // function so older stores can receive the rules without a version bump.
    for rule in skill_outcome_rules() {
        onto = onto.with_rule(rule);
    }

    // ---- Activities ------------------------------------------------------
    //
    // v2 keeps only activities the runtime actually schedules. v1 declared
    // seven; five of them (per-turn relevance, periodic extract, topic-change
    // extract, summarize, contradiction sweep) had no `ActivityStep`
    // implementation in the runtime at all, so they were documentation.
    let write_reconcile = Activity::new(
        ACTIVITY_WRITE_RECONCILE,
        "Reconcile an incoming instance against its store (reinforce / supersede / contradict / relate / insert).",
    )
    .on_event(EVENT_REMEMBER)
    .with_step(ActivityStep::ReconcileOnWrite {
        reinforce_at: 0.85,
        supersede_at: 0.5,
        relate_at: 0.3,
    });

    let final_extract = Activity::new(
        ACTIVITY_FINAL_EXTRACT,
        "Run LLM extraction at session finalization.",
    )
    .on_event(EVENT_FINALIZE)
    .with_step(ActivityStep::ExtractMemories {
        scope: Some("project".to_string()),
        reason: Some("finalize".to_string()),
    });

    let periodic_gc = Activity::new(
        ACTIVITY_PERIODIC_GC,
        "Archive instances whose type TTL elapsed without access.",
    )
    .on_event(EVENT_FINALIZE)
    .with_step(ActivityStep::GcArchived {
        age_days: 30,
        scope: Some("all".to_string()),
    });

    let skill_distill = Activity::new(
        ACTIVITY_SKILL_DISTILL,
        "Aggregate evidence memories into skill proposals (memory -> skill learning).",
    )
    .on_event(EVENT_FINALIZE)
    .with_step(ActivityStep::DistillSkillEvidence {
        min_evidence: 3,
        min_stores: 2,
    });

    onto = onto
        .with_activity(write_reconcile)
        .with_activity(final_extract)
        .with_activity(periodic_gc)
        .with_activity(skill_distill);

    onto
}

/// Lightweight list of all event names that the runtime currently emits.
/// Useful for tests, debug dumps, and tooling that introspects the ontology.
pub fn default_event_names() -> Vec<&'static str> {
    vec![
        EVENT_REMEMBER,
        EVENT_UPSERT,
        EVENT_DEDUP,
        EVENT_SUPERSEDE,
        EVENT_CONTRADICT,
        EVENT_TAG,
        EVENT_UNTAG,
        EVENT_LINK,
        EVENT_TOPIC_CHANGE,
        EVENT_TURN_TICK,
        EVENT_FINALIZE,
        EVENT_ACTION_OUTCOME,
    ]
}

/// Sanity check: the default ontology is internally consistent.
pub fn validate_default_ontology() -> Result<(), Vec<String>> {
    let mut errs: Vec<String> = Vec::new();
    let onto = default_ontology();

    let mut seen_events: HashSet<&str> = HashSet::new();
    for rule in &onto.rules {
        if rule.id.is_empty() {
            errs.push("rule with empty id".to_string());
        }
        if !seen_events.insert(rule.event.as_str()) {
            // Duplicate events are fine, just informational.
        }
        for effect in &rule.effects {
            if let Effect::TransitionLifecycle { to, .. } = effect {
                if let MemoryStatus::Active = to {
                    errs.push(format!(
                        "rule '{}' transitions to Active; ontology should not require explicit Active transitions",
                        rule.id
                    ));
                }
            }
        }
    }

    for activity in &onto.activities {
        if activity.id.is_empty() {
            errs.push("activity with empty id".to_string());
        }
    }

    for type_def in onto.types.values() {
        if type_def.lifecycle.initial_state != MemoryStatus::Active
            && !type_def
                .lifecycle
                .states
                .contains(&type_def.lifecycle.initial_state)
        {
            errs.push(format!(
                "type '{}' initial state {:?} is not in declared states",
                type_def.id, type_def.lifecycle.initial_state
            ));
        }
    }

    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ontology_loads() {
        let onto = default_ontology();
        assert_eq!(onto.id, DEFAULT_ONTOLOGY_ID);
        assert_eq!(onto.version, DEFAULT_ONTOLOGY_VERSION);
        assert_eq!(onto.version, 2);
        for type_id in [
            TYPE_ENV_FACT,
            TYPE_CONTRACT,
            TYPE_PROCESS_RULE,
            TYPE_OBSERVATION,
            TYPE_PREFERENCE,
            TYPE_ENTITY,
            TYPE_CORRECTION,
        ] {
            assert!(onto.types.contains_key(type_id), "missing type {type_id}");
        }
        assert!(onto.relationships.contains_key(REL_HAS_TAG));
        assert!(onto.relationships.contains_key(REL_RELATES_TO));
        assert!(onto.relationships.contains_key(REL_SUPERSEDES));
        assert!(onto.rules.iter().any(|r| r.id == RULE_WRITE_RECONCILE));
        assert!(onto.rules.iter().any(|r| r.id == RULE_REJECT));
        assert!(!onto.rules.is_empty());
        assert!(!onto.activities.is_empty());
        // v2 rule: every declared type must carry a TTL or an explicit
        // retention justification, and the TTL-bearing types are the two
        // perishable ones.
        assert_eq!(onto.types[TYPE_ENV_FACT].ttl_days, Some(30.0));
        assert_eq!(onto.types[TYPE_OBSERVATION].ttl_days, Some(90.0));
        assert_eq!(onto.types[TYPE_CONTRACT].ttl_days, None);
        assert_eq!(onto.types[TYPE_PROCESS_RULE].ttl_days, None);
        // Types with no producer/consumer were removed in v2.
        for removed in ["note", "skill", "goal", "fact"] {
            assert!(!onto.types.contains_key(removed), "{removed} must not be a type");
        }
    }

    #[test]
    fn v2_activities_are_the_scheduled_ones() {
        let onto = default_ontology();
        let ids: Vec<&str> = onto.activities.iter().map(|a| a.id.as_str()).collect();
        assert!(ids.contains(&ACTIVITY_WRITE_RECONCILE));
        assert!(ids.contains(&ACTIVITY_FINAL_EXTRACT));
        assert!(ids.contains(&ACTIVITY_PERIODIC_GC));
        assert!(ids.contains(&ACTIVITY_SKILL_DISTILL));
        assert_eq!(ids.len(), 4, "v2 schedules exactly four activities");
    }

    #[test]
    fn link_ids_are_recognized() {
        assert!(is_link_id("goal:kia203"));
        assert!(is_link_id("skill:optimization"));
        assert!(!is_link_id("mem_1790_1234"));
        assert_eq!(LINK_PREFIXES.len(), 2);
    }

    #[test]
    fn validate_default_ontology_passes() {
        validate_default_ontology().expect("default ontology must be valid");
    }

    #[test]
    fn rules_for_event_lookup() {
        let onto = default_ontology();
        let remember: Vec<_> = onto.rules_for_event(EVENT_REMEMBER).collect();
        assert!(remember.iter().any(|r| r.id == RULE_REMEMBER));
    }

    #[test]
    fn activities_for_event_lookup() {
        let onto = default_ontology();
        let final_extract: Vec<_> = onto.activities_for_event(EVENT_FINALIZE).collect();
        assert!(final_extract.iter().any(|a| a.id == ACTIVITY_FINAL_EXTRACT));
    }

    #[test]
    fn category_aliases_for_legacy_categories() {
        let onto = default_ontology();
        // The overloaded v1 `fact` bucket migrates to the residual type,
        // and `Skills` maps to the durable procedure type.
        assert_eq!(onto.type_from_category(&MemoryCategory::Fact), Some(TYPE_OBSERVATION));
        assert_eq!(
            onto.type_from_category(&MemoryCategory::Custom("Skills".to_string())),
            Some(TYPE_PROCESS_RULE)
        );
        assert_eq!(
            onto.type_from_category(&MemoryCategory::Custom("env_fact".to_string())),
            Some(TYPE_ENV_FACT)
        );
        assert_eq!(onto.type_from_category(&MemoryCategory::Correction), Some(TYPE_CORRECTION));
    }

    #[test]
    fn lifecycle_policy_transitions() {
        let policy = LifecyclePolicy::default();
        assert!(policy.can_transition(MemoryStatus::Active, MemoryStatus::Archived));
        assert!(!policy.can_transition(MemoryStatus::Archived, MemoryStatus::Expired));
    }

    #[test]
    fn rule_applies_to_type() {
        let r = Rule::new("test", "event").with_applies_to("fact");
        assert!(r.applies_to_type("fact"));
        assert!(!r.applies_to_type("preference"));

        let r_all = Rule::new("test", "event");
        assert!(r_all.applies_to_type("fact"));
        assert!(r_all.applies_to_type("preference"));
    }
}
