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
pub const DEFAULT_ONTOLOGY_VERSION: u32 = 1;
pub const DEFAULT_ONTOLOGY_ID: &str = "jcode/default/v1";

pub const TYPE_FACT: &str = "fact";
pub const TYPE_PREFERENCE: &str = "preference";
pub const TYPE_ENTITY: &str = "entity";
pub const TYPE_CORRECTION: &str = "correction";
pub const TYPE_NOTE: &str = "note";
pub const TYPE_SKILL: &str = "skill";
pub const TYPE_GOAL: &str = "goal";

pub const REL_HAS_TAG: &str = "has_tag";
pub const REL_RELATES_TO: &str = "relates_to";
pub const REL_SUPERSEDES: &str = "supersedes";
pub const REL_CONTRADICTS: &str = "contradicts";
pub const REL_DERIVED_FROM: &str = "derived_from";
pub const REL_IN_CLUSTER: &str = "in_cluster";

pub const RULE_REMEMBER: &str = "rule.remember";
pub const RULE_UPSERT: &str = "rule.upsert";
pub const RULE_DEDUP_REINFORCE: &str = "rule.dedup_reinforce";
pub const RULE_SUPERSEDE: &str = "rule.supersede";
pub const RULE_CONTRADICT: &str = "rule.contradict";
pub const RULE_TAG: &str = "rule.tag";
pub const RULE_UNTAG: &str = "rule.untag";
pub const RULE_LINK: &str = "rule.link";

pub const ACTIVITY_PER_TURN_RELEVANCE: &str = "activity.per_turn_relevance";
pub const ACTIVITY_PERIODIC_EXTRACT: &str = "activity.periodic_extract";
pub const ACTIVITY_TOPIC_CHANGE_EXTRACT: &str = "activity.topic_change_extract";
pub const ACTIVITY_FINAL_EXTRACT: &str = "activity.final_extract";
pub const ACTIVITY_GC_ARCHIVED: &str = "activity.gc_archived";
pub const ACTIVITY_SUMMARIZE: &str = "activity.summarize";

pub const EVENT_REMEMBER: &str = "event.remember";
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
        self.activities
            .iter()
            .filter(move |a| match &a.trigger {
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
            extraction_methods: vec![ExtractionMethod::UserStated],
            persistent: true,
            embeddable: true,
            present_in_prompt: true,
        }
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
        self.transitions.iter().any(|t| t.from == from && t.to == to)
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
    DeriveFrom { source_id: String, relation_kind: String },
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

// ---------------------------------------------------------------------------
// Built-in ontology
// ---------------------------------------------------------------------------

/// The default ontology installed at startup.  This captures the behavior
/// that the existing hardcoded `MemoryCategory`/`MemoryStatus`/`EdgeKind`
/// branches express so that the runtime can run off the ontology data
/// without changing user-visible behavior.
pub fn default_ontology() -> Ontology {
    let mut onto = Ontology::new(
        DEFAULT_ONTOLOGY_ID,
        DEFAULT_ONTOLOGY_VERSION,
        "Default memory ontology for jcode: facts, preferences, entities, corrections, notes, skills, and goals.",
    );

    // ---- Types -----------------------------------------------------------
    let fact = OntologyType::new(TYPE_FACT, "Fact")
        .with_legacy_category("fact")
        .with_extraction_method(ExtractionMethod::UserStated)
        .with_extraction_method(ExtractionMethod::StructuredMapping)
        .with_extraction_method(ExtractionMethod::RuleExtraction)
        .with_extraction_method(ExtractionMethod::LlmExtraction)
        .with_extraction_method(ExtractionMethod::RuleInference)
        .with_scoring(ScoringPolicy {
            base_score: 20.0,
            ..Default::default()
        })
        .with_decay(DecayPolicy {
            half_life_days: 30.0,
            access_boost: true,
        });

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
            base_score: 10.0,
            ..Default::default()
        })
        .with_decay(DecayPolicy {
            half_life_days: 60.0,
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

    let note = OntologyType::new(TYPE_NOTE, "Note")
        .with_legacy_category("note")
        .with_extraction_method(ExtractionMethod::UserStated)
        .with_decay(DecayPolicy {
            half_life_days: 45.0,
            access_boost: true,
        });

    let skill = OntologyType::new(TYPE_SKILL, "Skill")
        .with_legacy_category("Skills")
        .with_extraction_method(ExtractionMethod::StructuredMapping)
        .with_persistent(false)
        .with_embeddable(true)
        .with_present_in_prompt(false)
        .with_decay(DecayPolicy {
            half_life_days: 365.0,
            access_boost: false,
        });

    let goal = OntologyType::new(TYPE_GOAL, "Goal")
        .with_legacy_category("goal")
        .with_extraction_method(ExtractionMethod::UserStated)
        .with_extraction_method(ExtractionMethod::StructuredMapping)
        .with_embeddable(false)
        .with_present_in_prompt(false)
        .with_decay(DecayPolicy {
            half_life_days: 30.0,
            access_boost: true,
        });

    onto = onto
        .with_type(fact)
        .with_type(preference)
        .with_type(entity)
        .with_type(correction)
        .with_type(note)
        .with_type(skill)
        .with_type(goal);

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
        display_name: "In Cluster".to_string(),
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
        .with_rule(upsert_rule)
        .with_rule(dedup_rule)
        .with_rule(supersede_rule)
        .with_rule(contradict_rule)
        .with_rule(tag_rule)
        .with_rule(untag_rule)
        .with_rule(link_rule);

    // ---- Activities ------------------------------------------------------
    let per_turn_relevance = Activity::new(
        ACTIVITY_PER_TURN_RELEVANCE,
        "Compute relevance of stored memories against the current user turn.",
    )
    .on_event(EVENT_TURN_TICK)
    .with_session_state("fresh_user_turn")
    .with_step(ActivityStep::ComputeRelevance {
        max_candidates: 30,
        max_results: 10,
        scope: Some("all".to_string()),
    });

    let periodic_extract = Activity::new(
        ACTIVITY_PERIODIC_EXTRACT,
        "Run LLM extraction every N turns to capture incremental memories.",
    )
    .periodic(12)
    .with_step(ActivityStep::ExtractMemories {
        scope: Some("project".to_string()),
        reason: Some("periodic".to_string()),
    });

    let topic_change_extract = Activity::new(
        ACTIVITY_TOPIC_CHANGE_EXTRACT,
        "Run LLM extraction when the topic changes mid-session.",
    )
    .on_topic_change(0.3)
    .with_step(ActivityStep::ExtractMemories {
        scope: Some("project".to_string()),
        reason: Some("topic_change".to_string()),
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

    let gc_archived = Activity::new(
        ACTIVITY_GC_ARCHIVED,
        "Garbage-collect archived memories that have aged out.",
    )
    .periodic(50)
    .with_step(ActivityStep::GcArchived {
        age_days: 90,
        scope: Some("all".to_string()),
    });

    let summarize = Activity::new(
        ACTIVITY_SUMMARIZE,
        "Summarize active memories to keep prompt payloads compact.",
    )
    .periodic(20)
    .with_step(ActivityStep::Summarize {
        scope: Some("all".to_string()),
        target_type: None,
    });

    let check_contradictions = Activity::new(
        "activity.check_contradictions",
        "Re-check surface for contradictions via sidecar.",
    )
    .periodic(15)
    .with_step(ActivityStep::CheckContradictions {
        similarity_threshold: 0.5,
    });

    onto = onto
        .with_activity(per_turn_relevance)
        .with_activity(periodic_extract)
        .with_activity(topic_change_extract)
        .with_activity(final_extract)
        .with_activity(gc_archived)
        .with_activity(summarize)
        .with_activity(check_contradictions);

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

    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ontology_loads() {
        let onto = default_ontology();
        assert_eq!(onto.id, DEFAULT_ONTOLOGY_ID);
        assert!(onto.types.contains_key(TYPE_FACT));
        assert!(onto.types.contains_key(TYPE_PREFERENCE));
        assert!(onto.types.contains_key(TYPE_GOAL));
        assert!(onto.types.contains_key(TYPE_SKILL));
        assert!(onto.relationships.contains_key(REL_HAS_TAG));
        assert!(onto.relationships.contains_key(REL_RELATES_TO));
        assert!(onto.relationships.contains_key(REL_SUPERSEDES));
        assert!(!onto.rules.is_empty());
        assert!(!onto.activities.is_empty());
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
        let fact = MemoryCategory::Fact;
        assert_eq!(onto.type_from_category(&fact), Some(TYPE_FACT));
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
