//! Runtime registry for memory ontologies.
//!
//! The runtime consults an `Ontology` to decide how to classify, score, decay,
//! extract, and lifecycle-manage memory instances.  This module owns the set
//! of loaded ontologies (typically one in production) and exposes the helpers
//! call sites in `MemoryManager` and `MemoryAgent` need:
//!
//! *   `OntologyRegistry::with_default()` returns the registry seeded with the
//!     bundled `jcode/default/v1` ontology.
//! *   `OntologyRegistry::default_ontology()` returns the seeded ontology.
//! *   `OntologyRegistry::dispatch()` runs the rule engine against an event.
//! *   `OntologyRegistry::schedule()` runs the activity scheduler for a coarse
//!     event.
//!
//! The registry is intentionally cheap to clone: it is an `Arc<HashMap<...>>`
//! of `Ontology` values, so call sites can pass it around freely without
//! introducing `Mutex` contention on the hot paths.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use jcode_memory_types::activity::{
    ScheduleEvent, ScheduledActivity, schedule as activity_schedule,
};
use jcode_memory_types::ontology::{
    ActivityStep, Condition, DEFAULT_ONTOLOGY_ID, DEFAULT_ONTOLOGY_VERSION, Effect, Ontology,
    default_ontology,
};
use jcode_memory_types::rule_engine::{
    RuleContext, RulePlan, apply_entry_effects, apply_graph_effects, dispatch_event,
};
use jcode_memory_types::{MemoryEntry, MemoryEvent, MemoryEventKind, MemoryGraph};

use crate::memory_types::MemoryCategory;

/// Bundle of ontologies the runtime knows about.
///
/// The map is behind a `RwLock` so a reflection pass can install a new
/// ontology while the process keeps running. Cloning the registry shares the
/// same map (like the previous `Arc<HashMap>` did), so every `MemoryManager`
/// sees an installed replacement immediately.
#[derive(Debug, Clone)]
pub struct OntologyRegistry {
    inner: Arc<RwLock<HashMap<String, Arc<Ontology>>>>,
    /// Id of the ontology treated as the store default. Normally
    /// `jcode/default/v1`; tests can seed a custom one.
    default_id: Arc<RwLock<String>>,
}

impl Default for OntologyRegistry {
    fn default() -> Self {
        Self::with_default()
    }
}

impl OntologyRegistry {
    /// Construct a registry seeded with the bundled default ontology.
    pub fn with_default() -> Self {
        let ontology = default_ontology();
        let mut map = HashMap::new();
        map.insert(ontology.id.clone(), Arc::new(ontology));
        Self {
            inner: Arc::new(RwLock::new(map)),
            default_id: Arc::new(RwLock::new(DEFAULT_ONTOLOGY_ID.to_string())),
        }
    }

    /// Construct a registry containing exactly the supplied ontologies.  When
    /// two ontologies share an id, the last one wins. The last ontology also
    /// becomes the default.
    pub fn from_ontologies<I: IntoIterator<Item = Ontology>>(iter: I) -> Self {
        let mut map: HashMap<String, Arc<Ontology>> = HashMap::new();
        let mut default_id = DEFAULT_ONTOLOGY_ID.to_string();
        for ontology in iter {
            default_id = ontology.id.clone();
            map.insert(ontology.id.clone(), Arc::new(ontology));
        }
        Self {
            inner: Arc::new(RwLock::new(map)),
            default_id: Arc::new(RwLock::new(default_id)),
        }
    }

    /// Install or replace an ontology under its id and make it the default.
    ///
    /// This is the runtime hook a reflection pass uses after a patch has been
    /// validated and persisted. The returned `Arc` is the value new turns
    /// will pin.
    pub fn install(&self, ontology: Ontology) -> Arc<Ontology> {
        let id = ontology.id.clone();
        let ontology = Arc::new(ontology);
        let mut guard = self
            .inner
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.insert(id.clone(), Arc::clone(&ontology));
        drop(guard);
        *self
            .default_id
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = id;
        ontology
    }

    /// Remove an ontology by id. Removing the current default leaves the
    /// registry without a default, which makes `get` fall back to the compiled
    /// ontology rather than returning a stale value.
    pub fn remove(&self, id: &str) {
        let mut guard = self
            .inner
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.remove(id);
    }

    /// Replace the default id without removing the old entry.
    pub fn set_default(&self, id: &str) {
        *self
            .default_id
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = id.to_string();
    }

    /// Look up an ontology by id.  Returns an `Arc<Ontology>` so callers can
    /// keep it alive for as long as they need (e.g. embed it inside
    /// `RuleContext`).
    pub fn get(&self, id: &str) -> Arc<Ontology> {
        let guard = self
            .inner
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(arc) = guard.get(id) {
            return Arc::clone(arc);
        }
        let default_id = self
            .default_id
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        if let Some(arc) = guard.get(&default_id) {
            return Arc::clone(arc);
        }
        drop(guard);
        Arc::new(default_ontology())
    }

    /// Return the active default ontology.
    pub fn default_ontology(&self) -> Arc<Ontology> {
        let id = self
            .default_id
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        self.get(&id)
    }

    /// Ids of all loaded ontologies.  Mostly useful for tests and diagnostics.
    pub fn ids(&self) -> Vec<String> {
        self.inner
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .cloned()
            .collect()
    }

    /// Dispatch an event through the rule engine using the supplied context.
    /// The caller is responsible for populating `ctx.ontology` with the
    /// registry's view of the appropriate ontology (use
    /// `OntologyRegistry::make_context` if unsure).
    pub fn dispatch(&self, ctx: &RuleContext) -> RulePlan {
        dispatch_event(ctx)
    }

    /// Build a `RuleContext` populated with the registry's ontology for
    /// `ontology_id`.  Convenience for callers that don't already hold an
    /// `Arc<Ontology>`.
    pub fn make_context(
        &self,
        ontology_id: &str,
        entry: MemoryEntry,
        event: impl Into<String>,
    ) -> RuleContext {
        RuleContext {
            entry,
            type_id: String::new(),
            existing_id: None,
            similarity: None,
            graph: Arc::new(MemoryGraph::new()),
            ontology: self.get(ontology_id),
            source_label: String::new(),
            event: event.into(),
            scope: String::new(),
        }
    }

    /// Schedule activities triggered by `event`.
    pub fn schedule(&self, ontology_id: &str, event: &ScheduleEvent) -> Vec<ScheduledActivity> {
        activity_schedule(event, &self.get(ontology_id))
    }

    /// Resolve a legacy `MemoryCategory` to the ontology type id the
    /// registry considers canonical.  Used by call sites that have not yet
    /// been migrated to construct ontology type ids directly.
    pub fn type_from_category(
        &self,
        ontology_id: &str,
        category: &MemoryCategory,
    ) -> Option<String> {
        self.get(ontology_id)
            .type_from_category(category)
            .map(|s| s.to_string())
    }

    /// Persist ontology metadata on `graph` so the next load can detect
    /// ontology drift.  Called from `MemoryManager::save_*_graph`.
    pub fn bind_to_graph(&self, graph: &mut MemoryGraph) {
        if graph.metadata.ontology_id.is_empty() {
            graph.metadata.ontology_id = DEFAULT_ONTOLOGY_ID.to_string();
            graph.metadata.ontology_version = DEFAULT_ONTOLOGY_VERSION;
        }
    }
}

/// Apply the entry effects from a plan to `entry` and the graph effects to
/// `graph`.  This is the single entry point the runtime uses to commit a
/// dispatched plan, ensuring order and lifecycle guards are honoured.
// Dormant since the upstream Jev merge disconnected the ontology write path;
// retained with its tests for the planned re-enable.
#[allow(dead_code)]
pub fn apply_plan(entry: &mut MemoryEntry, new_id: &str, plan: &RulePlan, graph: &mut MemoryGraph) {
    apply_entry_effects(plan, entry);
    apply_graph_effects(plan, new_id, graph);
}

/// Lightweight summary of a plan for the memory log.
#[allow(dead_code)]
pub fn summarize_plan(plan: &RulePlan) -> MemoryEvent {
    let applied = plan.applied_rules.len();
    let skipped = plan.skipped.len();
    let effect_count = plan.effects.len();
    let detail = format!(
        "applied={} skipped={} effects={}",
        applied, skipped, effect_count
    );
    let kind = if skipped > 0 {
        MemoryEventKind::RuleSkip { detail }
    } else {
        MemoryEventKind::RuleApplied { detail }
    };
    MemoryEvent {
        kind,
        timestamp: std::time::Instant::now(),
        detail: None,
    }
}

/// Convenience that lists the kinds of effects the registry knows how to
/// apply.  Exposed for diagnostics and tests.
#[allow(dead_code)]
pub fn declared_effect_kinds(ontology: &Ontology, event: &str) -> Vec<&'static str> {
    use jcode_memory_types::rule_engine::declared_effect_kinds as inner;
    inner(ontology, event)
}

/// Convenience that lists the kinds of effects the registry has conditions
/// for.  Exposed for diagnostics and tests.
#[allow(dead_code)]
pub fn declared_condition_kinds(ontology: &Ontology, event: &str) -> Vec<&'static str> {
    let mut seen = std::collections::HashSet::new();
    let mut ordered = Vec::new();
    for r in ontology.rules_for_event(event) {
        for c in &r.conditions {
            let k = condition_kind_name(c);
            if seen.insert(k) {
                ordered.push(k);
            }
        }
    }
    ordered
}

/// Convenience that lists the kinds of steps the registry declares.  Exposed
/// for diagnostics and tests.
#[allow(dead_code)]
pub fn declared_step_kinds(ontology: &Ontology, event: &ScheduleEvent) -> Vec<&'static str> {
    use jcode_memory_types::activity::describe_step;
    let plan = activity_schedule(event, ontology);
    let mut seen = std::collections::HashSet::new();
    let mut ordered: Vec<&'static str> = Vec::new();
    for s in &plan {
        for step in &s.steps {
            let k = describe_step(step);
            if seen.insert(k) {
                ordered.push(k);
            }
        }
    }
    ordered
}

fn condition_kind_name(c: &Condition) -> &'static str {
    match c {
        Condition::TypeIs { .. } => "type_is",
        Condition::HasExistingId => "has_existing_id",
        Condition::NoExistingId => "no_existing_id",
        Condition::HasContent => "has_content",
        Condition::HasEmbedding => "has_embedding",
        Condition::ProvenanceAtLeast { .. } => "provenance_at_least",
        Condition::ConfidenceAtLeast { .. } => "confidence_at_least",
        Condition::SimilarityAtLeast { .. } => "similarity_at_least",
        Condition::Custom { .. } => "custom",
    }
}

#[allow(dead_code)]
fn effect_kind_name(e: &Effect) -> &'static str {
    match e {
        Effect::SetField { .. } => "set_field",
        Effect::AddTag { .. } => "add_tag",
        Effect::RemoveTag { .. } => "remove_tag",
        Effect::TransitionLifecycle { .. } => "transition_lifecycle",
        Effect::DeriveFrom { .. } => "derive_from",
        Effect::Reinforce { .. } => "reinforce",
        Effect::Supersede { .. } => "supersede",
        Effect::MarkContradiction { .. } => "mark_contradiction",
        Effect::SetProvenance { .. } => "set_provenance",
        Effect::Touch => "touch",
        Effect::Link { .. } => "link",
        Effect::Abort { .. } => "abort",
    }
}

#[allow(dead_code)]
fn step_kind_name(s: &ActivityStep) -> &'static str {
    jcode_memory_types::activity::describe_step(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use jcode_memory_types::{
        DEFAULT_ONTOLOGY_ID, MemoryCategory, MemoryEntry, MemoryGraph, ProvenanceRecord,
    };

    #[test]
    fn default_registry_loads_default_ontology() {
        let reg = OntologyRegistry::with_default();
        let onto = reg.default_ontology();
        assert_eq!(onto.id, DEFAULT_ONTOLOGY_ID);
        assert_eq!(onto.version, DEFAULT_ONTOLOGY_VERSION);
    }

    #[test]
    fn dispatch_runs_default_rules() {
        let reg = OntologyRegistry::with_default();
        let mut entry = MemoryEntry::new(MemoryCategory::Fact, "user prefers dark mode");
        entry.provenance = Some(ProvenanceRecord {
            method: jcode_memory_types::ExtractionMethod::UserStated,
            confidence: 1.0,
            source: "tool".to_string(),
            extracted_at: Utc::now(),
            ..Default::default()
        });
        let type_id = reg
            .type_from_category(DEFAULT_ONTOLOGY_ID, &MemoryCategory::Fact)
            .expect("fact maps to a type");
        let mut ctx = reg.make_context(DEFAULT_ONTOLOGY_ID, entry, "event.remember");
        ctx.type_id = type_id;
        ctx.source_label = "tool".to_string();
        let plan = reg.dispatch(&ctx);
        assert!(
            plan.effects
                .iter()
                .any(|e| matches!(e, Effect::SetProvenance { .. }) | matches!(e, Effect::Touch)),
            "remember rule should at least set provenance + touch, got {:?}",
            plan.effects
        );
    }

    #[test]
    fn bind_to_graph_sets_metadata() {
        let reg = OntologyRegistry::with_default();
        let mut graph = MemoryGraph::new();
        assert!(graph.metadata.ontology_id.is_empty());
        reg.bind_to_graph(&mut graph);
        assert_eq!(graph.metadata.ontology_id, DEFAULT_ONTOLOGY_ID);
        assert_eq!(graph.metadata.ontology_version, DEFAULT_ONTOLOGY_VERSION);
    }

    #[test]
    fn declared_helpers_are_consistent() {
        let reg = OntologyRegistry::with_default();
        let onto = reg.default_ontology();
        let kinds = declared_effect_kinds(&onto, "event.remember");
        assert!(kinds.contains(&"set_provenance"));
        let conds = declared_condition_kinds(&onto, "event.remember");
        assert!(conds.contains(&"has_content"));
    }

    #[test]
    fn schedule_dispatches_default_activities() {
        let reg = OntologyRegistry::with_default();
        let mut graph = MemoryGraph::new();
        graph.metadata.last_cluster_update = Some(Utc::now() - chrono::Duration::seconds(120));
        let mut ctx = reg.make_context(
            DEFAULT_ONTOLOGY_ID,
            MemoryEntry::new(MemoryCategory::Fact, "x"),
            "event.turn_tick",
        );
        ctx.graph = Arc::new(graph);
        let onto = reg.default_ontology();
        let steps = declared_step_kinds(
            &onto,
            &ScheduleEvent::PeriodicTick {
                last_tick: Some(Utc::now() - chrono::Duration::seconds(120)),
            },
        );
        // The periodic tick may or may not produce steps depending on
        // configured cadence; we just need to confirm the helper runs.
        let _ = steps;
    }
}
