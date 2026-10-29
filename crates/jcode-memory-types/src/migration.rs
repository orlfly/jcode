//! Runtime ontology evolution: patches, validation, and instance migration.
//!
//! The ontology is a value object (`Ontology`) that can be persisted beside a
//! memory graph and replaced at runtime. This module provides the pieces that
//! make that safe:
//!
//! * [`OntologyPatch`] - a declarative diff proposed by a reflection pass.
//! * [`apply_patch`] - checks the base version/hash, applies the operations,
//!   validates the result, and requires migration coverage for destructive
//!   operations.
//! * [`MigrationStep`] - the data-side changes that keep stored instances
//!   consistent with the new ontology (category remap, tag sanitization,
//!   subject-key recomputation, edge rewiring, lifecycle normalization).
//! * [`apply_migrations`] - the deterministic executor over a `MemoryGraph`.
//!
//! Everything here is local and deterministic: no embedder, no LLM, no
//! network. The same properties that make `reconcile` usable on every machine
//! make migrations reviewable and reproducible.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::graph::{EdgeKind, MemoryGraph};
use crate::instance::{ExtractionMethod, MemoryStatus};
use crate::ontology::{
    Activity, DecayPolicy, Ontology, OntologyType, RelationshipDef, Rule, ScoringPolicy,
    TYPE_CONTRACT, TYPE_ENV_FACT, TYPE_OBSERVATION, TYPE_PROCESS_RULE, is_link_id,
};
use crate::{MemoryCategory, MemoryEntry};

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

/// Deterministic content hash of an ontology.
///
/// `Ontology` uses a `HashMap` for `category_aliases`, so serializing the value
/// directly is not order-stable. This builds a canonical view with sorted
/// aliases and hashes it with FNV-1a. The result is stable across processes and
/// Rust versions, which is what drift detection needs.
pub fn ontology_content_hash(ontology: &Ontology) -> String {
    #[derive(Serialize)]
    struct Canonical<'a> {
        id: &'a str,
        version: u32,
        description: &'a str,
        types: &'a BTreeMap<String, OntologyType>,
        relationships: &'a BTreeMap<String, RelationshipDef>,
        rules: &'a [Rule],
        activities: &'a [Activity],
        category_aliases: BTreeMap<&'a str, &'a str>,
    }

    let aliases: BTreeMap<&str, &str> = ontology
        .category_aliases
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let canonical = Canonical {
        id: &ontology.id,
        version: ontology.version,
        description: &ontology.description,
        types: &ontology.types,
        relationships: &ontology.relationships,
        rules: &ontology.rules,
        activities: &ontology.activities,
        category_aliases: aliases,
    };
    let bytes = serde_json::to_vec(&canonical).unwrap_or_default();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

impl Ontology {
    /// Deterministic content hash (see [`ontology_content_hash`]).
    pub fn content_hash(&self) -> String {
        ontology_content_hash(self)
    }
}

// ---------------------------------------------------------------------------
// Patch model
// ---------------------------------------------------------------------------

/// A proposed change to an ontology.
///
/// The reflection pass produces this value, not a mutated `Ontology`. Applying
/// it is a separate, audited step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OntologyPatch {
    /// Ontology id this patch targets.
    #[serde(default = "default_ontology_id")]
    pub base_id: String,
    /// Version the patch was authored against.
    pub base_version: u32,
    /// Content hash the patch was authored against. Empty disables the check
    /// for hand-written patches.
    #[serde(default)]
    pub base_hash: String,
    /// Why the change is proposed.
    #[serde(default)]
    pub rationale: String,
    /// Evidence: session ids, memory ids, metric names.
    #[serde(default)]
    pub evidence: Vec<String>,
    /// Ordered ontology operations.
    #[serde(default)]
    pub ops: Vec<OntologyOp>,
    /// Data-side migrations required by structural operations.
    #[serde(default)]
    pub migrations: Vec<MigrationStep>,
}

fn default_ontology_id() -> String {
    crate::ontology::DEFAULT_ONTOLOGY_ID.to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum OntologyOp {
    AddType {
        type_def: OntologyType,
    },
    UpdateType {
        id: String,
        #[serde(default)]
        patch: TypePatch,
    },
    RemoveType {
        id: String,
        #[serde(default = "default_residual_type")]
        fallback: String,
    },
    AddRelationship {
        relationship: RelationshipDef,
    },
    RemoveRelationship {
        id: String,
    },
    AddRule {
        rule: Rule,
    },
    RemoveRule {
        id: String,
    },
    AddActivity {
        activity: Activity,
    },
    RemoveActivity {
        id: String,
    },
    SetCategoryAlias {
        category: String,
        type_id: String,
    },
    RemoveCategoryAlias {
        category: String,
    },
}

fn default_residual_type() -> String {
    TYPE_OBSERVATION.to_string()
}

/// Fields of an `OntologyType` that are safe to change at runtime.
///
/// `id`, `parent`, and `lifecycle` are structural and deliberately absent:
/// changing them requires a code change plus an explicit migration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TypePatch {
    /// Set a new TTL in days.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_days: Option<f32>,
    /// Remove the TTL (retain indefinitely).
    #[serde(default)]
    pub clear_ttl: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scoring: Option<ScoringPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decay: Option<DecayPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub present_in_prompt: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extraction_methods: Option<Vec<ExtractionMethod>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_category: Option<String>,
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate an ontology before it is installed.
///
/// Returns human-readable problems. An empty vector means the ontology is
/// structurally sound. The checks mirror `validate_default_ontology` and add
/// the invariants a runtime replacement needs.
pub fn validate_ontology(ontology: &Ontology) -> Vec<String> {
    let mut errors = Vec::new();
    if ontology.id.trim().is_empty() {
        errors.push("ontology id is empty".to_string());
    }
    if ontology.version == 0 {
        errors.push("ontology version must be > 0".to_string());
    }
    if ontology.types.is_empty() {
        errors.push("ontology declares no types".to_string());
    }

    for (key, type_def) in &ontology.types {
        if key != &type_def.id {
            errors.push(format!(
                "type map key '{key}' does not match type id '{}'",
                type_def.id
            ));
        }
        if type_def.id.trim().is_empty() {
            errors.push("type with empty id".to_string());
        }
        if is_link_id(&type_def.id) {
            errors.push(format!(
                "type '{}' uses a reserved link prefix; link nodes are not types",
                type_def.id
            ));
        }
        if let Some(ttl) = type_def.ttl_days
            && (!ttl.is_finite() || ttl <= 0.0)
        {
            errors.push(format!("type '{}' has invalid ttl_days {ttl}", type_def.id));
        }
        if !type_def.decay.half_life_days.is_finite() || type_def.decay.half_life_days <= 0.0 {
            errors.push(format!(
                "type '{}' has invalid decay half_life_days {}",
                type_def.id, type_def.decay.half_life_days
            ));
        }
        if !type_def.scoring.base_score.is_finite() {
            errors.push(format!("type '{}' has non-finite base_score", type_def.id));
        }
        if let Some(parent) = &type_def.parent
            && !ontology.types.contains_key(parent)
        {
            errors.push(format!(
                "type '{}' declares unknown parent '{parent}'",
                type_def.id
            ));
        }
    }

    for (category, type_id) in &ontology.category_aliases {
        if !ontology.types.contains_key(type_id) {
            errors.push(format!(
                "category alias '{category}' points at unknown type '{type_id}'"
            ));
        }
    }

    let known_relationships: std::collections::HashSet<&str> =
        ontology.relationships.keys().map(String::as_str).collect();
    for rule in &ontology.rules {
        if rule.id.trim().is_empty() {
            errors.push("rule with empty id".to_string());
        }
        if rule.event.trim().is_empty() {
            errors.push(format!("rule '{}' has no event", rule.id));
        }
        if rule.effects.is_empty() {
            errors.push(format!("rule '{}' has no effects", rule.id));
        }
        for type_id in &rule.applies_to {
            if !ontology.types.contains_key(type_id) {
                errors.push(format!(
                    "rule '{}' applies to unknown type '{type_id}'",
                    rule.id
                ));
            }
        }
        for effect in &rule.effects {
            match effect {
                crate::ontology::Effect::Link { relationship, .. } => {
                    if !known_relationships.contains(relationship.as_str()) {
                        errors.push(format!(
                            "rule '{}' links with unknown relationship '{relationship}'",
                            rule.id
                        ));
                    }
                }
                crate::ontology::Effect::DeriveFrom { relation_kind, .. } => {
                    if !known_relationships.contains(relation_kind.as_str()) {
                        errors.push(format!(
                            "rule '{}' derives with unknown relationship '{relation_kind}'",
                            rule.id
                        ));
                    }
                }
                _ => {}
            }
        }
    }

    for activity in &ontology.activities {
        if activity.id.trim().is_empty() {
            errors.push("activity with empty id".to_string());
        }
        if activity.steps.is_empty() {
            errors.push(format!("activity '{}' has no steps", activity.id));
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// Applying a patch
// ---------------------------------------------------------------------------

/// Apply `patch` to a clone of `base`, validating the result.
///
/// On success the returned ontology has `version = base.version + 1`. The base
/// is untouched, which makes dry runs and rollback trivial.
pub fn apply_patch(base: &Ontology, patch: &OntologyPatch) -> Result<Ontology, Vec<String>> {
    let mut errors = Vec::new();
    if patch.base_id != base.id {
        errors.push(format!(
            "patch targets ontology '{}' but base is '{}'",
            patch.base_id, base.id
        ));
    }
    if patch.base_version != base.version {
        errors.push(format!(
            "patch expects version {} but base is {}",
            patch.base_version, base.version
        ));
    }
    if !patch.base_hash.is_empty() && patch.base_hash != base.content_hash() {
        errors.push(format!(
            "patch expects hash {} but base is {}",
            patch.base_hash,
            base.content_hash()
        ));
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    let mut next = base.clone();
    for op in &patch.ops {
        if let Err(mut op_errors) = apply_op(&mut next, op) {
            errors.append(&mut op_errors);
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    next.version = base.version.saturating_add(1);
    let mut validation = validate_ontology(&next);
    errors.append(&mut validation);
    errors.append(&mut migration_coverage_errors(patch, base));
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(next)
}

fn apply_op(ontology: &mut Ontology, op: &OntologyOp) -> Result<(), Vec<String>> {
    match op {
        OntologyOp::AddType { type_def } => {
            if ontology.types.contains_key(&type_def.id) {
                return Err(vec![format!("type '{}' already exists", type_def.id)]);
            }
            let mut type_def = type_def.clone();
            if type_def.legacy_category.is_none() {
                type_def.legacy_category = Some(type_def.id.clone());
            }
            if let Some(alias) = &type_def.legacy_category {
                ontology
                    .category_aliases
                    .insert(alias.clone(), type_def.id.clone());
            }
            ontology.types.insert(type_def.id.clone(), type_def);
            Ok(())
        }
        OntologyOp::UpdateType { id, patch } => {
            let Some(type_def) = ontology.types.get_mut(id) else {
                return Err(vec![format!("type '{id}' does not exist")]);
            };
            if let Some(ttl) = patch.ttl_days {
                type_def.ttl_days = Some(ttl);
            }
            if patch.clear_ttl {
                type_def.ttl_days = None;
            }
            if let Some(scoring) = &patch.scoring {
                type_def.scoring = scoring.clone();
            }
            if let Some(decay) = &patch.decay {
                type_def.decay = decay.clone();
            }
            if let Some(present) = patch.present_in_prompt {
                type_def.present_in_prompt = present;
            }
            if let Some(methods) = &patch.extraction_methods {
                type_def.extraction_methods = methods.clone();
            }
            if let Some(legacy) = &patch.legacy_category {
                type_def.legacy_category = Some(legacy.clone());
                ontology.category_aliases.insert(legacy.clone(), id.clone());
            }
            Ok(())
        }
        OntologyOp::RemoveType { id, fallback } => {
            if !ontology.types.contains_key(id) {
                return Err(vec![format!("type '{id}' does not exist")]);
            }
            if !ontology.types.contains_key(fallback) {
                return Err(vec![format!("fallback type '{fallback}' does not exist")]);
            }
            ontology.types.remove(id);
            ontology.category_aliases.retain(|_, target| target != id);
            Ok(())
        }
        OntologyOp::AddRelationship { relationship } => {
            if ontology.relationships.contains_key(&relationship.id) {
                return Err(vec![format!(
                    "relationship '{}' already exists",
                    relationship.id
                )]);
            }
            ontology
                .relationships
                .insert(relationship.id.clone(), relationship.clone());
            Ok(())
        }
        OntologyOp::RemoveRelationship { id } => {
            if ontology.relationships.remove(id).is_none() {
                return Err(vec![format!("relationship '{id}' does not exist")]);
            }
            Ok(())
        }
        OntologyOp::AddRule { rule } => {
            if ontology.rules.iter().any(|existing| existing.id == rule.id) {
                return Err(vec![format!("rule '{}' already exists", rule.id)]);
            }
            ontology.rules.push(rule.clone());
            Ok(())
        }
        OntologyOp::RemoveRule { id } => {
            let before = ontology.rules.len();
            ontology.rules.retain(|rule| rule.id != *id);
            if ontology.rules.len() == before {
                return Err(vec![format!("rule '{id}' does not exist")]);
            }
            Ok(())
        }
        OntologyOp::AddActivity { activity } => {
            if ontology
                .activities
                .iter()
                .any(|existing| existing.id == activity.id)
            {
                return Err(vec![format!("activity '{}' already exists", activity.id)]);
            }
            ontology.activities.push(activity.clone());
            Ok(())
        }
        OntologyOp::RemoveActivity { id } => {
            let before = ontology.activities.len();
            ontology.activities.retain(|activity| activity.id != *id);
            if ontology.activities.len() == before {
                return Err(vec![format!("activity '{id}' does not exist")]);
            }
            Ok(())
        }
        OntologyOp::SetCategoryAlias { category, type_id } => {
            if !ontology.types.contains_key(type_id) {
                return Err(vec![format!(
                    "category alias '{category}' points at unknown type '{type_id}'"
                )]);
            }
            ontology
                .category_aliases
                .insert(category.clone(), type_id.clone());
            Ok(())
        }
        OntologyOp::RemoveCategoryAlias { category } => {
            ontology.category_aliases.remove(category);
            Ok(())
        }
    }
}

/// Destructive operations must ship with the migration that keeps stored rows
/// meaningful. Without this a reflection pass could silently orphan data.
fn migration_coverage_errors(patch: &OntologyPatch, base: &Ontology) -> Vec<String> {
    let mut errors = Vec::new();
    for op in &patch.ops {
        match op {
            OntologyOp::RemoveType { id, .. } => {
                let legacy = base
                    .types
                    .get(id)
                    .and_then(|type_def| type_def.legacy_category.clone());
                let covered = patch.migrations.iter().any(|step| match step {
                    MigrationStep::RemapCategory { from, .. }
                    | MigrationStep::RetireCategory { from, .. }
                    | MigrationStep::SplitCategory { from, .. } => {
                        from == id || legacy.as_deref() == Some(from.as_str())
                    }
                    _ => false,
                });
                if !covered {
                    errors.push(format!(
                        "removing type '{id}' requires a RemapCategory, RetireCategory or SplitCategory migration for that type"
                    ));
                }
            }
            OntologyOp::RemoveRelationship { id } => {
                let covered = patch.migrations.iter().any(|step| match step {
                    MigrationStep::RewireEdge { from_kind, .. } => from_kind == id,
                    _ => false,
                });
                if !covered {
                    errors.push(format!(
                        "removing relationship '{id}' requires a RewireEdge migration for that kind"
                    ));
                }
            }
            _ => {}
        }
    }
    errors
}

// ---------------------------------------------------------------------------
// Data migrations
// ---------------------------------------------------------------------------

/// One deterministic data-side change applied to a stored graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "step", rename_all = "snake_case")]
pub enum MigrationStep {
    /// Rewrite one category name into another.
    RemapCategory { from: String, to: String },
    /// Split one category into several using first-match content markers.
    SplitCategory {
        from: String,
        #[serde(default)]
        rules: Vec<SplitRule>,
        #[serde(default = "default_residual_type")]
        default: String,
    },
    /// Send all rows of a category to a fallback type, or archive them.
    RetireCategory {
        from: String,
        #[serde(default = "default_residual_type")]
        fallback: String,
        #[serde(default)]
        archive: bool,
    },
    /// Derive `subject_key` for rows that lack one, or recompute all of them.
    RecomputeSubjectKey {
        #[serde(default)]
        overwrite: bool,
    },
    /// Re-run the tag vocabulary and refresh `search_text`.
    ResanitizeTags,
    /// Repair illegal archived-but-active rows.
    NormalizeLifecycle,
    /// Demote `entity` rows that carry no identity.
    DemoteIdentitylessEntities,
    /// Drop edges whose endpoints no longer exist.
    DropDanglingEdges,
    /// Remove an edge kind, or replace it with another.
    RewireEdge {
        from_kind: String,
        #[serde(default)]
        to_kind: Option<String>,
    },
    /// Re-run the deterministic residual classifier.
    ReclassifyResidual,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SplitRule {
    /// Case-insensitive markers. The first rule with any marker wins.
    pub contains: Vec<String>,
    pub to: String,
}

/// Counts produced by a migration pass. All fields are additive.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MigrationReport {
    pub remapped: usize,
    pub split: usize,
    pub retired: usize,
    pub archived: usize,
    pub subject_keys: usize,
    pub tags_resanitized: usize,
    pub lifecycle_fixed: usize,
    pub entities_demoted: usize,
    pub edges_dropped: usize,
    pub edges_rewired: usize,
    pub reclassified: usize,
}

impl MigrationReport {
    pub fn changed(&self) -> usize {
        self.remapped
            + self.split
            + self.retired
            + self.archived
            + self.subject_keys
            + self.tags_resanitized
            + self.lifecycle_fixed
            + self.entities_demoted
            + self.edges_dropped
            + self.edges_rewired
            + self.reclassified
    }

    pub fn merge(&mut self, other: MigrationReport) {
        self.remapped += other.remapped;
        self.split += other.split;
        self.retired += other.retired;
        self.archived += other.archived;
        self.subject_keys += other.subject_keys;
        self.tags_resanitized += other.tags_resanitized;
        self.lifecycle_fixed += other.lifecycle_fixed;
        self.entities_demoted += other.entities_demoted;
        self.edges_dropped += other.edges_dropped;
        self.edges_rewired += other.edges_rewired;
        self.reclassified += other.reclassified;
    }
}

/// Execute `steps` in order over `graph`.
///
/// The ontology is used only by `ReclassifyResidual` (to resolve residual
/// categories) and is otherwise inert, which keeps the executor testable
/// without a registry.
pub fn apply_migrations(
    graph: &mut MemoryGraph,
    steps: &[MigrationStep],
    ontology: &Ontology,
) -> MigrationReport {
    let mut report = MigrationReport::default();
    for step in steps {
        match step {
            MigrationStep::RemapCategory { from, to } => {
                report.remapped += remap_category(graph, from, to);
            }
            MigrationStep::SplitCategory {
                from,
                rules,
                default,
            } => {
                report.split += split_category(graph, from, rules, default);
            }
            MigrationStep::RetireCategory {
                from,
                fallback,
                archive,
            } => {
                let (retired, archived) = retire_category(graph, from, fallback, *archive);
                report.retired += retired;
                report.archived += archived;
            }
            MigrationStep::RecomputeSubjectKey { overwrite } => {
                report.subject_keys += recompute_subject_keys(graph, *overwrite);
            }
            MigrationStep::ResanitizeTags => {
                report.tags_resanitized += resanitize_tags(graph);
            }
            MigrationStep::NormalizeLifecycle => {
                report.lifecycle_fixed += normalize_lifecycle(graph);
            }
            MigrationStep::DemoteIdentitylessEntities => {
                report.entities_demoted += demote_identityless_entities(graph);
            }
            MigrationStep::DropDanglingEdges => {
                report.edges_dropped += drop_dangling_edges(graph);
            }
            MigrationStep::RewireEdge { from_kind, to_kind } => {
                report.edges_rewired += rewire_edges(graph, from_kind, to_kind.as_deref());
            }
            MigrationStep::ReclassifyResidual => {
                report.reclassified += reclassify_residual(graph, ontology);
            }
        }
    }
    report
}

fn category_matches(entry: &MemoryEntry, from: &str) -> bool {
    entry.category.to_string() == from
}

fn remap_category(graph: &mut MemoryGraph, from: &str, to: &str) -> usize {
    let target: MemoryCategory = to
        .parse()
        .unwrap_or_else(|_| MemoryCategory::Custom(to.to_string()));
    let mut changed = 0;
    for entry in graph.memories.values_mut() {
        if category_matches(entry, from) {
            entry.category = target.clone();
            entry.updated_at = Utc::now();
            changed += 1;
        }
    }
    changed
}

fn split_category(
    graph: &mut MemoryGraph,
    from: &str,
    rules: &[SplitRule],
    default: &str,
) -> usize {
    let fallback: MemoryCategory = default
        .parse()
        .unwrap_or_else(|_| MemoryCategory::Custom(default.to_string()));
    let mut changed = 0;
    for entry in graph.memories.values_mut() {
        if !category_matches(entry, from) || is_link_id(&entry.id) {
            continue;
        }
        let lower = entry.content.to_ascii_lowercase();
        let target = rules
            .iter()
            .find(|rule| {
                rule.contains
                    .iter()
                    .any(|marker| lower.contains(&marker.to_ascii_lowercase()))
            })
            .map(|rule| {
                rule.to
                    .parse()
                    .unwrap_or_else(|_| MemoryCategory::Custom(rule.to.clone()))
            })
            .unwrap_or_else(|| fallback.clone());
        entry.category = target;
        entry.updated_at = Utc::now();
        changed += 1;
    }
    changed
}

fn retire_category(
    graph: &mut MemoryGraph,
    from: &str,
    fallback: &str,
    archive: bool,
) -> (usize, usize) {
    if archive {
        let mut archived = 0;
        for entry in graph.memories.values_mut() {
            if category_matches(entry, from) {
                entry.lifecycle.status = MemoryStatus::Archived;
                entry.active = false;
                entry.updated_at = Utc::now();
                archived += 1;
            }
        }
        (0, archived)
    } else {
        (remap_category(graph, from, fallback), 0)
    }
}

fn recompute_subject_keys(graph: &mut MemoryGraph, overwrite: bool) -> usize {
    let mut changed = 0;
    for entry in graph.memories.values_mut() {
        let missing = entry.subject_key.as_deref().is_none_or(str::is_empty);
        if !missing && !overwrite {
            continue;
        }
        if is_link_id(&entry.id) {
            continue;
        }
        let derived = crate::reconcile::derive_subject_key(&entry.content);
        if derived != entry.subject_key {
            entry.subject_key = derived;
            entry.updated_at = Utc::now();
            changed += 1;
        }
    }
    changed
}

fn resanitize_tags(graph: &mut MemoryGraph) -> usize {
    // Plan first, then mutate edges and entries. `tag_memory`/`untag_memory`
    // maintain tag nodes, `has_tag` edges and counts, so a direct field write
    // alone would leave the graph's tag layer stale.
    let planned: Vec<(String, Vec<String>, Vec<String>)> = graph
        .memories
        .iter()
        .filter(|(id, _)| !is_link_id(id))
        .filter_map(|(id, entry)| {
            let sanitized = crate::reconcile::sanitize_tags(entry.tags.iter().map(String::as_str));
            (sanitized != entry.tags).then(|| (id.clone(), entry.tags.clone(), sanitized))
        })
        .collect();

    for (id, old_tags, new_tags) in &planned {
        for tag in old_tags {
            if !new_tags.contains(tag) {
                graph.untag_memory(id, tag);
            }
        }
        for tag in new_tags {
            if !old_tags.contains(tag) {
                graph.tag_memory(id, tag);
            }
        }
        if let Some(entry) = graph.memories.get_mut(id) {
            entry.tags = new_tags.clone();
            entry.refresh_search_text();
            entry.updated_at = Utc::now();
        }
    }
    planned.len()
}

fn normalize_lifecycle(graph: &mut MemoryGraph) -> usize {
    let mut changed = 0;
    for entry in graph.memories.values_mut() {
        if entry.lifecycle.status == MemoryStatus::Archived && entry.active {
            entry.active = false;
            entry.updated_at = Utc::now();
            changed += 1;
        }
    }
    changed
}

fn demote_identityless_entities(graph: &mut MemoryGraph) -> usize {
    let mut changed = 0;
    for entry in graph.memories.values_mut() {
        if entry.category == MemoryCategory::Entity && entry.identity.is_empty() {
            entry.category = MemoryCategory::Fact;
            entry.updated_at = Utc::now();
            changed += 1;
        }
    }
    changed
}

fn drop_dangling_edges(graph: &mut MemoryGraph) -> usize {
    let mut valid: std::collections::HashSet<&str> =
        graph.memories.keys().map(String::as_str).collect();
    valid.extend(graph.tags.keys().map(String::as_str));
    valid.extend(graph.clusters.keys().map(String::as_str));

    let mut dropped = 0;
    let mut edges = std::mem::take(&mut graph.edges);
    edges.retain(|source, targets| {
        if !valid.contains(source.as_str()) {
            dropped += targets.len();
            return false;
        }
        let before = targets.len();
        targets.retain(|edge| valid.contains(edge.target.as_str()));
        dropped += before - targets.len();
        !targets.is_empty()
    });
    graph.edges = edges;
    rebuild_reverse_edges(graph);
    dropped
}

fn rewire_edges(graph: &mut MemoryGraph, from_kind: &str, to_kind: Option<&str>) -> usize {
    let mut changed = 0;
    let mut edges = std::mem::take(&mut graph.edges);
    for targets in edges.values_mut() {
        targets.retain_mut(|edge| {
            if edge_kind_name(&edge.kind) != from_kind {
                return true;
            }
            match to_kind {
                Some(name) => match parse_edge_kind(name) {
                    Some(kind) => {
                        edge.kind = kind;
                        changed += 1;
                        true
                    }
                    None => {
                        changed += 1;
                        false
                    }
                },
                None => {
                    changed += 1;
                    false
                }
            }
        });
    }
    edges.retain(|_, targets| !targets.is_empty());
    graph.edges = edges;
    rebuild_reverse_edges(graph);
    changed
}

fn rebuild_reverse_edges(graph: &mut MemoryGraph) {
    let mut reverse: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    let mut tag_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for (source, targets) in &graph.edges {
        for edge in targets {
            reverse
                .entry(edge.target.clone())
                .or_default()
                .push(source.clone());
            if matches!(edge.kind, EdgeKind::HasTag) {
                *tag_counts.entry(edge.target.clone()).or_default() += 1;
            }
        }
    }
    graph.reverse_edges = reverse;
    for (id, tag) in graph.tags.iter_mut() {
        tag.count = tag_counts.get(id).copied().unwrap_or(0);
    }
}

fn edge_kind_name(kind: &EdgeKind) -> &'static str {
    match kind {
        EdgeKind::HasTag => "has_tag",
        EdgeKind::InCluster => "in_cluster",
        EdgeKind::RelatesTo { .. } => "relates_to",
        EdgeKind::Supersedes => "supersedes",
        EdgeKind::Contradicts => "contradicts",
        EdgeKind::DerivedFrom => "derived_from",
    }
}

fn parse_edge_kind(name: &str) -> Option<EdgeKind> {
    Some(match name {
        "has_tag" => EdgeKind::HasTag,
        "in_cluster" => EdgeKind::InCluster,
        "relates_to" => EdgeKind::RelatesTo { weight: 1.0 },
        "supersedes" => EdgeKind::Supersedes,
        "contradicts" => EdgeKind::Contradicts,
        "derived_from" => EdgeKind::DerivedFrom,
        _ => return None,
    })
}

fn reclassify_residual(graph: &mut MemoryGraph, ontology: &Ontology) -> usize {
    let residual = TYPE_OBSERVATION;
    let mut changed = 0;
    for entry in graph.memories.values_mut() {
        if is_link_id(&entry.id) {
            continue;
        }
        let is_residual = ontology
            .type_from_category(&entry.category)
            .is_some_and(|type_id| type_id == residual)
            || matches!(
                entry.category,
                MemoryCategory::Fact | MemoryCategory::Observation
            );
        if !is_residual {
            continue;
        }
        let Some(target) = classify_residual_type(&entry.content) else {
            continue;
        };
        if entry.category.to_string() == target {
            continue;
        }
        entry.category = target
            .parse()
            .unwrap_or_else(|_| MemoryCategory::Custom(target.to_string()));
        entry.updated_at = Utc::now();
        changed += 1;
    }
    changed
}

// ---------------------------------------------------------------------------
// Residual classifier
// ---------------------------------------------------------------------------
//
// Moved here from `jcode-base/src/memory.rs` so migrations and the runtime
// classification pass share one implementation.

/// Deterministic classifier for residual (`fact` / `observation`) instances.
///
/// Order matters and mirrors what the corpus actually contains: procedures
/// first (a rule that mentions an endpoint is still a rule), then interface
/// contracts, then perishable environment facts. Returns `None` when the
/// instance should stay in the residual bucket.
pub fn classify_residual_type(content: &str) -> Option<&'static str> {
    let lower = content.to_ascii_lowercase();
    const RULE_MARKERS: [&str; 24] = [
        "必须",
        "不得",
        "禁止",
        "应该",
        "正确做法",
        "约定",
        "流程",
        "规范",
        "always",
        "never",
        "must ",
        "must\n",
        "do not",
        "instead of",
        "prefer ",
        "workflow",
        "rule ",
        "convention",
        "should ",
        "required",
        "不允许",
        "只能",
        "务必",
        "禁止",
    ];
    if RULE_MARKERS.iter().any(|m| lower.contains(m)) {
        return Some(TYPE_PROCESS_RULE);
    }
    const CONTRACT_MARKERS: [&str; 16] = [
        "/api/",
        "get ",
        "post ",
        "put ",
        "patch ",
        "delete ",
        "endpoint",
        "schema",
        "字段",
        "枚举",
        "enum",
        "payload",
        "接口",
        "路由",
        "status code",
        "http ",
    ];
    if CONTRACT_MARKERS.iter().any(|m| lower.contains(m)) {
        return Some(TYPE_CONTRACT);
    }
    const ENV_MARKERS: [&str; 8] = [
        "localhost:",
        "127.0.0.1",
        "/home/",
        "/opt/workspace/",
        "/tmp/",
        "task #",
        "session_",
        "mem_",
    ];
    if ENV_MARKERS.iter().any(|m| lower.contains(m)) || looks_like_host_port(&lower) {
        return Some(TYPE_ENV_FACT);
    }
    None
}

/// `host:port` or `ip:port` anywhere in the text.
fn looks_like_host_port(lower: &str) -> bool {
    lower
        .split(|c: char| c.is_whitespace() || matches!(c, '`' | '"' | '\'' | '(' | ')'))
        .any(|token| {
            let token = token.trim_matches(|c: char| c == ',' || c == ';');
            let Some((host, port)) = token.rsplit_once(':') else {
                return false;
            };
            !host.is_empty()
                && !port.is_empty()
                && port.chars().all(|c| c.is_ascii_digit())
                && (host == "localhost"
                    || host.chars().all(|c| c.is_ascii_digit() || c == '.')
                    || host
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.'))
        })
}

// ---------------------------------------------------------------------------
// History / audit types stored on the graph
// ---------------------------------------------------------------------------

/// A compact, persisted snapshot of an installed ontology.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OntologySnapshot {
    pub version: u32,
    pub hash: String,
    pub ontology: Ontology,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub applied_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub rationale: String,
}

impl OntologySnapshot {
    pub fn new(ontology: Ontology, rationale: impl Into<String>) -> Self {
        let hash = ontology.content_hash();
        Self {
            version: ontology.version,
            hash,
            ontology,
            applied_at: Some(Utc::now()),
            rationale: rationale.into(),
        }
    }
}

/// Migration state persisted on the graph so a crash mid-migration is
/// detectable instead of looking like a clean store.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MigrationState {
    pub from_version: u32,
    pub to_version: u32,
    pub started_at: DateTime<Utc>,
    pub completed_steps: usize,
    pub total_steps: usize,
}

/// The outcome of applying a patch to one store.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OntologyApplyReport {
    pub store: String,
    pub old_version: u32,
    pub new_version: u32,
    pub old_hash: String,
    pub new_hash: String,
    pub dry_run: bool,
    pub validation_ok: bool,
    pub migrations: MigrationReport,
    pub patch: OntologyPatch,
}

/// Reconcile thresholds declared by `activity.write_reconcile`.
///
/// Falls back to the compile-time constants when the ontology does not declare
/// the activity, which keeps hand-built test ontologies working.
pub fn reconcile_thresholds(ontology: &Ontology) -> (f32, f32, f32) {
    for activity in &ontology.activities {
        for step in &activity.steps {
            if let crate::ontology::ActivityStep::ReconcileOnWrite {
                reinforce_at,
                supersede_at,
                relate_at,
            } = step
            {
                return (*reinforce_at, *supersede_at, *relate_at);
            }
        }
    }
    (
        crate::reconcile::REINFORCE_AT,
        crate::reconcile::SUPERSEDE_AT,
        crate::reconcile::RELATE_AT,
    )
}

/// Build a human-readable summary of an ontology for CLI/tool output.
pub fn describe_ontology(ontology: &Ontology) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{} v{} ({})\n",
        ontology.id,
        ontology.version,
        ontology.content_hash()
    ));
    out.push_str(&format!("types: {}\n", ontology.types.len()));
    for type_def in ontology.types.values() {
        out.push_str(&format!(
            "  - {:<14} ttl={:<6} base={:<5} half_life={:<5} methods={}\n",
            type_def.id,
            type_def
                .ttl_days
                .map(|days| format!("{days:.0}d"))
                .unwrap_or_else(|| "-".to_string()),
            type_def.scoring.base_score,
            type_def.decay.half_life_days,
            type_def.extraction_methods.len()
        ));
    }
    out.push_str(&format!(
        "relationships: {}\nrules: {}\nactivities: {}\naliases: {}\n",
        ontology.relationships.len(),
        ontology.rules.len(),
        ontology.activities.len(),
        ontology.category_aliases.len()
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::{DEFAULT_ONTOLOGY_VERSION, default_ontology};

    fn patch(
        base: &Ontology,
        ops: Vec<OntologyOp>,
        migrations: Vec<MigrationStep>,
    ) -> OntologyPatch {
        OntologyPatch {
            base_id: base.id.clone(),
            base_version: base.version,
            base_hash: base.content_hash(),
            rationale: "test".to_string(),
            evidence: Vec::new(),
            ops,
            migrations,
        }
    }

    #[test]
    fn hash_is_stable_and_sensitive() {
        let a = default_ontology();
        let b = default_ontology();
        assert_eq!(a.content_hash(), b.content_hash());
        let mut c = a.clone();
        c.description.push_str(" changed");
        assert_ne!(a.content_hash(), c.content_hash());
    }

    #[test]
    fn patch_updates_ttl_and_bumps_version() {
        let base = default_ontology();
        let patch = patch(
            &base,
            vec![OntologyOp::UpdateType {
                id: TYPE_ENV_FACT.to_string(),
                patch: TypePatch {
                    ttl_days: Some(45.0),
                    ..Default::default()
                },
            }],
            Vec::new(),
        );
        let next = apply_patch(&base, &patch).expect("patch applies");
        assert_eq!(next.version, DEFAULT_ONTOLOGY_VERSION + 1);
        assert_eq!(next.types[TYPE_ENV_FACT].ttl_days, Some(45.0));
        // Base is untouched.
        assert_eq!(base.types[TYPE_ENV_FACT].ttl_days, Some(30.0));
    }

    #[test]
    fn destructive_patch_requires_migration() {
        let base = default_ontology();
        let patch = patch(
            &base,
            vec![OntologyOp::RemoveType {
                id: TYPE_OBSERVATION.to_string(),
                fallback: TYPE_ENV_FACT.to_string(),
            }],
            Vec::new(),
        );
        let errors = apply_patch(&base, &patch).expect_err("must reject");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("requires a RemapCategory")),
            "{errors:?}"
        );
    }

    #[test]
    fn stale_base_hash_is_rejected() {
        let base = default_ontology();
        let mut patch = patch(
            &base,
            vec![OntologyOp::UpdateType {
                id: TYPE_ENV_FACT.to_string(),
                patch: TypePatch {
                    ttl_days: Some(45.0),
                    ..Default::default()
                },
            }],
            Vec::new(),
        );
        patch.base_hash = "deadbeef".to_string();
        let errors = apply_patch(&base, &patch).expect_err("must reject");
        assert!(
            errors.iter().any(|error| error.contains("expects hash")),
            "{errors:?}"
        );
    }

    #[test]
    fn migrations_retype_retag_reanchor_and_rewire() {
        let mut graph = MemoryGraph::new();
        let mut entry = MemoryEntry::new(
            MemoryCategory::Fact,
            "You must call POST /api/task/claim with the team id",
        );
        entry.tags = vec!["assistant".to_string(), "api".to_string()];
        let id = graph.add_memory(entry);

        let mut other = MemoryEntry::new(MemoryCategory::Fact, "unrelated note");
        other.id = "mem_other".to_string();
        let other_id = graph.add_memory(other);
        graph.link_memories(&id, &other_id, 0.9);
        graph.remove_memory(&other_id);
        // Simulate a stale edge whose endpoint vanished without cleanup.
        graph.add_edge(&id, "mem_missing", EdgeKind::RelatesTo { weight: 0.5 });

        let ontology = default_ontology();
        let report = apply_migrations(
            &mut graph,
            &[
                MigrationStep::RemapCategory {
                    from: "fact".to_string(),
                    to: "observation".to_string(),
                },
                MigrationStep::ReclassifyResidual,
                MigrationStep::ResanitizeTags,
                MigrationStep::RecomputeSubjectKey { overwrite: true },
                MigrationStep::DropDanglingEdges,
            ],
            &ontology,
        );

        let entry = graph.get_memory(&id).expect("entry survives");
        assert_eq!(entry.category, MemoryCategory::ProcessRule);
        assert!(!entry.tags.contains(&"assistant".to_string()));
        assert_eq!(entry.subject_key.as_deref(), Some("api:/api/task/claim"));
        assert!(
            !graph
                .get_edges(&id)
                .iter()
                .any(|edge| edge.target == "tag:assistant"),
            "removed tag must lose its has_tag edge"
        );
        assert_eq!(
            graph.tags.get("tag:assistant").map(|tag| tag.count),
            Some(0),
            "removed tag count must be recomputed"
        );
        assert!(
            graph
                .get_edges(&id)
                .iter()
                .any(|edge| edge.target == "tag:api"),
            "kept tag must retain its has_tag edge"
        );
        assert!(report.edges_dropped >= 1);
        assert!(
            graph
                .get_edges(&id)
                .iter()
                .all(|edge| edge.target != other_id && edge.target != "mem_missing")
        );
        assert!(graph.reverse_edges.get(&other_id).is_none_or(Vec::is_empty));
    }

    #[test]
    fn split_category_uses_first_matching_rule() {
        let mut graph = MemoryGraph::new();
        let mut rule = MemoryEntry::new(MemoryCategory::Fact, "must always validate the token");
        rule.category = MemoryCategory::Custom("legacy".to_string());
        let rule_id = graph.add_memory(rule);
        let mut env = MemoryEntry::new(MemoryCategory::Fact, "server is on localhost:1337");
        env.category = MemoryCategory::Custom("legacy".to_string());
        let env_id = graph.add_memory(env);

        let ontology = default_ontology();
        apply_migrations(
            &mut graph,
            &[MigrationStep::SplitCategory {
                from: "legacy".to_string(),
                rules: vec![
                    SplitRule {
                        contains: vec!["must ".to_string()],
                        to: "process_rule".to_string(),
                    },
                    SplitRule {
                        contains: vec!["localhost:".to_string()],
                        to: "env_fact".to_string(),
                    },
                ],
                default: "observation".to_string(),
            }],
            &ontology,
        );
        assert_eq!(
            graph.get_memory(&rule_id).unwrap().category,
            MemoryCategory::ProcessRule
        );
        assert_eq!(
            graph.get_memory(&env_id).unwrap().category,
            MemoryCategory::EnvFact
        );
    }
}
