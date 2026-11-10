//! Layered ontology composition.
//!
//! An [`OntologyLayer`] is a partial ontology: it can add types, patch fields of
//! existing types, disable types, and merge relationships, rules, activities and
//! category aliases. Layers are folded onto a base ontology in priority order.
//!
//! This is the data model behind global domain packs and project overlays. The
//! runtime keeps a materialized effective ontology per store; this module owns
//! the deterministic fold and the inverse diff used when a project patch has to
//! be written back as an overlay file.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::migration::{TypePatch, validate_ontology};
use crate::ontology::{Activity, Ontology, OntologyType, RelationshipDef, Rule};

/// A partial ontology applied on top of a base ontology.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OntologyLayer {
    pub id: String,
    #[serde(default = "default_layer_version")]
    pub version: u32,
    /// Higher priority wins when two layers touch the same entry.
    #[serde(default)]
    pub priority: i32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Recommended id prefix for types in this layer, e.g. `cloud` or
    /// `acme.billing`. Informational; composition does not rewrite ids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// Full type definitions to add or replace.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub types: BTreeMap<String, OntologyType>,
    /// Field-level overrides for types defined in the base or an earlier layer.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub type_patches: BTreeMap<String, TypePatch>,
    /// Types removed from the effective ontology.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disabled_types: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub relationships: BTreeMap<String, RelationshipDef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disabled_relationships: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rules: Vec<Rule>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disabled_rules: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub activities: Vec<Activity>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disabled_activities: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub category_aliases: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disabled_aliases: Vec<String>,
}

fn default_layer_version() -> u32 {
    1
}

impl OntologyLayer {
    pub fn new(id: impl Into<String>, version: u32) -> Self {
        Self {
            id: id.into(),
            version,
            priority: 0,
            description: String::new(),
            namespace: None,
            types: BTreeMap::new(),
            type_patches: BTreeMap::new(),
            disabled_types: Vec::new(),
            relationships: BTreeMap::new(),
            disabled_relationships: Vec::new(),
            rules: Vec::new(),
            disabled_rules: Vec::new(),
            activities: Vec::new(),
            disabled_activities: Vec::new(),
            category_aliases: HashMap::new(),
            disabled_aliases: Vec::new(),
        }
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

/// Fold `layers` onto `base` in priority order.
///
/// The operation is deterministic and idempotent for absolute operations.
/// Disabling an absent entry is ignored, so the same overlay can be applied on
/// top of an already materialized ontology without failing.
pub fn compose_ontology_layers(
    base: &Ontology,
    layers: &[OntologyLayer],
) -> Result<Ontology, Vec<String>> {
    let mut ontology = base.clone();
    let mut errors = Vec::new();

    let mut ordered: Vec<(usize, &OntologyLayer)> = layers.iter().enumerate().collect();
    ordered.sort_by_key(|(index, layer)| (layer.priority, *index));

    for (_, layer) in ordered {
        apply_layer(&mut ontology, layer, &mut errors);
    }

    if errors.is_empty() {
        let validation = validate_ontology(&ontology);
        if !validation.is_empty() {
            return Err(validation);
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(ontology)
}

fn apply_layer(ontology: &mut Ontology, layer: &OntologyLayer, errors: &mut Vec<String>) {
    for id in &layer.disabled_types {
        if layer.types.contains_key(id) || layer.type_patches.contains_key(id) {
            errors.push(format!(
                "layer '{}' both defines and disables type '{id}'",
                layer.id
            ));
            continue;
        }
        ontology.types.remove(id);
        ontology.category_aliases.retain(|_, target| target != id);
    }

    for (id, type_def) in &layer.types {
        let mut type_def = type_def.clone();
        if type_def.id.is_empty() {
            type_def.id = id.clone();
        }
        if id != &type_def.id {
            errors.push(format!(
                "layer '{}' key '{id}' does not match type id '{}'",
                layer.id, type_def.id
            ));
            continue;
        }
        if let Some(alias) = &type_def.legacy_category {
            ontology
                .category_aliases
                .insert(alias.clone(), type_def.id.clone());
        }
        ontology.types.insert(id.clone(), type_def);
    }

    for (id, patch) in &layer.type_patches {
        let Some(type_def) = ontology.types.get_mut(id) else {
            errors.push(format!("layer '{}' patches missing type '{id}'", layer.id));
            continue;
        };
        apply_type_patch(type_def, patch, &mut ontology.category_aliases, id);
    }

    for id in &layer.disabled_relationships {
        ontology.relationships.remove(id);
    }
    for (id, relationship) in &layer.relationships {
        ontology
            .relationships
            .insert(id.clone(), relationship.clone());
    }

    for id in &layer.disabled_rules {
        ontology.rules.retain(|rule| &rule.id != id);
    }
    for rule in &layer.rules {
        ontology.rules.retain(|existing| existing.id != rule.id);
        ontology.rules.push(rule.clone());
    }

    for id in &layer.disabled_activities {
        ontology.activities.retain(|activity| &activity.id != id);
    }
    for activity in &layer.activities {
        ontology
            .activities
            .retain(|existing| existing.id != activity.id);
        ontology.activities.push(activity.clone());
    }

    for alias in &layer.disabled_aliases {
        ontology.category_aliases.remove(alias);
    }
    for (alias, target) in &layer.category_aliases {
        ontology
            .category_aliases
            .insert(alias.clone(), target.clone());
    }

    ontology.version = ontology.version.max(layer.version);
}

fn apply_type_patch(
    type_def: &mut OntologyType,
    patch: &TypePatch,
    aliases: &mut HashMap<String, String>,
    id: &str,
) {
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
        aliases.insert(legacy.clone(), id.to_string());
    }
    if let Some(properties) = &patch.properties {
        type_def.properties = properties.clone();
    }
}

/// Compute the layer that turns `base` into `next`.
///
/// Used to write a project patch back as a reviewable overlay. Field-level
/// changes become `type_patches` so unrelated future base changes still flow
/// through. Structural changes fall back to a full type definition.
pub fn diff_ontology_layer(base: &Ontology, next: &Ontology) -> OntologyLayer {
    let mut layer = OntologyLayer::new("project/overlay", next.version);
    layer.priority = 100;

    for (id, type_def) in &next.types {
        match base.types.get(id) {
            None => {
                layer.types.insert(id.clone(), type_def.clone());
            }
            Some(previous) if previous != type_def => {
                if structural_change(previous, type_def) {
                    layer.types.insert(id.clone(), type_def.clone());
                } else if let Some(patch) = diff_type_patch(previous, type_def) {
                    layer.type_patches.insert(id.clone(), patch);
                }
            }
            Some(_) => {}
        }
    }
    for id in base.types.keys() {
        if !next.types.contains_key(id) {
            layer.disabled_types.push(id.clone());
        }
    }

    for (id, relationship) in &next.relationships {
        match base.relationships.get(id) {
            None => {
                layer.relationships.insert(id.clone(), relationship.clone());
            }
            Some(previous) if previous != relationship => {
                layer.relationships.insert(id.clone(), relationship.clone());
            }
            Some(_) => {}
        }
    }
    for id in base.relationships.keys() {
        if !next.relationships.contains_key(id) {
            layer.disabled_relationships.push(id.clone());
        }
    }

    diff_rules(&base.rules, &next.rules, &mut layer);
    diff_activities(&base.activities, &next.activities, &mut layer);

    for (alias, target) in &next.category_aliases {
        if base.category_aliases.get(alias) != Some(target) {
            layer.category_aliases.insert(alias.clone(), target.clone());
        }
    }
    for alias in base.category_aliases.keys() {
        if !next.category_aliases.contains_key(alias) {
            layer.disabled_aliases.push(alias.clone());
        }
    }

    layer
}

fn structural_change(base: &OntologyType, next: &OntologyType) -> bool {
    base.display_name != next.display_name
        || base.description != next.description
        || base.parent != next.parent
        || base.lifecycle != next.lifecycle
        || base.persistent != next.persistent
        || base.embeddable != next.embeddable
        // TypePatch cannot clear a legacy category, so use a full replacement.
        || (base.legacy_category.is_some() && next.legacy_category.is_none())
}

fn diff_type_patch(base: &OntologyType, next: &OntologyType) -> Option<TypePatch> {
    let mut patch = TypePatch::default();
    let mut changed = false;

    if base.ttl_days != next.ttl_days {
        if let Some(ttl) = next.ttl_days {
            patch.ttl_days = Some(ttl);
        } else {
            patch.clear_ttl = true;
        }
        changed = true;
    }
    if base.scoring != next.scoring {
        patch.scoring = Some(next.scoring.clone());
        changed = true;
    }
    if base.decay != next.decay {
        patch.decay = Some(next.decay.clone());
        changed = true;
    }
    if base.present_in_prompt != next.present_in_prompt {
        patch.present_in_prompt = Some(next.present_in_prompt);
        changed = true;
    }
    if base.extraction_methods != next.extraction_methods {
        patch.extraction_methods = Some(next.extraction_methods.clone());
        changed = true;
    }
    if base.legacy_category != next.legacy_category {
        patch.legacy_category = next.legacy_category.clone();
        changed = true;
    }
    if base.properties != next.properties {
        patch.properties = Some(next.properties.clone());
        changed = true;
    }

    changed.then_some(patch)
}

fn diff_rules(base: &[Rule], next: &[Rule], layer: &mut OntologyLayer) {
    let base_by_id: HashMap<&str, &Rule> =
        base.iter().map(|rule| (rule.id.as_str(), rule)).collect();
    let next_by_id: HashMap<&str, &Rule> =
        next.iter().map(|rule| (rule.id.as_str(), rule)).collect();
    for (id, rule) in &next_by_id {
        if base_by_id.get(id) != Some(rule) {
            layer.rules.push((*rule).clone());
        }
    }
    for id in base_by_id.keys() {
        if !next_by_id.contains_key(id) {
            layer.disabled_rules.push((*id).to_string());
        }
    }
}

fn diff_activities(base: &[Activity], next: &[Activity], layer: &mut OntologyLayer) {
    let base_by_id: HashMap<&str, &Activity> = base
        .iter()
        .map(|activity| (activity.id.as_str(), activity))
        .collect();
    let next_by_id: HashMap<&str, &Activity> = next
        .iter()
        .map(|activity| (activity.id.as_str(), activity))
        .collect();
    for (id, activity) in &next_by_id {
        if base_by_id.get(id) != Some(activity) {
            layer.activities.push((*activity).clone());
        }
    }
    for id in base_by_id.keys() {
        if !next_by_id.contains_key(id) {
            layer.disabled_activities.push((*id).to_string());
        }
    }
}

/// Deterministic hash of a layer stack, used for materialization invalidation.
pub fn layer_stack_hash(layers: &[OntologyLayer]) -> String {
    #[derive(Serialize)]
    struct Canonical<'a> {
        id: &'a str,
        version: u32,
        priority: i32,
        description: &'a str,
        namespace: Option<&'a str>,
        types: &'a BTreeMap<String, OntologyType>,
        type_patches: &'a BTreeMap<String, TypePatch>,
        disabled_types: &'a [String],
        relationships: &'a BTreeMap<String, RelationshipDef>,
        disabled_relationships: &'a [String],
        rules: &'a [Rule],
        disabled_rules: &'a [String],
        activities: &'a [Activity],
        disabled_activities: &'a [String],
        category_aliases: BTreeMap<&'a str, &'a str>,
        disabled_aliases: &'a [String],
    }

    let mut bytes = Vec::new();
    for layer in layers {
        let aliases: BTreeMap<&str, &str> = layer
            .category_aliases
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        let canonical = Canonical {
            id: &layer.id,
            version: layer.version,
            priority: layer.priority,
            description: &layer.description,
            namespace: layer.namespace.as_deref(),
            types: &layer.types,
            type_patches: &layer.type_patches,
            disabled_types: &layer.disabled_types,
            relationships: &layer.relationships,
            disabled_relationships: &layer.disabled_relationships,
            rules: &layer.rules,
            disabled_rules: &layer.disabled_rules,
            activities: &layer.activities,
            disabled_activities: &layer.disabled_activities,
            category_aliases: aliases,
            disabled_aliases: &layer.disabled_aliases,
        };
        if let Ok(encoded) = serde_json::to_vec(&canonical) {
            bytes.extend_from_slice(&encoded);
        }
    }

    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::{TYPE_CONTRACT, TYPE_ENV_FACT, default_ontology};

    #[test]
    fn project_layer_adds_namespaced_type_and_patches_core_field() {
        let base = default_ontology();
        let mut layer = OntologyLayer::new("project/acme", 4).with_priority(100);
        layer.types.insert(
            "acme.billing.invoice_flow".to_string(),
            OntologyType::new("acme.billing.invoice_flow", "Invoice flow")
                .with_parent(TYPE_CONTRACT)
                .with_description("Invoice generation and settlement procedure"),
        );
        layer.type_patches.insert(
            TYPE_ENV_FACT.to_string(),
            TypePatch {
                ttl_days: Some(45.0),
                ..Default::default()
            },
        );

        let effective = compose_ontology_layers(&base, &[layer]).expect("valid composition");
        assert_eq!(effective.version, 4);
        assert_eq!(
            effective.types["acme.billing.invoice_flow"]
                .parent
                .as_deref(),
            Some(TYPE_CONTRACT)
        );
        assert_eq!(effective.types[TYPE_ENV_FACT].ttl_days, Some(45.0));
        // Unpatched core fields survive.
        assert_eq!(
            effective.types[TYPE_ENV_FACT].scoring,
            base.types[TYPE_ENV_FACT].scoring
        );
    }

    #[test]
    fn composition_is_idempotent_and_disable_removes_type() {
        let base = default_ontology();
        let mut layer = OntologyLayer::new("project/acme", 1);
        layer.disabled_types.push(TYPE_ENV_FACT.to_string());
        let once = compose_ontology_layers(&base, std::slice::from_ref(&layer)).expect("compose");
        assert!(!once.types.contains_key(TYPE_ENV_FACT));
        let twice = compose_ontology_layers(&once, &[layer]).expect("recompose");
        assert_eq!(once.content_hash(), twice.content_hash());
    }

    #[test]
    fn later_priority_wins_on_conflicting_patch() {
        let base = default_ontology();
        let mut low = OntologyLayer::new("pack/low", 1).with_priority(10);
        low.type_patches.insert(
            TYPE_ENV_FACT.to_string(),
            TypePatch {
                ttl_days: Some(30.0),
                ..Default::default()
            },
        );
        let mut high = OntologyLayer::new("project/high", 1).with_priority(100);
        high.type_patches.insert(
            TYPE_ENV_FACT.to_string(),
            TypePatch {
                ttl_days: Some(90.0),
                ..Default::default()
            },
        );

        let effective = compose_ontology_layers(&base, &[high, low]).expect("compose");
        assert_eq!(effective.types[TYPE_ENV_FACT].ttl_days, Some(90.0));
    }

    #[test]
    fn diff_writes_field_patch_not_full_type() {
        let base = default_ontology();
        let mut next = base.clone();
        next.version += 1;
        next.types.get_mut(TYPE_ENV_FACT).unwrap().ttl_days = Some(120.0);

        let layer = diff_ontology_layer(&base, &next);
        assert!(layer.types.is_empty(), "field change must stay a patch");
        assert_eq!(layer.type_patches[TYPE_ENV_FACT].ttl_days, Some(120.0));
        let recomposed = compose_ontology_layers(&base, &[layer]).expect("recompose");
        assert_eq!(recomposed.content_hash(), next.content_hash());
    }

    #[test]
    fn diff_round_trips_added_and_removed_types() {
        let base = default_ontology();
        let mut next = base.clone();
        next.version += 1;
        next.types.remove(TYPE_ENV_FACT);
        next.category_aliases
            .retain(|_, target| target != TYPE_ENV_FACT);
        next.types.insert(
            "acme.flow".to_string(),
            OntologyType::new("acme.flow", "Flow").with_parent(TYPE_CONTRACT),
        );

        let layer = diff_ontology_layer(&base, &next);
        assert!(layer.disabled_types.contains(&TYPE_ENV_FACT.to_string()));
        assert!(layer.types.contains_key("acme.flow"));
        let recomposed = compose_ontology_layers(&base, &[layer]).expect("recompose");
        assert_eq!(recomposed.content_hash(), next.content_hash());
    }

    #[test]
    fn layer_stack_hash_changes_with_content() {
        let mut layer = OntologyLayer::new("project/acme", 1);
        let first = layer_stack_hash(std::slice::from_ref(&layer));
        layer.priority = 100;
        let second = layer_stack_hash(std::slice::from_ref(&layer));
        assert_ne!(first, second);
        assert_eq!(
            first,
            layer_stack_hash(&[OntologyLayer::new("project/acme", 1)])
        );
    }
}
