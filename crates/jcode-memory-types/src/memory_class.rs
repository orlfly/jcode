//! Code-defined memory classes.
//!
//! The ontology store is for *business concepts* summarized while jcode runs:
//! domain entities, contracts, procedures and the like. The classes that
//! govern memory management itself are not business data. They are defined
//! here, in code, so their lifecycle and scoring behavior ships with the
//! runtime and cannot be edited or silently overwritten by an ontology patch.
//!
//! [`Ontology::get_type`](crate::Ontology::get_type) and
//! [`Ontology::all_types`](crate::Ontology::all_types) resolve these classes
//! alongside the persisted business types. Old stores that still persist the
//! core classes are normalized on read by [`strip_core_types`].

use std::collections::{BTreeMap, HashSet};
use std::sync::LazyLock;

use crate::MemoryCategory;
use crate::instance::ExtractionMethod;
use crate::ontology::{
    DEFAULT_ONTOLOGY_ID, DEFAULT_ONTOLOGY_VERSION, DecayPolicy, Ontology, OntologyType,
    ScoringPolicy, TYPE_CONTRACT, TYPE_CORRECTION, TYPE_ENTITY, TYPE_ENV_FACT, TYPE_OBSERVATION,
    TYPE_PREFERENCE, TYPE_PROCESS_RULE,
};

/// Core memory classes in declaration order.
pub const CORE_MEMORY_CLASS_IDS: [&str; 7] = [
    TYPE_ENV_FACT,
    TYPE_CONTRACT,
    TYPE_PROCESS_RULE,
    TYPE_OBSERVATION,
    TYPE_PREFERENCE,
    TYPE_ENTITY,
    TYPE_CORRECTION,
];

static CORE_MEMORY_CLASSES: LazyLock<BTreeMap<String, OntologyType>> = LazyLock::new(|| {
    let mut types = BTreeMap::new();

    // Authority ordering: correction > process_rule > contract > observation
    // > env_fact. `env_fact` and `observation` carry hard TTLs so the store
    // converges instead of growing monotonically.
    types.insert(
        TYPE_ENV_FACT.to_string(),
        OntologyType::new(TYPE_ENV_FACT, "Environment Fact")
            .with_legacy_category("env_fact")
            .with_extraction_method(ExtractionMethod::StructuredMapping)
            .with_extraction_method(ExtractionMethod::RuleExtraction)
            .with_scoring(ScoringPolicy {
                base_score: 8.0,
                ..Default::default()
            })
            .with_decay(DecayPolicy {
                half_life_days: 7.0,
                access_boost: true,
            })
            .with_ttl_days(14.0),
    );
    types.insert(
        TYPE_CONTRACT.to_string(),
        OntologyType::new(TYPE_CONTRACT, "Contract")
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
            }),
    );
    types.insert(
        TYPE_PROCESS_RULE.to_string(),
        OntologyType::new(TYPE_PROCESS_RULE, "Process Rule")
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
            }),
    );
    types.insert(
        TYPE_OBSERVATION.to_string(),
        OntologyType::new(TYPE_OBSERVATION, "Observation")
            .with_legacy_category("observation")
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
                half_life_days: 45.0,
                access_boost: true,
            })
            .with_ttl_days(90.0),
    );
    types.insert(
        TYPE_PREFERENCE.to_string(),
        OntologyType::new(TYPE_PREFERENCE, "Preference")
            .with_legacy_category("preference")
            .with_extraction_method(ExtractionMethod::UserStated)
            .with_extraction_method(ExtractionMethod::LlmExtraction)
            .with_scoring(ScoringPolicy {
                base_score: 20.0,
                ..Default::default()
            })
            .with_decay(DecayPolicy {
                half_life_days: 90.0,
                access_boost: true,
            }),
    );
    types.insert(
        TYPE_ENTITY.to_string(),
        OntologyType::new(TYPE_ENTITY, "Entity")
            .with_legacy_category("entity")
            .with_extraction_method(ExtractionMethod::UserStated)
            .with_extraction_method(ExtractionMethod::StructuredMapping)
            .with_extraction_method(ExtractionMethod::LlmExtraction)
            .with_scoring(ScoringPolicy {
                base_score: 35.0,
                ..Default::default()
            })
            .with_decay(DecayPolicy {
                half_life_days: 180.0,
                access_boost: true,
            }),
    );
    types.insert(
        TYPE_CORRECTION.to_string(),
        OntologyType::new(TYPE_CORRECTION, "Correction")
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
            }),
    );

    types
});

/// Legacy `MemoryCategory` spellings that resolve to a code-defined class.
///
/// This mapping is code, not ontology data. It exists so old rows and old
/// producers keep working; it is not a business-concept catalog.
static CORE_CATEGORY_ALIASES: LazyLock<BTreeMap<String, String>> = LazyLock::new(|| {
    let mut aliases = BTreeMap::new();
    for type_def in CORE_MEMORY_CLASSES.values() {
        if let Some(legacy) = &type_def.legacy_category {
            aliases.insert(legacy.to_ascii_lowercase(), type_def.id.clone());
        }
    }
    // The overloaded v1 `fact` bucket lands in `observation` (the residual
    // class); `Skills`/`skill` lands in `process_rule`; `note` is residual;
    // older serde spellings keep resolving too.
    for (alias, target) in [
        ("fact", TYPE_OBSERVATION),
        ("note", TYPE_OBSERVATION),
        ("skill", TYPE_PROCESS_RULE),
        ("skills", TYPE_PROCESS_RULE),
        ("medium", TYPE_OBSERVATION),
        ("envfact", TYPE_ENV_FACT),
        ("processrule", TYPE_PROCESS_RULE),
    ] {
        aliases.insert(alias.to_string(), target.to_string());
    }
    aliases
});

/// All code-defined memory classes, keyed by type id.
pub fn core_memory_classes() -> &'static BTreeMap<String, OntologyType> {
    &CORE_MEMORY_CLASSES
}

/// Look up one code-defined memory class.
pub fn core_memory_class(id: &str) -> Option<&'static OntologyType> {
    CORE_MEMORY_CLASSES.get(id)
}

/// Whether `id` names a code-defined memory class.
pub fn is_core_memory_class(id: &str) -> bool {
    CORE_MEMORY_CLASSES.contains_key(id)
}

/// Code-defined mapping from a legacy category to a core class.
pub fn core_class_for_category(category: &MemoryCategory) -> Option<&'static OntologyType> {
    let key = match category {
        MemoryCategory::Fact => "fact".to_string(),
        MemoryCategory::EnvFact => "env_fact".to_string(),
        MemoryCategory::Contract => "contract".to_string(),
        MemoryCategory::ProcessRule => "process_rule".to_string(),
        MemoryCategory::Observation => "observation".to_string(),
        MemoryCategory::Preference => "preference".to_string(),
        MemoryCategory::Entity => "entity".to_string(),
        MemoryCategory::Correction => "correction".to_string(),
        MemoryCategory::Custom(custom) => custom.to_ascii_lowercase(),
    };
    CORE_CATEGORY_ALIASES
        .get(&key)
        .and_then(|id| core_memory_class(id))
}

/// Whether `category` is owned by a code-defined memory class.
pub fn is_core_category(category: &str) -> bool {
    CORE_CATEGORY_ALIASES.contains_key(&category.to_ascii_lowercase())
}

/// Remove core classes and their aliases from a persisted ontology.
///
/// Returns `true` when the value changed. A default ontology stamped with an
/// older schema version is bumped to the current version. Custom ontologies
/// keep their own version.
pub fn strip_core_types(ontology: &mut Ontology) -> bool {
    let core_ids: HashSet<&str> = CORE_MEMORY_CLASSES.keys().map(String::as_str).collect();
    let mut changed = false;

    let before = ontology.types.len();
    ontology.types.retain(|id, type_def| {
        !core_ids.contains(id.as_str())
            && !is_core_category(id)
            && type_def
                .legacy_category
                .as_deref()
                .is_none_or(|legacy| !is_core_category(legacy))
    });
    if ontology.types.len() != before {
        changed = true;
    }

    let before = ontology.category_aliases.len();
    ontology.category_aliases.retain(|category, target| {
        !is_core_category(category) && !core_ids.contains(target.as_str())
    });
    if ontology.category_aliases.len() != before {
        changed = true;
    }

    if ontology.id == DEFAULT_ONTOLOGY_ID && ontology.version < DEFAULT_ONTOLOGY_VERSION {
        ontology.version = DEFAULT_ONTOLOGY_VERSION;
        changed = true;
    }

    changed
}
