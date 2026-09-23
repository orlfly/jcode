//! Rule engine for the ontology.
//!
//! At runtime, [`MemoryManager::remember_project`] and friends dispatch an
//! `event` (e.g. `event.remember`, `event.upsert`, `event.dedup`) into the
//! engine along with an [`RuleContext`].  The engine walks the active
//! ontology's rules, applies [`Condition`]s, and produces an ordered list of
//! [`Effect`]s to execute against the candidate instance.
//!
//! The engine is intentionally pure: it does not mutate the graph.  Callers
//! translate effects into mutations on the [`MemoryEntry`] /
//! [`MemoryGraph`].  This keeps the engine testable and lets the runtime
//! decide what to do when an effect produces an error (e.g. suppress, log,
//! retry).

use std::collections::HashSet;
use std::sync::Arc;

use crate::instance::MemoryStatus;
use crate::ontology::{Condition, Effect, Ontology, OntologyType, Rule};
use crate::{MemoryEntry, MemoryGraph};

/// The candidate + metadata passed to the rule engine for a single event.
#[derive(Debug, Clone)]
pub struct RuleContext {
    /// The candidate memory entry (already constructed by the caller).
    pub entry: MemoryEntry,
    /// The ontology type id the candidate resolves to.
    pub type_id: String,
    /// Optional existing instance id (when the rule fires on update/upsert).
    pub existing_id: Option<String>,
    /// Similarity to the matching existing instance (0.0..=1.0).
    pub similarity: Option<f32>,
    /// The graph used for edge / status inspection.
    pub graph: Arc<MemoryGraph>,
    /// The ontology to evaluate against.  Held as `Arc<Ontology>` so the
    /// runtime registry can hand out the ontology without constraining the
    /// call site's lifetimes.
    pub ontology: Arc<Ontology>,
    /// The label of the source that triggered this event (e.g. "dedup",
    /// "extraction", "tool").  Surfaced in `Effect::Reinforce`.
    pub source_label: String,
    /// The event being dispatched.
    pub event: String,
    /// Target storage scope for the write ("project" or "global").  Lets
    /// scope-gating rules fire only on global writes; empty means unknown.
    pub scope: String,
}

impl RuleContext {
    pub fn type_def(&self) -> Option<&OntologyType> {
        self.ontology.get_type(&self.type_id)
    }

    /// Construct a minimal context for testing or scripted callers.
    pub fn minimal(entry: MemoryEntry, event: impl Into<String>) -> Self {
        let ontology = crate::ontology::default_ontology();
        let type_id = match &entry.category {
            crate::MemoryCategory::Fact => crate::ontology::TYPE_FACT,
            crate::MemoryCategory::Preference => crate::ontology::TYPE_PREFERENCE,
            crate::MemoryCategory::Entity => crate::ontology::TYPE_ENTITY,
            crate::MemoryCategory::Correction => crate::ontology::TYPE_CORRECTION,
            crate::MemoryCategory::Custom(label) => match label.as_str() {
                "goal" => crate::ontology::TYPE_GOAL,
                "note" => crate::ontology::TYPE_NOTE,
                "skill" | "Skills" => crate::ontology::TYPE_SKILL,
                _ => "custom",
            },
        };
        Self {
            entry,
            type_id: type_id.to_string(),
            existing_id: None,
            similarity: None,
            graph: Arc::new(MemoryGraph::new()),
            ontology: Arc::new(ontology),
            source_label: "minimal".to_string(),
            event: event.into(),
            scope: String::new(),
        }
    }
}

/// Result of dispatching a rule chain.
#[derive(Debug, Default, Clone)]
pub struct RulePlan {
    pub effects: Vec<Effect>,
    pub applied_rules: Vec<String>,
    pub skipped: Vec<RuleSkip>,
}

#[derive(Debug, Clone)]
pub struct RuleSkip {
    pub rule_id: String,
    pub reason: String,
}

impl RulePlan {
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }
}

/// Dispatch an event through the rule engine.
///
/// The engine:
/// 1. Looks up all rules that target `event`.
/// 2. Filters by `applies_to_type` (if non-empty).
/// 3. Evaluates each `Condition` against the `RuleContext`.
/// 4. Collects `Effect`s into a `RulePlan` in declared order.
/// 5. Applies ontology-level sanity guards (e.g. lifecycle transition
///    legality).  Illegal transitions become `RuleSkip`s, not errors.
pub fn dispatch_event(ctx: &RuleContext) -> RulePlan {
    let mut plan = RulePlan::default();

    for rule in ctx.ontology.rules_for_event(&ctx.event) {
        if !rule.applies_to_type(&ctx.type_id) {
            plan.skipped.push(RuleSkip {
                rule_id: rule.id.clone(),
                reason: format!("type '{}' not in applies_to", ctx.type_id),
            });
            continue;
        }

        let mut ok = true;
        for cond in &rule.conditions {
            if !evaluate_condition(cond, ctx) {
                plan.skipped.push(RuleSkip {
                    rule_id: rule.id.clone(),
                    reason: format!("condition {:?} failed", cond),
                });
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }

        for effect in &rule.effects {
            if let Some(sanitized) = sanitize_effect(effect, ctx) {
                plan.effects.push(sanitized);
            } else {
                plan.skipped.push(RuleSkip {
                    rule_id: rule.id.clone(),
                    reason: format!("effect {:?} rejected by guard", effect),
                });
            }
        }
        plan.applied_rules.push(rule.id.clone());
    }

    plan
}

/// Evaluate a single `Condition` against the `RuleContext`.
pub fn evaluate_condition(cond: &Condition, ctx: &RuleContext) -> bool {
    match cond {
        Condition::TypeIs { values } => values.iter().any(|v| *v == ctx.type_id),
        Condition::HasExistingId => ctx
            .existing_id
            .as_ref()
            .map(|id| ctx.graph.memories.contains_key(id))
            .unwrap_or(false),
        Condition::NoExistingId => ctx.existing_id.is_none()
            || !ctx.graph.memories.contains_key(ctx.existing_id.as_deref().unwrap_or("")),
        Condition::HasContent => !ctx.entry.content.trim().is_empty(),
        Condition::HasEmbedding => ctx.entry.embedding.is_some(),
        Condition::ProvenanceAtLeast { min_rank } => ctx
            .entry
            .effective_provenance()
            .method
            .authority_rank()
            >= *min_rank,
        Condition::ConfidenceAtLeast { min } => ctx.entry.confidence >= *min,
        Condition::SimilarityAtLeast { threshold } => ctx
            .similarity
            .map(|s| s >= *threshold)
            .unwrap_or(false),
        Condition::Custom { expression } => match expression.as_str() {
            // Global-scope gate: only generalized, cross-project content may
            // be written to global memory. See `is_generalized_content`.
            "global_scope_generalized" => {
                ctx.scope == "global" && is_generalized_content(&ctx.entry.content)
                    || ctx.scope != "global"
            }
            _ => true,
        },
    }
}

/// Apply ontology-level guards to a single effect.  Returns `None` if the
/// effect should be skipped (e.g. an illegal lifecycle transition).
fn sanitize_effect(effect: &Effect, ctx: &RuleContext) -> Option<Effect> {
    match effect {
        Effect::TransitionLifecycle { to, .. } => {
            let current = ctx.entry.lifecycle.status;
            if let Some(type_def) = ctx.type_def() {
                if !type_def.lifecycle.can_transition(current, *to) {
                    return None;
                }
            } else {
                // Unknown type — fall back to the global lifecycle policy.
                let policy = crate::instance::default_lifecycle_transition_table();
                if !policy.iter().any(|t| t.0 == current && t.1 == *to) && current != *to {
                    return None;
                }
            }
            Some(effect.clone())
        }
        _ => Some(effect.clone()),
    }
}

/// Heuristic gate for global-scope memory writes.
///
/// Global memory is injected into every project's sessions, so it must be
/// conceptual / generalized / cross-project knowledge (processing patterns,
/// general rules, workflow preferences). Content that binds to one specific
/// environment - concrete hosts, IPs, ports, absolute paths, task ids, MR
/// numbers, container or registry references - belongs in project scope, not
/// global. The detector is intentionally conservative: it counts concrete
/// identifiers and flags the content as environment-specific only when
/// several categories appear. Credential-shaped content is always rejected.
pub fn is_generalized_content(content: &str) -> bool {
    let text = content.trim();
    if text.is_empty() {
        return false;
    }

    // Credential-shaped content is never acceptable in global scope.
    let credential_like = ["api_key", "apiKey", "API_KEY", "Bearer ", "password"]
        .iter()
        .any(|needle| text.contains(needle));
    if credential_like {
        return false;
    }

    let mut specificity = 0usize;

    // Private-network addresses.
    let has_private_ip = ["192.168.", "10.0.", "127.0.0.1"]
        .iter()
        .any(|needle| text.contains(needle));
    if has_private_ip {
        specificity += 2;
    }

    // Absolute filesystem paths.
    if ["/opt/", "/home/", "C:\\"]
        .iter()
        .any(|needle| text.contains(needle))
    {
        specificity += 1;
    }

    // Ticket / task ids: KIA-123, !19 (MR), #48330.
    let digits_after = |prefix: &str| {
        text.match_indices(prefix)
            .any(|(i, _)| text[i + prefix.len()..].starts_with(|c: char| c.is_ascii_digit()))
    };
    if digits_after("KIA-") || digits_after("!") || digits_after("#") {
        specificity += 1;
    }

    // Deploy artefacts: container / image / registry references.
    if ["容器", "镜像", "harbor", "docker ", "部署机"]
        .iter()
        .any(|needle| text.contains(needle))
    {
        specificity += 1;
    }

    // Explicit host:port endpoints.
    if ["localhost:", ":1337", ".cn:"]
        .iter()
        .any(|needle| text.contains(needle))
    {
        specificity += 1;
    }

    specificity < 2
}

/// Apply a `RulePlan` directly to a `MemoryEntry` in place.  Effects that
/// refer to graph-level changes (link, supersede, derive_from) are not
/// executed here — callers should call [`apply_graph_effects`] for those.
pub fn apply_entry_effects(plan: &RulePlan, entry: &mut MemoryEntry) -> Vec<Effect> {
    let mut deferred = Vec::new();
    for effect in &plan.effects {
        match effect {
            Effect::SetField { property, value } => {
                if let Some(value) = value {
                    set_field(entry, property, value);
                }
            }
            Effect::AddTag { tag } => {
                if tag != "__self__" && !entry.tags.iter().any(|t| t == tag) {
                    entry.tags.push(tag.clone());
                }
            }
            Effect::RemoveTag { tag } => {
                if tag != "__self__" {
                    entry.tags.retain(|t| t != tag);
                }
            }
            Effect::TransitionLifecycle { to, reason } => {
                if entry.lifecycle.transition_to(*to) {
                    if let Some(reason) = reason {
                        entry.lifecycle.deprecated_reason = Some(reason.clone());
                    }
                    if matches!(to, MemoryStatus::Archived) {
                        entry.active = false;
                    }
                }
            }
            Effect::Reinforce { source_label } => {
                entry.reinforce(source_label, 0);
            }
            Effect::Supersede { .. } => {
                deferred.push(effect.clone());
            }
            Effect::MarkContradiction { .. } => {
                deferred.push(effect.clone());
            }
            Effect::DeriveFrom { .. } => {
                deferred.push(effect.clone());
            }
            Effect::Link { .. } => {
                deferred.push(effect.clone());
            }
            Effect::SetProvenance {
                source,
                method,
                confidence,
            } => {
                if entry.provenance.is_none() {
                    entry.provenance = Some(
                        crate::instance::ProvenanceRecord::new(source.clone(), *method)
                            .with_confidence(*confidence),
                    );
                }
            }
            Effect::Touch => {
                entry.touch();
            }
            Effect::Abort { .. } => {
                // Abort is purely declarative; the runtime short-circuits on
                // it before calling `apply_entry_effects`.
            }
        }
    }
    entry.refresh_search_text();
    deferred
}

/// Apply graph-level effects to a `MemoryGraph`.  Returns the effects that
/// were successfully applied.  Used by `MemoryManager` after
/// `apply_entry_effects`.
pub fn apply_graph_effects(
    plan: &RulePlan,
    new_id: &str,
    graph: &mut MemoryGraph,
) -> Vec<Effect> {
    let mut applied = Vec::new();
    for effect in &plan.effects {
        match effect {
            Effect::MarkContradiction { other_id } => {
                if other_id != "__self__" && graph.memories.contains_key(other_id) {
                    graph.mark_contradiction(new_id, other_id);
                    applied.push(effect.clone());
                }
            }
            Effect::Supersede { old_id, new_id } => {
                if old_id != "__self__" && new_id != "__self__" && graph.memories.contains_key(old_id) {
                    graph.supersede(new_id, old_id);
                    applied.push(effect.clone());
                }
            }
            Effect::DeriveFrom { source_id, relation_kind } => {
                if graph.memories.contains_key(source_id) {
                    let edge_kind = match relation_kind.as_str() {
                        "relates_to" => crate::graph::EdgeKind::RelatesTo { weight: 0.5 },
                        "supersedes" => crate::graph::EdgeKind::Supersedes,
                        "contradicts" => crate::graph::EdgeKind::Contradicts,
                        _ => crate::graph::EdgeKind::DerivedFrom,
                    };
                    graph.add_edge(source_id, new_id, edge_kind);
                    applied.push(effect.clone());
                }
            }
            Effect::Link {
                other_id,
                relationship,
                weight,
            } => {
                if other_id != "__self__" && graph.memories.contains_key(other_id) {
                    let edge_kind = match relationship.as_str() {
                        "relates_to" => crate::graph::EdgeKind::RelatesTo {
                            weight: weight.unwrap_or(0.5),
                        },
                        "supersedes" => crate::graph::EdgeKind::Supersedes,
                        "contradicts" => crate::graph::EdgeKind::Contradicts,
                        "derived_from" => crate::graph::EdgeKind::DerivedFrom,
                        _ => crate::graph::EdgeKind::RelatesTo {
                            weight: weight.unwrap_or(0.5),
                        },
                    };
                    graph.add_edge(new_id, other_id, edge_kind);
                    applied.push(effect.clone());
                }
            }
            _ => {}
        }
    }
    applied
}

fn set_field(entry: &mut MemoryEntry, property: &str, value: &serde_json::Value) {
    match property {
        "content" => {
            if let Some(s) = value.as_str() {
                entry.content = s.to_string();
            }
        }
        "confidence" => {
            if let Some(n) = value.as_f64() {
                entry.confidence = n as f32;
            }
        }
        "trust" => {
            if let Some(s) = value.as_str() {
                entry.trust = match s {
                    "high" => crate::TrustLevel::High,
                    "low" => crate::TrustLevel::Low,
                    _ => crate::TrustLevel::Medium,
                };
            }
        }
        "active" => {
            if let Some(b) = value.as_bool() {
                entry.active = b;
            }
        }
        "critical" => {
            if let Some(b) = value.as_bool() {
                entry.critical = b;
            }
        }
        "tags" => {
            if let Some(arr) = value.as_array() {
                entry.tags = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
            }
        }
        "source" => {
            if let Some(s) = value.as_str() {
                entry.source = Some(s.to_string());
            }
        }
        _ => {
            // Unrecognized field — no-op.  Surfacing this through a log
            // would be useful but is intentionally left to the runtime.
        }
    }
}

/// Find the rule that should handle a given event when the legacy
/// `MemoryCategory` API is used.  Used by the run-time fallback path.
pub fn rule_for_event<'a>(ontology: &'a Ontology, event: &str) -> Option<&'a Rule> {
    ontology.rules_for_event(event).next()
}

/// Return the set of `[Effect]` kinds that have been declared in rules for
/// the given event.  Useful for assertions in tests.  Returns `Vec<&'static
/// str>` rather than `HashSet<String>` so callers can use them as match arms
/// without allocations.
pub fn declared_effect_kinds(ontology: &Ontology, event: &str) -> Vec<&'static str> {
    let mut kinds = HashSet::new();
    for r in ontology.rules_for_event(event) {
        for e in &r.effects {
            kinds.insert(effect_kind_name(e));
        }
    }
    kinds.into_iter().collect()
}

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

#[cfg(test)]
mod generality_gate_tests {
    use super::is_generalized_content;

    #[test]
    fn rejects_credential_shaped_content() {
        assert!(!is_generalized_content(
            "Kaneo devops API key is XFsuku and must be sent as Bearer token"
        ));
        assert!(!is_generalized_content("the password is hunter2"));
    }

    #[test]
    fn rejects_environment_specific_content() {
        assert!(!is_generalized_content(
            "部署机 = 192.168.6.33, SSH root 免密, live 容器 kiagents-v2-chat-node"
        ));
        assert!(!is_generalized_content(
            "chat-service 部署在 /opt/ki-agent-v2/infra/nginx/, KIA-243 已修复"
        ));
        assert!(!is_generalized_content(
            "kaneo 实例在 localhost:1337, DB 容器 kaneo-postgres, 镜像 2.0.2-kia211"
        ));
    }

    #[test]
    fn accepts_generalized_processing_rules() {
        assert!(is_generalized_content(
            "教训：绝不在共享宿主机用 xargs 对容器执行删除——必须先列明确认再逐个操作"
        ));
        assert!(is_generalized_content(
            "更新记忆前必须核对事实准确性：与已提交代码矛盾时应先修正旧条目"
        ));
        assert!(is_generalized_content(
            "Spring shadow controller 完全替换原版映射，绝不能只新增端点"
        ));
    }

    #[test]
    fn rejects_empty_content() {
        assert!(!is_generalized_content("   "));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::{default_ontology, EVENT_REMEMBER, TYPE_FACT, TYPE_GOAL};
    use crate::{MemoryCategory, MemoryEntry, MemoryGraph};
    use std::sync::Arc;

    fn ctx(event: &'static str, type_id: &'static str) -> RuleContext {
        let ontology = default_ontology();
        RuleContext {
            entry: MemoryEntry::new(MemoryCategory::Fact, "hello"),
            type_id: type_id.to_string(),
            existing_id: None,
            similarity: None,
            graph: Arc::new(MemoryGraph::new()),
            ontology: Arc::new(ontology),
            source_label: "test".to_string(),
            event: event.to_string(),
            scope: String::new(),
        }
    }

    #[test]
    fn empty_content_skips_remember() {
        let mut ctx = ctx(EVENT_REMEMBER, TYPE_FACT);
        ctx.entry.content = "   ".to_string();
        let plan = dispatch_event(&ctx);
        assert!(plan.effects.is_empty());
    }

    #[test]
    fn fact_remember_adds_provenance_and_touch() {
        let ctx = ctx(EVENT_REMEMBER, TYPE_FACT);
        let plan = dispatch_event(&ctx);
        assert!(plan.effects.iter().any(|e| matches!(e, Effect::SetProvenance { .. })));
        assert!(plan.effects.iter().any(|e| matches!(e, Effect::Touch)));
    }

    #[test]
    fn rule_applies_to_type_filters() {
        // TYPE_GOAL rule has no applies_to restriction, so it should match.
        let ctx = ctx(EVENT_REMEMBER, TYPE_GOAL);
        let plan = dispatch_event(&ctx);
        assert!(!plan.applied_rules.is_empty());
    }

    #[test]
    fn apply_entry_effects_touches_entry() {
        let ctx = ctx(EVENT_REMEMBER, TYPE_FACT);
        let mut entry = ctx.entry.clone();
        let plan = dispatch_event(&ctx);
        let before = entry.updated_at;
        let _ = apply_entry_effects(&plan, &mut entry);
        assert!(entry.updated_at >= before);
    }

    #[test]
    fn effects_kinds_for_event() {
        let onto = default_ontology();
        let kinds = declared_effect_kinds(&onto, EVENT_REMEMBER);
        assert!(kinds.contains(&"set_provenance"));
        assert!(kinds.contains(&"touch"));
    }
}
