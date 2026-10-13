use super::*;
use crate::instance::{ExtractionMethod, ProvenanceRecord};
use crate::graph::MemoryGraph;
use crate::{MemoryCategory, MemoryEntry};

fn entry(category: MemoryCategory, content: &str) -> MemoryEntry {
    MemoryEntry::new(category, content)
}

fn with_method(mut entry: MemoryEntry, method: ExtractionMethod) -> MemoryEntry {
    entry.provenance = Some(ProvenanceRecord::new("test", method));
    entry
}

fn graph_with(entries: Vec<MemoryEntry>) -> MemoryGraph {
    let mut graph = MemoryGraph::new();
    for e in entries {
        graph.add_memory(e);
    }
    graph
}

// ---------------------------------------------------------------------------
// similarity
// ---------------------------------------------------------------------------

#[test]
fn identical_content_is_identical_similarity() {
    let a = entry(MemoryCategory::Fact, "Kaneo claim endpoint is POST /api/task/claim/{id}");
    let b = entry(MemoryCategory::Fact, "Kaneo claim endpoint is POST /api/task/claim/{id}");
    assert!((lexical_similarity(&a, &b) - 1.0).abs() < f32::EPSILON);
}

#[test]
fn unrelated_content_has_low_similarity() {
    let a = entry(MemoryCategory::Fact, "Kaneo claim endpoint is POST /api/task/claim/{id}");
    let b = entry(MemoryCategory::Fact, "Kitty keyboard protocol needs enable_keyboard_enhancement");
    assert!(lexical_similarity(&a, &b) < RELATE_AT);
}

#[test]
fn paraphrase_is_similar_but_not_identical() {
    let a = entry(
        MemoryCategory::Fact,
        "Kaneo claim endpoint: POST /api/task/claim/{taskId} with a full internal id",
    );
    let b = entry(
        MemoryCategory::Fact,
        "Kaneo claim endpoint is POST /api/task/claim/{taskId}; truncated ids fail silently",
    );
    let sim = lexical_similarity(&a, &b);
    assert!((SUPERSEDE_AT..REINFORCE_AT).contains(&sim), "sim={sim}");
}

// ---------------------------------------------------------------------------
// reconcile actions
// ---------------------------------------------------------------------------

#[test]
fn empty_graph_inserts() {
    let graph = MemoryGraph::new();
    let candidate = entry(MemoryCategory::Fact, "Brand new observation about the build");
    let outcome = reconcile(&candidate, &graph);
    assert_eq!(outcome.action, ReconcileAction::Insert);
}

#[test]
fn duplicate_reinforces() {
    let existing = entry(MemoryCategory::Fact, "Kaneo API requires the x-team-id header");
    let id = existing.id.clone();
    let graph = graph_with(vec![existing]);
    let candidate = entry(MemoryCategory::Fact, "Kaneo API requires the x-team-id header");
    let outcome = reconcile(&candidate, &graph);
    assert_eq!(
        outcome.action,
        ReconcileAction::Reinforce {
            existing_id: id.clone()
        }
    );
    assert!(outcome.similarity >= REINFORCE_AT);
    assert_eq!(outcome.related.len(), 1);
}

#[test]
fn same_subject_update_supersedes_when_content_differs() {
    // Same anchor, materially different content: the newer description is the
    // current truth about that subject, so it replaces the stored one instead
    // of being folded into it.
    let mut existing = entry(
        MemoryCategory::Fact,
        "The API is reachable at /api/task/claim and needs a bearer token",
    );
    existing.subject_key = Some("api:/api/task/claim".to_string());
    let id = existing.id.clone();
    let graph = graph_with(vec![existing]);

    let mut candidate = entry(
        MemoryCategory::Fact,
        "Claiming uses a POST to the claim route with a team identifier",
    );
    candidate.subject_key = Some("api:/api/task/claim".to_string());

    let outcome = reconcile(&candidate, &graph);
    assert_eq!(
        outcome.action,
        ReconcileAction::Supersede {
            existing_id: id.clone()
        },
        "sim={}",
        outcome.similarity
    );
}

#[test]
fn same_subject_restatement_reinforces() {
    let mut existing = entry(
        MemoryCategory::Fact,
        "Kaneo claim endpoint requires a full internal task id and the team id",
    );
    existing.subject_key = Some("api:/api/task/claim".to_string());
    let id = existing.id.clone();
    let graph = graph_with(vec![existing]);

    let mut candidate = entry(
        MemoryCategory::Fact,
        "Kaneo claim endpoint requires a full internal task id and the team id",
    );
    candidate.subject_key = Some("api:/api/task/claim".to_string());

    let outcome = reconcile(&candidate, &graph);
    assert_eq!(
        outcome.action,
        ReconcileAction::Reinforce {
            existing_id: id.clone()
        }
    );
}

#[test]
fn correction_supersedes_a_related_fact() {
    let existing = with_method(
        entry(
            MemoryCategory::Fact,
            "Kaneo claim endpoint: POST /api/task/claim/{taskId} with a full internal id",
        ),
        ExtractionMethod::LlmExtraction,
    );
    let id = existing.id.clone();
    let graph = graph_with(vec![existing]);

    let candidate = with_method(
        entry(
            MemoryCategory::Correction,
            "Kaneo claim endpoint is POST /api/task/claim/{taskId}; truncated ids fail silently",
        ),
        ExtractionMethod::LlmExtraction,
    );
    let outcome = reconcile(&candidate, &graph);
    assert_eq!(
        outcome.action,
        ReconcileAction::Supersede {
            existing_id: id.clone()
        },
        "sim={}",
        outcome.similarity
    );
}

#[test]
fn loose_fact_paraphrase_links_instead_of_retiring_the_original() {
    let existing = entry(
        MemoryCategory::Fact,
        "The deploy script copies the bundle to the host and restarts the service unit",
    );
    let graph = graph_with(vec![existing]);

    let candidate = entry(
        MemoryCategory::Fact,
        "The release script copies the build output to the host before restarting the unit",
    );
    let outcome = reconcile(&candidate, &graph);
    assert_eq!(
        outcome.action.label(),
        "relate",
        "a 0.5-0.65 fact overlap must not auto-retire the stored instance (sim={})",
        outcome.similarity
    );
}

#[test]
fn spelling_variants_of_a_code_identifier_are_not_merged() {
    // `foo/bar` and `foo-bar` are different spellings of different things; a
    // token-level comparison that dropped the separator would merge them.
    let existing = entry(MemoryCategory::Fact, "path: foo/bar");
    let graph = graph_with(vec![existing]);
    let candidate = entry(MemoryCategory::Fact, "path: foo-bar");
    let outcome = reconcile(&candidate, &graph);
    assert_ne!(
        outcome.action.label(),
        "reinforce",
        "separator-only differences must stay distinct (sim={})",
        outcome.similarity
    );
}

#[test]
fn same_subject_update_supersedes() {
    let mut existing = entry(MemoryCategory::Fact, "The API base is http://localhost:1337");
    existing.subject_key = Some("host:localhost:1337".to_string());
    let id = existing.id.clone();
    let graph = graph_with(vec![existing]);

    let mut candidate = entry(MemoryCategory::Fact, "The API now runs on http://localhost:1337");
    candidate.subject_key = Some("host:localhost:1337".to_string());

    let outcome = reconcile(&candidate, &graph);
    assert_eq!(
        outcome.action,
        ReconcileAction::Supersede {
            existing_id: id.clone()
        },
        "sim={}",
        outcome.similarity
    );
}

#[test]
fn weaker_source_relates_instead_of_superseding_a_user_statement() {
    let existing = with_method(
        entry(
            MemoryCategory::Fact,
            "Kaneo claim endpoint: POST /api/task/claim/{taskId} with a full internal id",
        ),
        ExtractionMethod::UserStated,
    );
    let id = existing.id.clone();
    let graph = graph_with(vec![existing]);

    let candidate = with_method(
        entry(
            MemoryCategory::Fact,
            "Kaneo claim endpoint is POST /api/task/claim/{taskId}; truncated ids fail silently",
        ),
        ExtractionMethod::LlmExtraction,
    );
    let outcome = reconcile(&candidate, &graph);
    assert_eq!(
        outcome.action,
        ReconcileAction::Relate {
            existing_id: id.clone()
        },
        "an LLM paraphrase must not replace a user-stated memory (sim={})",
        outcome.similarity
    );
}

#[test]
fn negation_asymmetry_contradicts() {
    let existing = entry(
        MemoryCategory::Fact,
        "CI always rebuilds the binary before running the smoke suite",
    );
    let id = existing.id.clone();
    let graph = graph_with(vec![existing]);

    let candidate = entry(
        MemoryCategory::Fact,
        "CI does not rebuild the binary before running the smoke suite",
    );
    let outcome = reconcile(&candidate, &graph);
    assert_eq!(
        outcome.action,
        ReconcileAction::Contradict {
            existing_id: id.clone()
        }
    );
}

#[test]
fn same_subject_with_different_value_contradicts() {
    let mut existing = entry(MemoryCategory::Fact, "The API listens on port 6432 for requests");
    existing.subject_key = Some("host:localhost:6432".to_string());
    let graph = graph_with(vec![existing]);

    let mut candidate = entry(MemoryCategory::Fact, "The API listens on port 5432 for requests");
    candidate.subject_key = Some("host:localhost:6432".to_string());

    let outcome = reconcile(&candidate, &graph);
    assert_eq!(outcome.action.label(), "contradict");
}

#[test]
fn link_nodes_are_not_reconciled() {
    let mut skill_link = entry(
        MemoryCategory::Custom("Skills".to_string()),
        "Use skill `/optimization` when tuning latency.",
    );
    skill_link.id = "skill:optimization".to_string();
    let graph = graph_with(vec![skill_link.clone()]);

    let outcome = reconcile(&skill_link, &graph);
    assert_eq!(outcome.action, ReconcileAction::Insert);
    assert!(outcome.related.is_empty());
}

// ---------------------------------------------------------------------------
// noise gate
// ---------------------------------------------------------------------------

#[test]
fn noise_gate_rejects_tool_payloads() {
    assert!(noise_reason("#Tool: bash input: {\"command\":\"curl http://x\"}").is_some());
    assert!(noise_reason("{\"command\": \"ls -la\", \"timeout\": 10}").is_some());
}

#[test]
fn noise_gate_rejects_stubs_and_identifiers() {
    assert!(noise_reason("mem_123").is_some());
    assert!(noise_reason("7a4573fc2").is_some());
    assert!(noise_reason("---").is_some());
    assert!(noise_reason("").is_none(), "empty is validation's job (E20)");
}

#[test]
fn noise_gate_admits_real_knowledge() {
    assert!(noise_reason("Kaneo claim endpoint is POST /api/task/claim/{id}").is_none());
    assert!(noise_reason("当任务完成时，必须记录验收证据").is_none());
}

#[test]
fn noise_gate_agrees_with_the_ontology_condition() {
    // `Condition::Custom { expression: "content_is_noise" }` must resolve to
    // the same verdict the write path uses.
    let noisy = entry(MemoryCategory::Fact, "{\"command\": \"ls\"}");
    assert!(noise_reason(&noisy.content).is_some());
    let ctx = crate::rule_engine::RuleContext::minimal(noisy, crate::ontology::EVENT_REMEMBER);
    let plan = crate::rule_engine::dispatch_event(&ctx);
    assert!(
        plan.effects
            .iter()
            .any(|e| matches!(e, crate::ontology::Effect::Abort { .. })),
        "ontology reject rule must abort noise"
    );

    let clean = entry(
        MemoryCategory::Fact,
        "Kaneo claim endpoint is POST /api/task/claim/{id}",
    );
    let ctx = crate::rule_engine::RuleContext::minimal(clean, crate::ontology::EVENT_REMEMBER);
    let plan = crate::rule_engine::dispatch_event(&ctx);
    assert!(
        !plan
            .effects
            .iter()
            .any(|e| matches!(e, crate::ontology::Effect::Abort { .. })),
        "clean content must not be aborted"
    );
}

// ---------------------------------------------------------------------------
// subject keys
// ---------------------------------------------------------------------------

#[test]
fn subject_key_prefers_api_route() {
    assert_eq!(
        derive_subject_key("Kaneo claim: POST /api/task/claim/{taskId} requires teamId"),
        Some("api:/api/task/claim/{taskid}".to_string())
    );
}

#[test]
fn subject_key_extracts_host_port() {
    assert_eq!(
        derive_subject_key("The Kaneo dev server runs at localhost:1337 for local runs"),
        Some("host:localhost:1337".to_string())
    );
}

#[test]
fn subject_key_extracts_file_paths() {
    assert_eq!(
        derive_subject_key("The daemon lives in crates/jcode-base/src/memory.rs"),
        Some("file:crates/jcode-base/src/memory.rs".to_string())
    );
}

#[test]
fn subject_key_extracts_task_and_mr_ids() {
    assert_eq!(
        derive_subject_key("Shadow drill results were backfilled for task #157 条件③"),
        Some("task:#157".to_string())
    );
    assert_eq!(
        derive_subject_key("Authorization split landed in MR !358 across batches"),
        Some("mr:!358".to_string())
    );
}

#[test]
fn subject_key_is_absent_for_pure_prose() {
    assert_eq!(
        derive_subject_key("User prefers responses in Chinese and short summaries"),
        None
    );
}

// ---------------------------------------------------------------------------
// tag vocabulary
// ---------------------------------------------------------------------------

#[test]
fn tag_vocabulary_drops_transcript_roles() {
    let cleaned = sanitize_tags(["assistant", "Kaneo", "tool", "api", "API"]);
    assert_eq!(cleaned, vec!["kaneo".to_string(), "api".to_string()]);
}

#[test]
fn tag_vocabulary_keeps_namespaced_tags() {
    let cleaned = sanitize_tags(["goal:kia203-followup-1", "goal_status:active", "skill:deploy"]);
    assert_eq!(cleaned.len(), 3);
    assert!(cleaned.contains(&"skill:deploy".to_string()));
}

// ---------------------------------------------------------------------------
// prepare_for_write
// ---------------------------------------------------------------------------

#[test]
fn prepare_for_write_stamps_subject_and_cleans_tags() {
    let prepared = prepare_for_write(
        entry(
            MemoryCategory::Fact,
            "Kaneo claim endpoint is POST /api/task/claim/{id}",
        )
        .with_tags(vec!["assistant".to_string(), "Kaneo".to_string()]),
    );
    assert_eq!(prepared.subject_key.as_deref(), Some("api:/api/task/claim/{id}"));
    assert_eq!(prepared.tags, vec!["kaneo".to_string()]);
    assert!(prepared.search_text.contains("kaneo"));
}

#[test]
fn prepare_for_write_is_idempotent() {
    let once = prepare_for_write(entry(
        MemoryCategory::Fact,
        "Kaneo claim endpoint is POST /api/task/claim/{id}",
    ));
    let twice = prepare_for_write(once.clone());
    assert_eq!(once.subject_key, twice.subject_key);
    assert_eq!(once.tags, twice.tags);
    assert_eq!(once.category, twice.category);
}

#[test]
fn prepare_for_write_demotes_identityless_entities() {
    let demoted = prepare_for_write(entry(
        MemoryCategory::Entity,
        "ki-agent-v2 project env: GitLab kingsware group and internal hosts",
    ));
    assert_eq!(demoted.category, MemoryCategory::Fact);
}

#[test]
fn prepare_for_write_keeps_entities_with_identity() {
    let kept = prepare_for_write(
        entry(MemoryCategory::Entity, "Kaneo workspace Ku67b6 belongs to the KIA migration")
            .with_identity_key("kaneo:workspace:Ku67b6"),
    );
    assert_eq!(kept.category, MemoryCategory::Entity);
}
