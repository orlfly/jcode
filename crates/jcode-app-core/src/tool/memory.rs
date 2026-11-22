#![cfg_attr(test, allow(clippy::await_holding_lock))]
//! Memory tool for storing and recalling information across sessions

use super::{Tool, ToolContext, ToolOutput};
use crate::memory::{MemoryCategory, MemoryEntry, MemoryManager, MemoryScope};
use anyhow::Result;
use async_trait::async_trait;
use jcode_base::engineering_store::EngineeringStore;
use serde::Deserialize;
use serde_json::{Value, json};

pub struct MemoryTool {
    manager: MemoryManager,
}

impl MemoryTool {
    pub fn new() -> Self {
        Self {
            manager: MemoryManager::new(),
        }
    }

    /// Create a memory tool in test mode (isolated storage)
    pub fn new_test() -> Self {
        Self {
            manager: MemoryManager::new_test(),
        }
    }

    fn parse_scope(scope: Option<&str>, default: MemoryScope) -> Result<MemoryScope> {
        match scope.unwrap_or(match default {
            MemoryScope::Project => "project",
            MemoryScope::Global => "global",
            MemoryScope::All => "all",
        }) {
            "project" => Ok(MemoryScope::Project),
            "global" => Ok(MemoryScope::Global),
            "all" => Ok(MemoryScope::All),
            other => Err(anyhow::anyhow!(
                "Unknown scope: {}. Use project, global, or all",
                other
            )),
        }
    }

    /// Scope the manager to the per-call working directory so project-scoped
    /// memories resolve to the right `projects/<hash>.json` store. The base
    /// manager is built once in `new()` with `project_dir: None`, which made
    /// project writes silently no-op and reads come back empty (issue #491).
    fn scoped_manager(&self, ctx: &ToolContext) -> MemoryManager {
        match ctx.working_dir.as_deref() {
            Some(dir) if !dir.as_os_str().is_empty() => self.manager.clone().with_project_dir(dir),
            _ => self.manager.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct MemoryInput {
    action: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    tags: Option<Vec<String>>,
    #[serde(default)]
    scope: Option<String>,
    /// For link action: source memory ID
    #[serde(default)]
    from_id: Option<String>,
    /// For link action: target memory ID
    #[serde(default)]
    to_id: Option<String>,
    /// For path action: additional start anchors.
    #[serde(default)]
    from_anchors: Option<Vec<String>>,
    /// For path action: additional goal anchors.
    #[serde(default)]
    to_anchors: Option<Vec<String>>,
    /// For path action: canonical EdgeKind names to allow. Empty means all.
    #[serde(default)]
    relation_allow: Option<Vec<String>>,
    /// For path action: memory (default), engineering, or ontology.
    #[serde(default)]
    graph: Option<String>,
    /// For engineering or ontology paths: include the deterministic PathPlan.
    #[serde(default)]
    plan: Option<bool>,
    /// For action=path: structured PathQuery IR.
    #[serde(default)]
    path_query: Option<crate::memory_types::PathQuery>,
    /// For path action: admit pending shadow edges as hypothesized edges.
    #[serde(default)]
    include_hypotheses: Option<bool>,
    /// For engineering paths: join the ontology concept layer.
    #[serde(default)]
    include_ontology: Option<bool>,
    /// For action=path: wall-clock budget for the query and L0 retries.
    #[serde(default)]
    query_budget_ms: Option<u64>,
    /// For action=path: minimum score for auto-grounding unresolved anchors.
    #[serde(default)]
    semantic_min_score: Option<f32>,
    /// For link action: relationship weight (0.0-1.0)
    #[serde(default)]
    weight: Option<f32>,
    /// For shadow propose: candidate confidence (0.0-1.0).
    #[serde(default)]
    confidence: Option<f32>,
    /// For shadow propose: relation name. Default relates_to.
    #[serde(default)]
    relation: Option<String>,
    /// For action=shadow: list | propose | accept | reject.
    #[serde(default)]
    shadow_action: Option<String>,
    /// For related action: traversal depth (default: 2). For path: max hops
    /// (default: 4, capped at 8).
    #[serde(default)]
    depth: Option<usize>,
    /// Maximum results. Recall defaults to 10, list/search remain uncapped when omitted.
    #[serde(default)]
    limit: Option<usize>,
    /// For recall: `recent` or `jev`. For search: `hybrid` (FTS + vector + tags).
    #[serde(default)]
    mode: Option<String>,
    /// For recall action: engineering graph anchor such as `file:src/lib.rs`.
    #[serde(default)]
    anchor: Option<String>,
    /// For ontology action: show | propose | apply | history | rollback
    #[serde(default)]
    ontology_action: Option<String>,
    /// For ontology action: inline patch object
    #[serde(default)]
    patch: Option<Value>,
    /// For ontology_action=gate: structured six-question evidence.
    #[serde(default)]
    gate: Option<Value>,
    /// For ontology action: path to a patch JSON file
    #[serde(default)]
    patch_file: Option<String>,
    /// For ontology apply/rollback: explicitly confirm the change
    #[serde(default)]
    confirm: Option<bool>,
    /// For ontology apply: report what would change without writing
    #[serde(default)]
    dry_run: Option<bool>,
    /// For ontology rollback: target version
    #[serde(default)]
    version: Option<u32>,
    /// For review action: list | accept | reject | defer
    #[serde(default)]
    review_action: Option<String>,
    /// For review list: pending | accepted | rejected | deferred | all
    #[serde(default)]
    status: Option<String>,
    /// For health action: resume automatic extraction after a health pause.
    #[serde(default)]
    extraction_action: Option<String>,
    /// Optional reviewer note
    #[serde(default)]
    note: Option<String>,
    /// For action=belief: {state, observation, goal, max_steps}.
    #[serde(default)]
    belief: Option<Value>,
    /// For action=decision: {action, problem, start_node, relation, ...}.
    #[serde(default)]
    decision: Option<Value>,
    /// For action=domain: {action, text, facts, id, limit}.
    #[serde(default)]
    domain: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct BeliefUpdateInput {
    #[serde(default)]
    state: Option<crate::memory_types::BeliefState>,
    #[serde(default)]
    observation: Option<crate::memory_types::BeliefObservation>,
    #[serde(default)]
    goal: Option<String>,
    #[serde(default)]
    max_steps: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct DecisionTraceInput {
    action: String,
    #[serde(default)]
    problem: Option<String>,
    #[serde(default)]
    start_node: Option<String>,
    #[serde(default)]
    relation: Option<String>,
    #[serde(default)]
    direction: Option<String>,
    #[serde(default)]
    goal_node: Option<String>,
    #[serde(default)]
    model_version: Option<String>,
    #[serde(default)]
    outcome: Option<String>,
    #[serde(default)]
    provenance: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct DomainFactInput {
    action: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    subject: Option<String>,
    #[serde(default)]
    predicate: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    facts: Vec<crate::memory_types::DomainFact>,
}

#[async_trait]
impl Tool for MemoryTool {
    fn name(&self) -> &str {
        "memory"
    }

    fn description(&self) -> &str {
        // Keep this under the registry's description token cap (tool/tests.rs).
        // Full scope-gate rules live in `remember`'s implementation and the
        // memory scope enum docs; a longer description bloats every request.
        "Memory CRUD, links/paths, alignment, belief/decision/domain. Default: project."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "intent": super::intent_schema_property(),
                "action": {
                    "type": "string",
                    "enum": ["remember", "recall", "readback", "search", "list", "forget", "tag", "link", "bridge", "unbridge", "align", "shadow", "related", "path", "compile", "quality", "belief", "decision", "domain", "embed", "ontology", "review", "health"],
                    "description": "Action."
                },
                "content": { "type": "string" },
                "category": {
                    "type": "string",
                    "enum": ["fact", "preference", "entity", "correction", "env_fact", "contract", "process_rule", "observation"]
                },
                "query": { "type": "string" },
                "id": {
                    "type": "string",
                    "description": "Unique id [A-Za-z0-9_:.-]{1,256}; omit to auto-generate."
                },
                "tags": { "type": "array", "items": { "type": "string" } },
                "scope": {
                    "type": "string",
                    "enum": ["project", "global", "all"],
                    "description": "project (default; repo/env facts), global (any-project facts only), all. Env details stay project."
                },
                "from_id": {
                    "type": "string",
                    "description": "For link/bridge: source id. For path: start anchor."
                },
                "to_id": {
                    "type": "string",
                    "description": "For link/bridge: target id. For path: goal anchor."
                },
                "from_anchors": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "For action=path: additional start anchors. Every source-target pair is solved."
                },
                "to_anchors": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "For action=path: additional goal anchors. Every source-target pair is solved."
                },
                "relation_allow": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "For path: allowed relation names; omit to allow all."
                },
                "graph": {
                    "type": "string",
                    "enum": ["memory", "engineering", "ontology"],
                    "description": "For path: memory, engineering, or ontology graph."
                },
                "plan": {
                    "type": "boolean",
                    "description": "For engineering/ontology path: include PathPlan."
                },
                "path_query": {
                    "type": "object",
                    "description": "For action=path: structured PathQuery object."
                },
                "include_hypotheses": {
                    "type": "boolean",
                    "description": "For action=path: include pending shadow edges as hypotheses."
                },
                "include_ontology": {
                    "type": "boolean",
                    "description": "For engineering path: join ontology concept layer."
                },
                "query_budget_ms": {
                    "type": "integer",
                    "description": "For path: wall-clock query budget in milliseconds."
                },
                "semantic_min_score": {
                    "type": "number",
                    "description": "For path: minimum anchor grounding score (0.0-1.0)."
                },
                "shadow_action": {
                    "type": "string",
                    "enum": ["list", "propose", "accept", "reject"],
                    "description": "For action=shadow: list, propose, accept, reject."
                },
                "relation": {
                    "type": "string",
                    "description": "For action=shadow: edge relation. Default relates_to."
                },
                "confidence": {
                    "type": "number",
                    "description": "For action=shadow: candidate confidence 0.0-1.0."
                },
                "limit": { "type": "integer", "minimum": 0, "description": "Max results for recall/search/list/path. 0 returns none." },
                "mode": {
                    "type": "string",
                    "enum": ["recent", "jev", "hybrid"],
                    "description": "Retrieval mode. For recall: recent or jev. For search: hybrid (FTS + vector + tags)."
                },
                "anchor": {
                    "type": "string",
                    "description": "For action=recall/readback: engineering graph anchor (file:, error:, command:, task:)."
                },
                "ontology_action": {
                    "type": "string",
                    "enum": ["show", "propose", "validate", "gate", "apply", "history", "rollback"],
                    "description": "For action=ontology. Review sessions may show/validate/propose only."
                },
                "gate": {
                    "type": "object",
                    "description": "For ontology_action=gate: structured six-question evidence.",
                    "properties": {
                        "validation_path": { "type": "boolean" },
                        "machine_consumed": { "type": "boolean" },
                        "compliance_risk": { "type": "boolean" },
                        "cross_scope_consistency": { "type": "boolean" },
                        "closed_examples": { "type": "boolean" },
                        "unresolved_questions": { "type": "array", "items": { "type": "string" } },
                        "conflicting_evidence": { "type": "boolean" }
                    }
                },
                "patch": {
                    "type": "object",
                    "description": "OntologyPatch JSON object (base_id, base_version, base_hash, rationale, ops, migrations)."
                },
                "patch_file": {
                    "type": "string",
                    "description": "Path to an OntologyPatch JSON file instead of `patch`."
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Required true to apply or roll back an ontology change from a normal session."
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "For ontology apply: report migration counts without writing."
                },
                "version": {
                    "type": "integer",
                    "description": "For ontology rollback: target version."
                },
                "review_action": {
                    "type": "string",
                    "enum": ["list", "accept", "reject", "defer"],
                    "description": "For action=review."
                },
                "status": {
                    "type": "string",
                    "description": "For review listing: pending, accepted, rejected, deferred, all."
                },
                "extraction_action": {
                    "type": "string",
                    "enum": ["resume"],
                    "description": "For action=health: resume automatic memory extraction after a source-health pause."
                },
                "note": {
                    "type": "string",
                    "description": "Optional reviewer note."
                },
                "belief": {
                    "type": "object",
                    "description": "For action=belief: {state, observation, goal, max_steps}."
                },
                "decision": {
                    "type": "object",
                    "description": "For action=decision: {action: record|search|list|forget, ...}."
                },
                "domain": {
                    "type": "object",
                    "description": "For action=domain: {action, text, facts, id, limit}."
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, input: Value, ctx: ToolContext) -> Result<ToolOutput> {
        use crate::memory;
        use crate::memory_types::{MemoryEventKind, MemoryState};

        let input: MemoryInput = serde_json::from_value(input)?;
        let action_label = input.action.clone();
        let session_id = ctx.session_id.clone();
        let manager = self.scoped_manager(&ctx);

        match input.action.as_str() {
            "remember" => {
                let content = input
                    .content
                    .ok_or_else(|| anyhow::anyhow!("content required"))?;
                let category: MemoryCategory = input
                    .category
                    .as_deref()
                    .unwrap_or("fact")
                    .parse()
                    .map_err(|err| anyhow::anyhow!("invalid memory category: {}", err))?;
                let scope = input.scope.as_deref().unwrap_or("project");

                // If a custom id is supplied, validate it and ensure it does not
                // collide with an existing memory in the target scope. Honoring
                // the supplied id silently here would let `add_memory`'s
                // `HashMap::insert` overwrite a previously remembered entry.
                if let Some(ref id) = input.id {
                    validate_custom_memory_id(id)?;
                    if manager.has_memory(id, scope)? {
                        return Err(anyhow::anyhow!(
                            "memory id '{}' already exists in {} scope; \
                             forget it first or omit id to auto-generate",
                            id, scope
                        ));
                    }
                }

                memory::set_state(MemoryState::ToolAction {
                    action: "remember".into(),
                    detail: truncate_for_widget(&content, 40),
                });
                let session_id = ctx.session_id.clone();
                let mut entry = MemoryEntry::new(category.clone(), &content)
                    .with_source(session_id.clone())
                    .with_provenance(
                        crate::memory::ProvenanceRecord::new(
                            session_id.clone(),
                            crate::memory::ExtractionMethod::UserStated,
                        )
                        .with_locator(format!("session:{session_id}")),
                    );
                if let Some(tags) = input.tags {
                    entry = entry.with_tags(tags);
                }
                if let Some(id) = input.id {
                    entry = entry.with_id(id);
                }
                let outcome = if scope == "global" {
                    match manager.remember_global_governed(entry) {
                        Ok(id) => id,
                        Err(error) if error.to_string().contains("scope gate") => {
                            memory::set_state(MemoryState::Idle);
                            return Ok(ToolOutput::new(format!(
                                "Rejected: {error}. \
                                 Re-submit the same memory with scope \"project\" \
                                 if it belongs to this environment, or generalize it."
                            )));
                        }
                        Err(error) => return Err(error),
                    }
                } else {
                    manager.remember_project_governed(entry)?
                };
                memory::add_event(MemoryEventKind::ToolRemembered {
                    content: truncate_for_widget(&content, 60),
                    scope: scope.to_string(),
                    category: category.to_string(),
                });
                memory::set_state(MemoryState::Idle);
                match outcome {
                    memory::MemoryWriteOutcome::Stored { id, needs_review } => {
                        // The agent just wrote this memory itself; the content is
                        // in the transcript, so auto-recall should not inject it
                        // back into this session.
                        memory::mark_memories_known(
                            &session_id,
                            std::slice::from_ref(&id),
                            "stored via memory tool in this session",
                        );
                        Ok(ToolOutput::new(format!(
                            "Remembered {} ({}): \"{}\" [id: {}]{}",
                            category,
                            scope,
                            content,
                            id,
                            if needs_review {
                                " (marked needs review)"
                            } else {
                                ""
                            }
                        )))
                    }
                    memory::MemoryWriteOutcome::PendingReview {
                        review_id,
                        target_id,
                    } => Ok(ToolOutput::new(format!(
                        "Queued for review [{review_id}]: {category} ({scope}) \"{content}\"{}",
                        target_id
                            .map(|id| format!("; target {id} stays unchanged"))
                            .unwrap_or_default()
                    ))),
                }
            }
            "readback" => {
                let mut anchors = input.from_anchors.unwrap_or_default();
                if let Some(anchor) = input.anchor {
                    anchors.push(anchor);
                }
                if let Some(query) = input.query.as_deref() {
                    anchors.extend(parse_checkpoint_anchors(query));
                }
                Self::readback_checkpoint(
                    anchors,
                    input.depth.unwrap_or(1),
                    input.limit.unwrap_or(8),
                )
            }
            "recall" => {
                let limit = input.limit.unwrap_or(10);
                if let Some(anchor) = input
                    .anchor
                    .as_deref()
                    .map(str::trim)
                    .filter(|anchor| !anchor.is_empty())
                {
                    let depth = input.depth.unwrap_or(2);
                    return Self::recall_engineering_anchor(anchor, depth, limit);
                }
                let scope = Self::parse_scope(input.scope.as_deref(), MemoryScope::All)?;
                let mode = input.mode.as_deref().unwrap_or_else(|| {
                    if input.query.is_some() {
                        "jev"
                    } else {
                        "recent"
                    }
                });

                match mode {
                    "recent" => {
                        memory::set_state(MemoryState::ToolAction {
                            action: "recall".into(),
                            detail: "recent".into(),
                        });
                        let result = match manager.get_prompt_memories_scoped(limit, scope) {
                            Some(memories) => {
                                let count =
                                    memories.lines().filter(|l| l.starts_with("- ")).count();
                                memory::add_event(MemoryEventKind::ToolRecalled {
                                    query: "(recent)".into(),
                                    count,
                                });
                                Ok(ToolOutput::new(format!("Recent memories:\n{}", memories)))
                            }
                            None => {
                                memory::add_event(MemoryEventKind::ToolRecalled {
                                    query: "(recent)".into(),
                                    count: 0,
                                });
                                Ok(ToolOutput::new("No memories stored yet."))
                            }
                        };
                        memory::set_state(MemoryState::Idle);
                        result
                    }
                    "jev" | "semantic" | "cascade" => {
                        let query = match &input.query {
                            Some(q) => q.clone(),
                            None => {
                                return Err(anyhow::anyhow!(
                                    "query required for Jev recall (including semantic/cascade aliases)"
                                ));
                            }
                        };
                        memory::set_state(MemoryState::ToolAction {
                            action: "recall".into(),
                            detail: truncate_for_widget(&query, 40),
                        });

                        let results = if limit == 0 {
                            Ok(Vec::new())
                        } else {
                            crate::memory_jev::recall(&manager, &query, limit, scope).await
                        };
                        // A missing Jev credential or failed request must not leave
                        // the widget busy or silently fall back to unjudged recall.
                        memory::set_state(MemoryState::Idle);
                        let results = results?;

                        memory::add_event(MemoryEventKind::ToolRecalled {
                            query: truncate_for_widget(&query, 40),
                            count: results.len(),
                        });
                        memory::set_state(MemoryState::Idle);

                        if results.is_empty() {
                            Ok(ToolOutput::new(format!(
                                "No memories found matching '{}'. Try recall without query to see recent memories.",
                                query
                            )))
                        } else {
                            let mut out = format!(
                                "Found {} relevant memories for '{}':\n\n",
                                results.len(),
                                query
                            );
                            for (entry, score) in results {
                                let tags_str = if entry.tags.is_empty() {
                                    String::new()
                                } else {
                                    format!(" [{}]", entry.tags.join(", "))
                                };
                                out.push_str(&format!(
                                    "- [{}] {}{}\n  id: {} (relevance: {:.0}%)\n\n",
                                    entry.category,
                                    entry.content,
                                    tags_str,
                                    entry.id,
                                    score * 100.0
                                ));
                            }
                            Ok(ToolOutput::new(out))
                        }
                    }
                    other => Err(anyhow::anyhow!(
                        "Unknown mode: {}. Use recent or jev (semantic/cascade are Jev aliases)",
                        other
                    )),
                }
            }
            "search" => {
                let query = input
                    .query
                    .ok_or_else(|| anyhow::anyhow!("query required"))?;
                let scope = Self::parse_scope(input.scope.as_deref(), MemoryScope::All)?;
                memory::set_state(MemoryState::ToolAction {
                    action: "search".into(),
                    detail: truncate_for_widget(&query, 40),
                });
                let limit = input.limit.unwrap_or(10);
                let results = if input.mode.as_deref() == Some("hybrid") {
                    manager
                        .hybrid_search_scoped(&query, limit, scope)?
                        .into_iter()
                        .map(|(entry, _score)| entry)
                        .collect()
                } else {
                    let mut results = manager.search_scoped(&query, scope)?;
                    results.truncate(limit);
                    results
                };
                memory::add_event(MemoryEventKind::ToolRecalled {
                    query: truncate_for_widget(&query, 40),
                    count: results.len(),
                });
                memory::set_state(MemoryState::Idle);
                if results.is_empty() {
                    Ok(ToolOutput::new(format!("No memories matching '{}'", query)))
                } else {
                    let mut out = format!("Found {} memories:\n\n", results.len());
                    for e in results {
                        out.push_str(&format!(
                            "- [{}] {}\n  id: {}\n\n",
                            e.category, e.content, e.id
                        ));
                    }
                    Ok(ToolOutput::new(out))
                }
            }
            "list" => {
                let scope = Self::parse_scope(input.scope.as_deref(), MemoryScope::All)?;
                memory::set_state(MemoryState::ToolAction {
                    action: "list".into(),
                    detail: String::new(),
                });
                let mut all = manager.list_all_scoped(scope)?;
                if let Some(limit) = input.limit {
                    all.truncate(limit);
                }
                memory::add_event(MemoryEventKind::ToolListed { count: all.len() });
                memory::set_state(MemoryState::Idle);
                if all.is_empty() {
                    Ok(ToolOutput::new("No memories stored."))
                } else {
                    let mut out = format!("Memories ({}):\n\n", all.len());
                    for e in all {
                        out.push_str(&format!(
                            "- [{}] {}\n  id: {}\n\n",
                            e.category, e.content, e.id
                        ));
                    }
                    Ok(ToolOutput::new(out))
                }
            }
            "forget" => {
                let id = input.id.ok_or_else(|| anyhow::anyhow!("id required"))?;
                memory::set_state(MemoryState::ToolAction {
                    action: "forget".into(),
                    detail: truncate_for_widget(&id, 30),
                });
                let found = manager.forget(&id)?;
                memory::add_event(MemoryEventKind::ToolForgot { id: id.clone() });
                memory::set_state(MemoryState::Idle);
                if found {
                    Ok(ToolOutput::new(format!("Forgot: {}", id)))
                } else {
                    Ok(ToolOutput::new(format!("Not found: {}", id)))
                }
            }
            "tag" => {
                let id = input.id.ok_or_else(|| anyhow::anyhow!("id required"))?;
                let tags = input.tags.ok_or_else(|| anyhow::anyhow!("tags required"))?;

                if tags.is_empty() {
                    return Err(anyhow::anyhow!("At least one tag required"));
                }

                memory::set_state(MemoryState::ToolAction {
                    action: "tag".into(),
                    detail: format!("{} +{}", truncate_for_widget(&id, 20), tags.join(",")),
                });
                for tag in &tags {
                    manager.tag_memory(&id, tag)?;
                }
                let tags_str = tags.join(", ");
                memory::add_event(MemoryEventKind::ToolTagged {
                    id: id.clone(),
                    tags: tags_str.clone(),
                });
                memory::set_state(MemoryState::Idle);

                Ok(ToolOutput::new(format!(
                    "Tagged memory {} with: {}",
                    id, tags_str
                )))
            }
            "link" => {
                let from_id = input
                    .from_id
                    .ok_or_else(|| anyhow::anyhow!("from_id required"))?;
                let to_id = input
                    .to_id
                    .ok_or_else(|| anyhow::anyhow!("to_id required"))?;
                let weight = input.weight.unwrap_or(0.5);

                memory::set_state(MemoryState::ToolAction {
                    action: "link".into(),
                    detail: format!(
                        "{} -> {}",
                        truncate_for_widget(&from_id, 15),
                        truncate_for_widget(&to_id, 15)
                    ),
                });
                manager.link_memories(&from_id, &to_id, weight)?;
                memory::add_event(MemoryEventKind::ToolLinked {
                    from: from_id.clone(),
                    to: to_id.clone(),
                });
                memory::set_state(MemoryState::Idle);
                Ok(ToolOutput::new(format!(
                    "Linked memories {} -> {} (weight {:.2})",
                    from_id, to_id, weight
                )))
            }
            "bridge" => {
                let project_id = input
                    .from_id
                    .ok_or_else(|| anyhow::anyhow!("from_id (project memory) required"))?;
                let global_id = input
                    .to_id
                    .ok_or_else(|| anyhow::anyhow!("to_id (global memory) required"))?;
                let weight = input.weight.unwrap_or(0.8);
                memory::set_state(MemoryState::ToolAction {
                    action: "bridge".into(),
                    detail: format!(
                        "{} -> {}",
                        truncate_for_widget(&project_id, 15),
                        truncate_for_widget(&global_id, 15)
                    ),
                });
                let bridge = manager.link_bridge(&project_id, &global_id, weight, "user")?;
                memory::set_state(MemoryState::Idle);
                Ok(ToolOutput::new(format!(
                    "Bridged project memory {} to global memory {} (id {}, weight {:.2}, confidence {:.2}).",
                    bridge.project_id,
                    bridge.global_id,
                    bridge.id,
                    bridge.weight,
                    bridge.confidence
                )))
            }
            "align" => {
                if input.confirm != Some(true) {
                    return Err(anyhow::anyhow!(
                        "alignment promotion requires confirm=true after reviewing the candidate"
                    ));
                }
                let from_id = input
                    .from_id
                    .ok_or_else(|| anyhow::anyhow!("from_id required"))?;
                let to_id = input
                    .to_id
                    .ok_or_else(|| anyhow::anyhow!("to_id required"))?;
                let relation = input.relation.as_deref().unwrap_or("same_as");
                let score = input
                    .confidence
                    .or(input.weight)
                    .ok_or_else(|| anyhow::anyhow!("confidence (reviewed score) required"))?;
                let evidence = input
                    .note
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("note (promotion evidence) required"))?;
                memory::set_state(MemoryState::ToolAction {
                    action: "align".into(),
                    detail: format!(
                        "{} -> {}",
                        truncate_for_widget(&from_id, 15),
                        truncate_for_widget(&to_id, 15)
                    ),
                });
                let promotion = manager.promote_alignment(
                    &from_id, &to_id, relation, score, evidence,
                )?;
                memory::set_state(MemoryState::Idle);
                Ok(ToolOutput::new(format!(
                    "Promoted alignment {} -> {} as {} in {} (score {:.2}). Evidence: {}",
                    promotion.from_id,
                    promotion.to_id,
                    promotion.edge,
                    promotion.scope,
                    promotion.score,
                    promotion.evidence
                )))
            }
            "unbridge" => {
                let id = input
                    .id
                    .ok_or_else(|| anyhow::anyhow!("id required for unbridge"))?;
                memory::set_state(MemoryState::ToolAction {
                    action: "unbridge".into(),
                    detail: truncate_for_widget(&id, 30),
                });
                let removed = manager.unlink_bridge(&id)?;
                memory::set_state(MemoryState::Idle);
                if removed {
                    Ok(ToolOutput::new(format!("Removed bridge {id}.")))
                } else {
                    Ok(ToolOutput::new(format!("No bridge found with id {id}.")))
                }
            }
            "shadow" => {
                use crate::memory::ShadowEdgeStatus;

                let shadow_action = input.shadow_action.as_deref().unwrap_or("list");
                match shadow_action {
                    "list" => {
                        let scope =
                            Self::parse_scope(input.scope.as_deref(), MemoryScope::All)?;
                        let status = input
                            .status
                            .as_deref()
                            .map(|value| {
                                ShadowEdgeStatus::parse(value).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "Unknown shadow status: {value}. Use pending, accepted, or rejected"
                                    )
                                })
                            })
                            .transpose()?;
                        let mut edges = manager.list_shadow_edges_scoped(status, scope)?;
                        if let Some(limit) = input.limit {
                            edges.truncate(limit);
                        }
                        if edges.is_empty() {
                            return Ok(ToolOutput::new("No shadow edges found.".to_string()));
                        }
                        let mut out = format!("Shadow edges ({}):", edges.len());
                        for edge in edges {
                            out.push_str(&format!(
                                "\n- {} [{}] {} -{}-> {} (weight {:.2}, confidence {:.2})",
                                edge.id,
                                edge.status.as_str(),
                                edge.from_id,
                                edge.relation,
                                edge.to_id,
                                edge.weight,
                                edge.confidence
                            ));
                        }
                        Ok(ToolOutput::new(out))
                    }
                    "propose" => {
                        let from_id = input
                            .from_id
                            .ok_or_else(|| anyhow::anyhow!("from_id required for shadow propose"))?;
                        let to_id = input
                            .to_id
                            .ok_or_else(|| anyhow::anyhow!("to_id required for shadow propose"))?;
                        let scope =
                            Self::parse_scope(input.scope.as_deref(), MemoryScope::Project)?;
                        let relation = input.relation.as_deref().unwrap_or("relates_to");
                        let edge = manager.propose_shadow_edge(
                            &from_id,
                            &to_id,
                            relation,
                            input.weight.unwrap_or(0.5),
                            input.confidence.unwrap_or(0.5),
                            input.content.as_deref(),
                            "user",
                            scope,
                        )?;
                        Ok(ToolOutput::new(format!(
                            "Proposed shadow edge {} [pending] {} -{}-> {} (weight {:.2}, confidence {:.2}). Accept with action=shadow shadow_action=accept id={}.",
                            edge.id,
                            edge.from_id,
                            edge.relation,
                            edge.to_id,
                            edge.weight,
                            edge.confidence,
                            edge.id
                        )))
                    }
                    "accept" | "reject" => {
                        let id = input.id.ok_or_else(|| {
                            anyhow::anyhow!("id required for shadow {shadow_action}")
                        })?;
                        let accept = shadow_action == "accept";
                        let edge = manager.review_shadow_edge(&id, accept)?;
                        if accept {
                            Ok(ToolOutput::new(format!(
                                "Accepted shadow edge {}; wrote {} -> {} ({}).",
                                edge.id, edge.from_id, edge.to_id, edge.relation
                            )))
                        } else {
                            Ok(ToolOutput::new(format!("Rejected shadow edge {}.", edge.id)))
                        }
                    }
                    other => Err(anyhow::anyhow!(
                        "Unknown shadow_action: {other}. Use list, propose, accept, or reject"
                    )),
                }
            }
            "related" => {
                let id = input.id.ok_or_else(|| anyhow::anyhow!("id required"))?;
                let depth = input.depth.unwrap_or(2);

                memory::set_state(MemoryState::ToolAction {
                    action: "related".into(),
                    detail: truncate_for_widget(&id, 30),
                });
                let related = manager.get_related(&id, depth)?;
                memory::add_event(MemoryEventKind::ToolRecalled {
                    query: format!("related:{}", truncate_for_widget(&id, 20)),
                    count: related.len(),
                });
                memory::set_state(MemoryState::Idle);

                if related.is_empty() {
                    Ok(ToolOutput::new(format!(
                        "No related memories found for {}",
                        id
                    )))
                } else {
                    let mut out = format!(
                        "Found {} memories related to {} (depth {}):\n\n",
                        related.len(),
                        id,
                        depth
                    );
                    for e in related {
                        out.push_str(&format!(
                            "- [{}] {}\n  id: {}\n\n",
                            e.category, e.content, e.id
                        ));
                    }
                    Ok(ToolOutput::new(out))
                }
            }
            "path" => {
                let mut auto_ground = false;
                let mut semantic_min_score = None;
                let mut query_filters: Option<(
                    Vec<String>,
                    Option<f32>,
                    Option<crate::memory_types::PathObjective>,
                )> = None;
                let (
                    scope,
                    depth,
                    limit,
                    froms,
                    tos,
                    waypoints,
                    relations,
                    graph_kind,
                    include_plan,
                    include_hypotheses,
                    include_ontology,
                    query_budget_ms,
                ) = if let Some(query) = input.path_query.clone() {
                    query.validate().map_err(|errors| {
                        anyhow::anyhow!("invalid path_query: {}", errors.join("; "))
                    })?;
                    let scope = match query.scope() {
                        crate::memory_types::PathQueryScope::Project => MemoryScope::Project,
                        crate::memory_types::PathQueryScope::Global => MemoryScope::Global,
                        crate::memory_types::PathQueryScope::All => MemoryScope::All,
                    };
                    if !query.constraints.deny_relations.is_empty()
                        || query.constraints.min_confidence.is_some()
                        || query.constraints.objective.is_some()
                    {
                        query_filters = Some((
                            query.constraints.deny_relations.clone(),
                            query.constraints.min_confidence,
                            query.constraints.objective,
                        ));
                    }
                    auto_ground = query.constraints.auto_ground;
                    semantic_min_score = query.constraints.semantic_min_score;
                    let graph_kind = query.graph_kind().as_str().to_string();
                    (
                        scope,
                        query.depth(4),
                        query.limit(3),
                        query.start,
                        query.goal,
                        query.waypoints,
                        query.constraints.relations,
                        graph_kind,
                        query.constraints.plan,
                        query.constraints.include_hypotheses,
                        query.constraints.include_ontology,
                        query.constraints.query_budget_ms,
                    )
                } else {
                    let scope = Self::parse_scope(input.scope.as_deref(), MemoryScope::All)?;
                    let depth = input.depth.unwrap_or(4);
                    let limit = input.limit.unwrap_or(3);
                    let mut froms = input.from_anchors.unwrap_or_default();
                    if let Some(from) = input.from_id {
                        froms.insert(0, from);
                    }
                    let mut tos = input.to_anchors.unwrap_or_default();
                    if let Some(to) = input.to_id {
                        tos.insert(0, to);
                    }
                    let relations = input.relation_allow.unwrap_or_default();
                    let graph_kind = input.graph.unwrap_or_else(|| "memory".to_string());
                    let include_plan = input.plan.unwrap_or(false);
                    let include_hypotheses = input.include_hypotheses.unwrap_or(false);
                    let include_ontology = input.include_ontology.unwrap_or(false);
                    let query_budget_ms = input.query_budget_ms;
                    semantic_min_score = input.semantic_min_score;
                    (
                        scope,
                        depth,
                        limit,
                        froms,
                        tos,
                        Vec::new(),
                        relations,
                        graph_kind,
                        include_plan,
                        include_hypotheses,
                        include_ontology,
                        query_budget_ms,
                    )
                };
                if froms.is_empty() {
                    return Err(anyhow::anyhow!(
                        "from_id, from_anchors, or path_query.start required for path"
                    ));
                }
                if tos.is_empty() {
                    return Err(anyhow::anyhow!(
                        "to_id, to_anchors, or path_query.goal required for path"
                    ));
                }
                if let Some(score) = semantic_min_score
                    && !(0.0..=1.0).contains(&score)
                {
                    return Err(anyhow::anyhow!(
                        "semantic_min_score must be between 0.0 and 1.0"
                    ));
                }
                if graph_kind == "engineering" || graph_kind == "ontology" {
                    if query_filters.is_some() {
                        return Err(anyhow::anyhow!(
                            "path_query deny_relations/min_confidence/objective are only supported for the memory graph"
                        ));
                    }
                    return Self::path_engineering(
                        &froms,
                        &tos,
                        depth,
                        limit,
                        &relations,
                        include_plan,
                        &graph_kind,
                        include_ontology,
                    );
                }
                if graph_kind != "memory" {
                    return Err(anyhow::anyhow!(
                        "Unknown path graph: {graph_kind}. Use memory, engineering, or ontology"
                    ));
                }

                memory::set_state(MemoryState::ToolAction {
                    action: "path".into(),
                    detail: format!(
                        "{} -> {}",
                        truncate_for_widget(&froms.join(","), 15),
                        truncate_for_widget(&tos.join(","), 15)
                    ),
                });
                let query_options = crate::memory::MemoryPathQueryOptions {
                    include_hypotheses,
                    deny_relations: query_filters
                        .as_ref()
                        .map(|filters| filters.0.clone())
                        .unwrap_or_default(),
                    min_confidence: query_filters.as_ref().and_then(|filters| filters.1),
                    objective: query_filters.as_ref().and_then(|filters| filters.2),
                    auto_ground_anchors: auto_ground,
                    query_budget_ms,
                    semantic_min_score,
                };
                let outcome = if waypoints.is_empty() {
                    manager.paths_between_anchors_with_options(
                        &froms,
                        &tos,
                        depth,
                        limit,
                        scope,
                        &relations,
                        query_options,
                    )?
                } else {
                    manager.paths_via_waypoints(
                        &froms,
                        &tos,
                        &waypoints,
                        depth,
                        limit,
                        scope,
                        &relations,
                        query_options.clone(),
                    )?
                };
                let mut outcome = outcome;
                if outcome.paths.is_empty() && query_filters.is_some() {
                    outcome.next_actions.push(
                        "query constraints excluded all candidate paths; relax deny_relations or min_confidence".to_string(),
                    );
                }
                let paths = &outcome.paths;
                memory::add_event(MemoryEventKind::ToolRecalled {
                    query: format!(
                        "path:{}->{}",
                        truncate_for_widget(&froms.join(","), 15),
                        truncate_for_widget(&tos.join(","), 15)
                    ),
                    count: paths.len(),
                });
                memory::set_state(MemoryState::Idle);

                let mut diagnostics = Vec::new();
                if !outcome.unresolved_from.is_empty() {
                    diagnostics.push(format!(
                        "unresolved start anchor(s): {}",
                        outcome.unresolved_from.join(", ")
                    ));
                }
                if !outcome.unresolved_to.is_empty() {
                    diagnostics.push(format!(
                        "unresolved goal anchor(s): {}",
                        outcome.unresolved_to.join(", ")
                    ));
                }
                for suggestion in &outcome.anchor_suggestions {
                    diagnostics.push(format!(
                        "anchor candidate for `{}`: `{}` ({}) score {:.2}: {}",
                        suggestion.anchor,
                        suggestion.id,
                        suggestion.store,
                        suggestion.score,
                        suggestion.content
                    ));
                }
                if !outcome.unknown_relations.is_empty() {
                    diagnostics.push(format!(
                        "unknown relation(s) ignored: {}",
                        outcome.unknown_relations.join(", ")
                    ));
                }
                if !outcome.diagnosis.is_empty() {
                    diagnostics.push(format!("diagnosis: {}", outcome.diagnosis.join(", ")));
                }
                if !outcome.missing_classes.is_empty() {
                    diagnostics.push(format!(
                        "missing class: {}",
                        outcome.missing_classes.join(", ")
                    ));
                }
                for action in &outcome.next_actions {
                    diagnostics.push(format!("next action: {action}"));
                }
                for relaxation in &outcome.relaxations {
                    diagnostics.push(format!("relaxation: {relaxation}"));
                }
                for candidate in &outcome.alignment_candidates {
                    diagnostics.push(format!(
                        "alignment candidate: {} ({}) <-> {} ({}) score {:.2}; propose shadow relation={} {} -> {}",
                        candidate.from_id,
                        candidate.from_store,
                        candidate.to_id,
                        candidate.to_store,
                        candidate.score,
                        candidate.relation,
                        candidate.from_id,
                        candidate.to_id
                    ));
                }
                let mut quality_flags = Vec::new();
                for quality in &outcome.quality {
                    for flag in &quality.flags {
                        if !quality_flags.contains(flag) {
                            quality_flags.push(*flag);
                        }
                    }
                }
                let decision = crate::memory_types::select_next_action(
                    &crate::memory_types::DecisionSignals {
                        has_paths: !paths.is_empty(),
                        path_confidence: paths
                            .iter()
                            .map(|path| path.confidence)
                            .fold(0.0, f32::max),
                        diagnosis: outcome.diagnosis.clone(),
                        quality_flags,
                        planning_requested: include_plan,
                        attempts_remaining: 1,
                    },
                );
                diagnostics.push(format!(
                    "decision: {} (priority {}): {}",
                    decision.action.as_str(),
                    decision.priority,
                    decision.reason
                ));

                let rerank_problem = input
                    .query
                    .clone()
                    .or_else(|| input.content.clone())
                    .filter(|problem| !problem.trim().is_empty());
                let decision_examples = rerank_problem
                    .as_deref()
                    .map(|problem| {
                        let mut query =
                            crate::memory_types::DecisionStepQuery::new(problem).with_limit(3);
                        if let Some(start_node) = froms.first() {
                            query = query.with_start_node(start_node.clone());
                        }
                        manager.rank_decision_steps(&query)
                    })
                    .transpose()?
                    .unwrap_or_default();
                let domain_facts = rerank_problem
                    .as_deref()
                    .map(|problem| {
                        manager.rank_domain_facts(
                            &crate::memory_types::DomainFactQuery::new(problem).with_limit(5),
                        )
                    })
                    .transpose()?
                    .unwrap_or_default();

                if paths.is_empty() {
                    let mut out = format!(
                        "No memory path found from [{}] to [{}] within depth {} (scope {:?}, stores searched: {}).",
                        froms.join(", "),
                        tos.join(", "),
                        outcome.depth,
                        scope,
                        outcome.stores_searched
                    );
                    for diagnostic in &diagnostics {
                        out.push_str(&format!("\n- {diagnostic}"));
                    }
                    append_decision_examples(&mut out, &decision_examples);
                    append_domain_facts(&mut out, &domain_facts);
                    out.push_str(
                        "\nTry different anchors or a wider scope; E05 depth is relaxed once from 8 to 16 automatically.",
                    );
                    Ok(ToolOutput::new(out))
                } else {
                    let mut out = format!(
                        "Found {} memory path(s) from [{}] to [{}] (depth {}, stores searched: {}):\n",
                        paths.len(),
                        froms.join(", "),
                        tos.join(", "),
                        outcome.depth,
                        outcome.stores_searched
                    );
                    for diagnostic in &diagnostics {
                        out.push_str(&format!("\n- {diagnostic}"));
                    }
                    for (index, path) in paths.iter().enumerate() {
                        out.push_str(&format!(
                            "\nPath {}: cost {:.2}, confidence {:.0}%, {} node(s)\n",
                            index + 1,
                            path.cost,
                            path.confidence * 100.0,
                            path.nodes.len()
                        ));
                        if let Some(quality) = outcome.quality.get(index)
                            && !quality.is_empty()
                        {
                            out.push_str(&format!("  quality: {}\n", quality.label()));
                            for reason in &quality.reasons {
                                out.push_str(&format!("    - {reason}\n"));
                            }
                        }
                        let flags = outcome
                            .quality
                            .get(index)
                            .map(|quality| quality.flags.as_slice())
                            .unwrap_or(&[]);
                        let constraint = if outcome.diagnosis.is_empty() {
                            1.0
                        } else if outcome
                            .diagnosis
                            .iter()
                            .any(|code| code == "E04" || code == "E10")
                        {
                            0.0
                        } else {
                            0.5
                        };
                        let reward = crate::memory_types::shaped_reward(
                            &crate::memory_types::RewardSignals {
                                accuracy: path.confidence,
                                efficiency: 1.0 / (1.0 + path.cost.max(0.0)),
                                semantic: path.confidence,
                                constraint,
                                completion: if flags.contains(
                                    &crate::memory_types::PathQualityFlag::HypothesizedEdge,
                                ) {
                                    0.0
                                } else {
                                    1.0
                                },
                                plan: 0.5,
                                quality: crate::memory_types::quality_signal(flags),
                            },
                            &crate::memory_types::RewardWeights::default(),
                        );
                        out.push_str(&format!("  reward: {reward:.2}\n"));
                        for (node_index, node_id) in path.nodes.iter().enumerate() {
                            match manager.get_by_id(node_id, scope)? {
                                Some(entry) => out.push_str(&format!(
                                    "  [{}] {}\n    id: {}\n",
                                    entry.category,
                                    truncate_for_widget(&entry.content, 200),
                                    entry.id
                                )),
                                None if node_id.starts_with("tag:") => out.push_str(&format!(
                                    "  [tag] {}\n",
                                    node_id.trim_start_matches("tag:")
                                )),
                                None => out.push_str(&format!("  [node] {}\n", node_id)),
                            }
                            if let Some(edge) = path.edges.get(node_index) {
                                out.push_str(&format!(
                                    "  --{} (cost {:.2})-->\n",
                                    memory_edge_label(&edge.kind),
                                    edge.cost
                                ));
                            }
                        }
                    }
                    append_decision_examples(&mut out, &decision_examples);
                    append_domain_facts(&mut out, &domain_facts);
                    Ok(ToolOutput::new(out))
                }
            }
            "embed" => {
                let limit = input.limit.unwrap_or(64).min(512);
                memory::set_state(MemoryState::ToolAction {
                    action: "embed".into(),
                    detail: format!("limit {limit}"),
                });
                let (generated, failed, remaining) =
                    manager.backfill_embeddings_bounded(limit)?;
                memory::set_state(MemoryState::Idle);
                Ok(ToolOutput::new(format!(
                    "Embedding backfill: {generated} generated, {failed} failed, {remaining} still missing (limit {limit})."
                )))
            }
            "review" => {
                let review_action = input.review_action.as_deref().unwrap_or("list");
                match review_action {
                    "list" => {
                        let filter = match input.status.as_deref().unwrap_or("pending") {
                            "all" => None,
                            other => Some(crate::memory::ReviewStatus::parse(other).ok_or_else(
                                || anyhow::anyhow!("unknown review status: {other}"),
                            )?),
                        };
                        let items = manager.list_reviews(filter)?;
                        if items.is_empty() {
                            Ok(ToolOutput::new("No review items."))
                        } else {
                            let mut out = format!("{} review item(s):\n\n", items.len());
                            for item in items {
                                out.push_str(&format!(
                                    "- {} [{:?}]\n  reason: {}\n  diff: {}\n  proposed: {}\n  current: {}\n\n",
                                    item.id,
                                    item.status,
                                    item.reason,
                                    item.diff.join(", "),
                                    item.new.content,
                                    item.old
                                        .as_ref()
                                        .map(|old| old.content.as_str())
                                        .unwrap_or("<new>")
                                ));
                            }
                            Ok(ToolOutput::new(out))
                        }
                    }
                    "accept" | "reject" | "defer" => {
                        let id = input
                            .id
                            .as_deref()
                            .ok_or_else(|| anyhow::anyhow!("id required for review {review_action}"))?;
                        let reviewer = if session_id.is_empty() {
                            "agent"
                        } else {
                            session_id.as_str()
                        };
                        let item = match review_action {
                            "accept" => manager.accept_review(id, reviewer, input.note.as_deref())?,
                            "reject" => manager.reject_review(id, reviewer, input.note.as_deref())?,
                            _ => manager.defer_review(id, reviewer, input.note.as_deref())?,
                        };
                        Ok(ToolOutput::new(format!(
                            "{} review {}: {}",
                            review_action,
                            item.id,
                            item.new.content
                        )))
                    }
                    other => Err(anyhow::anyhow!(
                        "unknown review action: {other}; use list, accept, reject or defer"
                    )),
                }
            }
            "quality" => {
                let scope = Self::parse_scope(input.scope.as_deref(), MemoryScope::All)?;
                let report = manager.graph_quality(scope)?;
                Ok(ToolOutput::new(format!(
                    "Graph quality ({}):\n{}",
                    report.scope,
                    serde_json::to_string_pretty(&report)?
                )))
            }
            "compile" => {
                let text = input
                    .query
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("query required for compile"))?;
                let graph_kind = input.graph.as_deref().unwrap_or("memory");
                if !matches!(graph_kind, "memory" | "engineering" | "ontology") {
                    return Err(anyhow::anyhow!(
                        "Unknown compile graph: {graph_kind}. Use memory, engineering, or ontology"
                    ));
                }
                // Always offer the engineering vocabulary so a `graph
                // engineering` token inside the text can validate even when the
                // caller left the separate `graph` field at its default.
                let allowed: Vec<String> =
                    crate::memory_types::engineering_ontology::default_engineering_ontology()
                        .relations
                        .keys()
                        .cloned()
                        .collect();
                let allowed_refs: Vec<&str> = allowed.iter().map(String::as_str).collect();
                let compile_text = if graph_kind != "memory"
                    && !text.to_ascii_lowercase().contains("graph")
                {
                    format!("{text} graph {graph_kind}")
                } else {
                    text.to_string()
                };
                let query = crate::memory_types::path_query::compile_path_query_text(
                    &compile_text,
                    &allowed_refs,
                )
                .map_err(|errors| {
                    anyhow::anyhow!("path query compile failed: {}", errors.join("; "))
                })?;
                Ok(ToolOutput::new(format!(
                    "Compiled PathQuery:\n{}",
                    serde_json::to_string_pretty(&query)?
                )))
            }
            "belief" => {
                let raw = input
                    .belief
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("belief action requires `belief` object"))?;
                let request: BeliefUpdateInput = serde_json::from_value(raw)?;
                let BeliefUpdateInput {
                    state,
                    observation,
                    goal,
                    max_steps,
                } = request;
                let goal = goal
                    .or_else(|| input.content.clone())
                    .unwrap_or_else(|| "decision task".to_string());
                let mut state = state.unwrap_or_else(|| {
                    let max_steps = max_steps.unwrap_or(input.limit.unwrap_or(5).max(1) as u32);
                    crate::memory_types::BeliefState::new(goal, max_steps)
                });
                let step = observation.map(|observation| state.observe(observation));
                let best = state
                    .best_hypothesis()
                    .map(|hypothesis| {
                        format!("{} @ {:.3}", hypothesis.id, hypothesis.confidence)
                    })
                    .unwrap_or_else(|| "none".to_string());
                let action = step
                    .as_ref()
                    .map(|step| step.action.as_str())
                    .unwrap_or("none");
                let blocked = if state.blocked_anchors.is_empty() {
                    "none".to_string()
                } else {
                    state
                        .blocked_anchors
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let payload = json!({ "belief": &state, "step": &step });
                Ok(ToolOutput::new(format!(
                    "Belief step {} / {}: confidence {:.3}, best {best}, action {action}\nblocked anchors: {blocked}\n\n{}",
                    state.step,
                    state.max_steps,
                    state.confidence(),
                    serde_json::to_string_pretty(&payload)?
                )))
            }
            "decision" => {
                let raw = input
                    .decision
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("decision action requires `decision` object"))?;
                let request: DecisionTraceInput = serde_json::from_value(raw)?;
                match request.action.as_str() {
                    "record" => {
                        if let Some(source) = request.source.as_deref()
                            && !source.eq_ignore_ascii_case("verified")
                        {
                            return Err(anyhow::anyhow!(
                                "only verified decision steps may be recorded (source={source})"
                            ));
                        }
                        let problem = request
                            .problem
                            .ok_or_else(|| anyhow::anyhow!("decision record requires `problem`"))?;
                        let start_node = request.start_node.ok_or_else(|| {
                            anyhow::anyhow!("decision record requires `start_node`")
                        })?;
                        let relation = request.relation.ok_or_else(|| {
                            anyhow::anyhow!("decision record requires `relation`")
                        })?;
                        let direction = crate::memory_types::DecisionDirection::parse(
                            request.direction.as_deref().unwrap_or("outgoing"),
                        )
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "decision direction must be outgoing, incoming, or undirected"
                            )
                        })?;
                        let step = manager.record_verified_decision_step(
                            &problem,
                            &start_node,
                            &relation,
                            direction,
                            request.goal_node.as_deref(),
                            request.model_version.as_deref(),
                            request.outcome.as_deref(),
                            request.provenance.as_deref().unwrap_or("manual:record"),
                        )?;
                        Ok(ToolOutput::new(format!(
                            "Recorded verified decision {} ({})\n\n{}",
                            step.id,
                            step.problem,
                            serde_json::to_string_pretty(&step)?
                        )))
                    }
                    "search" => {
                        let problem = request
                            .problem
                            .ok_or_else(|| anyhow::anyhow!("decision search requires `problem`"))?;
                        let mut query = crate::memory_types::DecisionStepQuery::new(&problem)
                            .with_limit(request.limit.unwrap_or(5));
                        if let Some(start_node) = request.start_node {
                            query = query.with_start_node(start_node);
                        }
                        if let Some(relation) = request.relation {
                            query = query.with_relation(relation);
                        }
                        let hits = manager.rank_decision_steps(&query)?;
                        Ok(ToolOutput::new(format!(
                            "Found {} decision example(s) for `{}`\n\n{}",
                            hits.len(),
                            problem,
                            serde_json::to_string_pretty(&hits)?
                        )))
                    }
                    "list" => {
                        let steps = manager.list_decision_steps()?;
                        Ok(ToolOutput::new(format!(
                            "{} verified decision step(s)\n\n{}",
                            steps.len(),
                            serde_json::to_string_pretty(&steps)?
                        )))
                    }
                    "forget" => {
                        let id = request.id.ok_or_else(|| {
                            anyhow::anyhow!("decision forget requires `id`")
                        })?;
                        let removed = manager.forget_decision_step(&id)?;
                        Ok(ToolOutput::new(if removed {
                            format!("Forgot decision step {id}")
                        } else {
                            format!("Decision step {id} not found")
                        }))
                    }
                    other => Err(anyhow::anyhow!(
                        "unknown decision action: {other}; use record, search, list or forget"
                    )),
                }
            }
            "domain" => {
                let raw = input
                    .domain
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("domain action requires `domain` object"))?;
                let request: DomainFactInput = serde_json::from_value(raw)?;
                match request.action.as_str() {
                    "record" => {
                        if request.facts.is_empty() {
                            return Err(anyhow::anyhow!(
                                "domain record requires a non-empty `facts` array"
                            ));
                        }
                        let inserted = manager.record_domain_facts(&request.facts)?;
                        Ok(ToolOutput::new(format!(
                            "Recorded {} domain fact(s), {} new\n\n{}",
                            request.facts.len(),
                            inserted,
                            serde_json::to_string_pretty(&request.facts)?
                        )))
                    }
                    "search" => {
                        let text = request
                            .text
                            .ok_or_else(|| anyhow::anyhow!("domain search requires `text`"))?;
                        let mut query = crate::memory_types::DomainFactQuery::new(&text)
                            .with_limit(request.limit.unwrap_or(8));
                        if let Some(subject) = request.subject {
                            query = query.with_subject(subject);
                        }
                        if let Some(predicate) = request.predicate {
                            query = query.with_predicate(predicate);
                        }
                        let hits = manager.rank_domain_facts(&query)?;
                        Ok(ToolOutput::new(format!(
                            "Found {} domain fact(s) for `{}`\n\n{}",
                            hits.len(),
                            text,
                            serde_json::to_string_pretty(&hits)?
                        )))
                    }
                    "list" => {
                        let facts = manager.list_domain_facts()?;
                        Ok(ToolOutput::new(format!(
                            "{} domain fact(s)\n\n{}",
                            facts.len(),
                            serde_json::to_string_pretty(&facts)?
                        )))
                    }
                    "forget" => {
                        let id = request
                            .id
                            .ok_or_else(|| anyhow::anyhow!("domain forget requires `id`"))?;
                        let removed = manager.forget_domain_fact(&id)?;
                        Ok(ToolOutput::new(if removed {
                            format!("Forgot domain fact {id}")
                        } else {
                            format!("Domain fact {id} not found")
                        }))
                    }
                    other => Err(anyhow::anyhow!(
                        "unknown domain action: {other}; use record, search, list or forget"
                    )),
                }
            }
            "health" => {
                if let Some(extraction_action) = input.extraction_action.as_deref() {
                    match extraction_action {
                        "resume" => {
                            let source = manager.resume_extraction()?;
                            return Ok(ToolOutput::new(format!(
                                "Extraction resumed.\n{}",
                                source.summary()
                            )));
                        }
                        other => {
                            return Err(anyhow::anyhow!(
                                "Unknown extraction_action: {other}. Use resume"
                            ));
                        }
                    }
                }
                let report = manager.health_report()?;
                let source = manager.extraction_health()?;
                Ok(ToolOutput::new(format!(
                    "{}\n\nreview backlog: {} pending, {} deferred, {} accepted, {} rejected\nlow confidence: {}, missing locator: {}, contradictions: {}\nextraction: {}",
                    report.summary(),
                    report.reviews.pending,
                    report.reviews.deferred,
                    report.reviews.accepted,
                    report.reviews.rejected,
                    report.low_confidence,
                    report.missing_locator,
                    report.contradictions,
                    source.summary()
                )))
            }
            "ontology" => {
                use crate::memory::OntologyScope;
                use crate::memory_types::migration::{OntologyPatch, apply_patch};

                let scope = OntologyScope::parse(input.scope.as_deref().unwrap_or("project"))?;
                let ontology_action = input.ontology_action.as_deref().unwrap_or("show");

                if ontology_action == "gate" {
                    use crate::memory_types::{
                        SixQuestionSignals, route_ontology_definition,
                    };

                    let signals = input
                        .gate
                        .clone()
                        .map(serde_json::from_value::<SixQuestionSignals>)
                        .transpose()?
                        .unwrap_or_default();
                    let decision = route_ontology_definition(&signals);
                    let definition = input.content.as_deref().unwrap_or("(unnamed definition)");
                    let matched = if decision.matched_questions.is_empty() {
                        "(none)".to_string()
                    } else {
                        decision.matched_questions.join(", ")
                    };
                    let mut out = format!(
                        "Six-question gate: {}\ndefinition: {}\nmatched: {}\nrule layer: {}",
                        decision.exit.as_str(),
                        definition,
                        matched,
                        if decision.rule_layer_required {
                            "required"
                        } else {
                            "not required"
                        }
                    );
                    for reason in &decision.reasons {
                        out.push_str(&format!("\n- {reason}"));
                    }
                    return Ok(ToolOutput::new(out));
                }

                let patch = if let Some(value) = input.patch.clone() {
                    Some(serde_json::from_value::<OntologyPatch>(value)?)
                } else if let Some(path) = input.patch_file.as_deref() {
                    let raw = std::fs::read_to_string(path)?;
                    Some(serde_json::from_str::<OntologyPatch>(&raw)?)
                } else {
                    None
                };

                match ontology_action {
                    "show" => {
                        let ontology = manager.stored_ontology(scope)?;
                        let history = manager.ontology_history(scope)?;
                        Ok(ToolOutput::new(format!(
                            "{} scope\nbase_id: {}\nbase_version: {}\nbase_hash: {}\n{}\nhistory: {}",
                            scope.label(),
                            ontology.id,
                            ontology.version,
                            ontology.content_hash(),
                            crate::memory_types::migration::describe_ontology(&ontology),
                            history
                                .iter()
                                .map(|snapshot| format!("v{}", snapshot.version))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )))
                    }
                    "propose" | "validate" => {
                        let patch = patch.ok_or_else(|| {
                            anyhow::anyhow!(
                                "ontology {ontology_action} requires `patch` or `patch_file`"
                            )
                        })?;
                        let base = manager.stored_ontology(scope)?;
                        match apply_patch(&base, &patch) {
                            Ok(next) => {
                                if ontology_action == "validate" {
                                    return Ok(ToolOutput::new(format!(
                                        "Patch valid: {} v{} -> v{} ({})",
                                        next.id,
                                        base.version,
                                        next.version,
                                        next.content_hash()
                                    )));
                                }
                                let dir = crate::storage::jcode_dir()?
                                    .join("memory")
                                    .join("ontology")
                                    .join(".proposals");
                                std::fs::create_dir_all(&dir)?;
                                let slug: String = patch
                                    .base_id
                                    .chars()
                                    .map(|c| {
                                        if c.is_ascii_alphanumeric() {
                                            c
                                        } else {
                                            '-'
                                        }
                                    })
                                    .collect();
                                let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");
                                let path = dir.join(format!("{stamp}-{slug}.json"));
                                std::fs::write(&path, serde_json::to_string_pretty(&patch)?)?;
                                Ok(ToolOutput::new(format!(
                                    "Ontology proposal written to {}\n{} v{} -> v{}\nApply with: jcode memory ontology apply {} --scope {} --apply",
                                    path.display(),
                                    next.id,
                                    base.version,
                                    next.version,
                                    path.display(),
                                    scope.label()
                                )))
                            }
                            Err(errors) => Ok(ToolOutput::new(format!(
                                "Patch rejected:\n- {}",
                                errors.join("\n- ")
                            ))),
                        }
                    }
                    "apply" => {
                        if input.confirm != Some(true) {
                            return Err(anyhow::anyhow!(
                                "ontology apply requires `confirm: true`"
                            ));
                        }
                        if is_read_only_review_session(&session_id) {
                            return Err(anyhow::anyhow!(
                                "review sessions may propose ontology patches but not apply them; \
                                 use `jcode memory ontology apply <proposal> --apply`"
                            ));
                        }
                        let patch = patch.ok_or_else(|| {
                            anyhow::anyhow!("ontology apply requires `patch` or `patch_file`")
                        })?;
                        if input.dry_run != Some(true) {
                            crate::skill::queue_skill_reconcile(
                                "ontology_apply",
                                ctx.working_dir.as_deref(),
                            );
                        }
                        let report = manager.apply_ontology_patch(
                            scope,
                            &patch,
                            input.dry_run.unwrap_or(false),
                        )?;
                        let reconciled = if report.dry_run {
                            String::new()
                        } else {
                            reconcile_skill_links_after_ontology(ctx.working_dir.as_deref())
                        };
                        Ok(ToolOutput::new(format!(
                            "{} ontology: v{} -> v{} ({})\nmigrations: {}{}",
                            if report.dry_run { "Dry run" } else { "Applied" },
                            report.old_version,
                            report.new_version,
                            report.new_hash,
                            serde_json::to_string(&report.migrations)?,
                            reconciled
                        )))
                    }
                    "history" => {
                        let history = manager.ontology_history(scope)?;
                        if history.is_empty() {
                            Ok(ToolOutput::new(format!(
                                "No ontology history for {} scope.",
                                scope.label()
                            )))
                        } else {
                            let mut out = format!("Ontology history ({} scope):\n", scope.label());
                            for snapshot in history {
                                out.push_str(&format!(
                                    "- v{} {} {}\n",
                                    snapshot.version, snapshot.hash, snapshot.rationale
                                ));
                            }
                            Ok(ToolOutput::new(out))
                        }
                    }
                    "rollback" => {
                        if input.confirm != Some(true) {
                            return Err(anyhow::anyhow!(
                                "ontology rollback requires `confirm: true`"
                            ));
                        }
                        if is_read_only_review_session(&session_id) {
                            return Err(anyhow::anyhow!(
                                "review sessions cannot roll back the ontology"
                            ));
                        }
                        let version = input.version.ok_or_else(|| {
                            anyhow::anyhow!("ontology rollback requires `version`")
                        })?;
                        crate::skill::queue_skill_reconcile(
                            "ontology_rollback",
                            ctx.working_dir.as_deref(),
                        );
                        let snapshot = manager.rollback_ontology(scope, version)?;
                        let reconciled =
                            reconcile_skill_links_after_ontology(ctx.working_dir.as_deref());
                        Ok(ToolOutput::new(format!(
                            "Restored {} ontology v{} ({}){}",
                            scope.label(),
                            snapshot.version,
                            snapshot.hash,
                            reconciled
                        )))
                    }
                    other => Err(anyhow::anyhow!(
                        "Unknown ontology action: {}. Use show, propose, validate, apply, history or rollback",
                        other
                    )),
                }
            }
            other => Err(anyhow::anyhow!("Unknown action: {}", other)),
        }
        .map_err(|err| {
            crate::logging::warn(&format!(
                "[tool:memory] action failed action={} session_id={} error={}",
                action_label, session_id, err
            ));
            err
        })
    }
}

impl MemoryTool {
    /// Resolve checkpoint anchors against the persisted engineering graphs.
    fn readback_checkpoint(anchors: Vec<String>, depth: usize, limit: usize) -> Result<ToolOutput> {
        let mut unique = Vec::new();
        for anchor in anchors {
            let anchor = anchor.trim();
            if !anchor.is_empty() && !unique.iter().any(|seen| seen == anchor) {
                unique.push(anchor.to_string());
            }
        }
        if unique.is_empty() {
            return Err(anyhow::anyhow!(
                "readback requires anchor, from_anchors, or query containing `- `id` | Class | summary` lines"
            ));
        }
        if limit == 0 {
            return Ok(ToolOutput::new("No graph anchors requested."));
        }

        let store = EngineeringStore::open_default()
            .map_err(|error| anyhow::anyhow!("open engineering graph store: {error}"))?;
        let anchors: Vec<String> = unique.into_iter().take(limit.min(8)).collect();
        let mut out = format!(
            "Graph readback for {} anchor(s) at depth {depth}:\n",
            anchors.len()
        );
        let mut missing = 0usize;
        for anchor in &anchors {
            let recalls = store
                .recall_anchor(anchor, depth, 2)
                .map_err(|error| anyhow::anyhow!("readback engineering anchor: {error}"))?;
            if recalls.is_empty() {
                missing += 1;
                out.push_str(&format!(
                    "\n## `{anchor}`\nmissing: no stored engineering graph contains this anchor.\n"
                ));
                continue;
            }
            for recall in recalls.iter().take(2) {
                out.push_str(&format!(
                    "\n## `{anchor}` (session {})\n",
                    recall.session_id
                ));
                match recall.graph.entities.get(anchor) {
                    Some(entity) => {
                        out.push_str(&format!("class: {}\nfact: {}\n", entity.class, entity.name));
                    }
                    None => out.push_str("class: unresolved\n"),
                }
                let mut edges = 0usize;
                for relation in recall.graph.relations.values() {
                    if relation.from == *anchor || relation.to == *anchor {
                        out.push_str(&format!(
                            "- `{}` -{}-> `{}`\n",
                            relation.from, relation.relation, relation.to
                        ));
                        edges += 1;
                        if edges >= 8 {
                            break;
                        }
                    }
                }
            }
        }
        if missing > 0 {
            out.push_str(&format!(
                "\nmissing anchors: {missing}; try action=recall anchor=<id> depth=2 or memory path.\n"
            ));
        }
        Ok(ToolOutput::new(out))
    }

    /// Search persisted engineering episode graphs for typed anchor paths.
    fn path_engineering(
        froms: &[String],
        tos: &[String],
        depth: usize,
        limit: usize,
        relations: &[String],
        include_plan: bool,
        graph_kind: &str,
        include_ontology: bool,
    ) -> Result<ToolOutput> {
        if limit == 0 {
            return Ok(ToolOutput::new("No graph paths requested."));
        }
        let allowed: Vec<&str> = relations.iter().map(String::as_str).collect();
        let mut hits = Vec::new();
        if graph_kind == "ontology" {
            for from in froms {
                for to in tos {
                    hits.extend(
                        EngineeringStore::ontology_path_between_anchors(
                            from, to, depth, limit, &allowed,
                        )
                        .map_err(|error| anyhow::anyhow!("ontology path query: {error}"))?,
                    );
                }
            }
        } else {
            let store = EngineeringStore::open_default()
                .map_err(|error| anyhow::anyhow!("open engineering graph store: {error}"))?;
            for from in froms {
                for to in tos {
                    hits.extend(
                        if include_ontology {
                            store.path_between_anchors_with_ontology(
                                from, to, depth, limit, &allowed,
                            )
                        } else {
                            store.path_between_anchors(from, to, depth, limit, &allowed)
                        }
                        .map_err(|error| anyhow::anyhow!("engineering path query: {error}"))?,
                    );
                }
            }
        }
        hits.sort_by(|left, right| {
            left.path
                .cost
                .total_cmp(&right.path.cost)
                .then_with(|| left.session_id.cmp(&right.session_id))
                .then_with(|| left.path.nodes.cmp(&right.path.nodes))
        });
        let mut seen = std::collections::HashSet::new();
        hits.retain(|hit| seen.insert((hit.session_id.clone(), hit.path.nodes.clone())));
        hits.truncate(limit);

        if hits.is_empty() {
            let layer = if graph_kind == "ontology" {
                "ontology concept"
            } else {
                "engineering"
            };
            let mut out = format!(
                "No {layer} path found from [{}] to [{}] within depth {}.",
                froms.join(", "),
                tos.join(", "),
                depth.clamp(1, 16)
            );
            if !relations.is_empty() {
                out.push_str(&format!(
                    "\n- relation allow-list: {} (unknown names match nothing)",
                    relations.join(", ")
                ));
            }
            if graph_kind == "ontology" {
                let ontology =
                    crate::memory_types::engineering_ontology::default_engineering_ontology();
                let graph = crate::memory_types::EngineeringGraph::concept_graph(&ontology);
                let from_resolved = froms
                    .iter()
                    .all(|anchor| graph.resolve_concept_anchor(anchor).is_some());
                let to_resolved = tos
                    .iter()
                    .all(|anchor| graph.resolve_concept_anchor(anchor).is_some());
                let (code, reason) = if !from_resolved || !to_resolved {
                    (
                        "E01",
                        "one or more concept anchors are not declared in the ontology",
                    )
                } else {
                    (
                        "E02",
                        "concept anchors exist but no relation-signature path connects them",
                    )
                };
                out.push_str(&format!("\n- {code}: {reason}"));
            }
            return Ok(ToolOutput::new(out));
        }

        let layer = if graph_kind == "ontology" {
            "ontology concept"
        } else {
            "engineering"
        };
        let mut out = format!(
            "Found {} {layer} path(s) from [{}] to [{}] (depth {}):\n",
            hits.len(),
            froms.join(", "),
            tos.join(", "),
            depth.clamp(1, 16)
        );
        let mut current_session = None;
        for (index, hit) in hits.iter().enumerate() {
            if graph_kind != "ontology"
                && current_session.as_deref() != Some(hit.session_id.as_str())
            {
                out.push_str(&format!("\n## session {}\n", hit.session_id));
                if let Some(working_dir) = hit.working_dir.as_deref() {
                    out.push_str(&format!("working_dir: {working_dir}\n"));
                }
                current_session = Some(hit.session_id.clone());
            }
            out.push_str(&format!(
                "\nPath {}: cost {:.2}, confidence {:.0}%, {} hop(s)\n",
                index + 1,
                hit.path.cost,
                hit.path.confidence * 100.0,
                hit.path.hop_count()
            ));
            for (node_index, node_id) in hit.path.nodes.iter().enumerate() {
                let label = hit
                    .node_labels
                    .get(node_index)
                    .map(String::as_str)
                    .unwrap_or(node_id);
                out.push_str(&format!("  `{node_id}` ({label})\n"));
                if let Some(relation) = hit.path.relations.get(node_index) {
                    let relation_label = hit
                        .relation_labels
                        .get(node_index)
                        .map(String::as_str)
                        .unwrap_or(relation);
                    out.push_str(&format!("  --{relation_label}-->\n"));
                }
            }
            if include_plan {
                out.push_str(&format!(
                    "  plan: confidence {:.0}%, {} task(s), {} stage(s), {} verification(s)",
                    hit.plan.confidence * 100.0,
                    hit.plan.tasks.len(),
                    hit.plan.stage_count(),
                    hit.plan.verification_queries.len()
                ));
                if hit.plan.approval_gates.is_empty() {
                    out.push_str(", no approval gates\n");
                } else {
                    out.push_str(&format!(
                        ", approval gates: {}\n",
                        hit.plan.approval_gates.join(", ")
                    ));
                }
                for task in &hit.plan.tasks {
                    let depends = if task.depends_on.is_empty() {
                        "none".to_string()
                    } else {
                        task.depends_on.join(", ")
                    };
                    out.push_str(&format!(
                        "  {} [{}] {} (depends: {depends}, risk: {}, approval: {})\n    success: {}\n    evidence: {}\n",
                        task.id,
                        task.action_type.as_str(),
                        task.title,
                        task.risk.as_str(),
                        if task.approval { "yes" } else { "no" },
                        task.success_check,
                        task.evidence_ref
                    ));
                }
                for warning in &hit.plan.warnings {
                    out.push_str(&format!("  warning: {warning}\n"));
                }
            }
        }
        Ok(ToolOutput::new(out))
    }

    /// Read back persisted episode subgraphs around a stable graph id.
    fn recall_engineering_anchor(anchor: &str, depth: usize, limit: usize) -> Result<ToolOutput> {
        if limit == 0 {
            return Ok(ToolOutput::new("No graph anchors requested."));
        }
        let store = EngineeringStore::open_default()
            .map_err(|error| anyhow::anyhow!("open engineering graph store: {error}"))?;
        let recalls = store
            .recall_anchor(anchor, depth, limit)
            .map_err(|error| anyhow::anyhow!("recall engineering anchor: {error}"))?;
        if recalls.is_empty() {
            return Ok(ToolOutput::new(format!(
                "No stored engineering graph contains anchor `{anchor}`."
            )));
        }

        let mut out = format!(
            "Found {} stored session graph(s) for anchor `{anchor}` (depth {depth}):\n",
            recalls.len()
        );
        for recall in recalls {
            out.push_str(&format!("\n## session {}\n", recall.session_id));
            if let Some(working_dir) = recall.working_dir.as_deref() {
                out.push_str(&format!("working_dir: {working_dir}\n"));
            }
            out.push_str("entities:\n");
            for entity in recall.graph.entities.values() {
                out.push_str(&format!(
                    "- `{}` | {} | {}\n",
                    entity.id, entity.class, entity.name
                ));
            }
            if !recall.graph.relations.is_empty() {
                out.push_str("relations:\n");
                for relation in recall.graph.relations.values() {
                    out.push_str(&format!(
                        "- `{}` -{}-> `{}`\n",
                        relation.from, relation.relation, relation.to
                    ));
                }
            }
        }
        Ok(ToolOutput::new(out))
    }
}

/// Review/autoreview/judge sessions are analysis-only. They may propose an
/// ontology patch (which writes to the memory proposal directory, not the
/// repo) but must never apply or roll one back.
fn is_read_only_review_session(session_id: &str) -> bool {
    let Ok(session) = crate::session::Session::load(session_id) else {
        return false;
    };
    matches!(
        session.title.as_deref(),
        Some("review" | "autoreview" | "judge" | "autojudge")
    )
}

fn truncate_for_widget(s: &str, max: usize) -> String {
    if s.chars().count() > max {
        let truncated: String = s.chars().take(max).collect();
        format!("{}…", truncated)
    } else {
        s.to_string()
    }
}

/// Extract anchor ids from `## Graph Memory State` bullet lines.
fn parse_checkpoint_anchors(text: &str) -> Vec<String> {
    let mut anchors = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("- `") else {
            continue;
        };
        let Some((id, _)) = rest.split_once('`') else {
            continue;
        };
        let id = id.trim();
        if !id.is_empty() && !anchors.iter().any(|seen| seen == id) {
            anchors.push(id.to_string());
        }
    }
    anchors
}

fn append_decision_examples(
    out: &mut String,
    examples: &[crate::memory_types::DecisionStepMatch],
) {
    if examples.is_empty() {
        return;
    }
    out.push_str("\nDecision examples (verified, for relation reranking):\n");
    for hit in examples {
        let step = &hit.step;
        let model = step.model_version.as_deref().unwrap_or("unrecorded");
        out.push_str(&format!(
            "- score {:.2} | {} | {} -[{} / {}]-> {} | model {model} | provenance {}\n",
            hit.score,
            step.problem,
            step.start_node,
            step.relation,
            step.direction.as_str(),
            step.goal_node.as_deref().unwrap_or("(open)"),
            step.provenance
        ));
    }
}

fn append_domain_facts(out: &mut String, facts: &[crate::memory_types::DomainFactMatch]) {
    if facts.is_empty() {
        return;
    }
    out.push_str("\nDomain facts (explicit user statements):\n");
    for hit in facts {
        let fact = &hit.fact;
        out.push_str(&format!(
            "- score {:.2} | {} -[{}]-> {} | {} | provenance {}\n",
            hit.score,
            fact.subject,
            fact.predicate,
            fact.object,
            fact.statement,
            fact.source_locator
        ));
    }
}

fn memory_edge_label(kind: &crate::memory_graph::EdgeKind) -> String {
    use crate::memory_graph::EdgeKind;
    match kind {
        EdgeKind::HasTag => "has_tag".to_string(),
        EdgeKind::InCluster => "in_cluster".to_string(),
        EdgeKind::RelatesTo { weight } => format!("relates_to({weight:.2})"),
        EdgeKind::Supersedes => "supersedes".to_string(),
        EdgeKind::Contradicts => "contradicts".to_string(),
        EdgeKind::DerivedFrom => "derived_from".to_string(),
        EdgeKind::Contains => "contains".to_string(),
        EdgeKind::NextStep => "next_step".to_string(),
        EdgeKind::Bridge => "bridge".to_string(),
        EdgeKind::SameAs => "same_as".to_string(),
        EdgeKind::Hypothesized { relation, .. } => format!("hypothesized:{relation}"),
    }
}

/// Validate a user-supplied memory id. Keeps the id set small enough to be
/// safe in storage paths and printable in tool output, and avoids whitespace,
/// control characters, and JSON/scope-relevant punctuation that would let a
/// caller smuggle graph-internal separators or empty segments.
///
/// Allowed: `[A-Za-z0-9_:.-]`, length 1..=256, must not be empty after trim.
fn validate_custom_memory_id(id: &str) -> Result<()> {
    let trimmed = id.trim();
    if trimmed.is_empty() {
        return Err(anyhow::anyhow!("custom memory id must not be empty"));
    }
    if id.chars().count() > 256 {
        return Err(anyhow::anyhow!(
            "custom memory id must be at most 256 characters (got {})",
            id.chars().count()
        ));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | ':' | '-' | '.'))
    {
        return Err(anyhow::anyhow!(
            "custom memory id must contain only [A-Za-z0-9_:.\\-]; got {:?}",
            id
        ));
    }
    Ok(())
}

/// Ontology migrations can retype or archive the memories a skill cites as
/// evidence. Re-run the reconciler so mirrors record the new resolved set and
/// the next recall does not advertise a stale justification.
fn reconcile_skill_links_after_ontology(working_dir: Option<&std::path::Path>) -> String {
    let Ok(registry) = crate::skill::SkillRegistry::load_global() else {
        return String::new();
    };
    let report = crate::skill::reconcile_skill_memory_links(&registry, working_dir);
    format!(
        "\nskill-memory reconcile: {} mirror(s), {} part(s), {} pruned, {} part(s) pruned, +{} part edges, -{} part edges, +{} edges, -{} edges, {} need review",
        report.mirrored,
        report.parts_mirrored,
        report.pruned,
        report.parts_pruned,
        report.part_edges_added,
        report.part_edges_removed,
        report.edges_added,
        report.edges_removed,
        report.needs_review
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_only_advertises_core_memory_fields() {
        let schema = MemoryTool::new().parameters_schema();
        let props = schema["properties"]
            .as_object()
            .expect("memory schema should have properties");

        assert!(props.contains_key("action"));
        assert!(props.contains_key("content"));
        assert!(props.contains_key("category"));
        assert!(props.contains_key("query"));
        assert!(props.contains_key("id"));
        assert!(props.contains_key("tags"));
        assert!(props.contains_key("scope"));
        assert!(props.contains_key("from_id"));
        assert!(props.contains_key("to_id"));
        assert!(props.contains_key("limit"));
        assert!(props.contains_key("mode"));
        assert!(props.contains_key("anchor"));
        assert!(props.contains_key("extraction_action"));
        assert!(!props.contains_key("weight"));
        assert!(!props.contains_key("depth"));
    }

    #[tokio::test]
    async fn path_action_finds_weighted_route_between_memories() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let project = tempfile::tempdir().unwrap();
        let tool = MemoryTool::new();
        let ctx = test_ctx(Some(project.path().to_path_buf()));

        for (id, content) in [
            ("mem_path_a", "alpha quasar calibration note"),
            ("mem_path_b", "beta nebula observation note"),
            ("mem_path_c", "gamma pulsar timing note"),
        ] {
            let output = tool
                .execute(
                    json!({"action":"remember","id":id,"content":content,"scope":"project"}),
                    ctx.clone(),
                )
                .await
                .unwrap()
                .output;
            assert!(output.contains("Remembered"), "{output}");
        }
        for (from, to, weight) in [
            ("mem_path_a", "mem_path_c", 0.95),
            ("mem_path_c", "mem_path_b", 0.95),
        ] {
            let output = tool
                .execute(
                    json!({"action":"link","from_id":from,"to_id":to,"weight":weight,"scope":"project"}),
                    ctx.clone(),
                )
                .await
                .unwrap()
                .output;
            assert!(output.contains("Linked memories"), "{output}");
        }

        let output = tool
            .execute(
                json!({
                    "action":"path",
                    "from_id":"mem_path_a",
                    "to_id":"mem_path_b",
                    "scope":"project",
                    "depth":3,
                    "limit":2
                }),
                ctx.clone(),
            )
            .await
            .unwrap()
            .output;

        assert!(output.contains("Found 1 memory path(s)"), "{output}");
        assert!(output.contains("mem_path_c"), "{output}");
        assert!(output.contains("relates_to"), "{output}");
        assert!(output.contains("decision: terminate"), "{output}");
        assert!(output.contains("reward:"), "{output}");

        let structured = tool
            .execute(
                json!({
                    "action":"path",
                    "path_query":{
                        "start":["mem_path_a"],
                        "goal":["mem_path_b"],
                        "graph":"memory",
                        "constraints":{
                            "scope":"project",
                            "depth":3,
                            "limit":2,
                            "relations":["relates_to"]
                        }
                    }
                }),
                ctx.clone(),
            )
            .await
            .unwrap()
            .output;
        assert!(
            structured.contains("Found 1 memory path(s)"),
            "{structured}"
        );
        assert!(structured.contains("mem_path_c"), "{structured}");

        let denied = tool
            .execute(
                json!({
                    "action":"path",
                    "path_query":{
                        "start":["mem_path_a"],
                        "goal":["mem_path_b"],
                        "graph":"memory",
                        "constraints":{
                            "scope":"project",
                            "depth":3,
                            "limit":2,
                            "deny_relations":["relates_to"]
                        }
                    }
                }),
                ctx,
            )
            .await
            .expect("deny relation query")
            .output;
        assert!(denied.contains("No memory path found"), "{denied}");
        assert!(
            denied.contains("query constraints excluded all candidate paths"),
            "{denied}"
        );

        let invalid = tool
            .execute(
                json!({
                    "action":"path",
                    "path_query":{
                        "start":["mem_path_a"],
                        "goal":["mem_path_b"],
                        "constraints":{"relations":["invented_relation"]}
                    }
                }),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(invalid.contains("unknown memory relation"), "{invalid}");
    }

    #[tokio::test]
    async fn path_output_injects_verified_decision_examples() {
        use jcode_base::memory_graph::MemoryGraph;

        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let project = tempfile::tempdir().unwrap();
        let mut tool = MemoryTool::new();
        tool.manager = tool
            .manager
            .clone()
            .with_project_dir(project.path().to_path_buf());
        let ctx = test_ctx(Some(project.path().to_path_buf()));

        let mut graph = MemoryGraph::new();
        let from = graph.add_memory(MemoryEntry::new(
            MemoryCategory::Fact,
            "parser crash evidence",
        ));
        let to = graph.add_memory(MemoryEntry::new(
            MemoryCategory::Fact,
            "parser module owner",
        ));
        graph.link_memories(&from, &to, 0.9);
        tool.manager
            .save_project_graph(&graph)
            .expect("save path graph");
        tool.manager
            .record_verified_decision_step(
                "trace the parser crash to its owner",
                &from,
                "relates_to",
                crate::memory_types::DecisionDirection::Outgoing,
                Some(&to),
                Some("model-v1"),
                Some("accepted"),
                "eval:path",
            )
            .expect("record decision example");
        tool.manager
            .record_domain_facts(&[jcode_base::memory_types::DomainFact {
                id: "domain:path".to_string(),
                subject: "parser crash".to_string(),
                predicate: "uses".to_string(),
                object: "module owner".to_string(),
                statement: "parser crash uses module owner".to_string(),
                source_session: "session-path".to_string(),
                source_role: "user".to_string(),
                source_locator: "message:0#sentence:0".to_string(),
                confidence: 0.75,
                created_at: chrono::Utc::now(),
            }])
            .expect("record domain fact");

        let output = tool
            .execute(
                json!({
                    "action":"path",
                    "from_id":from,
                    "to_id":to,
                    "scope":"project",
                    "query":"parser crash owner",
                    "depth":3,
                    "limit":1
                }),
                ctx,
            )
            .await
            .unwrap()
            .output;
        assert!(
            output.contains("Decision examples (verified, for relation reranking)"),
            "{output}"
        );
        assert!(output.contains("eval:path"), "{output}");
        assert!(output.contains("relates_to / outgoing"), "{output}");
        assert!(output.contains("score "), "{output}");
        assert!(output.contains("model model-v1"), "{output}");
        assert!(
            output.contains("Domain facts (explicit user statements)"),
            "{output}"
        );
        assert!(
            output.contains("parser crash -[uses]-> module owner"),
            "{output}"
        );
    }

    #[tokio::test]
    async fn bridge_action_stitches_project_and_global_paths() {
        use jcode_base::memory_graph::MemoryGraph;

        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let project = tempfile::tempdir().unwrap();
        let mut tool = MemoryTool::new();
        tool.manager = tool
            .manager
            .clone()
            .with_project_dir(project.path().to_path_buf());
        let p0 = tool
            .manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "project bridge zero",
            ))
            .expect("p0");
        let p1 = tool
            .manager
            .remember_project(MemoryEntry::new(MemoryCategory::Fact, "project bridge one"))
            .expect("p1");
        tool.manager
            .link_memories(&p0, &p1, 0.9)
            .expect("project edge");
        let mut global = MemoryGraph::new();
        let g1 = global.add_memory(MemoryEntry::new(MemoryCategory::Fact, "global bridge one"));
        let g2 = global.add_memory(MemoryEntry::new(MemoryCategory::Fact, "global bridge two"));
        global.link_memories(&g1, &g2, 0.9);
        tool.manager
            .save_global_graph(&global)
            .expect("save global");

        let output = tool
            .execute(
                json!({"action":"bridge","from_id":p1,"to_id":g1,"weight":0.9}),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("bridge");
        assert!(
            output.output.contains("Bridged project memory"),
            "{}",
            output.output
        );

        let output = tool
            .execute(
                json!({"action":"path","from_id":p0,"to_id":g2,"scope":"all","depth":4}),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("bridge path");
        assert!(output.output.contains("bridge"), "{}", output.output);
        assert!(
            output.output.contains("global bridge two"),
            "{}",
            output.output
        );
    }

    #[tokio::test]
    async fn shadow_action_proposes_lists_and_accepts() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let project = tempfile::tempdir().unwrap();
        let tool = MemoryTool::new();
        let ctx = test_ctx(Some(project.path().to_path_buf()));

        for (id, content) in [
            ("mem_shadow_a", "alpha quasar calibration"),
            ("mem_shadow_b", "zulu nebula observation"),
        ] {
            let output = tool
                .execute(
                    json!({"action":"remember","id":id,"content":content,"scope":"project"}),
                    ctx.clone(),
                )
                .await
                .unwrap()
                .output;
            assert!(output.contains("Remembered"), "{output}");
        }

        let proposed = tool
            .execute(
                json!({
                    "action":"shadow",
                    "shadow_action":"propose",
                    "from_id":"mem_shadow_a",
                    "to_id":"mem_shadow_b",
                    "scope":"project",
                    "content":"test evidence"
                }),
                ctx.clone(),
            )
            .await
            .unwrap()
            .output;
        assert!(proposed.contains("[pending]"), "{proposed}");
        let id = proposed
            .split("shadow edge ")
            .nth(1)
            .and_then(|rest| rest.split(" [pending]").next())
            .expect("shadow edge id")
            .to_string();

        let listed = tool
            .execute(
                json!({"action":"shadow","shadow_action":"list","scope":"project"}),
                ctx.clone(),
            )
            .await
            .unwrap()
            .output;
        assert!(listed.contains(&id), "{listed}");

        let hypothesized = tool
            .execute(
                json!({
                    "action":"path",
                    "from_id":"mem_shadow_a",
                    "to_id":"mem_shadow_b",
                    "scope":"project",
                    "depth":2,
                    "include_hypotheses":true
                }),
                ctx.clone(),
            )
            .await
            .unwrap()
            .output;
        assert!(
            hypothesized.contains("Found 1 memory path(s)"),
            "{hypothesized}"
        );
        assert!(hypothesized.contains("hypothesized"), "{hypothesized}");
        assert!(hypothesized.contains("decision: reason"), "{hypothesized}");

        let accepted = tool
            .execute(
                json!({"action":"shadow","shadow_action":"accept","id":id}),
                ctx.clone(),
            )
            .await
            .unwrap()
            .output;
        assert!(accepted.contains("Accepted shadow edge"), "{accepted}");

        let path = tool
            .execute(
                json!({
                    "action":"path",
                    "from_id":"mem_shadow_a",
                    "to_id":"mem_shadow_b",
                    "scope":"project",
                    "depth":2
                }),
                ctx,
            )
            .await
            .unwrap()
            .output;
        assert!(path.contains("Found 1 memory path(s)"), "{path}");
    }

    #[tokio::test]
    async fn align_action_promotes_reviewed_same_as_edge() {
        use jcode_base::memory_graph::MemoryGraph;

        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let project = tempfile::tempdir().unwrap();
        let mut tool = MemoryTool::new();
        tool.manager = tool
            .manager
            .clone()
            .with_project_dir(project.path().to_path_buf());
        let ctx = test_ctx(Some(project.path().to_path_buf()));

        let mut graph = MemoryGraph::new();
        let mut a = MemoryEntry::new(MemoryCategory::Fact, "invoice export job owner");
        a.id = "mem_align_a".to_string();
        graph.add_memory(a);
        let mut b = MemoryEntry::new(MemoryCategory::Fact, "invoice export worker owner");
        b.id = "mem_align_b".to_string();
        graph.add_memory(b);
        tool.manager
            .save_project_graph(&graph)
            .expect("save align graph");

        let blocked = tool
            .execute(
                json!({
                    "action":"align",
                    "from_id":"mem_align_a",
                    "to_id":"mem_align_b",
                    "confidence":0.93,
                    "note":"shared invoice export ownership in runbook"
                }),
                ctx.clone(),
            )
            .await
            .expect_err("confirm is required");
        assert!(blocked.to_string().contains("confirm=true"), "{blocked}");

        let promoted = tool
            .execute(
                json!({
                    "action":"align",
                    "from_id":"mem_align_a",
                    "to_id":"mem_align_b",
                    "confidence":0.93,
                    "note":"shared invoice export ownership in runbook",
                    "confirm":true
                }),
                ctx.clone(),
            )
            .await
            .expect("align")
            .output;
        assert!(promoted.contains("Promoted alignment"), "{promoted}");
        assert!(promoted.contains("same_as"), "{promoted}");

        let path = tool
            .execute(
                json!({
                    "action":"path",
                    "from_id":"mem_align_a",
                    "to_id":"mem_align_b",
                    "scope":"project",
                    "depth":2,
                    "limit":1
                }),
                ctx,
            )
            .await
            .expect("same_as path")
            .output;
        assert!(path.contains("same_as"), "{path}");
    }

    #[tokio::test]
    async fn embed_action_reports_empty_store_without_loading_a_model() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let project = tempfile::tempdir().unwrap();
        let tool = MemoryTool::new();
        let output = tool
            .execute(
                json!({"action":"embed","scope":"project","limit":1}),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .unwrap()
            .output;
        assert!(output.contains("0 generated"), "{output}");
        assert!(output.contains("0 still missing"), "{output}");
    }

    fn test_ctx(working_dir: Option<std::path::PathBuf>) -> ToolContext {
        ToolContext {
            session_id: "test-session".to_string(),
            message_id: "test-message".to_string(),
            tool_call_id: "test-tool-call".to_string(),
            working_dir,
            stdin_request_tx: None,
            graceful_shutdown_signal: None,
            execution_mode: crate::tool::ToolExecutionMode::Direct,
        }
    }

    /// Sandbox disk credentials as well as environment keys. Restore even when
    /// an assertion fails, so these public-interface tests never use real keys.
    struct OfflineEnv(Vec<(&'static str, Option<std::ffi::OsString>)>);

    impl OfflineEnv {
        fn new(home: &std::path::Path) -> Self {
            let keys = [
                "JCODE_HOME",
                "OPENROUTER_API_KEY",
                "JCODE_API_KEY",
                "TYPESAFE_API_KEY",
                "AIMLAPI_API_KEY",
                "JCODE_MEMORY_JEV_PROVIDER",
            ];
            let previous = keys
                .into_iter()
                .map(|key| (key, std::env::var_os(key)))
                .collect();
            crate::env::set_var("JCODE_HOME", home);
            for key in &keys[1..] {
                crate::env::remove_var(key);
            }
            Self(previous)
        }
    }

    impl Drop for OfflineEnv {
        fn drop(&mut self) {
            for (key, previous) in &self.0 {
                if let Some(value) = previous {
                    crate::env::set_var(key, value);
                } else {
                    crate::env::remove_var(key);
                }
            }
        }
    }

    #[tokio::test]
    async fn recall_anchor_reads_persisted_engineering_graph() {
        use crate::message::{ContentBlock, Message, Role};

        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".to_string(),
                    name: "edit".to_string(),
                    input: json!({
                        "file_path": "src/lib.rs",
                        "old_string": "a",
                        "new_string": "b",
                    }),
                    thought_signature: None,
                }]
                .into(),
                timestamp: None,
                tool_duration_ms: None,
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call-1".to_string(),
                    content: "ok".to_string(),
                    is_error: Some(false),
                }]
                .into(),
                timestamp: None,
                tool_duration_ms: None,
            },
        ];
        let extraction = jcode_base::engineering_episode::extract_engineering_episode(
            "session-anchor",
            &messages,
            Some("/repo"),
        );
        EngineeringStore::open_default()
            .expect("store")
            .save_graph("session-anchor", Some("/repo"), &extraction.graph)
            .expect("save graph");

        let tool = MemoryTool::new();
        let output = tool
            .execute(
                json!({"action":"recall", "anchor":"file:src/lib.rs", "depth":1}),
                test_ctx(None),
            )
            .await
            .expect("anchor recall");
        assert!(
            output.output.contains("session-anchor"),
            "{}",
            output.output
        );
        assert!(
            output.output.contains("file:src/lib.rs"),
            "{}",
            output.output
        );
        assert!(output.output.contains("modifies"), "{}", output.output);

        let readback = tool
            .execute(
                json!({
                    "action":"readback",
                    "query":"## Graph Memory State\n- `file:src/lib.rs` | File | src/lib.rs\n- `missing:anchor` | File | missing.rs",
                    "depth":1,
                    "limit":4
                }),
                test_ctx(None),
            )
            .await
            .expect("checkpoint readback");
        assert!(
            readback.output.contains("Graph readback for 2 anchor(s)"),
            "{}",
            readback.output
        );
        assert!(
            readback.output.contains("class: file"),
            "{}",
            readback.output
        );
        assert!(readback.output.contains("modifies"), "{}", readback.output);
        assert!(
            readback
                .output
                .contains("missing: no stored engineering graph"),
            "{}",
            readback.output
        );
    }

    #[tokio::test]
    async fn path_action_reads_persisted_engineering_graph() {
        use crate::message::{ContentBlock, Message, Role};

        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".to_string(),
                    name: "edit".to_string(),
                    input: json!({
                        "file_path": "src/lib.rs",
                        "old_string": "a",
                        "new_string": "b",
                    }),
                    thought_signature: None,
                }]
                .into(),
                timestamp: None,
                tool_duration_ms: None,
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call-1".to_string(),
                    content: "ok".to_string(),
                    is_error: Some(false),
                }]
                .into(),
                timestamp: None,
                tool_duration_ms: None,
            },
        ];
        let extraction = jcode_base::engineering_episode::extract_engineering_episode(
            "session-path",
            &messages,
            Some("/repo"),
        );
        EngineeringStore::open_default()
            .expect("store")
            .save_graph("session-path", Some("/repo"), &extraction.graph)
            .expect("save graph");

        let tool = MemoryTool::new();
        let output = tool
            .execute(
                json!({
                    "action":"path",
                    "graph":"engineering",
                    "from_id":"action:call-1",
                    "to_id":"file:src/lib.rs",
                    "relation_allow":["modifies"],
                    "plan":true,
                    "depth":2
                }),
                test_ctx(None),
            )
            .await
            .expect("engineering path");
        assert!(
            output.output.contains("Found 1 engineering path(s)"),
            "{}",
            output.output
        );
        assert!(
            output.output.contains("session-path"),
            "{}",
            output.output
        );
        assert!(output.output.contains("modifies"), "{}", output.output);
        assert!(
            output.output.contains("file:src/lib.rs"),
            "{}",
            output.output
        );
        assert!(output.output.contains("plan: confidence"), "{}", output.output);
        assert!(output.output.contains("success:"), "{}", output.output);
        assert!(output.output.contains("evidence:"), "{}", output.output);
    }

    #[tokio::test]
    async fn path_action_queries_ontology_concept_layer() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let tool = MemoryTool::new();
        let output = tool
            .execute(
                json!({
                    "action":"path",
                    "graph":"ontology",
                    "from_id":"File",
                    "to_id":"API",
                    "relation_allow":["defines"],
                    "plan":true,
                    "depth":2
                }),
                test_ctx(None),
            )
            .await
            .expect("ontology path");
        assert!(
            output.output.contains("Found 1 ontology concept path(s)"),
            "{}",
            output.output
        );
        assert!(output.output.contains("concept:file"), "{}", output.output);
        assert!(output.output.contains("concept:api"), "{}", output.output);
        assert!(output.output.contains("defines"), "{}", output.output);
        assert!(output.output.contains("plan: confidence"), "{}", output.output);
    }

    #[tokio::test]
    async fn ontology_path_reports_e01_and_e02() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let tool = MemoryTool::new();

        let e02 = tool
            .execute(
                json!({
                    "action":"path",
                    "graph":"ontology",
                    "from_id":"Agent",
                    "to_id":"File",
                    "depth":3
                }),
                test_ctx(None),
            )
            .await
            .expect("e02 query")
            .output;
        assert!(e02.contains("E02"), "{e02}");
        assert!(e02.contains("relation-signature path"), "{e02}");

        let e01 = tool
            .execute(
                json!({
                    "action":"path",
                    "graph":"ontology",
                    "from_id":"File",
                    "to_id":"NotAClass",
                    "depth":3
                }),
                test_ctx(None),
            )
            .await
            .expect("e01 query")
            .output;
        assert!(e01.contains("E01"), "{e01}");
    }

    #[tokio::test]
    async fn path_action_joins_engineering_instances_through_ontology() {
        use crate::memory_types::{EngineeringEntity, EngineeringGraph};

        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let mut graph = EngineeringGraph::new();
        graph.insert_entity(EngineeringEntity::new("file", "file:joined", "joined.rs"));
        graph.insert_entity(EngineeringEntity::new("api", "api:joined", "joined api"));
        EngineeringStore::open_default()
            .expect("store")
            .save_graph("ontology-join", None, &graph)
            .expect("save graph");

        let tool = MemoryTool::new();
        let output = tool
            .execute(
                json!({
                    "action":"path",
                    "graph":"engineering",
                    "include_ontology":true,
                    "from_id":"file:joined",
                    "to_id":"api:joined",
                    "depth":4
                }),
                test_ctx(None),
            )
            .await
            .expect("ontology-joined engineering path");
        assert!(
            output.output.contains("Found 1 engineering path(s)"),
            "{}",
            output.output
        );
        assert!(output.output.contains("concept:file"), "{}", output.output);
        assert!(output.output.contains("concept:api"), "{}", output.output);
        assert!(output.output.contains("instance_of"), "{}", output.output);
        assert!(output.output.contains("has_instance"), "{}", output.output);
    }

    #[tokio::test]
    async fn compile_action_returns_structured_path_query() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let tool = MemoryTool::new();
        let output = tool
            .execute(
                json!({
                    "action":"compile",
                    "query":"from \"file:src/lib.rs\" to \"api:/task\" graph engineering relations defines implements depth 3 limit 2"
                }),
                test_ctx(None),
            )
            .await
            .expect("compile");
        assert!(output.output.contains("Compiled PathQuery"), "{}", output.output);
        assert!(output.output.contains("\"engineering\""), "{}", output.output);
        assert!(output.output.contains("file:src/lib.rs"), "{}", output.output);
        assert!(output.output.contains("api:/task"), "{}", output.output);
        assert!(output.output.contains("\"defines\""), "{}", output.output);
        assert!(output.output.contains("\"depth\": 3"), "{}", output.output);
    }

    #[tokio::test]
    async fn path_action_reports_e07_when_budget_is_exhausted() {
        use crate::memory_graph::MemoryGraph;

        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let project = tempfile::tempdir().unwrap();
        let mut tool = MemoryTool::new();
        tool.manager = tool
            .manager
            .clone()
            .with_project_dir(project.path().to_path_buf());
        let mut graph = MemoryGraph::new();
        let a = graph.add_memory(MemoryEntry::new(MemoryCategory::Fact, "budget one"));
        let b = graph.add_memory(MemoryEntry::new(MemoryCategory::Fact, "budget two"));
        graph.add_edge(
            &a,
            &b,
            crate::memory_graph::EdgeKind::RelatesTo { weight: 0.9 },
        );
        tool.manager
            .save_project_graph(&graph)
            .expect("save budget graph");

        let output = tool
            .execute(
                json!({
                    "action":"path",
                    "from_id":a,
                    "to_id":b,
                    "scope":"project",
                    "depth":2,
                    "query_budget_ms":0
                }),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("budgeted path")
            .output;
        assert!(output.contains("E07"), "{output}");
    }

    #[tokio::test]
    async fn quality_action_reports_graph_metrics() {
        use crate::memory_graph::MemoryGraph;

        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let project = tempfile::tempdir().unwrap();
        let mut tool = MemoryTool::new();
        tool.manager = tool
            .manager
            .clone()
            .with_project_dir(project.path().to_path_buf());
        let mut graph = MemoryGraph::new();
        let a = graph.add_memory(MemoryEntry::new(MemoryCategory::Fact, "quality one"));
        let b = graph.add_memory(MemoryEntry::new(MemoryCategory::Fact, "quality two"));
        graph.add_memory(MemoryEntry::new(MemoryCategory::Fact, "quality orphan"));
        graph.add_edge(
            &a,
            &b,
            crate::memory_graph::EdgeKind::SameAs,
        );
        tool.manager
            .save_project_graph(&graph)
            .expect("save quality graph");

        let output = tool
            .execute(
                json!({"action":"quality","scope":"project"}),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("quality report")
            .output;
        assert!(output.contains("Graph quality (project)"), "{output}");
        assert!(output.contains("\"memories\": 3"), "{output}");
        assert!(output.contains("\"alignment_edges\": 1"), "{output}");
        assert!(output.contains("\"orphan_ratio\""), "{output}");
        assert!(output.contains("\"provenance_coverage\""), "{output}");
        assert!(output.contains("\"findings\""), "{output}");
        assert!(output.contains("\"debt\""), "{output}");
        assert!(output.contains("\"sla_days\": 7"), "{output}");
        assert!(output.contains("\"status\": \"open\""), "{output}");
        assert!(output.contains("Q01"), "{output}");
        assert!(output.contains("\"healthy\": false"), "{output}");
    }

    #[tokio::test]
    async fn health_action_resumes_paused_extraction() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let memory_dir = home.path().join("memory");
        std::fs::create_dir_all(&memory_dir).unwrap();
        let health_path = memory_dir.join("extraction_health.json");
        std::fs::write(
            &health_path,
            r#"{"paused":true,"pause_reason":"test pause","batches":[]}"#,
        )
        .unwrap();

        let tool = MemoryTool::new();
        let output = tool
            .execute(
                json!({"action":"health", "extraction_action":"resume"}),
                test_ctx(None),
            )
            .await
            .expect("resume extraction");
        assert!(
            output.output.contains("Extraction resumed"),
            "{}",
            output.output
        );

        let health: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&health_path).unwrap()).unwrap();
        assert_eq!(health["paused"], false);
        assert!(health["pause_reason"].is_null());
    }

    #[tokio::test]
    async fn queried_recall_uses_jev_for_default_and_legacy_aliases() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let project = tempfile::tempdir().unwrap();
        let tool = MemoryTool::new();
        let ctx = || test_ctx(Some(project.path().to_path_buf()));
        tool.execute(
            json!({"action":"remember", "content":"jev-only-recall-probe", "scope":"project"}),
            ctx(),
        )
        .await
        .unwrap();

        for mode in [None, Some("jev"), Some("semantic"), Some("cascade")] {
            let mut input =
                json!({"action":"recall", "query":"jev-only-recall-probe", "scope":"project"});
            if let Some(mode) = mode {
                input["mode"] = json!(mode);
            }
            let error = tool
                .execute(input.clone(), ctx())
                .await
                .expect_err("populated query recall without credentials must fail closed");
            assert!(error.to_string().contains("Jev"), "{mode:?}: {error}");
            assert!(!error.to_string().contains("jev-only-recall-probe"));

            input["limit"] = json!(0);
            let output = tool
                .execute(input, ctx())
                .await
                .expect("zero limit must not require a key or a network request");
            assert!(!output.output.contains("id:"), "{}", output.output);
        }

        // An empty selected scope is not allowed to borrow candidates from
        // another scope (which would also require unavailable Jev credentials).
        let empty = tool
            .execute(
                json!({"action":"recall", "query":"probe", "scope":"global"}),
                ctx(),
            )
            .await
            .unwrap();
        assert!(empty.output.starts_with("No memories found"));
        let other_project = tempfile::tempdir().unwrap();
        let empty = tool
            .execute(
                json!({"action":"recall", "query":"probe", "scope":"project"}),
                test_ctx(Some(other_project.path().to_path_buf())),
            )
            .await
            .unwrap();
        assert!(empty.output.starts_with("No memories found"));
    }

    #[tokio::test]
    async fn local_memory_operations_honor_scope_and_limits_without_credentials() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let project = tempfile::tempdir().unwrap();
        let tool = MemoryTool::new();
        let ctx = || test_ctx(Some(project.path().to_path_buf()));
        let mut remembered_id = None;
        for (scope, content) in [
            ("project", "local-probe project alpha storage 4311"),
            ("project", "local-probe project beta release 4312"),
            ("global", "local-probe global gamma"),
        ] {
            let output = tool
                .execute(
                    json!({"action":"remember", "content":content, "scope":scope}),
                    ctx(),
                )
                .await
                .unwrap();
            assert!(
                output.output.contains("Remembered") && output.output.contains(content),
                "{}",
                output.output
            );
            if scope == "global" {
                remembered_id = output
                    .output
                    .rsplit_once("[id: ")
                    .map(|(_, tail)| tail.trim_end_matches(']').to_string());
            }
        }

        for action in ["list", "search"] {
            for (scope, expected) in [("project", 2), ("global", 1), ("all", 3)] {
                for limit in [None, Some(0), Some(1), Some(10)] {
                    let mut input = json!({"action":action, "query":"local-probe", "scope":scope});
                    if let Some(limit) = limit {
                        input["limit"] = json!(limit);
                    }
                    let output = tool.execute(input, ctx()).await.unwrap().output;
                    assert_eq!(
                        output.matches("\n  id: ").count(),
                        limit.unwrap_or(expected).min(expected),
                        "{action} {scope} {limit:?}: {output}"
                    );
                    if scope == "project" {
                        assert!(!output.contains("global gamma"));
                    } else if scope == "global" {
                        assert!(
                            !output.contains("project alpha") && !output.contains("project beta")
                        );
                    }
                }
            }
        }

        for mode in [None, Some("recent")] {
            let mut input = json!({"action":"recall", "scope":"project", "limit":1});
            if let Some(mode) = mode {
                input["mode"] = json!(mode);
            }
            let output = tool.execute(input.clone(), ctx()).await.unwrap().output;
            assert_eq!(output.matches("local-probe").count(), 1);
            assert!(!output.contains("global gamma"));
            input["limit"] = json!(0);
            let output = tool.execute(input, ctx()).await.unwrap().output;
            assert!(!output.contains("local-probe"));
        }

        let forgotten = tool
            .execute(
                json!({"action":"forget", "id":remembered_id.expect("remember returned an ID")}),
                ctx(),
            )
            .await
            .unwrap();
        assert!(forgotten.output.starts_with("Forgot:"));
        let remaining = tool
            .execute(json!({"action":"list", "scope":"all"}), ctx())
            .await
            .unwrap();
        assert_eq!(remaining.output.matches("\n  id: ").count(), 2);
        assert!(!remaining.output.contains("global gamma"));
    }

    #[tokio::test]
    async fn remember_records_user_stated_provenance_and_locator() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let project = tempfile::tempdir().unwrap();
        let tool = MemoryTool::new();

        let output = tool
            .execute(
                json!({"action":"remember", "content":"manual provenance probe"}),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .unwrap()
            .output;
        let id = output
            .rsplit_once("[id: ")
            .and_then(|(_, tail)| tail.split(']').next())
            .map(str::to_string)
            .expect("remember returned an ID");

        let graph = tool
            .manager
            .clone()
            .with_project_dir(project.path())
            .load_project_graph()
            .unwrap();
        let provenance = graph
            .get_memory(&id)
            .expect("remembered memory")
            .provenance
            .as_ref()
            .expect("manual remember should carry provenance");
        assert_eq!(
            provenance.method,
            crate::memory::ExtractionMethod::UserStated
        );
        assert_eq!(provenance.locator.as_deref(), Some("session:test-session"));
    }

    /// Issue #491 regression: project-scoped remember followed by list must
    /// round-trip through the real (non-test-mode) manager when the tool
    /// context carries a working dir.
    #[tokio::test]
    async fn project_scope_round_trips_with_working_dir() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().expect("home");
        let project = tempfile::tempdir().expect("project");
        let prev_home = std::env::var_os("JCODE_HOME");
        crate::env::set_var("JCODE_HOME", home.path());

        let tool = MemoryTool::new();
        let remember = tool
            .execute(
                json!({
                    "action": "remember",
                    "content": "issue-491-probe",
                    "scope": "project"
                }),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("remember should succeed");
        assert!(remember.output.contains("issue-491-probe"));

        let list = tool
            .execute(
                json!({ "action": "list", "scope": "project" }),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("list should succeed");
        assert!(
            list.output.contains("issue-491-probe"),
            "project-scoped memory must persist and be listed, got: {}",
            list.output
        );

        if let Some(prev_home) = prev_home {
            crate::env::set_var("JCODE_HOME", prev_home);
        } else {
            crate::env::remove_var("JCODE_HOME");
        }
    }

    /// Issue #729 regression, behavioral rather than structural.
    ///
    /// `create_headless_session` used to call `enable_memory_test_mode()`
    /// unconditionally, so real swarm-spawned workers got throwaway storage and
    /// could never read what the session that spawned them remembered. The fix
    /// makes isolation an explicit per-caller choice, but the property that
    /// actually matters to a user is this: with the same working directory, a
    /// default registry's memory tool sees what was written, and a test-mode
    /// one does not.
    ///
    /// Driving `Tool::execute` (rather than inspecting a flag) means this stays
    /// honest even if the internals are refactored.
    #[tokio::test]
    async fn swarm_worker_memory_sees_the_spawning_session_only_without_isolation() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().expect("home");
        let project = tempfile::tempdir().expect("project");
        let prev_home = std::env::var_os("JCODE_HOME");
        crate::env::set_var("JCODE_HOME", home.path());

        // The session that spawns a worker records something project-scoped.
        let spawner = MemoryTool::new();
        spawner
            .execute(
                json!({
                    "action": "remember",
                    "content": "issue-729-spawner-note",
                    "scope": "project"
                }),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("spawner remember should succeed");

        // A worker that kept real memory (the fixed path) must see it.
        let worker = MemoryTool::new();
        let seen = worker
            .execute(
                json!({ "action": "list", "scope": "project" }),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("worker list should succeed");
        assert!(
            seen.output.contains("issue-729-spawner-note"),
            "a swarm worker must see the spawning session's project memory, got: {}",
            seen.output
        );

        // A worker forced into test mode (the pre-fix path) cannot, no matter
        // that it has the identical working directory. This is the defect.
        let isolated = MemoryTool::new_test();
        let blind = isolated
            .execute(
                json!({ "action": "list", "scope": "project" }),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("isolated list should succeed");
        assert!(
            !blind.output.contains("issue-729-spawner-note"),
            "test mode unexpectedly saw real project memory, so this test cannot \
             distinguish the two paths: {}",
            blind.output
        );

        if let Some(prev_home) = prev_home {
            crate::env::set_var("JCODE_HOME", prev_home);
        } else {
            crate::env::remove_var("JCODE_HOME");
        }
    }

    /// Ontology scope gate: environment-specific content must be rejected
    /// from global scope with a redirect message, not a hard error.
    #[tokio::test]
    async fn remember_global_gate_rejects_environment_specific() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().expect("home");
        let project = tempfile::tempdir().expect("project");
        let prev_home = std::env::var_os("JCODE_HOME");
        crate::env::set_var("JCODE_HOME", home.path());

        let tool = MemoryTool::new();
        let out = tool
            .execute(
                json!({
                    "action": "remember",
                    "content": "部署机 = 192.168.6.33, SSH root 免密, live 容器 kiagents-v2-chat-node",
                    "category": "fact",
                    "scope": "global"
                }),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("gate rejection must be a graceful redirect, not an error");
        let text = out.output.clone();
        assert!(
            text.contains("scope gate"),
            "expected redirect message, got: {text}"
        );

        let generalized = tool
            .execute(
                json!({
                    "action": "remember",
                    "content": "教训：更新记忆前必须核对事实准确性，矛盾时应先修正旧条目",
                    "category": "fact",
                    "scope": "global"
                }),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("generalized content should pass the gate");
        assert!(generalized.output.contains("Remembered"));

        if let Some(prev) = prev_home {
            crate::env::set_var("JCODE_HOME", prev);
        } else {
            crate::env::remove_var("JCODE_HOME");
        }
    }

    /// Regression: a user-supplied `id` must be honored by `remember`, not
    /// silently replaced by an auto-generated one. The pre-fix tool accepted
    /// the `id` field on input but never propagated it to `MemoryEntry`,
    /// so callers that tried to thread a stable id through (e.g. deterministic
    /// skill:NAME keys) got back a different id on every call.
    #[tokio::test]
    async fn remember_honors_supplied_id() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().expect("home");
        let project = tempfile::tempdir().expect("project");
        let prev_home = std::env::var_os("JCODE_HOME");
        crate::env::set_var("JCODE_HOME", home.path());

        let tool = MemoryTool::new();
        let out = tool
            .execute(
                json!({
                    "action": "remember",
                    "content": "issue-v8-supplied-id",
                    "category": "fact",
                    "scope": "project",
                    "id": "skill:deploy-pipeline"
                }),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("remember with custom id should succeed");
        assert!(
            out.output.contains("[id: skill:deploy-pipeline]"),
            "supplied id must round-trip to the result line, got: {}",
            out.output
        );

        // The stored memory must be reachable via the supplied id through the
        // real manager — proves the id is committed to the graph, not just
        // echoed in the response message.
        let list = tool
            .execute(
                json!({ "action": "list", "scope": "project" }),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("list should succeed");
        assert!(
            list.output.contains("skill:deploy-pipeline"),
            "memory with custom id must appear under that id in list, got: {}",
            list.output
        );

        if let Some(prev_home) = prev_home {
            crate::env::set_var("JCODE_HOME", prev_home);
        } else {
            crate::env::remove_var("JCODE_HOME");
        }
    }

    /// A supplied id that already exists must be rejected — silent overwrite
    /// via `add_memory`'s `HashMap::insert` would be a data-loss bug. Callers
    /// who really want to refresh content should `forget` first, or we can
    /// add a separate `upsert` action later.
    #[tokio::test]
    async fn remember_rejects_supplied_id_that_already_exists() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().expect("home");
        let project = tempfile::tempdir().expect("project");
        let prev_home = std::env::var_os("JCODE_HOME");
        crate::env::set_var("JCODE_HOME", home.path());

        let tool = MemoryTool::new();
        let first = tool
            .execute(
                json!({
                    "action": "remember",
                    "content": "first write",
                    "scope": "project",
                    "id": "skill:duplicate-probe"
                }),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("first remember should succeed");
        assert!(first.output.contains("skill:duplicate-probe"));

        let _second = tool
            .execute(
                json!({
                    "action": "remember",
                    "content": "second write should be rejected",
                    "scope": "project",
                    "id": "skill:duplicate-probe"
                }),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect_err("second remember with same id must error");
        // Confirm the original content was not overwritten: the surviving
        // memory must still contain "first write", not the second-write text.
        let list = tool
            .execute(
                json!({ "action": "list", "scope": "project" }),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("list should succeed");
        assert!(
            list.output.contains("first write"),
            "original content must survive a rejected collision, got: {}",
            list.output
        );
        assert!(
            !list.output.contains("second write"),
            "rejected collision must not have written new content, got: {}",
            list.output
        );

        if let Some(prev_home) = prev_home {
            crate::env::set_var("JCODE_HOME", prev_home);
        } else {
            crate::env::remove_var("JCODE_HOME");
        }
    }

    /// Defensive: malformed ids are rejected at validation time, not after a
    /// graph write. Pins the allowed character set so we don't accidentally
    /// widen it without thinking about graph-internal separators (the `tag:`
    /// prefix used by tag nodes, the `:` that joins kind+key, etc.).
    #[tokio::test]
    async fn remember_rejects_malformed_supplied_id() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().expect("home");
        let project = tempfile::tempdir().expect("project");
        let prev_home = std::env::var_os("JCODE_HOME");
        crate::env::set_var("JCODE_HOME", home.path());

        let tool = MemoryTool::new();
        for bad in [
            "",             // empty
            "   ",          // whitespace only
            "has space",    // embedded whitespace
            "has/slash",    // path separator
            "has\nnewline", // control character
            "has\"quote",   // JSON-relevant punctuation
        ] {
            let result = tool
                .execute(
                    json!({
                        "action": "remember",
                        "content": "should never persist",
                        "scope": "project",
                        "id": bad
                    }),
                    test_ctx(Some(project.path().to_path_buf())),
                )
                .await;
            assert!(
                result.is_err(),
                "malformed id {:?} must be rejected, got {:?}",
                bad,
                result.map(|o| o.output)
            );
        }

        // None of the rejected writes should have leaked into storage.
        let list = tool
            .execute(
                json!({ "action": "list", "scope": "project" }),
                test_ctx(Some(project.path().to_path_buf())),
            )
            .await
            .expect("list should succeed");
        assert!(
            !list.output.contains("should never persist"),
            "no malformed-id write should have committed, got: {}",
            list.output
        );

        if let Some(prev_home) = prev_home {
            crate::env::set_var("JCODE_HOME", prev_home);
        } else {
            crate::env::remove_var("JCODE_HOME");
        }
    }
    #[tokio::test]
    async fn ontology_tool_show_propose_and_confirmed_apply() {
        use serde_json::json;

        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let project = home.path().join("project");
        std::fs::create_dir_all(&project).unwrap();

        let tool = MemoryTool::new();
        let ctx = test_ctx(Some(project));

        let show = tool
            .execute(
                json!({"action":"ontology","ontology_action":"show","scope":"project"}),
                ctx.clone(),
            )
            .await
            .unwrap();
        assert!(show.output.contains("jcode/default/v1"), "{}", show.output);

        let patch = json!({
            "base_id": "jcode/default/v1",
            "base_version": crate::memory_types::ontology::DEFAULT_ONTOLOGY_VERSION,
            "base_hash": "",
            "rationale": "tool test: add a business concept",
            "ops": [{
                "op": "add_type",
                "type_def": {
                    "id": "test.business_contract",
                    "display_name": "Test Business Contract",
                    "description": "Tool test business concept.",
                    "ttl_days": 45.0
                }
            }]
        });

        let propose = tool
            .execute(
                json!({"action":"ontology","ontology_action":"propose","scope":"project","patch":patch}),
                ctx.clone(),
            )
            .await
            .unwrap();
        assert!(
            propose.output.contains("Ontology proposal written"),
            "{}",
            propose.output
        );
        let proposals = home.path().join("memory/ontology/.proposals");
        assert!(
            std::fs::read_dir(&proposals).unwrap().next().is_some(),
            "proposal file must exist"
        );

        let apply = json!({
            "action":"ontology",
            "ontology_action":"apply",
            "scope":"project",
            "confirm": true,
            "patch": patch
        });
        let applied = tool.execute(apply, ctx.clone()).await.unwrap();
        assert!(
            applied.output.contains("Applied ontology"),
            "{}",
            applied.output
        );

        let updated = tool
            .execute(
                json!({"action":"ontology","ontology_action":"show","scope":"project"}),
                ctx,
            )
            .await
            .unwrap();
        assert!(updated.output.contains("ttl=45d"), "{}", updated.output);
    }

    #[tokio::test]
    async fn belief_action_initializes_and_carries_state() {
        let _guard = crate::storage::lock_test_env();
        let tool = MemoryTool::new();
        let ctx = test_ctx(None);

        let first = tool
            .execute(
                json!({
                    "action":"belief",
                    "belief":{
                        "goal":"find the root cause",
                        "max_steps":4,
                        "observation":{
                            "kind":"path_missing",
                            "from":"error:E0425",
                            "to":"file:src/lib.rs",
                            "diagnosis":["E03"],
                            "missing_class":"change"
                        }
                    }
                }),
                ctx.clone(),
            )
            .await
            .unwrap();
        assert!(first.output.contains("Belief step 1 / 4"), "{}", first.output);
        assert!(first.output.contains("\"path_missing\""), "{}", first.output);
        assert!(first.output.contains("\"action\": \"complete\""), "{}", first.output);
        assert!(first.output.contains("\"open_diagnoses\""), "{}", first.output);

        let payload_start = first.output.find("\n\n").expect("summary separator") + 2;
        let payload: Value =
            serde_json::from_str(&first.output[payload_start..]).expect("belief payload JSON");
        let state = payload["belief"].clone();
        assert_eq!(state["step"], 1);

        let second = tool
            .execute(
                json!({
                    "action":"belief",
                    "belief":{
                        "state":state,
                        "observation":{
                            "kind":"path_found",
                            "from":"error:E0425",
                            "to":"file:src/lib.rs",
                            "path_confidence":0.9,
                            "cost":1.0,
                            "quality_flags":[]
                        }
                    }
                }),
                ctx,
            )
            .await
            .unwrap();
        assert!(second.output.contains("Belief step 2 / 4"), "{}", second.output);
        assert!(second.output.contains("\"path_found\""), "{}", second.output);
        assert!(
            second.output.contains("\"action\": \"terminate\""),
            "{}",
            second.output
        );
        assert!(
            second.output.contains("\"open_diagnoses\": []"),
            "{}",
            second.output
        );
    }

    #[tokio::test]
    async fn decision_action_records_searches_and_forgets() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let project = tempfile::tempdir().unwrap();
        let mut tool = MemoryTool::new();
        tool.manager = tool
            .manager
            .clone()
            .with_project_dir(project.path().to_path_buf());
        let ctx = test_ctx(Some(project.path().to_path_buf()));

        let recorded = tool
            .execute(
                json!({
                    "action":"decision",
                    "decision":{
                        "action":"record",
                        "problem":"fix parser crash",
                        "start_node":"module:parser",
                        "relation":"causes",
                        "direction":"outgoing",
                        "goal_node":"error:panic",
                        "model_version":"model-v1",
                        "outcome":"accepted",
                        "provenance":"eval:parser"
                    }
                }),
                ctx.clone(),
            )
            .await
            .unwrap();
        assert!(
            recorded.output.contains("Recorded verified decision"),
            "{}",
            recorded.output
        );
        let payload_start = recorded
            .output
            .find("\n\n")
            .expect("record summary separator")
            + 2;
        let recorded_json: Value =
            serde_json::from_str(&recorded.output[payload_start..]).expect("decision JSON");
        let id = recorded_json["id"].as_str().expect("decision id").to_string();
        assert_eq!(recorded_json["source"], "verified");

        let rejected = tool
            .execute(
                json!({
                    "action":"decision",
                    "decision":{
                        "action":"record",
                        "source":"candidate",
                        "problem":"unsafe example",
                        "start_node":"module:parser",
                        "relation":"causes"
                    }
                }),
                ctx.clone(),
            )
            .await;
        assert!(rejected.is_err(), "candidate traces must be rejected");

        let searched = tool
            .execute(
                json!({
                    "action":"decision",
                    "decision":{
                        "action":"search",
                        "problem":"parser crash root cause",
                        "start_node":"module:parser",
                        "limit":3
                    }
                }),
                ctx.clone(),
            )
            .await
            .unwrap();
        assert!(
            searched.output.contains("Found 1 decision example"),
            "{}",
            searched.output
        );
        assert!(searched.output.contains("eval:parser"), "{}", searched.output);
        assert!(searched.output.contains("\"score\""), "{}", searched.output);

        let listed = tool
            .execute(
                json!({"action":"decision","decision":{"action":"list"}}),
                ctx.clone(),
            )
            .await
            .unwrap();
        assert!(
            listed.output.contains("1 verified decision step"),
            "{}",
            listed.output
        );

        let forgotten = tool
            .execute(
                json!({"action":"decision","decision":{"action":"forget","id":id}}),
                ctx,
            )
            .await
            .unwrap();
        assert!(
            forgotten.output.contains("Forgot decision step"),
            "{}",
            forgotten.output
        );
    }

    #[tokio::test]
    async fn domain_action_records_searches_and_forgets() {
        let _guard = crate::storage::lock_test_env();
        let home = tempfile::tempdir().unwrap();
        let _env = OfflineEnv::new(home.path());
        let project = tempfile::tempdir().unwrap();
        let mut tool = MemoryTool::new();
        tool.manager = tool
            .manager
            .clone()
            .with_project_dir(project.path().to_path_buf());
        let ctx = test_ctx(Some(project.path().to_path_buf()));

        let recorded = tool
            .execute(
                json!({
                    "action":"domain",
                    "domain":{
                        "action":"record",
                        "facts":[{
                            "id":"domain:billing",
                            "subject":"Billing service",
                            "predicate":"uses",
                            "object":"Stripe",
                            "statement":"Billing service uses Stripe",
                            "source_session":"session-domain",
                            "source_role":"user",
                            "source_locator":"message:0#sentence:0",
                            "confidence":0.75,
                            "created_at":"2026-01-01T00:00:00Z"
                        }]
                    }
                }),
                ctx.clone(),
            )
            .await
            .unwrap();
        assert!(
            recorded.output.contains("Recorded 1 domain fact(s), 1 new"),
            "{}",
            recorded.output
        );

        let searched = tool
            .execute(
                json!({
                    "action":"domain",
                    "domain":{"action":"search","text":"Stripe billing","limit":3}
                }),
                ctx.clone(),
            )
            .await
            .unwrap();
        assert!(
            searched.output.contains("Found 1 domain fact"),
            "{}",
            searched.output
        );
        assert!(
            searched.output.contains("domain:billing"),
            "{}",
            searched.output
        );
        assert!(searched.output.contains("\"score\""), "{}", searched.output);

        let listed = tool
            .execute(
                json!({"action":"domain","domain":{"action":"list"}}),
                ctx.clone(),
            )
            .await
            .unwrap();
        assert!(
            listed.output.contains("1 domain fact(s)"),
            "{}",
            listed.output
        );

        let forgotten = tool
            .execute(
                json!({
                    "action":"domain",
                    "domain":{"action":"forget","id":"domain:billing"}
                }),
                ctx,
            )
            .await
            .unwrap();
        assert!(
            forgotten.output.contains("Forgot domain fact domain:billing"),
            "{}",
            forgotten.output
        );
    }

    #[tokio::test]
    async fn ontology_gate_routes_structured_signals() {
        let _guard = crate::storage::lock_test_env();
        let tool = MemoryTool::new();
        let ctx = test_ctx(None);

        let pass = tool
            .execute(
                json!({
                    "action":"ontology",
                    "ontology_action":"gate",
                    "content":"invoice status",
                    "gate":{"validation_path":true,"compliance_risk":true}
                }),
                ctx.clone(),
            )
            .await
            .unwrap();
        assert!(
            pass.output.contains("Six-question gate: auto_pass"),
            "{}",
            pass.output
        );
        assert!(pass.output.contains("Q1, Q3"), "{}", pass.output);
        assert!(
            pass.output.contains("rule layer: required"),
            "{}",
            pass.output
        );

        let clarify = tool
            .execute(
                json!({
                    "action":"ontology",
                    "ontology_action":"gate",
                    "gate":{"unresolved_questions":["Q3"]}
                }),
                ctx.clone(),
            )
            .await
            .unwrap();
        assert!(
            clarify.output.contains("Six-question gate: clarify"),
            "{}",
            clarify.output
        );

        let fallback = tool
            .execute(json!({"action":"ontology","ontology_action":"gate"}), ctx)
            .await
            .unwrap();
        assert!(
            fallback.output.contains("Six-question gate: llm_only"),
            "{}",
            fallback.output
        );
    }
}
