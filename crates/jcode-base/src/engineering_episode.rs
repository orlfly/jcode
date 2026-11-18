//! Deterministic extraction of engineering episode graphs from tool traces.
//!
//! This is the first producer for [`jcode_memory_types::engineering_ontology`].
//! It deliberately does not use an LLM: `bash`, `edit`, `write`, `apply_patch`,
//! `read` and `todo` already carry structured evidence. Semantic extraction can
//! be layered on later for decisions, preferences and root causes.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use jcode_memory_types::engineering_ontology::{
    CLASS_ACTION, CLASS_CHANGE, CLASS_COMMAND, CLASS_ERROR, CLASS_FILE, CLASS_OBSERVATION,
    CLASS_SESSION, CLASS_TASK, CLASS_TURN, EngineeringEntity, EngineeringGraph,
    EngineeringProvenance, EngineeringRelation, REL_CALLS, REL_CAUSES, REL_CONTAINS,
    REL_IMPLEMENTS, REL_MODIFIES, REL_PRODUCES, REL_USES,
};
use serde_json::{Map, Value, json};

use crate::message::{ContentBlock, Message, Role};

/// Result of one deterministic extraction pass.
#[derive(Debug, Clone, Default)]
pub struct EpisodeExtraction {
    pub graph: EngineeringGraph,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct PendingAction {
    action_id: String,
    tool: String,
}

/// Extract a session-level engineering graph from tool calls and results.
///
/// The graph is intentionally conservative: every edge is backed by a real
/// tool call, a tool result, or a file path in the call input. Unknown tools
/// still produce an `Action` entity but no speculative artefact edges.
pub fn extract_engineering_episode(
    session_id: &str,
    messages: &[Message],
    working_dir: Option<&str>,
) -> EpisodeExtraction {
    let mut extraction = EpisodeExtraction::default();
    let mut pending: HashMap<String, PendingAction> = HashMap::new();
    let mut turn_index = 0usize;

    let mut session = EngineeringEntity::new(CLASS_SESSION, session_id, session_id)
        .with_property("session_id", json!(session_id))
        .with_provenance(EngineeringProvenance::new(format!("session:{session_id}")));
    if let Some(working_dir) = working_dir {
        session
            .properties
            .insert("working_dir".to_string(), json!(working_dir));
    }
    extraction.graph.insert_entity(session);

    for (message_index, message) in messages.iter().enumerate() {
        match message.role {
            Role::Assistant => {
                let mut turn_id = None;
                for block in message.content.iter() {
                    let ContentBlock::ToolUse {
                        id, name, input, ..
                    } = block
                    else {
                        continue;
                    };
                    if turn_id.is_none() {
                        turn_index += 1;
                        let id = format!("turn:{session_id}:{turn_index}");
                        let mut turn =
                            EngineeringEntity::new(CLASS_TURN, &id, format!("Turn {turn_index}"))
                                .with_property("turn_id", json!(turn_index.to_string()))
                                .with_provenance(
                                    EngineeringProvenance::new(format!("session:{session_id}"))
                                        .with_locator(format!("message:{message_index}")),
                                );
                        turn.valid_from = message.timestamp;
                        extraction.graph.insert_entity(turn);
                        extraction.graph.insert_relation(EngineeringRelation::new(
                            session_id,
                            REL_CONTAINS,
                            &id,
                        ));
                        turn_id = Some(id);
                    }

                    let action_id = format!("action:{id}");
                    let mut action = EngineeringEntity::new(CLASS_ACTION, &action_id, name)
                        .with_property("tool", json!(name))
                        .with_property("input", compact_tool_input(name, input))
                        .with_provenance(
                            EngineeringProvenance::new(format!("session:{session_id}"))
                                .with_locator(format!("message:{message_index}"))
                                .with_activity(format!("tool:{name}")),
                        );
                    action.valid_from = message.timestamp;
                    extraction.graph.insert_entity(action);
                    if let Some(turn_id) = &turn_id {
                        extraction.graph.insert_relation(EngineeringRelation::new(
                            turn_id,
                            REL_CONTAINS,
                            &action_id,
                        ));
                    }

                    attach_tool_input(
                        &mut extraction,
                        &action_id,
                        name,
                        input,
                        session_id,
                        message.timestamp,
                        message_index,
                    );
                    pending.insert(
                        id.clone(),
                        PendingAction {
                            action_id,
                            tool: name.clone(),
                        },
                    );
                }
            }
            Role::User => {
                for block in message.content.iter() {
                    let ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } = block
                    else {
                        continue;
                    };
                    let Some(action) = pending.remove(tool_use_id) else {
                        extraction.warnings.push(format!(
                            "tool result {tool_use_id} has no matching tool call"
                        ));
                        continue;
                    };

                    let observation_id = format!("observation:{tool_use_id}");
                    let summary = non_empty_preview(content, 800);
                    let mut observation = EngineeringEntity::new(
                        CLASS_OBSERVATION,
                        &observation_id,
                        format!("{} result", action.tool),
                    )
                    .with_property("summary", json!(summary))
                    .with_property("is_error", json!(is_error.unwrap_or(false)))
                    .with_provenance(
                        EngineeringProvenance::new(format!("session:{session_id}"))
                            .with_locator(format!("message:{message_index}"))
                            .with_activity(format!("tool:{}", action.tool)),
                    );
                    observation.valid_from = message.timestamp;
                    extraction.graph.insert_entity(observation);
                    extraction.graph.insert_relation(EngineeringRelation::new(
                        &action.action_id,
                        REL_PRODUCES,
                        &observation_id,
                    ));

                    if is_error.unwrap_or(false) {
                        let signature = error_signature(content);
                        let error_id = stable_id("error", &signature);
                        if !extraction.graph.entities.contains_key(&error_id) {
                            let mut error =
                                EngineeringEntity::new(CLASS_ERROR, &error_id, signature.clone())
                                    .with_property("signature", json!(signature))
                                    .with_property(
                                        "message",
                                        json!(non_empty_preview(content, 800)),
                                    )
                                    .with_provenance(
                                        EngineeringProvenance::new(format!("session:{session_id}"))
                                            .with_locator(format!("message:{message_index}"))
                                            .with_activity(format!("tool:{}", action.tool)),
                                    );
                            error.valid_from = message.timestamp;
                            extraction.graph.insert_entity(error);
                        }
                        extraction.graph.insert_relation(EngineeringRelation::new(
                            &action.action_id,
                            REL_CAUSES,
                            &error_id,
                        ));
                    }
                }
            }
        }
    }

    extraction
}

fn attach_tool_input(
    extraction: &mut EpisodeExtraction,
    action_id: &str,
    tool: &str,
    input: &Value,
    session_id: &str,
    timestamp: Option<DateTime<Utc>>,
    message_index: usize,
) {
    match tool {
        "bash" => {
            let Some(command) = input.get("command").and_then(Value::as_str) else {
                extraction
                    .warnings
                    .push(format!("{action_id} bash call has no command"));
                return;
            };
            let cwd = input.get("cwd").and_then(Value::as_str).unwrap_or("");
            let command_id = stable_id("command", &format!("{cwd}\u{0}{command}"));
            if !extraction.graph.entities.contains_key(&command_id) {
                let mut entity = EngineeringEntity::new(CLASS_COMMAND, &command_id, command)
                    .with_property("command", json!(command))
                    .with_provenance(
                        EngineeringProvenance::new(format!("session:{session_id}"))
                            .with_locator(format!("message:{message_index}"))
                            .with_activity("tool:bash"),
                    );
                if !cwd.is_empty() {
                    entity.properties.insert("cwd".to_string(), json!(cwd));
                }
                entity.valid_from = timestamp;
                extraction.graph.insert_entity(entity);
            }
            extraction.graph.insert_relation(EngineeringRelation::new(
                action_id,
                REL_CALLS,
                &command_id,
            ));
        }
        "edit" | "replace" | "write" | "apply_patch" => {
            let Some(path) = file_path_from_input(input) else {
                extraction
                    .warnings
                    .push(format!("{action_id} {tool} call has no file path"));
                return;
            };
            let file_id = ensure_file(
                &mut extraction.graph,
                &path,
                session_id,
                timestamp,
                message_index,
            );
            extraction.graph.insert_relation(EngineeringRelation::new(
                action_id,
                REL_MODIFIES,
                &file_id,
            ));

            let change_id = format!("change:{action_id}");
            if !extraction.graph.entities.contains_key(&change_id) {
                let mut change =
                    EngineeringEntity::new(CLASS_CHANGE, &change_id, format!("{tool} {path}"))
                        .with_property("summary", json!(format!("{tool} {path}")))
                        .with_property("files", json!([path]))
                        .with_provenance(
                            EngineeringProvenance::new(format!("session:{session_id}"))
                                .with_locator(format!("message:{message_index}"))
                                .with_activity(format!("tool:{tool}")),
                        );
                change.valid_from = timestamp;
                extraction.graph.insert_entity(change);
            }
            extraction.graph.insert_relation(EngineeringRelation::new(
                action_id,
                REL_PRODUCES,
                &change_id,
            ));
            extraction.graph.insert_relation(EngineeringRelation::new(
                &change_id,
                REL_MODIFIES,
                &file_id,
            ));
        }
        "read" => {
            let Some(path) = file_path_from_input(input) else {
                extraction
                    .warnings
                    .push(format!("{action_id} read call has no file path"));
                return;
            };
            let file_id = ensure_file(
                &mut extraction.graph,
                &path,
                session_id,
                timestamp,
                message_index,
            );
            extraction
                .graph
                .insert_relation(EngineeringRelation::new(action_id, REL_USES, &file_id));
        }
        "todo" => attach_todo_tasks(
            extraction,
            action_id,
            input,
            session_id,
            timestamp,
            message_index,
        ),
        _ => {}
    }
}

fn attach_todo_tasks(
    extraction: &mut EpisodeExtraction,
    action_id: &str,
    input: &Value,
    session_id: &str,
    timestamp: Option<DateTime<Utc>>,
    message_index: usize,
) {
    let Some(items) = input.get("items").and_then(Value::as_array) else {
        return;
    };
    for (index, item) in items.iter().enumerate() {
        let Some(title) = item
            .get("name")
            .or_else(|| item.get("title"))
            .and_then(Value::as_str)
        else {
            extraction
                .warnings
                .push(format!("{action_id} todo item {index} has no name"));
            continue;
        };
        let task_id = format!("task:{action_id}:{index}");
        if !extraction.graph.entities.contains_key(&task_id) {
            let status = item
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("pending");
            let mut task = EngineeringEntity::new(CLASS_TASK, &task_id, title)
                .with_property("title", json!(title))
                .with_property("status", json!(status))
                .with_provenance(
                    EngineeringProvenance::new(format!("session:{session_id}"))
                        .with_locator(format!("message:{message_index}"))
                        .with_activity("tool:todo"),
                );
            task.valid_from = timestamp;
            extraction.graph.insert_entity(task);
        }
        extraction.graph.insert_relation(EngineeringRelation::new(
            action_id,
            REL_IMPLEMENTS,
            &task_id,
        ));
    }
}

fn ensure_file(
    graph: &mut EngineeringGraph,
    path: &str,
    session_id: &str,
    timestamp: Option<DateTime<Utc>>,
    message_index: usize,
) -> String {
    let id = format!("file:{path}");
    if !graph.entities.contains_key(&id) {
        let mut entity = EngineeringEntity::new(CLASS_FILE, &id, path)
            .with_property("path", json!(path))
            .with_provenance(
                EngineeringProvenance::new(format!("session:{session_id}"))
                    .with_locator(format!("message:{message_index}")),
            );
        entity.valid_from = timestamp;
        graph.insert_entity(entity);
    }
    id
}

fn file_path_from_input(input: &Value) -> Option<String> {
    for key in ["file_path", "path"] {
        if let Some(path) = input.get(key).and_then(Value::as_str)
            && !path.trim().is_empty()
        {
            return Some(path.trim().to_string());
        }
    }
    let patch = input.get("patch").and_then(Value::as_str)?;
    for line in patch.lines() {
        for prefix in ["*** Update File: ", "*** Add File: ", "*** Delete File: "] {
            if let Some(path) = line.strip_prefix(prefix) {
                let path = path.trim();
                if !path.is_empty() {
                    return Some(path.to_string());
                }
            }
        }
    }
    None
}

fn compact_tool_input(tool: &str, input: &Value) -> Value {
    let mut map = Map::new();
    for key in [
        "command",
        "cwd",
        "file_path",
        "path",
        "intent",
        "query",
        "url",
        "action",
    ] {
        if let Some(value) = input.get(key) {
            let value = match value.as_str() {
                Some(text) => json!(truncate(text, 500)),
                None => value.clone(),
            };
            map.insert(key.to_string(), value);
        }
    }
    if map.is_empty() {
        json!({
            "tool": tool,
            "preview": truncate(&input.to_string(), 500),
        })
    } else {
        Value::Object(map)
    }
}

fn error_signature(content: &str) -> String {
    let first = content
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("tool error");
    truncate(&first.split_whitespace().collect::<Vec<_>>().join(" "), 160)
}

fn non_empty_preview(content: &str, max_chars: usize) -> String {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        "(empty tool result)".to_string()
    } else {
        truncate(trimmed, max_chars)
    }
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max_chars).collect();
    out.push_str("...");
    out
}

fn stable_id(prefix: &str, input: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{prefix}:{hash:016x}")
}

/// Render a deterministic `Graph Memory State` block for a compaction summary.
///
/// The block contains only stable anchors that a later `shortest_path` or
/// `memory recall` call can resolve. Session, turn and observation nodes are
/// omitted because they are already represented by the summary itself.
pub fn render_graph_memory_state(messages: &[Message], max_anchors: usize) -> String {
    if max_anchors == 0 {
        return String::new();
    }
    let extraction = extract_engineering_episode("compaction", messages, None);
    let mut lines = Vec::new();
    for entity in extraction.graph.entities.values() {
        if matches!(
            entity.class.as_str(),
            CLASS_SESSION | CLASS_TURN | CLASS_OBSERVATION
        ) {
            continue;
        }
        let summary = entity.name.split_whitespace().collect::<Vec<_>>().join(" ");
        lines.push(format!(
            "- `{}` | {} | {}",
            entity.id,
            entity.class,
            truncate(&summary, 120)
        ));
        if lines.len() >= max_anchors {
            break;
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use jcode_memory_types::engineering_ontology::{
        CLASS_COMMAND, CLASS_FILE, CLASS_OBSERVATION, CLASS_SESSION, CLASS_TURN,
        default_engineering_ontology,
    };

    fn tool_use(id: &str, name: &str, input: Value) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
                input,
                thought_signature: None,
            }]
            .into(),
            timestamp: Some(Utc::now()),
            tool_duration_ms: None,
        }
    }

    fn tool_result(id: &str, content: &str, is_error: bool) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: id.to_string(),
                content: content.to_string(),
                is_error: Some(is_error),
            }]
            .into(),
            timestamp: Some(Utc::now()),
            tool_duration_ms: None,
        }
    }

    #[test]
    fn extracts_bash_error_episode() {
        let messages = vec![
            tool_use("call-1", "bash", json!({"command": "cargo test -p jcode"})),
            tool_result(
                "call-1",
                "error[E0425]: cannot find function truncate_with_ellipsis",
                true,
            ),
        ];
        let extraction = extract_engineering_episode("session-1", &messages, Some("/repo"));
        let graph = extraction.graph;
        assert!(graph.entities.contains_key("session-1"));
        assert!(graph.find_by_class(CLASS_ACTION).len() == 1);
        assert!(graph.find_by_class(CLASS_COMMAND).len() == 1);
        assert!(graph.find_by_class(CLASS_OBSERVATION).len() == 1);
        assert!(graph.find_by_class(CLASS_ERROR).len() == 1);
        assert!(
            graph
                .relations
                .values()
                .any(|relation| relation.relation == REL_CAUSES)
        );

        let ontology = default_engineering_ontology();
        let report = graph.validate(&ontology);
        assert!(report.is_valid(), "{:?}", report.errors());
    }

    #[test]
    fn extracts_edit_episode() {
        let messages = vec![
            tool_use(
                "call-2",
                "edit",
                json!({"file_path": "src/lib.rs", "old_string": "a", "new_string": "b"}),
            ),
            tool_result("call-2", "ok", false),
        ];
        let extraction = extract_engineering_episode("session-2", &messages, None);
        let graph = extraction.graph;
        assert!(graph.entities.contains_key("file:src/lib.rs"));
        assert!(graph.find_by_class(CLASS_FILE).len() == 1);
        assert!(graph.find_by_class(CLASS_CHANGE).len() == 1);
        assert!(
            graph
                .relations
                .values()
                .any(|relation| relation.relation == REL_MODIFIES)
        );
    }

    #[test]
    fn extracts_todo_tasks() {
        let messages = vec![
            tool_use(
                "call-3",
                "todo",
                json!({"items": [{"name": "implement extractor", "status": "in_progress"}]}),
            ),
            tool_result("call-3", "ok", false),
        ];
        let extraction = extract_engineering_episode("session-3", &messages, None);
        assert_eq!(extraction.graph.find_by_class(CLASS_TASK).len(), 1);
        assert!(
            extraction
                .graph
                .relations
                .values()
                .any(|relation| relation.relation == REL_IMPLEMENTS)
        );
    }

    #[test]
    fn unknown_tool_creates_action_without_artefacts() {
        let messages = vec![tool_use("call-4", "custom", json!({"x": 1}))];
        let extraction = extract_engineering_episode("session-4", &messages, None);
        assert_eq!(extraction.graph.find_by_class(CLASS_ACTION).len(), 1);
        assert!(extraction.graph.find_by_class(CLASS_FILE).is_empty());
        assert!(extraction.graph.find_by_class(CLASS_COMMAND).is_empty());
    }

    #[test]
    fn renders_graph_memory_state_anchors() {
        let messages = vec![
            tool_use(
                "call-6",
                "edit",
                json!({"file_path": "src/lib.rs", "old_string": "a", "new_string": "b"}),
            ),
            tool_result("call-6", "ok", false),
            tool_use("call-7", "bash", json!({"command": "cargo test -p jcode"})),
            tool_result("call-7", "error[E0425]: boom", true),
        ];
        let rendered = render_graph_memory_state(&messages, 24);
        assert!(rendered.contains("file:src/lib.rs"), "{rendered}");
        assert!(rendered.contains("command:"), "{rendered}");
        assert!(rendered.contains("error:"), "{rendered}");
        assert!(!rendered.contains("session:"), "{rendered}");
        assert!(!rendered.contains("turn:"), "{rendered}");
    }

    #[test]
    fn unmatched_result_is_a_warning() {
        let messages = vec![tool_result("missing", "ok", false)];
        let extraction = extract_engineering_episode("session-5", &messages, None);
        assert_eq!(extraction.warnings.len(), 1);
        assert_eq!(extraction.graph.find_by_class(CLASS_SESSION).len(), 1);
        assert!(extraction.graph.find_by_class(CLASS_TURN).is_empty());
    }
}
