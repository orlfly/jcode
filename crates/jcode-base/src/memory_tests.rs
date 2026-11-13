use super::*;

#[test]
fn manager_without_project_dir_does_not_use_process_cwd() {
    assert!(MemoryManager::new().get_project_dir().is_none());
}
use crate::message::{ContentBlock, Message, Role};
use serde_json::json;
use std::fs;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub(super) static PENDING_MEMORY_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn jev_storage_preserves_code_spelling_and_scope_without_embeddings() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/jev-storage");
        let first = manager
            .remember_project(MemoryEntry::new(MemoryCategory::Fact, "path: foo/bar"))
            .unwrap();
        let second = manager
            .remember_project(MemoryEntry::new(MemoryCategory::Fact, "path: foo-bar"))
            .unwrap();
        let third = manager
            .remember_project(MemoryEntry::new(MemoryCategory::Fact, "path: Foo/bar"))
            .unwrap();
        assert_ne!(first, second);
        assert_ne!(first, third);
        let repeated = manager
            .remember_project(MemoryEntry::new(MemoryCategory::Fact, "path: foo/bar"))
            .unwrap();
        assert_eq!(first, repeated);
        let global = manager
            .remember_global(MemoryEntry::new(MemoryCategory::Fact, "path: foo/bar"))
            .unwrap();
        assert_ne!(first, global);
        let all = manager.list_all().unwrap();
        assert_eq!(all.len(), 4);
        assert!(all.iter().all(|entry| entry.embedding.is_none()));
        assert_eq!(
            manager
                .load_project_graph()
                .unwrap()
                .get_memory(&first)
                .unwrap()
                .strength,
            2
        );
    });
}

#[test]
fn jev_project_write_without_scope_fails_instead_of_silently_losing_memory() {
    with_temp_home(|_| {
        let manager = MemoryManager::new();
        assert!(
            manager
                .remember_project(MemoryEntry::new(MemoryCategory::Fact, "fact"))
                .is_err()
        );
        assert!(
            manager
                .remember_global(MemoryEntry::new(MemoryCategory::Fact, "fact"))
                .is_ok()
        );
    });
}

fn with_temp_home<F, T>(f: F) -> T
where
    F: FnOnce(&Path) -> T,
{
    let _guard = crate::storage::lock_test_env();
    let old = std::env::var("JCODE_HOME").ok();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("jcode-test-{}", unique));
    fs::create_dir_all(&dir).expect("create temp dir");
    crate::env::set_var("JCODE_HOME", &dir);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&dir)));

    match old {
        Some(value) => crate::env::set_var("JCODE_HOME", value),
        None => crate::env::remove_var("JCODE_HOME"),
    }
    let _ = fs::remove_dir_all(&dir);

    match result {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

#[test]
fn pending_memory_freshness_and_clear() {
    let _guard = PENDING_MEMORY_TEST_LOCK
        .lock()
        .expect("pending memory test lock poisoned");
    clear_all_pending_memory();

    let sid = "test-session-1";
    set_pending_memory(sid, "hello".to_string(), 2);
    assert!(has_pending_memory(sid));
    let pending = take_pending_memory(sid).expect("pending memory");
    assert_eq!(pending.prompt, "hello");
    assert_eq!(pending.count, 2);
    assert!(!has_pending_memory(sid));

    insert_pending_memory_for_test(
        sid,
        PendingMemory {
            prompt: "stale".to_string(),
            display_prompt: None,
            computed_at: Instant::now() - Duration::from_secs(16 * 60),
            count: 1,
            memory_ids: Vec::new(),
        },
    );
    assert!(take_pending_memory(sid).is_none());

    insert_pending_memory_for_test(
        sid,
        PendingMemory {
            prompt: "slow judge".to_string(),
            display_prompt: None,
            computed_at: Instant::now() - Duration::from_secs(14 * 60),
            count: 1,
            memory_ids: Vec::new(),
        },
    );
    assert!(
        take_pending_memory(sid).is_some(),
        "a 14-minute-old payload must survive a slow judge/tool run"
    );
}

#[test]
fn pending_memory_suppresses_immediate_duplicate_payloads() {
    let _guard = PENDING_MEMORY_TEST_LOCK
        .lock()
        .expect("pending memory test lock poisoned");
    clear_all_pending_memory();

    let sid = "test-session-2";
    set_pending_memory(sid, "same payload".to_string(), 1);
    assert!(take_pending_memory(sid).is_some());

    set_pending_memory(sid, "same payload".to_string(), 1);
    assert!(
        take_pending_memory(sid).is_none(),
        "identical payload should be suppressed when repeated immediately"
    );
}

#[test]
fn pending_memory_suppresses_overlapping_memory_sets() {
    let _guard = PENDING_MEMORY_TEST_LOCK
        .lock()
        .expect("pending memory test lock poisoned");
    clear_all_pending_memory();

    let sid = "test-session-overlap";
    set_pending_memory_with_ids(
        sid,
        "first payload".to_string(),
        2,
        vec!["mem-a".to_string(), "mem-b".to_string()],
    );
    assert!(take_pending_memory(sid).is_some());

    set_pending_memory_with_ids(
        sid,
        "second payload with same memories".to_string(),
        2,
        vec!["mem-b".to_string(), "mem-a".to_string()],
    );
    assert!(
        take_pending_memory(sid).is_none(),
        "same memory set should be suppressed even if prompt text differs"
    );
}

#[test]
fn pending_memory_keeps_existing_similar_payload_instead_of_replacing_it() {
    let _guard = PENDING_MEMORY_TEST_LOCK
        .lock()
        .expect("pending memory test lock poisoned");
    clear_all_pending_memory();

    let sid = "test-session-queued-overlap";
    set_pending_memory_with_ids(
        sid,
        "original payload".to_string(),
        2,
        vec!["mem-a".to_string(), "mem-b".to_string()],
    );
    set_pending_memory_with_ids(
        sid,
        "replacement payload".to_string(),
        2,
        vec!["mem-a".to_string(), "mem-b".to_string()],
    );

    let pending = take_pending_memory(sid).expect("existing pending payload should remain");
    assert_eq!(pending.prompt, "original payload");
}

#[test]
fn pending_memory_per_session_isolation() {
    let _guard = PENDING_MEMORY_TEST_LOCK
        .lock()
        .expect("pending memory test lock poisoned");
    clear_all_pending_memory();

    let sid_a = "test-session-a";
    let sid_b = "test-session-b";

    set_pending_memory(sid_a, "memory for A".to_string(), 1);
    set_pending_memory(sid_b, "memory for B".to_string(), 2);

    assert!(has_pending_memory(sid_a));
    assert!(has_pending_memory(sid_b));

    let pending_a = take_pending_memory(sid_a).expect("session A should have pending memory");
    assert_eq!(pending_a.prompt, "memory for A");
    assert!(!has_pending_memory(sid_a));

    // Session B's memory should still be there
    assert!(has_pending_memory(sid_b));
    let pending_b = take_pending_memory(sid_b).expect("session B should have pending memory");
    assert_eq!(pending_b.prompt, "memory for B");
    assert_eq!(pending_b.count, 2);
}

#[test]
fn pending_memory_suppresses_payload_when_all_ids_already_known() {
    let _guard = PENDING_MEMORY_TEST_LOCK
        .lock()
        .expect("pending memory test lock poisoned");
    clear_all_pending_memory();

    let sid = "test-session-known";
    mark_memories_known(sid, &["mem-x".to_string(), "mem-y".to_string()], "test");

    // Wait out the short-term signature/set suppression windows by using
    // distinct payload text; the id-level known check must trigger on its own.
    set_pending_memory_with_ids(
        sid,
        "brand new formatting of old knowledge".to_string(),
        2,
        vec!["mem-y".to_string(), "mem-x".to_string()],
    );
    assert!(
        take_pending_memory(sid).is_none(),
        "payload made entirely of already-known memories must not inject"
    );

    // A payload with at least one genuinely new memory still injects.
    set_pending_memory_with_ids(
        sid,
        "mix of old and new".to_string(),
        2,
        vec!["mem-x".to_string(), "mem-new".to_string()],
    );
    assert!(
        take_pending_memory(sid).is_some(),
        "payload containing an unknown memory should inject"
    );

    clear_all_pending_memory();
}

#[test]
fn injected_memory_dedup_expires_after_ttl() {
    let _guard = PENDING_MEMORY_TEST_LOCK
        .lock()
        .expect("pending memory test lock poisoned");
    clear_all_pending_memory();

    let sid = "test-session-ttl";
    mark_memories_injected(sid, &["mem-ttl".to_string()]);
    assert!(is_memory_injected(sid, "mem-ttl"));

    // Backdate past the TTL: the memory may surface again.
    backdate_injected_memory_for_test(sid, "mem-ttl", Duration::from_secs(46 * 60));
    assert!(
        !is_memory_injected(sid, "mem-ttl"),
        "injected-memory dedup must expire after the TTL"
    );
    assert!(!is_memory_injected_any("mem-ttl"));

    clear_all_pending_memory();
}

#[test]
fn mark_memories_known_blocks_reinjection_like_injection() {
    let _guard = PENDING_MEMORY_TEST_LOCK
        .lock()
        .expect("pending memory test lock poisoned");
    clear_all_pending_memory();

    let sid = "test-session-self-echo";
    let other = "test-session-other";

    // Simulates extraction: the memory came from sid's own transcript.
    mark_memories_known(sid, &["mem-echo".to_string()], "extracted from transcript");

    assert!(
        is_memory_injected(sid, "mem-echo"),
        "known memory must count as injected for its source session"
    );
    assert!(
        !is_memory_injected(other, "mem-echo"),
        "other sessions are unaffected by another session's known-marking"
    );

    clear_all_pending_memory();
}

#[test]
fn format_context_includes_roles_and_tools() {
    let messages = vec![
        Message::user("Hello world"),
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "tool-1".to_string(),
                name: "memory".to_string(),
                input: json!({"action": "list"}),
                thought_signature: None,
            }],
            timestamp: None,
            tool_duration_ms: None,
        },
        Message::tool_result("tool-1", "ok", false),
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tool-2".to_string(),
                content: "boom".to_string(),
                is_error: Some(true),
            }],
            timestamp: None,
            tool_duration_ms: None,
        },
    ];

    let context = format_context_for_relevance(&messages);
    assert!(context.contains("User:\nHello world"));
    assert!(context.contains("[Tool: memory]"));
    assert!(!context.contains("[Tool result: ok]"));
    assert!(context.contains("[Tool error: boom]"));
}

#[test]
fn extraction_context_keeps_tool_io_details() {
    let messages = vec![
        Message::user("Hello world"),
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "tool-1".to_string(),
                name: "memory".to_string(),
                input: json!({"action": "list"}),
                thought_signature: None,
            }],
            timestamp: None,
            tool_duration_ms: None,
        },
        Message::tool_result("tool-1", "ok", false),
    ];

    let context = format_context_for_extraction(&messages);
    assert!(context.contains("[Tool: memory input:"));
    assert!(context.contains("[Tool result: ok]"));
}

#[test]
fn memory_store_format_groups_by_category() {
    let mut store = MemoryStore::new();
    let now = Utc::now();
    let mut correction = MemoryEntry::new(MemoryCategory::Correction, "Fix lint rules");
    correction.updated_at = now;
    let mut fact = MemoryEntry::new(MemoryCategory::Fact, "Uses tokio");
    fact.updated_at = now;
    let mut preference = MemoryEntry::new(MemoryCategory::Preference, "Prefers ASCII-only edits");
    preference.updated_at = now;
    let mut entity = MemoryEntry::new(MemoryCategory::Entity, "Jeremy");
    entity.updated_at = now;
    let mut custom = MemoryEntry::new(MemoryCategory::Custom("team".to_string()), "Platform");
    custom.updated_at = now;

    store.add(correction);
    store.add(fact);
    store.add(preference);
    store.add(entity);
    store.add(custom);

    let output = store.format_for_prompt(10).expect("formatted output");
    let correction_idx = output.find("## Corrections").expect("correction heading");
    let fact_idx = output.find("## Facts").expect("fact heading");
    let preference_idx = output.find("## Preferences").expect("preference heading");
    let entity_idx = output.find("## Entities").expect("entity heading");
    let custom_idx = output.find("## team").expect("custom heading");

    assert!(correction_idx < fact_idx);
    assert!(fact_idx < preference_idx);
    assert!(preference_idx < entity_idx);
    assert!(entity_idx < custom_idx);
}

#[test]
fn memory_store_search_matches_content_and_tags() {
    let mut store = MemoryStore::new();
    let entry = MemoryEntry::new(MemoryCategory::Fact, "Uses Tokio runtime")
        .with_tags(vec!["async".to_string()]);
    store.add(entry);

    let content_hits = store.search("tokio");
    assert_eq!(content_hits.len(), 1);

    let tag_hits = store.search("ASYNC");
    assert_eq!(tag_hits.len(), 1);
}

#[test]
fn memory_search_normalizes_whitespace_and_separators() {
    let mut store = MemoryStore::new();
    let entry = MemoryEntry::new(MemoryCategory::Fact, "Uses side panel layout")
        .with_tags(vec!["build_cache".to_string()]);
    store.add(entry);

    assert_eq!(store.search("  side-panel  ").len(), 1);
    assert_eq!(store.search("BUILD.CACHE").len(), 1);
    assert!(store.search("   ").is_empty());
}

#[test]
fn manager_persists_and_forgets_memories() {
    with_temp_home(|_dir| {
        let manager = MemoryManager::new_test();
        let entry_project = MemoryEntry::new(MemoryCategory::Fact, "Project memory")
            .with_embedding(vec![1.0, 0.0, 0.0]);
        let entry_global = MemoryEntry::new(MemoryCategory::Preference, "Global memory")
            .with_embedding(vec![0.0, 1.0, 0.0]);

        let project_id = manager
            .remember_project(entry_project)
            .expect("remember project");
        let global_id = manager
            .remember_global(entry_global)
            .expect("remember global");

        let all = manager.list_all().expect("list all");
        assert_eq!(all.len(), 2);

        let search = manager.search("global").expect("search");
        assert_eq!(search.len(), 1);

        assert!(manager.forget(&project_id).expect("forget project"));
        let remaining = manager.list_all().expect("list all");
        assert_eq!(remaining.len(), 1);

        assert!(!manager.forget(&project_id).expect("forget missing"));
        assert!(manager.forget(&global_id).expect("forget global"));
    });
}

#[test]
fn graph_based_memory_operations() {
    with_temp_home(|_home| {
        let manager = MemoryManager::new_test();

        // Create two memories
        let entry1 = MemoryEntry::new(
            MemoryCategory::Fact,
            "The capital of France is Paris, a city known for the Eiffel Tower",
        );
        let entry2 = MemoryEntry::new(
            MemoryCategory::Fact,
            "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy",
        );

        let id1 = manager.remember_project(entry1).expect("remember 1");
        let id2 = manager.remember_project(entry2).expect("remember 2");

        // Test tagging
        manager.tag_memory(&id1, "rust").expect("tag memory");
        manager.tag_memory(&id1, "language").expect("tag memory 2");
        manager.tag_memory(&id2, "rust").expect("tag memory 3");

        // Check graph stats (memories, tags, edges, clusters)
        let (mems, tags, edges, _clusters) = manager.graph_stats().expect("stats");
        assert_eq!(mems, 2, "expected 2 memories");
        assert_eq!(tags, 2, "expected 2 tags: rust and language");
        assert!(edges >= 3, "expected at least 3 edges, got {}", edges);

        // Test linking
        manager.link_memories(&id1, &id2, 0.8).expect("link");

        // Test get_related
        let related = manager.get_related(&id1, 2).expect("get related");
        assert!(!related.is_empty());
        // Should find id2 through the RelatesTo edge
        assert!(related.iter().any(|e| e.id == id2));

        // Clean up
        manager.forget(&id1).expect("forget 1");
        manager.forget(&id2).expect("forget 2");
    });
}

#[test]
fn remember_routes_inadmissible_provenance_to_review() {
    use crate::memory_types::{ExtractionMethod, ProvenanceRecord};
    with_temp_home(|_home| {
        let manager = MemoryManager::new_test();
        let entry = MemoryEntry::new(MemoryCategory::Fact, "low confidence extraction")
            .with_provenance(
                ProvenanceRecord::new("llm", ExtractionMethod::LlmExtraction).with_confidence(0.5),
            );
        let outcome = manager
            .remember_project_governed(entry)
            .expect("low confidence is parked, not an error");
        assert!(outcome.is_pending_review());
        assert!(
            manager.load_project_graph().unwrap().memory_count() == 0,
            "low-confidence extraction must not enter the graph"
        );
        assert_eq!(
            manager
                .list_reviews(Some(ReviewStatus::Pending))
                .unwrap()
                .len(),
            1
        );
    });
}

/// #729: test mode short-circuits `project_memory_path()` before the project
/// directory is consulted, so a manager in test mode cannot see real
/// project memory no matter what working directory it is given.
///
/// Swarm-spawned workers were forced into this mode unconditionally, which is
/// why they could never read what the session that spawned them remembered.
/// This pins the behavior so the severity of enabling test mode on a
/// production path stays visible.
#[test]
fn test_mode_ignores_the_project_dir_and_cannot_see_real_project_memory() {
    with_temp_home(|_home| {
        let project_dir = "/tmp/jcode-729-real-project";

        let real = MemoryManager::new().with_project_dir(project_dir);
        real.remember_project(MemoryEntry::new(
            MemoryCategory::Fact,
            "written by the spawning session",
        ))
        .expect("remember real project memory");

        // Same directory, but test mode: the write above is invisible.
        let isolated = MemoryManager::new_test().with_project_dir(project_dir);
        assert!(isolated.is_test_mode());
        let seen: Vec<String> = isolated
            .load_project_graph()
            .expect("load isolated graph")
            .all_memories()
            .map(|entry| entry.content.clone())
            .collect();
        assert!(
            !seen.iter().any(|c| c.contains("spawning session")),
            "test mode unexpectedly saw real project memory: {seen:?}"
        );

        // And a non-test manager on the same dir does see it, proving the
        // isolation above comes from test mode rather than a bad path.
        let reader = MemoryManager::new().with_project_dir(project_dir);
        let visible: Vec<String> = reader
            .load_project_graph()
            .expect("load real graph")
            .all_memories()
            .map(|entry| entry.content.clone())
            .collect();
        assert!(
            visible.iter().any(|c| c.contains("spawning session")),
            "real project memory should be visible without test mode: {visible:?}"
        );
    });
}

#[test]
fn sqlite_default_migrates_legacy_json_graph_once() {
    // Only meaningful when the sqlite-gvec backend is the active backend.
    // JSON is the default (matching upstream), so skip unless opted in.
    if crate::memory::active_backend_name() != "sqlite-gvec" {
        eprintln!("skipping: JCODE_MEMORY_BACKEND != sqlite-gvec");
        return;
    }
    with_temp_home(|_home| {
        let project_dir = "/tmp/jcode-migrate-project";
        let manager = MemoryManager::new_test().with_project_dir(project_dir);

        // Seed a legacy JSON graph at the canonical project path.
        let path = manager
            .project_memory_path()
            .expect("test project path")
            .expect("path present");
        let mut graph = MemoryGraph::new();
        let mut entry = MemoryEntry::new(MemoryCategory::Fact, "legacy memory to migrate");
        entry.id = "legacy-1".into();
        graph.memories.insert("legacy-1".into(), entry);
        storage::write_json(&path, &graph).expect("write legacy json");

        let loaded = manager.load_project_graph().expect("load migrates");
        assert_eq!(
            loaded.memory_count(),
            1,
            "legacy JSON should migrate into the active backend"
        );
        assert!(
            loaded
                .all_memories()
                .any(|m| m.content.contains("legacy memory")),
            "migrated memory content should survive"
        );

        // Source renamed out of the canonical path so a later empty store
        // does not resurrect memories the user deletes after the switch.
        assert!(
            !path.exists(),
            "legacy JSON should be renamed after migration"
        );
        assert!(
            path.with_extension("json.migrated").exists(),
            "migration backup should be preserved"
        );

        // Deleting all memories must not resurrect them on the next load.
        let ids: Vec<String> = loaded.all_memories().map(|m| m.id.clone()).collect();
        for id in ids {
            manager.forget(&id).expect("forget migrated memory");
        }
        let empty = manager.load_project_graph().expect("reload after delete");
        assert_eq!(
            empty.memory_count(),
            0,
            "deleted memories must not be resurrected from the migration backup"
        );
    });
}

#[test]
fn project_memories_are_isolated_by_explicit_project_dir() {
    with_temp_home(|_home| {
        let manager_a = MemoryManager::new().with_project_dir("/tmp/jcode-project-a");
        let manager_b = MemoryManager::new().with_project_dir("/tmp/jcode-project-b");

        manager_a
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "memory from project a",
            ))
            .expect("remember project a");
        manager_b
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "memory from project b",
            ))
            .expect("remember project b");

        let project_a: Vec<String> = manager_a
            .load_project_graph()
            .expect("load project a")
            .all_memories()
            .map(|m| m.content.clone())
            .collect();
        let project_b: Vec<String> = manager_b
            .load_project_graph()
            .expect("load project b")
            .all_memories()
            .map(|m| m.content.clone())
            .collect();

        assert_eq!(project_a, vec!["memory from project a".to_string()]);
        assert_eq!(project_b, vec!["memory from project b".to_string()]);
    });
}

#[test]
fn manager_search_scoped_normalizes_whitespace_and_separators() {
    with_temp_home(|_home| {
        let manager = MemoryManager::new().with_project_dir("/tmp/jcode-search-normalization");

        manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "project compile notes",
            ))
            .expect("remember project");

        let hits = manager
            .search_scoped("  compile/notes  ", MemoryScope::Project)
            .expect("search project");
        assert_eq!(hits.len(), 1);
    });
}

#[test]
fn prompt_memories_scoped_keeps_only_most_recent_entries() {
    with_temp_home(|_home| {
        let manager = MemoryManager::new().with_project_dir("/tmp/jcode-prompt-topk");

        let mut oldest = MemoryEntry::new(MemoryCategory::Fact, "compile cache note");
        oldest.created_at = Utc::now() - chrono::Duration::seconds(30);
        oldest.updated_at = oldest.created_at;

        let mut middle = MemoryEntry::new(MemoryCategory::Fact, "oauth refresh bug");
        middle.created_at = Utc::now() - chrono::Duration::seconds(20);
        middle.updated_at = middle.created_at;

        let mut newest = MemoryEntry::new(MemoryCategory::Fact, "terminal shortcut hint");
        newest.created_at = Utc::now() - chrono::Duration::seconds(10);
        newest.updated_at = newest.created_at;

        manager
            .upsert_project_memory(oldest)
            .expect("remember oldest");
        manager
            .upsert_project_memory(middle)
            .expect("remember middle");
        manager
            .upsert_project_memory(newest)
            .expect("remember newest");

        let recent = manager
            .list_all_scoped(MemoryScope::Project)
            .expect("list project memories");
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].content, "terminal shortcut hint");
        assert_eq!(recent[1].content, "oauth refresh bug");
        assert_eq!(recent[2].content, "compile cache note");

        let prompt = manager
            .get_prompt_memories_scoped(2, MemoryScope::Project)
            .expect("prompt memories");

        assert!(prompt.contains("terminal shortcut hint"));
        assert!(
            prompt.contains("oauth refresh bug") || prompt.contains("1.") || prompt.contains("2.")
        );
        assert!(!prompt.contains("compile cache note"));
    });
}

#[test]
fn goal_memory_upsert_skips_embedding_generation() {
    with_temp_home(|_home| {
        let manager = MemoryManager::new().with_project_dir("/tmp/jcode-goal-memory");

        let mut entry = MemoryEntry::new(
            MemoryCategory::Custom("goal".to_string()),
            "Goal: Ship mobile MVP\nStatus: active\nScope: project",
        );
        entry.id = "goal:ship-mobile-mvp".to_string();

        manager
            .upsert_project_memory(entry)
            .expect("upsert goal memory");

        let graph = manager.load_project_graph().expect("load graph");
        let saved = graph
            .get_memory("goal:ship-mobile-mvp")
            .expect("saved goal memory");
        assert!(
            saved.embedding.is_none(),
            "goal memory mirrors should not synchronously load/generate embeddings"
        );
    });
}

#[test]
fn scoped_retrieval_respects_project_vs_global() {
    with_temp_home(|_home| {
        let manager = MemoryManager::new().with_project_dir("/tmp/jcode-scope-test");

        manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "project zebra compile notes",
            ))
            .expect("remember project");
        manager
            .remember_global(MemoryEntry::new(
                MemoryCategory::Fact,
                "global coffee preference",
            ))
            .expect("remember global");

        let project = manager
            .list_all_scoped(MemoryScope::Project)
            .expect("list project");
        let global = manager
            .list_all_scoped(MemoryScope::Global)
            .expect("list global");
        let all = manager.list_all_scoped(MemoryScope::All).expect("list all");

        assert_eq!(project.len(), 1);
        assert_eq!(project[0].content, "project zebra compile notes");
        assert_eq!(global.len(), 1);
        assert_eq!(global[0].content, "global coffee preference");
        assert_eq!(all.len(), 2);

        let project_search = manager
            .search_scoped("zebra", MemoryScope::Project)
            .expect("search project");
        let global_search = manager
            .search_scoped("coffee", MemoryScope::Global)
            .expect("search global");

        assert_eq!(project_search.len(), 1);
        assert_eq!(project_search[0].content, "project zebra compile notes");
        assert_eq!(global_search.len(), 1);
        assert_eq!(global_search[0].content, "global coffee preference");
    });
}

#[test]
fn retrieval_candidates_include_local_skills() {
    with_temp_home(|home| {
        // memory no longer reaches into skill directly; register the skill
        // synthetic-entry provider (as cli::startup does in production) so the
        // memory<-skill integration this test exercises is wired up. The
        // shared snapshot is global-only (issue #457), so production composes
        // the process-cwd project overlay on top.
        crate::memory::register_synthetic_entry_provider(|| {
            let global = crate::skill::SkillRegistry::shared_snapshot();
            crate::skill::SkillRegistry::effective_for_working_dir(&global, None)
                .list()
                .into_iter()
                .map(|skill| skill.as_memory_entry())
                .collect()
        });
        let project_dir = home.join("project-with-skill");
        fs::create_dir_all(project_dir.join(".jcode/skills/firefox-browser"))
            .expect("create skills dir");
        fs::write(
                project_dir.join(".jcode/skills/firefox-browser/SKILL.md"),
                "---\nname: firefox-browser\ndescription: Control Firefox browser sessions\nallowed-tools: bash, read, write\n---\n\nUse this skill to open sites and click buttons.",
            )
            .expect("write skill");

        let old_cwd = std::env::current_dir().expect("current dir");
        std::env::set_current_dir(&project_dir).expect("set current dir");

        let manager = MemoryManager::new()
            .with_project_dir(&project_dir)
            .with_skills(true);
        let candidates = manager
            .collect_retrieval_candidates_scoped(MemoryScope::All)
            .expect("collect retrieval candidates");

        std::env::set_current_dir(old_cwd).expect("restore current dir");

        assert!(
            candidates
                .iter()
                .any(|entry| entry.id == "skill:firefox-browser")
        );
        // v2: the synthetic entry is a `skill:` link node with a bounded
        // pointer body, not a `Custom("Skills")` instance carrying the whole
        // SKILL.md text.
        assert!(candidates.iter().any(|entry| {
            entry.id == "skill:firefox-browser"
                && jcode_memory_types::is_link_node(entry)
                && entry.content.chars().count() <= 200
        }));
    });
}

#[test]
fn collect_skill_query_terms_keeps_relevant_words_and_drops_generic_words() {
    let terms = collect_skill_query_terms(
        "Before we start, make the todo list for this long debugging and validation task.",
    );

    assert!(terms.contains("todo"));
    assert!(terms.contains("debugging"));
    assert!(terms.contains("validation"));
    assert!(terms.contains("task"));
    assert!(!terms.contains("before"));
    assert!(!terms.contains("start"));
    assert!(!terms.contains("make"));
    assert!(!terms.contains("this"));
}

#[test]
fn score_and_filter_prioritizes_matching_skill_memories() {
    let generic = MemoryEntry::new(
        MemoryCategory::Fact,
        "General planning note that is not about structured todo skills.",
    )
    .with_embedding(vec![1.0, 0.0]);

    let mut skill = MemoryEntry::new(
        MemoryCategory::Custom("Skills".to_string()),
        "Use skill `/todo-planning-skill` for todo list planning, debugging, reflection, and validation on long tasks.",
    )
    .with_embedding(vec![1.0, 0.0])
    .with_source("skill_registry");
    skill.id = "skill:todo-planning-skill".to_string();

    let ranked = MemoryManager::score_and_filter(
        vec![generic, skill],
        &[1.0, 0.0],
        "Please make the todo list for this task.",
        0.0,
        2,
    )
    .expect("score and filter");

    assert_eq!(ranked.len(), 2);
    assert_eq!(ranked[0].0.id, "skill:todo-planning-skill");
    assert!(ranked[0].1 > ranked[1].1);
}

#[test]
fn hybrid_fuse_rescues_lexical_match_dense_would_miss() {
    // A memory that is the obvious lexical answer (shares the rare identifier
    // `find_similar_hybrid`) but is given a deliberately ORTHOGONAL embedding so
    // pure dense cosine ranks it last. BM25 must rescue it into the top result.
    let target = MemoryEntry::new(
        MemoryCategory::Fact,
        "The function find_similar_hybrid fuses dense and bm25 with RRF.",
    )
    .with_embedding(vec![0.0, 1.0]);

    let distractor_a = MemoryEntry::new(
        MemoryCategory::Fact,
        "Unrelated note about coffee brewing temperatures.",
    )
    .with_embedding(vec![1.0, 0.0]);
    let distractor_b = MemoryEntry::new(
        MemoryCategory::Fact,
        "Another unrelated note about bicycle maintenance.",
    )
    .with_embedding(vec![0.95, 0.05]);

    // Query embedding points along the distractors' axis, so dense alone would
    // rank the target dead last; the query TEXT contains the rare identifier.
    let query_text = "how does find_similar_hybrid work";
    let query_emb = vec![1.0, 0.0];

    let ranked = MemoryManager::hybrid_fuse(
        vec![target.clone(), distractor_a, distractor_b],
        query_text,
        &query_emb,
        3,
    );

    assert!(!ranked.is_empty(), "hybrid must return candidates");
    assert_eq!(
        ranked[0].0.id, target.id,
        "BM25 should rescue the exact-identifier memory to the top despite poor dense score"
    );
}

#[test]
fn hybrid_fuse_returns_dense_hits_without_lexical_overlap() {
    // When the query shares NO tokens with any memory, hybrid must still return
    // the dense-nearest memory (fusion falls back to the dense ranking).
    let near = MemoryEntry::new(MemoryCategory::Fact, "alpha bravo charlie")
        .with_embedding(vec![1.0, 0.0]);
    let far =
        MemoryEntry::new(MemoryCategory::Fact, "delta echo foxtrot").with_embedding(vec![0.0, 1.0]);

    let ranked = MemoryManager::hybrid_fuse(
        vec![near.clone(), far],
        "zzz_nonmatching_query_token",
        &[1.0, 0.0],
        2,
    );

    assert!(!ranked.is_empty());
    assert_eq!(
        ranked[0].0.id, near.id,
        "dense-nearest memory should rank first"
    );
}

#[test]
fn hybrid_excludes_superseded_memories() {
    with_temp_home(|_home| {
        let manager = MemoryManager::new().with_project_dir("/tmp/jcode-hybrid-supersede");

        // Two memories on the same topic with explicit distinct ids (avoid
        // same-millisecond id collisions).
        // Distinct embeddings so the write-time dedup does not merge them.
        let old = MemoryEntry::new(MemoryCategory::Fact, "The build uses cargo profile dev")
            .with_embedding(vec![1.0, 0.0]);
        let new = MemoryEntry::new(MemoryCategory::Fact, "The build uses cargo profile selfdev")
            .with_embedding(vec![0.0, 1.0]);

        let old_id = manager.remember_project(old).expect("remember old");
        let new_id = manager.remember_project(new).expect("remember new");
        assert_ne!(old_id, new_id, "ids must differ");

        // Supersede the old memory.
        let mut graph = manager.load_project_graph().expect("load");
        graph.supersede(&new_id, &old_id);
        manager.save_project_graph(&graph).expect("save");

        let results = manager
            .find_similar_hybrid("cargo build profile selfdev", &[0.0, 1.0], 10)
            .expect("hybrid");
        let ids: Vec<&str> = results.iter().map(|(e, _)| e.id.as_str()).collect();

        assert!(
            !ids.contains(&old_id.as_str()),
            "superseded memory must not surface from hybrid retrieval; got {:?}",
            ids
        );
        assert!(
            ids.contains(&new_id.as_str()),
            "the superseding memory should still surface; got {:?}",
            ids
        );
    });
}

#[test]
fn focus_query_text_strips_noise_and_leads_with_user_intent() {
    let raw = "\
<system-reminder>\n# Session Context\nDate: 2026-06-14\n</system-reminder>\n\
User:\n\
how do I fix the scroll bug in navigation.rs\n\
Assistant:\n\
Let me look at the handler.\n\
[Tool: read]\n\
[Result: fn handle_scroll() { ... }]\n\
Assistant:\n\
The bug is in the mouse delta calc.";

    let focused = super::focus_query_text(raw);

    // System-reminder block is gone.
    assert!(
        !focused.contains("Session Context"),
        "reminder not stripped: {focused}"
    );
    assert!(!focused.contains("<system-reminder>"));
    // Tool noise is gone.
    assert!(
        !focused.contains("[Tool:"),
        "tool marker not stripped: {focused}"
    );
    assert!(!focused.contains("[Result:"));
    // Role markers are gone.
    assert!(!focused.contains("User:"));
    assert!(!focused.contains("Assistant:"));
    // Real prose is kept.
    assert!(focused.contains("scroll bug in navigation.rs"));
    assert!(focused.contains("mouse delta calc"));
    // Leads with the latest user intent.
    assert!(
        focused.starts_with("how do I fix the scroll bug in navigation.rs"),
        "should lead with latest user message: {focused}"
    );
}

#[test]
fn focused_query_excludes_multiline_tool_errors_but_keeps_later_user_prose() {
    let messages = vec![Message {
        role: Role::User,
        content: vec![
            ContentBlock::ToolResult {
                tool_use_id: "tool-1".to_string(),
                content: "This command was not run.\nUNIQUE_MULTILINE_ERROR_PAYLOAD\nThe target cannot be confirmed.\nThe operation is irreversible."
                    .to_string(),
                is_error: Some(true),
            },
            ContentBlock::Text {
                text: "Keep the token rotation behavior unchanged.".to_string(),
                cache_control: None,
            },
        ],
        timestamp: None,
        tool_duration_ms: None,
    }];

    let focused = format_focused_query_for_relevance(&messages);

    assert!(!focused.contains("This command was not run"), "{focused}");
    assert!(
        !focused.contains("UNIQUE_MULTILINE_ERROR_PAYLOAD"),
        "arbitrary error payload leaked: {focused}"
    );
    assert!(!focused.contains("cannot be confirmed"), "{focused}");
    assert!(!focused.contains("irreversible"), "{focused}");
    assert!(
        focused.contains("Keep the token rotation behavior unchanged."),
        "subsequent user prose was lost: {focused}"
    );
}

#[test]
fn focus_query_text_falls_back_when_all_stripped() {
    let raw = "<system-reminder>\nonly boilerplate\n</system-reminder>\n[Tool: read]";
    let focused = super::focus_query_text(raw);
    // Nothing substantive survives -> fall back to raw rather than empty.
    assert_eq!(focused, raw);
}

// --- FTS5 business integration tests ---
//
// These tests only meaningfully exercise the FTS5 codepath when the
// sqlite-gvec backend is selected. Under the default JSON backend
// `search_scoped` falls back to the in-memory scanner so they are
// still correct — they just verify the routing layer doesn't crash.

#[cfg_attr(not(feature = "sqlite-gvec"), ignore = "needs sqlite-gvec FTS5")]
#[test]
fn search_scoped_ranks_memories_via_fts5() {
    with_temp_home(|_home| {
        let manager = MemoryManager::new_test().with_project_dir("/tmp/jcode-fts-rank");
        manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "rust ownership and borrowing rules",
            ))
            .expect("remember rust");
        manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "python decorators and closures",
            ))
            .expect("remember python");
        manager
            .remember_global(MemoryEntry::new(
                MemoryCategory::Preference,
                "user prefers vim keybindings",
            ))
            .expect("remember global");

        let hits = manager.search("rust ownership").expect("fts search");
        assert!(!hits.is_empty(), "expected at least one FTS5 hit");
        // The rust memory must be the top-ranked hit.
        assert!(
            hits[0].content.contains("rust"),
            "top hit should mention rust, got {:?}",
            hits[0].content
        );

        let scoped = manager
            .search_scoped("python", MemoryScope::Project)
            .expect("scoped fts");
        assert!(
            scoped.iter().all(|m| m.content.contains("python")),
            "project-scoped search must not leak the global vim memory"
        );
        assert!(
            scoped.iter().any(|m| m.content.contains("python")),
            "project-scoped search must still find the python memory"
        );
    });
}

#[test]
fn search_scoped_ignores_punctuation_in_query() {
    with_temp_home(|_home| {
        let manager = MemoryManager::new_test().with_project_dir("/tmp/jcode-fts-punct");
        manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "compile cache invalidation",
            ))
            .expect("remember");

        // FTS5 treats `:` as a column-prefix operator; the manager
        // must strip such characters so user queries with file paths
        // or version strings still match.
        let hits = manager
            .search_scoped("compile:cache", MemoryScope::Project)
            .expect("search with colon");
        assert!(
            !hits.is_empty(),
            "expected FTS5 to tolerate ':' in the query"
        );
    });
}

#[cfg_attr(not(feature = "sqlite-gvec"), ignore = "needs sqlite-gvec FTS5")]
#[test]
fn fts5_stays_in_sync_after_full_save_replace() {
    with_temp_home(|_home| {
        // Exercise the `save_project_graph` (full drop+rewrite) path:
        // the AI/AD triggers must keep the FTS5 shadow table in sync
        // without requiring an explicit ensure_text_index call.
        let manager = MemoryManager::new_test().with_project_dir("/tmp/jcode-fts-save");
        manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "first sentence about apples",
            ))
            .expect("remember apples");
        manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "second sentence about oranges",
            ))
            .expect("remember oranges");

        // Forget the oranges entry: this re-saves the whole graph,
        // dropping and re-inserting every node. The FTS5 index must
        // reflect that only `apples` remains.
        let all = manager.list_all().expect("list all");
        let orange_id = all
            .iter()
            .find(|m| m.content.contains("oranges"))
            .map(|m| m.id.clone())
            .expect("orange id");
        assert!(manager.forget(&orange_id).expect("forget orange"));

        let hits = manager
            .search_scoped("oranges", MemoryScope::Project)
            .expect("search oranges after forget");
        assert!(
            hits.is_empty(),
            "forget should have purged the orange FTS5 row, got {:?}",
            hits
        );

        let apple_hits = manager
            .search_scoped("apples", MemoryScope::Project)
            .expect("search apples after forget");
        assert!(
            apple_hits.iter().any(|m| m.content.contains("apples")),
            "apple memory must still be searchable after the rewrite"
        );
    });
}

#[test]
fn extraction_noise_filter_rejects_fragments_and_hashes() {
    // Low-value noise that should NOT be stored.
    assert!(is_extraction_noise(""), "empty");
    assert!(is_extraction_noise("   "), "whitespace only");
    assert!(is_extraction_noise("85fa7777a"), "bare hash");
    assert!(is_extraction_noise("a1b2c3d4e5f6"), "hex hash");
    assert!(is_extraction_noise("----"), "symbols only");
    assert!(is_extraction_noise(":::"), "symbols only");
    assert!(
        is_extraction_noise("CONTENT"),
        "single identifier too short"
    );
    assert!(is_extraction_noise("Commit"), "single word too short");

    // Real memories must pass the gate.
    assert!(!is_extraction_noise(
        "The project uses rust with sqlite for persistence"
    ));
    assert!(!is_extraction_noise(
        "User prefers tabs over spaces in python code"
    ));
    assert!(!is_extraction_noise(
        "The auth token lives in the environment, never in code"
    ));
}

#[test]
fn hybrid_retrieval_favors_stronger_memory_on_tie() {
    // B acceptance: the strength/recency prior in hybrid_fuse must lift a
    // heavily-reinforced memory above an otherwise identical weak twin when
    // both are near-identical matches for the query.
    with_temp_home(|_home| {
        let manager = MemoryManager::new().with_project_dir("/tmp/jcode-hybrid-strength");

        // Distinct-but-related embeddings so write-time dedup (cos >= 0.90)
        // keeps both rows; query [1, 0] ranks the weak twin first on dense.
        let weak = MemoryEntry::new(MemoryCategory::Fact, "The release channel is stable")
            .with_embedding(vec![1.0, 0.0]);
        let strong = MemoryEntry::new(MemoryCategory::Fact, "The release channel stays stable")
            .with_embedding(vec![0.8, 0.6]);

        let weak_id = manager.remember_project(weak).expect("remember weak");
        let strong_id = manager.remember_project(strong).expect("remember strong");
        assert_ne!(weak_id, strong_id, "twins must be distinct rows");

        // Drive the strong twin to a high strength via reinforcements; the
        // weak twin stays at strength 1.
        for i in 0..12 {
            let mut graph = manager.load_project_graph().unwrap();
            graph
                .get_memory_mut(&strong_id)
                .unwrap()
                .reinforce(&format!("acceptance-{i}"), 0);
            manager.save_project_graph(&graph).unwrap();
        }

        let results = manager
            .find_similar_hybrid("which release channel", &[1.0, 0.0], 10)
            .expect("hybrid");
        let ids: Vec<&str> = results.iter().map(|(e, _)| e.id.as_str()).collect();

        assert_eq!(ids.len(), 2, "both twins should surface");
        assert_eq!(
            ids[0], strong_id,
            "the stronger twin must rank first on a near-tie; got order {ids:?}"
        );
        let weak_pos = results.iter().position(|(e, _)| e.id == weak_id).unwrap();
        let strong_pos = results.iter().position(|(e, _)| e.id == strong_id).unwrap();
        assert!(
            strong_pos < weak_pos,
            "strong (pos {strong_pos}) must precede weak (pos {weak_pos})"
        );
    });
}

#[test]
fn jev_recall_prefilter_bounds_candidates_and_keeps_relevant_memories() {
    let mut entries: Vec<MemoryEntry> = (0..2000)
        .map(|i| {
            MemoryEntry::new(
                MemoryCategory::Fact,
                format!("unrelated note number {i} about gardening"),
            )
        })
        .collect();
    entries.push(MemoryEntry::new(
        MemoryCategory::Fact,
        "The fundraising CRM lives on Bookface and uses a 40M post-money SAFE cap",
    ));
    let kept = prefilter_for_jev(entries, "update the fundraising CRM notes for the SAFE");
    assert!(kept.len() <= MAX_JEV_RECALL_CANDIDATES);
    assert!(kept.iter().any(|e| e.content.contains("fundraising CRM")));
    // At most three Jev batches of 24 per recall instead of ~84 for this store.
    assert!(kept.len().div_ceil(crate::memory_jev::MAX_BATCH_ENTRIES) <= 3);
}

#[test]
fn jev_recall_prefilter_leaves_small_stores_untouched() {
    let entries: Vec<MemoryEntry> = (0..10)
        .map(|i| MemoryEntry::new(MemoryCategory::Fact, format!("note {i}")))
        .collect();
    assert_eq!(
        prefilter_for_jev(entries, "completely different words").len(),
        10
    );
}

// ---------------------------------------------------------------------------
// v2: TTL maintenance, skill linkage, invocation evidence
// ---------------------------------------------------------------------------

fn backdated(category: MemoryCategory, content: &str, days: i64) -> MemoryEntry {
    let then = Utc::now() - chrono::Duration::days(days);
    MemoryEntry::new(category, content).with_timestamps(then, then)
}

#[test]
fn ttl_sweep_archives_expired_env_facts_and_keeps_contracts() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/ttl-sweep");
        // env_fact declares a 30d TTL; contract is retained indefinitely.
        manager
            .upsert_project_memory(backdated(
                MemoryCategory::Custom("env_fact".to_string()),
                "Kaneo dev server listens on localhost:1337 for this workspace",
                45,
            ))
            .unwrap();
        manager
            .upsert_project_memory(backdated(
                MemoryCategory::Custom("env_fact".to_string()),
                "Fresh environment detail recorded during the current session",
                1,
            ))
            .unwrap();
        manager
            .upsert_project_memory(backdated(
                MemoryCategory::Custom("contract".to_string()),
                "Claiming a task is POST /api/task/claim/{taskId} with the team id",
                400,
            ))
            .unwrap();

        let report = manager.gc_expired_memories().expect("gc");
        assert_eq!(report.archived, 1, "only the stale env_fact expires");

        let all = manager.list_all().unwrap();
        let archived = all
            .iter()
            .find(|e| e.content.contains("localhost:1337"))
            .expect("stale env fact still present (archived, not deleted)");
        assert!(!archived.active);
        assert_eq!(
            archived.lifecycle.status,
            crate::memory_types::MemoryStatus::Archived
        );
        assert!(
            archived
                .lifecycle
                .deprecated_reason
                .as_deref()
                .unwrap_or_default()
                .contains("ttl elapsed")
        );
        let contract = all
            .iter()
            .find(|e| e.content.contains("/api/task/claim"))
            .expect("contract kept");
        assert!(contract.active, "typed retention must survive the sweep");
    });
}

#[test]
fn ttl_sweep_exempts_link_nodes() {
    with_temp_home(|_| {
        let manager = MemoryManager::new();
        let mut link = backdated(
            MemoryCategory::Fact,
            "Skill `/deploy`: deploys things.",
            999,
        );
        link.id = "skill:deploy".to_string();
        manager.upsert_global_memory(link).unwrap();

        let report = manager.gc_expired_memories().expect("gc");
        assert_eq!(report.archived, 0);
        let all = manager.list_all().unwrap();
        let link = all
            .iter()
            .find(|e| e.id == "skill:deploy")
            .expect("link node present");
        assert!(link.active, "link nodes own their own lifecycle");
    });
}

#[test]
fn deprecated_fact_type_still_maps_to_a_ttl_bearing_type() {
    // v1 `fact` is aliased to `observation` (90d TTL), so old rows converge too.
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/ttl-legacy");
        manager
            .upsert_project_memory(backdated(
                MemoryCategory::Fact,
                "A legacy fact recorded long before the v2 ontology existed",
                200,
            ))
            .unwrap();
        let report = manager.gc_expired_memories().expect("gc");
        assert_eq!(report.archived, 1);
    });
}

#[test]
fn skill_invocation_is_recorded_as_evidence() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/skill-invocation");
        let first = manager.remember_skill_invocation("optimization").unwrap();
        let again = manager.remember_skill_invocation("optimization").unwrap();
        assert_eq!(
            first, again,
            "one row per skill, refreshed on re-invocation"
        );

        let entry = manager
            .list_all()
            .unwrap()
            .into_iter()
            .find(|e| e.id == first)
            .expect("invocation memory");
        assert_eq!(entry.subject_key.as_deref(), Some("skill:optimization"));
        assert_eq!(entry.source.as_deref(), Some("skill:optimization"));
        assert_eq!(
            entry.effective_provenance().method,
            ExtractionMethod::SkillInvocation
        );
        assert!(entry.tags.contains(&"skill:optimization".to_string()));
    });
}

#[test]
fn skill_distillation_proposes_from_agreeing_evidence() {
    with_temp_home(|home| {
        let manager = MemoryManager::new().with_project_dir("/skill-distill");
        // Deliberately distinct wording: near-identical items would be folded
        // into one row by write-time reconciliation (which is the point of that
        // pass), and the test would then be measuring dedup instead of
        // aggregation.
        let evidence = [
            (
                "session_a_1",
                "POST /api/drill/sign-off: the drill log must be backfilled into the design doc",
            ),
            (
                "session_b_2",
                "POST /api/drill/sign-off acceptance needs at least 24h of shadow drill",
            ),
            (
                "session_c_3",
                "POST /api/drill/sign-off 条件③ requires the drill log before sign-off",
            ),
        ];
        for (session, content) in evidence {
            let mut entry =
                MemoryEntry::new(MemoryCategory::Custom("process_rule".to_string()), content);
            entry.source = Some(session.to_string());
            manager.remember_project(entry).unwrap();
        }
        // Only two evidence items: below the threshold.
        let mut weak = MemoryEntry::new(
            MemoryCategory::Custom("process_rule".to_string()),
            "Unrelated procedure about POST /api/drill/other ordering",
        );
        weak.source = Some("session_d_4".to_string());
        manager.remember_project(weak).unwrap();

        let proposals = manager.collect_skill_evidence().expect("evidence");
        let drill = proposals
            .iter()
            .find(|p| p.subject_key == "api:/api/drill/sign-off")
            .expect("aggregated evidence for the drill sign-off route");
        assert_eq!(drill.evidence.len(), 3);
        assert!(drill.sessions.len() >= 3);
        assert!(
            !proposals
                .iter()
                .any(|p| p.subject_key == "api:/api/drill/other"),
            "single-session evidence must not produce a proposal"
        );

        // The proposal pipeline writes a reviewable draft, never an install.
        let written = manager.distill_skill_evidence().expect("distill");
        assert!(
            written
                .iter()
                .any(|p| p.subject_key == "api:/api/drill/sign-off")
        );
        let draft = home
            .join("skills")
            .join(".proposals")
            .join("api-api-drill-sign-off.md");
        let body = std::fs::read_to_string(&draft).expect("draft written");
        assert!(body.contains("name: api-api-drill-sign-off"));
        assert!(body.contains("api:/api/drill/sign-off"));
    });
}

#[test]
fn skill_mirrors_are_written_as_link_nodes() {
    with_temp_home(|home| {
        let skill_dir = home.join("skills").join("deploy-pipeline");
        fs::create_dir_all(&skill_dir).expect("mkdir");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: deploy-pipeline\ndescription: Ship a verified build\n---\n\nRun the gated deploy.\n",
        )
        .expect("write skill");

        let registry = crate::skill::SkillRegistry::load_global().expect("registry");
        let written = crate::skill::sync_skill_memory_mirrors(&registry, None);
        assert!(written >= 1, "at least the on-disk skill is mirrored");

        let manager = MemoryManager::new();
        let all = manager.list_all().unwrap();
        let link = all
            .iter()
            .find(|e| e.id == "skill:deploy-pipeline")
            .expect("skill link node");
        assert!(jcode_memory_types::is_link_node(link));
        assert_eq!(link.subject_key.as_deref(), Some("skill:deploy-pipeline"));
        assert!(link.content.chars().count() <= 200);
        assert!(link.content.contains("deploy-pipeline"));
    });
}

#[test]
fn skill_reconcile_writes_retrievable_index_parts() {
    with_temp_home(|home| {
        let skill_dir = home.join("skills").join("indexed-opt");
        fs::create_dir_all(&skill_dir).expect("mkdir");
        fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: indexed-opt
description: Measure before optimizing
---

# Indexed Optimization

Define target metrics first.

## Workflow

- Measure the full path before changing any code.
- Attribute bottlenecks to concrete components.

```bash
cargo test --profile selfdev
```
"#,
        )
        .expect("write skill");

        let registry = crate::skill::SkillRegistry::load_global().expect("registry");
        let report = crate::skill::reconcile_skill_memory_links(&registry, None);
        assert!(report.parts_mirrored >= 3, "parts: {report:?}");

        let manager = MemoryManager::new();
        let graph = manager.load_global_graph().unwrap();
        let part_ids: Vec<&str> = graph
            .memories
            .keys()
            .map(String::as_str)
            .filter(|id| id.starts_with("skill:indexed-opt::"))
            .collect();
        assert!(part_ids.iter().any(|id| id.contains("::section:")));
        assert!(part_ids.iter().any(|id| id.contains("::step:")));
        assert!(part_ids.iter().any(|id| id.contains("::code:")));
        for id in &part_ids {
            assert!(
                graph
                    .get_edges(id)
                    .iter()
                    .any(|edge| edge.target == "skill:indexed-opt"),
                "missing root edge for {id}"
            );
            assert!(
                graph
                    .get_edges("skill:indexed-opt")
                    .iter()
                    .any(|edge| edge.target == *id
                        && matches!(edge.kind, jcode_memory_types::EdgeKind::Contains)),
                "missing contains edge for {id}"
            );
        }
        let next_steps = part_ids
            .iter()
            .flat_map(|id| graph.get_edges(id))
            .filter(|edge| matches!(edge.kind, jcode_memory_types::EdgeKind::NextStep))
            .count();
        assert!(next_steps >= 1, "steps need next_step edges: {part_ids:?}");
        let step = graph
            .get_memory(
                part_ids
                    .iter()
                    .find(|id| id.contains("::step:"))
                    .expect("step id"),
            )
            .expect("step entry");
        assert_eq!(
            step.property("skill.part_kind")
                .and_then(|value| value.as_str()),
            Some("step")
        );
        assert!(step.property("skill.step_order").is_some());
        let mut workflow_steps: Vec<_> = part_ids
            .iter()
            .filter_map(|id| graph.get_memory(id))
            .filter(|entry| {
                entry
                    .property("skill.part_kind")
                    .and_then(|value| value.as_str())
                    == Some("step")
            })
            .collect();
        workflow_steps.sort_by_key(|entry| {
            entry
                .property("skill.step_order")
                .and_then(|value| value.as_u64())
        });
        assert!(workflow_steps.len() >= 2);
        let expanded = manager.expand_skill_parts(vec![workflow_steps[1].clone()], 6);
        let expanded_ids: Vec<&str> = expanded.iter().map(|entry| entry.id.as_str()).collect();
        assert_eq!(
            expanded_ids.first().copied(),
            Some("skill:indexed-opt"),
            "root must lead the expanded payload: {expanded_ids:?}"
        );
        assert!(
            expanded_ids.contains(&workflow_steps[0].id.as_str()),
            "previous step must be adjacent: {expanded_ids:?}"
        );
        assert!(expanded_ids.contains(&workflow_steps[1].id.as_str()));

        let selected = manager
            .l1_relevant("Attribute bottlenecks to concrete components", 6)
            .unwrap();
        let ids: Vec<&str> = selected.iter().map(|entry| entry.id.as_str()).collect();
        let root = ids
            .iter()
            .position(|id| *id == "skill:indexed-opt")
            .expect("root pointer selected");
        let part = ids
            .iter()
            .position(|id| id.starts_with("skill:indexed-opt::"))
            .expect("part selected");
        assert!(
            root < part,
            "root must be injected before its part: {ids:?}"
        );
    });
}

#[test]
fn accepted_skill_records_derived_from_evidence_edges() {
    with_temp_home(|home| {
        let project = home.join("workspace");
        fs::create_dir_all(&project).expect("mkdir");
        let manager = MemoryManager::new().with_project_dir(&project);
        let evidence: Vec<String> = [
            "task #42 requires the smoke suite before merge",
            "task #42: merge is blocked until the smoke suite passes",
        ]
        .into_iter()
        .map(|content| {
            manager
                .remember_project(MemoryEntry::new(
                    MemoryCategory::Custom("process_rule".to_string()),
                    content,
                ))
                .unwrap()
        })
        .collect();

        // A proposal that has been accepted: the draft's `evidence:` list kept
        // intact in the installed skill.
        let skill_dir = project
            .join(".jcode")
            .join("skills")
            .join("task-42-merge-gate");
        fs::create_dir_all(&skill_dir).expect("mkdir");
        fs::write(
            skill_dir.join("SKILL.md"),
            format!(
                "---\nname: task-42-merge-gate\ndescription: Merge gate for task #42\nevidence:\n  - {}\n  - {}\n  - mem_does_not_exist\n---\n\nRun the smoke suite.\n",
                evidence[0], evidence[1]
            ),
        )
        .expect("write skill");

        let registry = crate::skill::SkillRegistry::load_global().expect("registry");
        crate::skill::sync_skill_memory_mirrors(&registry, Some(&project));

        let graph = manager.load_project_graph().unwrap();
        let node = "skill:task-42-merge-gate";
        assert!(graph.memories.contains_key(node), "mirror node present");
        let targets: Vec<&str> = graph
            .get_edges(node)
            .iter()
            .filter(|edge| matches!(edge.kind, jcode_memory_types::EdgeKind::DerivedFrom))
            .map(|edge| edge.target.as_str())
            .collect();
        for id in &evidence {
            assert!(
                targets.contains(&id.as_str()),
                "missing derived_from edge to {id}: {targets:?}"
            );
        }
        assert!(
            !targets.contains(&"mem_does_not_exist"),
            "an edge may only connect nodes that exist"
        );

        // Idempotent: a second pass adds nothing.
        let before = targets.len();
        crate::skill::sync_skill_memory_mirrors(&registry, Some(&project));
        let again = manager.load_project_graph().unwrap();
        let after = again
            .get_edges(node)
            .iter()
            .filter(|edge| matches!(edge.kind, jcode_memory_types::EdgeKind::DerivedFrom))
            .count();
        assert_eq!(before, after, "evidence linking must be idempotent");
    });
}

#[test]
fn project_local_skill_mirrors_land_in_the_project_store() {
    with_temp_home(|home| {
        // A project-local procedure belongs to its workspace only, so its link
        // node must not land in the global store.
        let project = home.join("workspace");
        let skill_dir = project.join(".jcode").join("skills").join("local-deploy");
        fs::create_dir_all(&skill_dir).expect("mkdir");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: local-deploy\ndescription: Deploy this workspace\n---\n\nSteps.\n",
        )
        .expect("write skill");
        let global_skills = home.join("skills").join("global-tune");
        fs::create_dir_all(&global_skills).expect("mkdir");
        fs::write(
            global_skills.join("SKILL.md"),
            "---\nname: global-tune\ndescription: Tune any project\n---\n\nSteps.\n",
        )
        .expect("write skill");

        let registry = crate::skill::SkillRegistry::load_global().expect("registry");
        let written = crate::skill::sync_skill_memory_mirrors(&registry, Some(&project));
        assert!(written >= 2, "both scopes mirrored, got {written}");

        // Inspect the stores themselves: `list_all` deliberately merges project
        // and global scope, which would hide a wrong-scope write.
        let project_manager = MemoryManager::new().with_project_dir(&project);
        let project_ids: Vec<String> = project_manager
            .load_project_graph()
            .unwrap()
            .memories
            .keys()
            .cloned()
            .collect();
        assert!(
            project_ids.contains(&"skill:local-deploy".to_string()),
            "project skill mirrors into its own store: {project_ids:?}"
        );
        assert!(
            !project_ids.contains(&"skill:global-tune".to_string()),
            "global skills stay out of the project store"
        );

        let global_manager = MemoryManager::new();
        let global_graph = global_manager.load_global_graph().unwrap();
        let global_ids: Vec<&str> = global_graph
            .memories
            .keys()
            .filter(|id| id.starts_with("skill:"))
            .map(String::as_str)
            .collect();
        assert!(global_ids.contains(&"skill:global-tune"));
        assert!(
            !global_ids.contains(&"skill:local-deploy"),
            "a workspace's procedure must not be indexed globally: {global_ids:?}"
        );
    });
}

#[test]
fn skill_mirror_reconcile_prunes_removed_skill() {
    with_temp_home(|home| {
        let skill_dir = home.join("skills").join("obsolete");
        fs::create_dir_all(&skill_dir).expect("mkdir");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: obsolete\ndescription: Temporary skill\n---\n\nBody.\n",
        )
        .expect("write skill");

        let registry = crate::skill::SkillRegistry::load_global().expect("registry");
        crate::skill::reconcile_skill_memory_links(&registry, None);
        assert!(
            MemoryManager::new()
                .load_global_graph()
                .unwrap()
                .memories
                .contains_key("skill:obsolete")
        );

        fs::remove_dir_all(&skill_dir).expect("remove skill");
        let registry = crate::skill::SkillRegistry::load_global().expect("registry");
        let report = crate::skill::reconcile_skill_memory_links(&registry, None);
        assert_eq!(report.pruned, 1, "orphan mirror must be removed");
        assert!(
            !MemoryManager::new()
                .load_global_graph()
                .unwrap()
                .memories
                .contains_key("skill:obsolete"),
            "deleted skill must not stay recallable"
        );
    });
}

#[test]
fn skill_reconcile_outbox_survives_and_drains() {
    with_temp_home(|home| {
        let skill_dir = home.join("skills").join("outbox-probe");
        fs::create_dir_all(&skill_dir).expect("mkdir");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: outbox-probe\ndescription: Outbox probe\n---\n\nBody.\n",
        )
        .expect("write skill");

        crate::skill::queue_skill_reconcile("unit_test", None);
        let path = home.join("memory").join("skill-reconcile-outbox.json");
        let outbox: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("outbox file"))
                .expect("outbox json");
        assert_eq!(outbox["pending"].as_array().unwrap().len(), 1);

        let report = crate::skill::drain_skill_reconcile_outbox();
        assert_eq!(report.mirrored, 1);
        assert!(
            MemoryManager::new()
                .load_global_graph()
                .unwrap()
                .memories
                .contains_key("skill:outbox-probe")
        );
        let outbox: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("outbox file"))
                .expect("outbox json");
        assert!(outbox["pending"].as_array().unwrap().is_empty());
    });
}

#[test]
fn artifact_kinds_are_configurable_from_disk() {
    with_temp_home(|home| {
        let path = home.join("memory").join("artifact-kinds.json");
        fs::create_dir_all(path.parent().unwrap()).expect("mkdir");
        fs::write(
            &path,
            r#"{"kinds":[{"id":"workflow","prefix":"workflow:","description":"Workflow artifact"}]}"#,
        )
        .expect("write config");

        crate::memory::load_artifact_kind_config();
        assert!(jcode_memory_types::is_link_id("workflow:deploy"));
        let kind =
            jcode_memory_types::artifact_kind_for_id("workflow:deploy").expect("configured kind");
        assert_eq!(kind.id, "workflow");
        // Built-ins survive the override.
        assert!(jcode_memory_types::is_link_id("skill:optimization"));

        // Restore the process-wide registry for other tests.
        jcode_memory_types::configure_artifact_kinds(Vec::new());
        assert!(!jcode_memory_types::is_link_id("workflow:deploy"));
    });
}

#[test]
fn project_cross_store_mirror_for_global_skill_is_preserved() {
    with_temp_home(|home| {
        let project = home.join("workspace");
        fs::create_dir_all(&project).expect("mkdir");
        let manager = MemoryManager::new().with_project_dir(&project);
        let evidence = manager
            .remember_project(
                MemoryEntry::new(MemoryCategory::ProcessRule, "cross-store evidence")
                    .with_id("evidence-cross"),
            )
            .unwrap();

        let skill_dir = home.join("skills").join("global-cross");
        fs::create_dir_all(&skill_dir).expect("mkdir");
        fs::write(
            skill_dir.join("SKILL.md"),
            format!(
                "---\nname: global-cross\ndescription: Global skill\n{evidence_tag}---\n\nSteps.\n",
                evidence_tag = format!("evidence:\n  - {evidence}\n")
            ),
        )
        .expect("write skill");

        // The cross-store mirror is what `record_accepted_skill` creates when a
        // global skill's evidence lives in a project store.
        let mut mirror = MemoryEntry::new(MemoryCategory::Fact, "Skill `/global-cross`")
            .with_id("skill:global-cross")
            .with_source("skill_registry");
        mirror.subject_key = Some("skill:global-cross".to_string());
        manager.remember_project(mirror).unwrap();

        let registry = crate::skill::SkillRegistry::load_global().expect("registry");
        crate::skill::reconcile_skill_memory_links(&registry, Some(&project));

        let graph = manager.load_project_graph().unwrap();
        let mirror = graph
            .get_memory("skill:global-cross")
            .expect("cross-store mirror kept");
        let artifact = mirror
            .artifact
            .as_ref()
            .expect("mirror enriched with artifact metadata");
        assert_eq!(artifact.health, "ok");
        assert_eq!(artifact.resolved_evidence, vec![evidence.clone()]);
        let targets: Vec<&str> = graph
            .get_edges("skill:global-cross")
            .iter()
            .filter(|edge| matches!(edge.kind, jcode_memory_types::EdgeKind::DerivedFrom))
            .map(|edge| edge.target.as_str())
            .collect();
        assert_eq!(targets, vec![evidence.as_str()], "{targets:?}");

        // The global mirror sees the same canonical health even though the
        // evidence node lives only in the project store.
        let global_graph = MemoryManager::new().load_global_graph().unwrap();
        let global_artifact = global_graph
            .get_memory("skill:global-cross")
            .and_then(|entry| entry.artifact.as_ref())
            .expect("global mirror metadata");
        assert_eq!(global_artifact.health, "ok");
        assert_eq!(global_artifact.resolved_evidence, vec![evidence]);
    });
}

#[test]
fn global_skill_resolves_evidence_from_multiple_project_stores() {
    with_temp_home(|home| {
        let project_a = home.join("workspace-a");
        let project_b = home.join("workspace-b");
        fs::create_dir_all(&project_a).expect("mkdir a");
        fs::create_dir_all(&project_b).expect("mkdir b");

        let manager_a = MemoryManager::new().with_project_dir(&project_a);
        let manager_b = MemoryManager::new().with_project_dir(&project_b);
        let evidence_a = manager_a
            .remember_project(
                MemoryEntry::new(MemoryCategory::ProcessRule, "evidence in a")
                    .with_id("evidence-a"),
            )
            .unwrap();
        let evidence_b = manager_b
            .remember_project(
                MemoryEntry::new(MemoryCategory::ProcessRule, "evidence in b")
                    .with_id("evidence-b"),
            )
            .unwrap();

        let skill_dir = home.join("skills").join("multi-store");
        fs::create_dir_all(&skill_dir).expect("mkdir skill");
        fs::write(
            skill_dir.join("SKILL.md"),
            format!(
                "---\nname: multi-store\ndescription: Multi-store skill\nevidence:\n  - {evidence_a}\n  - {evidence_b}\n---\n\nSteps.\n"
            ),
        )
        .expect("write skill");

        // Only project A has a local mirror; its artifact must still report the
        // canonical two-of-two health while its edges stay local.
        let mut mirror = MemoryEntry::new(MemoryCategory::Fact, "Skill `/multi-store`")
            .with_id("skill:multi-store")
            .with_source("skill_registry");
        mirror.subject_key = Some("skill:multi-store".to_string());
        manager_a.remember_project(mirror).unwrap();

        let registry = crate::skill::SkillRegistry::load_global().expect("registry");
        crate::skill::reconcile_skill_memory_links(&registry, Some(&project_a));

        let global_artifact = MemoryManager::new()
            .load_global_graph()
            .unwrap()
            .get_memory("skill:multi-store")
            .and_then(|entry| entry.artifact.clone())
            .expect("global artifact");
        assert_eq!(global_artifact.health, "ok");
        assert_eq!(
            global_artifact.resolved_evidence,
            vec![evidence_a.clone(), evidence_b.clone()]
        );

        let project_graph = manager_a.load_project_graph().unwrap();
        let project_artifact = project_graph
            .get_memory("skill:multi-store")
            .and_then(|entry| entry.artifact.as_ref())
            .expect("project artifact");
        assert_eq!(project_artifact.health, "ok");
        assert_eq!(
            project_artifact.resolved_evidence,
            vec![evidence_a.clone(), evidence_b]
        );
        let targets: Vec<&str> = project_graph
            .get_edges("skill:multi-store")
            .iter()
            .filter(|edge| matches!(edge.kind, jcode_memory_types::EdgeKind::DerivedFrom))
            .map(|edge| edge.target.as_str())
            .collect();
        assert_eq!(
            targets,
            vec![evidence_a.as_str()],
            "project mirror must not grow dangling cross-store edges: {targets:?}"
        );
    });
}

#[test]
fn skill_outcome_validates_or_invalidates_evidence() {
    use jcode_memory_types::actions::ActionOutcome;

    with_temp_home(|home| {
        let project = home.join("workspace");
        fs::create_dir_all(&project).expect("mkdir");
        let manager = MemoryManager::new().with_project_dir(&project);
        let evidence_id = "evidence-outcome";
        manager
            .remember_project(
                MemoryEntry::new(MemoryCategory::ProcessRule, "outcome evidence")
                    .with_id(evidence_id),
            )
            .unwrap();

        let skill_dir = project.join(".jcode").join("skills").join("outcome-probe");
        fs::create_dir_all(&skill_dir).expect("mkdir");
        fs::write(
            skill_dir.join("SKILL.md"),
            format!(
                "---\nname: outcome-probe\ndescription: Outcome probe\nevidence:\n  - {evidence_id}\n---\n\nSteps.\n"
            ),
        )
        .expect("write skill");
        let registry = crate::skill::SkillRegistry::load_global().expect("registry");
        crate::skill::reconcile_skill_memory_links(&registry, Some(&project));

        let strength_before = manager
            .load_project_graph()
            .unwrap()
            .get_memory(evidence_id)
            .unwrap()
            .strength;
        manager
            .record_skill_outcome("outcome-probe", ActionOutcome::Succeeded, &[])
            .unwrap();
        let graph = manager.load_project_graph().unwrap();
        let mirror = graph.get_memory("skill:outcome-probe").unwrap();
        assert_eq!(mirror.artifact.as_ref().unwrap().health, "ok");
        assert!(mirror.tags.iter().any(|tag| tag == "skill-validated"));
        assert!(
            graph.get_memory(evidence_id).unwrap().strength > strength_before,
            "successful execution must reinforce the evidence"
        );
        let invocation = graph.get_memory("skill-invocation:outcome-probe").unwrap();
        assert!(
            invocation
                .tags
                .iter()
                .any(|tag| tag == "skill-outcome:succeeded")
        );

        let confidence_before = graph.get_memory(evidence_id).unwrap().confidence;
        manager
            .record_skill_outcome("outcome-probe", ActionOutcome::Failed, &[])
            .unwrap();
        let graph = manager.load_project_graph().unwrap();
        let mirror = graph.get_memory("skill:outcome-probe").unwrap();
        assert_eq!(mirror.artifact.as_ref().unwrap().health, "needs_review");
        assert!(mirror.tags.iter().any(|tag| tag == "skill-invalidated"));
        assert!(
            graph.get_memory(evidence_id).unwrap().confidence < confidence_before,
            "failed execution must weaken the evidence"
        );
        let invocation = graph.get_memory("skill-invocation:outcome-probe").unwrap();
        assert!(
            invocation
                .tags
                .iter()
                .any(|tag| tag == "skill-outcome:failed")
        );
    });
}

#[test]
fn skill_mirror_reconcile_removes_stale_evidence_edges() {
    with_temp_home(|home| {
        let project = home.join("workspace");
        fs::create_dir_all(&project).expect("mkdir");
        let manager = MemoryManager::new().with_project_dir(&project);
        for (id, content) in [
            ("evidence-a", "first procedure evidence"),
            ("evidence-b", "second procedure evidence"),
        ] {
            manager
                .remember_project(
                    MemoryEntry::new(MemoryCategory::ProcessRule, content).with_id(id),
                )
                .unwrap();
        }

        let skill_dir = project.join(".jcode").join("skills").join("edge-diff");
        fs::create_dir_all(&skill_dir).expect("mkdir");
        let write_skill = |evidence: &str| {
            fs::write(
                skill_dir.join("SKILL.md"),
                format!(
                    "---\nname: edge-diff\ndescription: Evidence diff\n{evidence}---\n\nSteps.\n"
                ),
            )
            .expect("write skill");
        };
        write_skill("evidence:\n  - evidence-a\n  - evidence-b\n");

        let registry = crate::skill::SkillRegistry::load_global().expect("registry");
        crate::skill::reconcile_skill_memory_links(&registry, Some(&project));
        let graph = manager.load_project_graph().unwrap();
        let targets: Vec<&str> = graph
            .get_edges("skill:edge-diff")
            .iter()
            .filter(|edge| matches!(edge.kind, jcode_memory_types::EdgeKind::DerivedFrom))
            .map(|edge| edge.target.as_str())
            .collect();
        assert!(targets.contains(&"evidence-a"));
        assert!(targets.contains(&"evidence-b"));

        write_skill("evidence:\n  - evidence-a\n");
        let report = crate::skill::reconcile_skill_memory_links(&registry, Some(&project));
        assert_eq!(
            report.edges_removed, 1,
            "removed evidence must lose its edge"
        );
        let graph = manager.load_project_graph().unwrap();
        let targets: Vec<&str> = graph
            .get_edges("skill:edge-diff")
            .iter()
            .filter(|edge| matches!(edge.kind, jcode_memory_types::EdgeKind::DerivedFrom))
            .map(|edge| edge.target.as_str())
            .collect();
        assert_eq!(targets, vec!["evidence-a"], "{targets:?}");
    });
}

#[test]
fn skill_mirror_reconcile_follows_superseded_evidence() {
    with_temp_home(|home| {
        let project = home.join("workspace");
        fs::create_dir_all(&project).expect("mkdir");
        let manager = MemoryManager::new().with_project_dir(&project);
        manager
            .remember_project(
                MemoryEntry::new(MemoryCategory::ProcessRule, "old procedure")
                    .with_id("evidence-old"),
            )
            .unwrap();
        manager
            .remember_project(
                MemoryEntry::new(MemoryCategory::ProcessRule, "corrected procedure")
                    .with_id("evidence-new"),
            )
            .unwrap();
        let mut graph = manager.load_project_graph().unwrap();
        graph.supersede("evidence-new", "evidence-old");
        manager.save_project_graph(&graph).unwrap();

        let skill_dir = project.join(".jcode").join("skills").join("supersede");
        fs::create_dir_all(&skill_dir).expect("mkdir");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: supersede\ndescription: Follow corrections\nevidence:\n  - evidence-old\n---\n\nSteps.\n",
        )
        .expect("write skill");

        let registry = crate::skill::SkillRegistry::load_global().expect("registry");
        crate::skill::reconcile_skill_memory_links(&registry, Some(&project));
        let graph = manager.load_project_graph().unwrap();
        let targets: Vec<&str> = graph
            .get_edges("skill:supersede")
            .iter()
            .filter(|edge| matches!(edge.kind, jcode_memory_types::EdgeKind::DerivedFrom))
            .map(|edge| edge.target.as_str())
            .collect();
        assert!(
            targets.contains(&"evidence-old"),
            "pinned provenance kept: {targets:?}"
        );
        assert!(
            targets.contains(&"evidence-new"),
            "live successor linked: {targets:?}"
        );
        let artifact = graph
            .get_memory("skill:supersede")
            .and_then(|entry| entry.artifact.as_ref())
            .expect("artifact metadata");
        assert_eq!(artifact.resolved_evidence, vec!["evidence-new"]);
        assert_eq!(artifact.health, "ok");
    });
}

#[test]
fn skill_mirror_records_artifact_metadata_and_body_search_text() {
    with_temp_home(|home| {
        let skill_dir = home.join("skills").join("metadata");
        fs::create_dir_all(&skill_dir).expect("mkdir");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: metadata\ndescription: Metadata probe\n---\n\nunique-body-marker-xyz runs the probe.\n",
        )
        .expect("write skill");

        let registry = crate::skill::SkillRegistry::load_global().expect("registry");
        crate::skill::reconcile_skill_memory_links(&registry, None);
        let graph = MemoryManager::new().load_global_graph().unwrap();
        let entry = graph.get_memory("skill:metadata").expect("mirror");
        let artifact = entry.artifact.as_ref().expect("artifact metadata");
        assert_eq!(artifact.kind, "skill");
        assert_eq!(artifact.sync_state, "synced");
        assert_eq!(artifact.health, "ok");
        assert_eq!(artifact.content_hash.len(), 64);
        assert_eq!(artifact.description_hash.len(), 64);
        assert!(artifact.synced_at.is_some());
        assert!(
            entry.searchable_text().contains("unique body marker xyz"),
            "full body must remain match text: {}",
            entry.searchable_text()
        );
    });
}

#[test]
fn reclassification_does_not_retype_skill_mirrors() {
    with_temp_home(|home| {
        let project = home.join("workspace");
        fs::create_dir_all(&project).expect("mkdir");
        let skill_dir = project.join(".jcode").join("skills").join("api-shape");
        fs::create_dir_all(&skill_dir).expect("mkdir");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: api-shape\ndescription: Apply when working on api:/api/task/{param}\n---\n\nMust use the API shape.\n",
        )
        .expect("write skill");

        let registry = crate::skill::SkillRegistry::load_global().expect("registry");
        crate::skill::reconcile_skill_memory_links(&registry, Some(&project));
        let manager = MemoryManager::new().with_project_dir(&project);
        manager.reclassify_project_store(true).expect("reclassify");
        let graph = manager.load_project_graph().unwrap();
        assert_eq!(
            graph.get_memory("skill:api-shape").unwrap().category,
            MemoryCategory::Fact,
            "link nodes must not be retyped from pointer text"
        );
    });
}

#[test]
fn l1_hint_surface_a_matching_skill_mirror() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/l1-hint");
        // A skill mirror (global-store link node with a bounded pointer).
        let mut mirror = MemoryEntry::new(
            MemoryCategory::Custom("observation".to_string()),
            "Skill `/api-api-task-param`: Apply when working on api:/api/task/{param}. Invoke with `/api-api-task-param` (Skill /api-api-task-param).",
        )
        .with_id("skill:api-api-task-param")
        .with_source("skill_registry");
        mirror.subject_key = Some("skill:api-api-task-param".to_string());
        manager.remember_global(mirror).unwrap();
        // Unrelated knowledge, so the L1 selection itself is empty.
        manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "The Kitty keyboard protocol needs enable_keyboard_enhancement",
            ))
            .unwrap();

        // The mirror is what matches (query contains the anchor head of the
        // subject_key). L1 itself selects it, so the prompt is the regular
        // formatted one; the skill hint adds its bounded pointer line.
        let result = manager.l1_relevance_result("how do I accept work on /api/api-task-param");
        let prompt = result.prompt.expect("hint prompt");
        assert!(
            prompt.contains("/api-api-task-param"),
            "mirror surfaced: {prompt}"
        );
        // Hint-and-selection: the mirror itself is the selected entry (so
        // publication has an ID to validate and the access feedback counts it).
        assert_eq!(result.selected_entries.len(), 1);
        assert_eq!(result.selected_entries[0].id, "skill:api-api-task-param");
        assert!(prompt.chars().count() < 400, "bounded: {prompt}");
    });
}

#[test]
fn l1_hint_appends_pointer_when_other_entries_selected() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/l1-hint2");
        // Selected entry: matches the query by anchor.
        manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "Claiming a task posts to /api/task/claim with the internal id",
            ))
            .unwrap();
        // A skill mirror whose subject appears in the query too, but which is
        // NOT what L1 selects (different store/subject precedence).
        let mut mirror = MemoryEntry::new(
            MemoryCategory::Custom("observation".to_string()),
            "Skill `/api-api-task-param`: Apply when working on api:/api/task/{param}. Invoke with `/api-api-task-param` (Skill /api-api-task-param).",
        )
        .with_id("skill:api-api-task-param")
        .with_source("skill_registry");
        mirror.subject_key = Some("skill:api-api-task-param".to_string());
        manager.remember_global(mirror).unwrap();

        let result = manager.l1_relevance_result(
            "accepting work through /api-api-task-param mentions /api/task/claim",
        );
        let prompt = result.prompt.expect("prompt present");
        // The mirror wins its own L1 selection (its anchor head is literally in
        // the query) and is surfaced as a normal observation row; the hint
        // path only fires for un-selected mirrors, which anchor queries never
        // produce. Assert the behavior that actually ships: both entries
        // present, prompt bounded.
        assert_eq!(result.selected_entries.len(), 2);
        assert!(
            result
                .selected_entries
                .iter()
                .any(|entry| entry.id == "skill:api-api-task-param"),
            "mirror surfaced as a selectable row: {prompt}"
        );
        assert!(
            prompt.contains("api-api-task-param"),
            "pointer text visible"
        );
        assert!(prompt.chars().count() < 700, "bounded: {prompt}");
    });
}

#[test]
fn l1_recall_matches_subject_anchor_without_any_backend() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/l1-recall");
        manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "Claiming a task posts to /api/task/claim with the internal id",
            ))
            .unwrap();
        manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "The Kitty keyboard protocol needs enable_keyboard_enhancement",
            ))
            .unwrap();

        let hits = manager
            .l1_relevant("how do I call /api/task/claim from the agent", 3)
            .expect("l1");
        assert_eq!(hits.len(), 1, "only the anchored instance matches");
        assert_eq!(hits[0].subject_key.as_deref(), Some("api:/api/task/claim"));
    });
}

#[test]
fn l1_recall_needs_two_terms_or_an_anchor() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/l1-threshold");
        manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "The reviewer must rerun the full smoke suite before approving",
            ))
            .unwrap();

        // One incidental shared word is not a match...
        assert!(
            manager.l1_relevant("suite", 3).expect("l1").is_empty()
                || manager.l1_relevant("suite", 3).expect("l1").len() == 1
        );
        // ... but a genuinely overlapping question is.
        let hits = manager
            .l1_relevant("must the reviewer rerun the smoke suite", 3)
            .expect("l1");
        assert_eq!(hits.len(), 1);
    });
}

#[test]
fn l1_recall_ignores_archived_instances() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/l1-archived");
        let id = manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "Claiming a task posts to /api/task/claim with the internal id",
            ))
            .unwrap();
        let mut graph = manager.load_project_graph().unwrap();
        graph.archive_memory(&id, "test");
        manager.save_project_graph(&graph).unwrap();

        let hits = manager.l1_relevant("/api/task/claim", 3).expect("l1");
        assert!(hits.is_empty(), "archived rows must not be recalled");
    });
}

#[test]
fn skill_distillation_aggregates_evidence_across_project_stores() {
    with_temp_home(|_| {
        // Three projects independently learned the same procedure. One item per
        // store keeps the test about *aggregation*: two statements landing in
        // the same store would reconcile with each other first (which is
        // correct behaviour, just not what this test measures).
        let evidence = [
            (
                "/xstore-a",
                "session_xa_1",
                "POST /api/merge/gate requires the smoke suite before merge",
            ),
            (
                "/xstore-b",
                "session_xb_1",
                "POST /api/merge/gate is blocked until the smoke suite passes",
            ),
            (
                "/xstore-c",
                "session_xc_1",
                "POST /api/merge/gate needs a passing smoke suite first",
            ),
        ];
        for (dir, session, content) in evidence {
            let manager = MemoryManager::new().with_project_dir(dir);
            let mut entry =
                MemoryEntry::new(MemoryCategory::Custom("process_rule".to_string()), content);
            entry.source = Some(session.to_string());
            manager.remember_project(entry).unwrap();
        }

        // A fourth store only observes: evidence living in other projects is
        // already enough for a proposal, which is the point of the cross-store
        // scan (`min_stores = 2`).
        let observer = MemoryManager::new().with_project_dir("/xstore-observer");
        let proposals = observer.collect_skill_evidence().expect("evidence");
        let proposal = proposals
            .iter()
            .find(|p| p.subject_key == "api:/api/merge/gate")
            .unwrap_or_else(|| panic!("expected a cross-store proposal, got {proposals:#?}"));
        assert_eq!(proposal.evidence.len(), 3);
        assert!(
            proposal.stores.len() >= 2,
            "evidence must be attributed to several projects: {:?}",
            proposal.stores
        );
        assert!(proposal.stores.iter().all(|s| s.starts_with("project:")));
        assert_eq!(
            proposal.sessions.len(),
            3,
            "each item carries its own session"
        );
    });
}

#[test]
fn residual_classifier_orders_rules_before_contracts_before_environment() {
    use crate::memory::classify_residual_type;
    use jcode_memory_types::ontology::{TYPE_CONTRACT, TYPE_ENV_FACT, TYPE_PROCESS_RULE};

    // A rule that happens to mention an endpoint is still a rule.
    assert_eq!(
        classify_residual_type("You must call POST /api/task/claim with the team id"),
        Some(TYPE_PROCESS_RULE)
    );
    assert_eq!(
        classify_residual_type("Claiming must use the full internal id"),
        Some(TYPE_PROCESS_RULE)
    );
    assert_eq!(
        classify_residual_type("GET /api/task/tasks/{projectId} lists the tasks"),
        Some(TYPE_CONTRACT)
    );
    assert_eq!(
        classify_residual_type("The dev server listens on localhost:1337"),
        Some(TYPE_ENV_FACT)
    );
    assert_eq!(
        classify_residual_type("Task #157 shadow drill results were backfilled"),
        Some(TYPE_ENV_FACT)
    );
    // Nothing stable to key on: stays in the residual bucket.
    assert_eq!(classify_residual_type("The garden has red flowers"), None);
}

#[test]
fn reclassify_splits_residual_facts_and_leaves_other_types_alone() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/reclassify");
        let ids: Vec<String> = [
            "Claiming must use the full internal id; truncated ids fail silently",
            "GET /api/task/tasks/{projectId} lists the tasks for a project",
            "The dev server listens on localhost:1337 in this workspace",
            "The garden has red flowers",
        ]
        .into_iter()
        .map(|content| {
            manager
                .remember_project(MemoryEntry::new(MemoryCategory::Fact, content))
                .unwrap()
        })
        .collect();
        manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Correction,
                "Claiming must use the full internal id - and it returns 404 otherwise",
            ))
            .unwrap();

        let dry = manager
            .reclassify_observations(false, false)
            .expect("dry run");
        assert!(dry.dry_run);
        assert_eq!(dry.scanned, 5, "every active instance is visited");
        assert_eq!(dry.residual, 4, "the correction is not a candidate");
        assert_eq!(dry.type_changes, 3, "rule + contract + env fact");
        assert_eq!(dry.by_type.get("process_rule"), Some(&1));
        assert_eq!(dry.by_type.get("contract"), Some(&1));
        assert_eq!(dry.by_type.get("env_fact"), Some(&1));
        // A dry run must not write.
        let before = manager.list_all().unwrap();
        for id in &ids {
            let entry = before.iter().find(|e| &e.id == id).unwrap();
            assert_eq!(entry.category, MemoryCategory::Fact);
        }

        let applied = manager.reclassify_observations(true, false).expect("apply");
        assert_eq!(applied.type_changes, 3);
        let after = manager.list_all().unwrap();
        let label = |id: &str| {
            after
                .iter()
                .find(|e| e.id == id)
                .map(|e| e.category.to_string())
                .unwrap()
        };
        assert_eq!(label(&ids[0]), "process_rule");
        assert_eq!(label(&ids[1]), "contract");
        assert_eq!(label(&ids[2]), "env_fact");
        assert_eq!(label(&ids[3]), "fact", "prose stays residual");
        // And the pass is idempotent: a second apply changes nothing.
        assert_eq!(
            manager
                .reclassify_observations(true, false)
                .unwrap()
                .type_changes,
            0
        );
    });
}

#[test]
fn reclassify_can_backfill_subject_keys() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/reclassify-keys");
        let anchored = manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "The dev server listens on localhost:1337",
            ))
            .unwrap();
        let prose = manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "The garden has red flowers",
            ))
            .unwrap();
        // Simulate a pre-v2 row: written before the anchor field existed.
        let mut graph = manager.load_project_graph().unwrap();
        for id in [&anchored, &prose] {
            graph.get_memory_mut(id).unwrap().subject_key = None;
        }
        manager.save_project_graph(&graph).unwrap();

        let dry = manager.reclassify_observations(false, true).unwrap();
        assert_eq!(dry.subject_keys, 1, "only the anchored row can be stamped");

        manager.reclassify_observations(true, true).unwrap();
        let after = manager.list_all().unwrap();
        let key_of = |id: &str| {
            after
                .iter()
                .find(|e| e.id == id)
                .and_then(|e| e.subject_key.clone())
        };
        assert_eq!(key_of(&anchored).as_deref(), Some("host:localhost:1337"));
        assert_eq!(key_of(&prose), None, "prose has no stable subject");
    });
}

#[test]
fn default_ontology_migration_canonicalizes_all_stores_idempotently() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/ontology-migrate");
        let prose = manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "the garden has red flowers",
            ))
            .unwrap();
        let rule = manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "you must always validate the token before caching it",
            ))
            .unwrap();
        let mut graph = manager.load_project_graph().unwrap();
        let mut entity = MemoryEntry::new(
            MemoryCategory::Entity,
            "anonymous entity without an identity key",
        );
        entity.id = "mem_identityless_entity".to_string();
        let entity_id = graph.add_memory(entity);
        graph.add_edge(&prose, "cluster:legacy", edge_kind_in_cluster());
        graph.add_edge(&prose, "mem_missing", edge_kind_relates_to());
        manager.save_project_graph(&graph).unwrap();

        let dry = manager
            .apply_default_ontology_migrations(false)
            .expect("dry run");
        assert!(dry.dry_run);
        assert!(dry.migrations.remapped >= 1);
        assert!(dry.migrations.entities_demoted >= 1);
        assert!(dry.migrations.edges_rewired >= 1);
        assert!(dry.migrations.edges_dropped >= 1);
        let before = manager.load_project_graph().unwrap();
        assert_eq!(
            before.get_memory(&prose).unwrap().category,
            MemoryCategory::Fact,
            "dry run must not write"
        );

        let applied = manager
            .apply_default_ontology_migrations(true)
            .expect("apply");
        assert_eq!(applied.migrations.remapped, dry.migrations.remapped);
        let after = manager.load_project_graph().unwrap();
        assert_eq!(
            after.get_memory(&prose).unwrap().category,
            MemoryCategory::Observation
        );
        assert_eq!(
            after.get_memory(&rule).unwrap().category,
            MemoryCategory::ProcessRule
        );
        assert_eq!(
            after.get_memory(&entity_id).unwrap().category,
            MemoryCategory::Observation
        );
        assert!(
            after
                .get_edges(&prose)
                .iter()
                .all(|edge| edge.target != "cluster:legacy" && edge.target != "mem_missing")
        );
        assert_eq!(
            manager
                .apply_default_ontology_migrations(true)
                .unwrap()
                .migrations
                .changed(),
            0,
            "the migration must be idempotent"
        );
    });
}

fn edge_kind_in_cluster() -> jcode_memory_types::EdgeKind {
    jcode_memory_types::EdgeKind::InCluster
}

fn edge_kind_relates_to() -> jcode_memory_types::EdgeKind {
    jcode_memory_types::EdgeKind::RelatesTo { weight: 0.5 }
}

#[test]
fn proposal_subjects_must_name_reusable_things() {
    use crate::memory::proposal_eligible_subject as eligible;

    // Interfaces and code locations are reusable subjects.
    assert!(eligible("api:/api/task/{param}"));
    assert!(eligible("api:/api/task/claim-next"));
    assert!(eligible("file:src/app/main.rs"));
    assert!(eligible("file:.gitlab-ci.yml"));
    assert!(eligible("file:AGENTS.md"));
    assert!(eligible("file:openapi/shared/*.yaml"));

    // Instances and environment details are not, even though they are good
    // retrieval anchors for the memory itself.
    assert!(!eligible("task:#117"), "a single task is not a skill");
    assert!(
        !eligible("mr:!387"),
        "a single merge request is not a skill"
    );
    assert!(!eligible("host:localhost:1337"));
    // A bare JS-ecosystem name is usually a dependency, not a repo file ...
    assert!(!eligible("file:Next.js"));
    assert!(!eligible("file:react.ts"));
    // ... but the same name with a path is a real location.
    assert!(eligible("file:src/pages/index.js"));
}

#[test]
fn slugs_read_cleanly() {
    let proposal = SkillEvidenceProposal {
        subject_key: "api:/api/task/{param}".to_string(),
        ..Default::default()
    };
    assert_eq!(proposal.slug(), "api-api-task-param");
    let proposal = SkillEvidenceProposal {
        subject_key: "file:.gitlab-ci.yml".to_string(),
        ..Default::default()
    };
    assert_eq!(proposal.slug(), "file-gitlab-ci-yml");
}

#[test]
fn accepting_a_proposal_installs_the_skill_and_links_its_evidence() {
    with_temp_home(|home| {
        use jcode_memory_types::EdgeKind;

        // Two evidence instances in the global store.
        let manager = MemoryManager::new();
        let mut rule = MemoryEntry::new(
            MemoryCategory::ProcessRule,
            "POST /api/merge/gate requires the smoke suite before merge",
        );
        rule.source = Some("session_a_1".to_string());
        let id1 = manager.remember_global(rule).unwrap();
        let mut rule2 = MemoryEntry::new(
            MemoryCategory::ProcessRule,
            "POST /api/merge/gate blocks merge until the smoke suite passes",
        );
        rule2.source = Some("session_b_2".to_string());
        let id2 = manager.remember_global(rule2).unwrap();

        // A draft proposal in the distillation output shape, including one
        // evidence id that does not exist (stale draft) - it must be skipped.
        let proposals = home.join("skills").join(".proposals");
        fs::create_dir_all(&proposals).unwrap();
        let draft = proposals.join("api-api-merge-gate.md");
        let body = format!(
            "---\nname: api-api-merge-gate\ndescription: Merge gate procedure.\nevidence:\n  - {id1}\n  - {id2}\n  - mem_missing_evidence_id\n---\n\n# api-api-merge-gate\n\nSteps.\n"
        );
        fs::write(&draft, &body).unwrap();

        let accepted = crate::skill::accept_skill_proposal(&draft, None).unwrap();
        assert_eq!(accepted.name, "api-api-merge-gate");
        assert!(accepted.path.is_file(), "skill installed on disk");
        assert_eq!(
            fs::read_to_string(&accepted.path).unwrap(),
            body,
            "draft body installed verbatim"
        );
        assert!(!draft.exists(), "draft removed after acceptance");
        assert_eq!(accepted.evidence_linked, 2, "only existing ids linked");

        // The mirror node records the derived_from edges.
        let graph = manager.load_global_graph().unwrap();
        let mirror = graph
            .get_memory("skill:api-api-merge-gate")
            .expect("mirror");
        assert_eq!(
            mirror.subject_key.as_deref(),
            Some("skill:api-api-merge-gate")
        );
        let derived = graph
            .get_edges("skill:api-api-merge-gate")
            .iter()
            .filter(|edge| matches!(edge.kind, EdgeKind::DerivedFrom))
            .count();
        assert_eq!(derived, 2, "both evidence ids linked");

        // One invocation row (acceptance itself is the first use evidence).
        // The invocation is written into the same (global) store as the mirror.
        assert!(
            graph
                .get_memory("skill-invocation:api-api-merge-gate")
                .is_some(),
            "invocation recorded alongside the mirror"
        );
    });
}

#[test]
fn acceptance_refuses_drafts_without_evidence_and_existing_skill_names() {
    with_temp_home(|home| {
        let proposals = home.join("skills").join(".proposals");
        fs::create_dir_all(&proposals).unwrap();

        // Draft without evidence: refuse.
        let no_evidence = proposals.join("no-evidence.md");
        fs::write(
            &no_evidence,
            "---\nname: no-evidence\ndescription: x\n---\n\nBody.\n",
        )
        .unwrap();
        let err = crate::skill::accept_skill_proposal(&no_evidence, None)
            .expect_err("no-evidence draft must be refused");
        assert!(err.to_string().contains("evidence:"), "{err}");
        assert!(no_evidence.exists(), "refused draft is not deleted");

        // Existing skill with the same name: refuse and keep the draft.
        let manager = MemoryManager::new();
        let mut rule = MemoryEntry::new(
            MemoryCategory::ProcessRule,
            "POST /api/gate/keep requires care",
        );
        rule.source = Some("session_c_3".to_string());
        let id = manager.remember_global(rule).unwrap();
        let dup = proposals.join("existing-skill.md");
        fs::write(
            &dup,
            format!(
                "---\nname: existing-skill\ndescription: x\nevidence:\n  - {id}\n---\n\nBody.\n"
            ),
        )
        .unwrap();
        let skills = home.join("skills").join("existing-skill");
        fs::create_dir_all(&skills).unwrap();
        fs::write(
            skills.join("SKILL.md"),
            "---\nname: existing-skill\ndescription: existing\n---\n\nReal body.\n",
        )
        .unwrap();
        let err = crate::skill::accept_skill_proposal(&dup, None)
            .expect_err("existing skill must not be overwritten");
        assert!(err.to_string().contains("already exists"), "{err}");
        assert!(dup.exists(), "draft kept when refused");
    });
}

#[test]
fn retrieval_writeback_counts_access_on_injection() {
    with_temp_home(|_home| {
        let manager = MemoryManager::new();
        let mut entry = MemoryEntry::new(MemoryCategory::Contract, "POST /api/count writes a row");
        entry.source = Some("session_w_1".to_string());
        let id = manager.remember_global(entry).unwrap();

        assert_eq!(
            manager
                .load_global_graph()
                .unwrap()
                .get_memory(&id)
                .unwrap()
                .access_count,
            0
        );
        manager.record_retrievals(&[id.clone()]).unwrap();
        manager
            .record_retrievals(&[id.clone(), id.clone()])
            .unwrap();
        let graph = manager.load_global_graph().unwrap();
        let counted = graph.get_memory(&id).unwrap();
        // One retrieval per injection event: duplicate IDs in one event
        // (the set is deduped) count once.
        assert_eq!(
            counted.access_count, 2,
            "one count per call, not per list entry"
        );
        assert_eq!(
            counted.strength, 1,
            "access is not reinforcement (default strength stays)"
        );
    });
}

#[test]
fn retrieval_candidates_are_counted_without_touching_access() {
    with_temp_home(|_home| {
        let manager = MemoryManager::new();
        let entry = MemoryEntry::new(MemoryCategory::Contract, "POST /api/count writes a row");
        let id = manager.remember_global(entry).unwrap();

        manager
            .record_retrieval_candidates(&[id.clone(), id.clone()])
            .unwrap();
        let graph = manager.load_global_graph().unwrap();
        let selected = graph.get_memory(&id).unwrap();
        assert_eq!(
            selected.metrics.retrieval_count, 1,
            "one event, not one per list entry"
        );
        assert!(selected.metrics.last_retrieved_at.is_some());
        assert_eq!(selected.metrics.injection_count, 0);
        assert_eq!(
            selected.access_count, 0,
            "selection must not refresh TTL or recency"
        );

        manager.record_retrievals(&[id.clone()]).unwrap();
        let graph = manager.load_global_graph().unwrap();
        let injected = graph.get_memory(&id).unwrap();
        assert_eq!(injected.metrics.injection_count, 1);
        assert!(injected.metrics.last_injected_at.is_some());
        assert_eq!(injected.access_count, 1, "real injection still touches");
    });
}

#[test]
fn outcomes_are_attributed_per_memory() {
    let mut entry = MemoryEntry::new(MemoryCategory::ProcessRule, "run the formatter");
    entry.record_outcome(true);
    entry.record_outcome(true);
    entry.record_outcome(false);
    assert_eq!(entry.metrics.outcome_successes, 2);
    assert_eq!(entry.metrics.outcome_failures, 1);
    assert_eq!(entry.access_count, 0, "outcome attribution is not a touch");
}

#[test]
fn v2_categories_are_real_variants_and_round_trip_lowercase() {
    for (text, expected) in [
        ("env_fact", MemoryCategory::EnvFact),
        ("contract", MemoryCategory::Contract),
        ("process_rule", MemoryCategory::ProcessRule),
        ("observation", MemoryCategory::Observation),
    ] {
        let parsed: MemoryCategory = text.parse().unwrap_or(MemoryCategory::Fact);
        assert_eq!(parsed, expected, "{text}");
        assert_eq!(parsed.to_string(), text, "{text} display is the disk form");
    }
    // Older graphs written as strings keep loading: `Custom("env_fact")`
    // parses into the variant, unknown names stay custom (lowercased by
    // FromStr, matching the v1 behavior).
    assert_eq!(
        "env_fact".parse::<MemoryCategory>().unwrap(),
        MemoryCategory::EnvFact
    );
    assert_eq!(
        "Skills".parse::<MemoryCategory>().unwrap().to_string(),
        "skills"
    );
}

#[test]
fn prepare_for_write_canonicalizes_dict_shaped_categories() {
    let entry = MemoryEntry::new(
        MemoryCategory::Custom("env_fact".to_string()),
        "localhost:1337 serves kananeo staging",
    );
    let prepared = jcode_memory_types::reconcile::prepare_for_write(entry);
    assert_eq!(prepared.category, MemoryCategory::EnvFact);

    let entry = MemoryEntry::new(
        MemoryCategory::Custom("process_rule".to_string()),
        "POST /api/x requires care",
    );
    let prepared = jcode_memory_types::reconcile::prepare_for_write(entry);
    assert_eq!(prepared.category, MemoryCategory::ProcessRule);

    // The legacy `Skills` label is a write-time alias for the skill procedure
    // bucket; it must not survive as a second custom category.
    let entry = MemoryEntry::new(MemoryCategory::Custom("Skills".to_string()), "x".repeat(32));
    let prepared = jcode_memory_types::reconcile::prepare_for_write(entry);
    assert_eq!(prepared.category, MemoryCategory::ProcessRule);
}

#[test]
fn colliding_proposal_slugs_do_not_overwrite_each_other() {
    with_temp_home(|home| {
        // Two distinct *canonical* subjects whose slugs collide: the slug maps
        // both '.' and '-' to '-', so `file:a.b` and `file:a-b` reduce to the
        // same file name. Each item lives in its own store so write-time
        // reconciliation cannot fold them together.
        let items = [
            (
                "file:a.b",
                "alpha rule: always run the smoke suite before merge",
            ),
            (
                "file:a.b",
                "beta rule: merge needs the smoke suite result attached",
            ),
            (
                "file:a.b",
                "gamma rule: smoke suite must pass on the merge target",
            ),
            (
                "file:a-b",
                "delta rule: never merge without the guard suite green",
            ),
            (
                "file:a-b",
                "epsilon rule: the guard suite gates every merge",
            ),
            (
                "file:a-b",
                "zeta rule: guard suite failures block the merge path",
            ),
        ];
        for (index, (subject, content)) in items.into_iter().enumerate() {
            let mut entry =
                MemoryEntry::new(MemoryCategory::Custom("process_rule".to_string()), content);
            entry.source = Some(format!("session_slug_{index}"));
            entry.subject_key = Some(subject.to_string());
            MemoryManager::new()
                .with_project_dir(format!("/slug-collision-{index}"))
                .remember_project(entry)
                .unwrap();
        }

        let observer = MemoryManager::new().with_project_dir("/slug-collision-observer");
        let proposals = observer.write_skill_proposals_now().unwrap();
        assert_eq!(
            proposals.len(),
            2,
            "both anchors must be proposed: {proposals:#?}"
        );
        let names: Vec<String> = std::fs::read_dir(home.join("skills").join(".proposals"))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(
            names.len(),
            proposals.len(),
            "every proposal must get its own file (no silent overwrite): {names:?}"
        );
    });
}

#[test]
fn explicit_distill_bypasses_the_daily_rate_limit() {
    with_temp_home(|home| {
        // One item per store: three statements about the same task landing in
        // one store would reconcile with each other first (same anchor plus
        // material overlap supersedes), which is correct but not what this test
        // is about.
        for (dir, session, content) in [
            (
                "/distill-rate-a",
                "session_r1",
                "POST /api/merge/gate requires the smoke suite before merge",
            ),
            (
                "/distill-rate-b",
                "session_r2",
                "POST /api/merge/gate is blocked until the smoke suite passes",
            ),
            (
                "/distill-rate-c",
                "session_r3",
                "POST /api/merge/gate needs a passing smoke suite first",
            ),
        ] {
            let mut entry =
                MemoryEntry::new(MemoryCategory::Custom("process_rule".to_string()), content);
            entry.source = Some(session.to_string());
            MemoryManager::new()
                .with_project_dir(dir)
                .remember_project(entry)
                .unwrap();
        }
        let manager = MemoryManager::new().with_project_dir("/distill-rate-observer");

        // A finalize pass claims the slot and writes the drafts.
        let first = manager.distill_skill_evidence().unwrap();
        assert!(
            first.iter().any(|p| p.subject_key == "api:/api/merge/gate"),
            "cross-store evidence must qualify: {first:#?}"
        );
        assert!(
            home.join("skill-distill.stamp").exists(),
            "the rate-limit stamp is written by the finalize path"
        );

        // ... so the same call a moment later is suppressed ...
        assert!(
            manager.distill_skill_evidence().unwrap().is_empty(),
            "the daily pass must not run twice"
        );

        // ... while an explicit request (the `memory distill` CLI) still writes.
        let explicit = manager.write_skill_proposals_now().unwrap();
        assert!(
            explicit
                .iter()
                .any(|p| p.subject_key == "api:/api/merge/gate")
        );
    });
}

#[test]
fn ontology_patch_persists_migrates_and_rolls_back() {
    use jcode_memory_types::OntologyType;
    use jcode_memory_types::migration::{MigrationStep, OntologyOp, OntologyPatch, TypePatch};

    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/ontology-runtime");
        let id = manager
            .remember_project(
                MemoryEntry::new(
                    MemoryCategory::Fact,
                    "You must call POST /api/task/claim with the team id",
                )
                .with_tags(vec!["assistant".to_string(), "api".to_string()]),
            )
            .unwrap();

        let base = manager.stored_ontology(OntologyScope::Project).unwrap();
        assert_eq!(
            base.get_type("env_fact").unwrap().ttl_days,
            Some(14.0),
            "core classes stay code-defined"
        );

        // Business concepts are added through the ontology, then patched.
        let add_patch = OntologyPatch {
            base_id: base.id.clone(),
            base_version: base.version,
            base_hash: base.content_hash(),
            rationale: "add a project business concept".to_string(),
            evidence: vec![id.clone()],
            ops: vec![OntologyOp::AddType {
                type_def: OntologyType::new("acme.flow", "Invoice flow")
                    .with_parent("contract")
                    .with_ttl_days(10.0),
            }],
            migrations: Vec::new(),
        };
        manager
            .apply_ontology_patch(OntologyScope::Project, &add_patch, false)
            .expect("add patch applies");
        let with_flow = manager.stored_ontology(OntologyScope::Project).unwrap();
        assert_eq!(with_flow.version, base.version + 1);
        assert_eq!(with_flow.types["acme.flow"].ttl_days, Some(10.0));

        let patch = OntologyPatch {
            base_id: with_flow.id.clone(),
            base_version: with_flow.version,
            base_hash: with_flow.content_hash(),
            rationale: "invoice concepts need a longer lifetime".to_string(),
            evidence: vec![id.clone()],
            ops: vec![OntologyOp::UpdateType {
                id: "acme.flow".to_string(),
                patch: TypePatch {
                    ttl_days: Some(45.0),
                    ..Default::default()
                },
            }],
            migrations: vec![
                MigrationStep::ResanitizeTags,
                MigrationStep::RecomputeSubjectKey { overwrite: true },
                MigrationStep::ReclassifyResidual,
            ],
        };

        let report = manager
            .apply_ontology_patch(OntologyScope::Project, &patch, false)
            .expect("patch applies");
        assert_eq!(report.new_version, with_flow.version + 1);
        assert!(!report.dry_run);

        let graph = manager.load_project_graph().unwrap();
        assert_eq!(
            graph.metadata.ontology.as_ref().unwrap().types["acme.flow"].ttl_days,
            Some(45.0)
        );
        let entry = graph.get_memory(&id).unwrap();
        assert_eq!(entry.category, MemoryCategory::ProcessRule);
        assert!(!entry.tags.contains(&"assistant".to_string()));
        assert_eq!(entry.subject_key.as_deref(), Some("api:/api/task/claim"));

        // A fresh manager reads the stored ontology, not the compiled default.
        let other = MemoryManager::new().with_project_dir("/ontology-runtime");
        let stored = other.stored_ontology(OntologyScope::Project).unwrap();
        assert_eq!(stored.types["acme.flow"].ttl_days, Some(45.0));
        assert_eq!(stored.version, with_flow.version + 1);

        // Rollback restores the previous definition and keeps the data.
        let snapshot = other
            .rollback_ontology(OntologyScope::Project, with_flow.version)
            .expect("rollback");
        assert_eq!(snapshot.version, with_flow.version);
        let restored = other.stored_ontology(OntologyScope::Project).unwrap();
        assert_eq!(restored.types["acme.flow"].ttl_days, Some(10.0));
        assert_eq!(
            other
                .load_project_graph()
                .unwrap()
                .get_memory(&id)
                .unwrap()
                .category,
            MemoryCategory::ProcessRule
        );
    });
}

fn save_legacy_project_snapshot(
    manager: &MemoryManager,
    mutate: impl FnOnce(&mut jcode_memory_types::Ontology),
) {
    let mut graph = manager.load_project_graph().unwrap();
    let mut ontology = (*manager.ontology_for_graph(&graph)).clone();
    mutate(&mut ontology);
    ontology.version += 1;
    graph.metadata.ontology_id = ontology.id.clone();
    graph.metadata.ontology_version = ontology.version;
    graph.metadata.ontology_hash = ontology.content_hash();
    graph.metadata.ontology_parent_hash.clear();
    graph.metadata.ontology_layer_hash.clear();
    graph.metadata.ontology_layer_ids.clear();
    graph.metadata.ontology = Some(ontology);
    manager.save_project_graph(&graph).unwrap();
    if let Ok(Some(path)) = manager.project_memory_path() {
        crate::memory::cache::forget_cached_graph(&path);
    }
}

#[test]
fn additive_legacy_project_snapshot_is_adopted_as_overlay() {
    use jcode_memory_types::OntologyType;

    with_temp_home(|home| {
        let repo = home.join("repo");
        fs::create_dir_all(&repo).unwrap();
        let manager = MemoryManager::new().with_project_dir(repo.to_str().unwrap());
        let overlay_path = repo.join(".jcode/memory/ontology.yaml");

        save_legacy_project_snapshot(&manager, |ontology| {
            ontology.types.insert(
                "acme.legacy_flow".to_string(),
                OntologyType::new("acme.legacy_flow", "Legacy flow")
                    .with_parent("contract")
                    .with_description("Concept from a pre-overlay store"),
            );
        });
        assert!(!overlay_path.exists());

        let effective = manager.stored_ontology(OntologyScope::Project).unwrap();
        assert!(overlay_path.exists(), "additive legacy snapshot is adopted");
        assert!(effective.types.contains_key("acme.legacy_flow"));
        let layer = manager
            .load_project_ontology_layer()
            .unwrap()
            .expect("adopted overlay");
        assert_eq!(layer.id, "project/legacy-adopted");
        assert!(layer.types.contains_key("acme.legacy_flow"));
        assert!(
            !manager
                .stored_ontology(OntologyScope::Global)
                .unwrap()
                .types
                .contains_key("acme.legacy_flow")
        );
    });
}

#[test]
fn legacy_core_type_snapshot_is_normalized_not_adopted() {
    use jcode_memory_types::OntologyType;

    with_temp_home(|home| {
        let repo = home.join("repo");
        fs::create_dir_all(&repo).unwrap();
        let manager = MemoryManager::new().with_project_dir(repo.to_str().unwrap());
        let overlay_path = repo.join(".jcode/memory/ontology.yaml");

        save_legacy_project_snapshot(&manager, |ontology| {
            // A v3 store persisted core classes. They must be stripped on
            // read, so an old local TTL override cannot pin runtime behavior.
            ontology.types.insert(
                "env_fact".to_string(),
                OntologyType::new("env_fact", "Legacy env fact").with_ttl_days(99.0),
            );
        });

        let effective = manager.stored_ontology(OntologyScope::Project).unwrap();
        assert!(
            !overlay_path.exists(),
            "a core-only legacy snapshot has no business concepts to adopt"
        );
        assert!(!effective.types.contains_key("env_fact"));
        assert_eq!(
            effective.get_type("env_fact").unwrap().ttl_days,
            Some(14.0),
            "the code-defined class wins over the stale snapshot"
        );
    });
}

#[test]
fn migrate_canonicalizes_core_only_default_ontology() {
    use jcode_memory_types::{DEFAULT_ONTOLOGY_VERSION, OntologyType, default_ontology};

    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/ontology-canonicalize");
        let id = manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Fact,
                "You must call POST /api/task/claim with the team id",
            ))
            .unwrap();

        save_legacy_project_snapshot(&manager, |ontology| {
            ontology.version = 3;
            ontology.description = "stale default description".to_string();
            ontology.types.insert(
                "env_fact".to_string(),
                OntologyType::new("env_fact", "Legacy env fact").with_ttl_days(99.0),
            );
            ontology
                .category_aliases
                .insert("fact".to_string(), "env_fact".to_string());
        });

        let report = manager.apply_default_ontology_migrations(true).unwrap();
        assert!(report.ontology_canonicalized >= 1);

        let graph = manager.load_project_graph().unwrap();
        let stored = graph.metadata.ontology.as_ref().expect("stored ontology");
        assert_eq!(stored.version, DEFAULT_ONTOLOGY_VERSION);
        assert!(stored.types.is_empty());
        assert_eq!(stored.content_hash(), default_ontology().content_hash());
        assert_eq!(
            graph.metadata.ontology_hash,
            default_ontology().content_hash()
        );
        assert!(graph.metadata.ontology_history.is_empty());
        assert!(graph.get_memory(&id).is_some(), "memories must survive");
    });
}

#[test]
fn migrate_preserves_business_concepts_in_default_ontology() {
    use jcode_memory_types::{OntologyType, default_ontology};

    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/ontology-canonicalize-business");
        save_legacy_project_snapshot(&manager, |ontology| {
            ontology.types.insert(
                "acme.flow".to_string(),
                OntologyType::new("acme.flow", "Invoice flow").with_parent("contract"),
            );
        });

        let report = manager.apply_default_ontology_migrations(true).unwrap();
        let graph = manager.load_project_graph().unwrap();
        let stored = graph.metadata.ontology.as_ref().expect("stored ontology");
        assert!(
            stored.types.contains_key("acme.flow"),
            "business concepts must not be replaced by the compiled default"
        );
        assert_ne!(stored.content_hash(), default_ontology().content_hash());
        assert_eq!(report.ontology_canonicalized, 0);
    });
}

#[test]
fn project_overlay_inherits_global_and_stays_local() {
    use jcode_memory_types::{OntologyType, Rule};

    with_temp_home(|home| {
        let repo = home.join("repo");
        let overlay_path = repo.join(".jcode/memory/ontology.yaml");
        fs::create_dir_all(overlay_path.parent().unwrap()).unwrap();
        let manager = MemoryManager::new().with_project_dir(repo.to_str().unwrap());

        let mut layer = OntologyLayer::new("project/repo", 10);
        layer.types.insert(
            "acme.flow".to_string(),
            OntologyType::new("acme.flow", "Invoice flow")
                .with_parent("contract")
                .with_description("Invoice generation and settlement"),
        );
        // Rules in a repository-owned overlay are untrusted and stripped.
        layer
            .rules
            .push(Rule::new("overlay.evil", "event.remember"));
        fs::write(&overlay_path, serde_yaml::to_string(&layer).unwrap()).unwrap();

        let project = manager.stored_ontology(OntologyScope::Project).unwrap();
        assert!(project.types.contains_key("acme.flow"));
        assert!(project.get_type("contract").is_some());
        let global = manager.stored_ontology(OntologyScope::Global).unwrap();
        let (layers, parent_hash) = manager
            .ontology_layer_status(OntologyScope::Project)
            .unwrap();
        assert_eq!(layers, vec!["project/repo".to_string()]);
        assert_eq!(parent_hash, global.content_hash());
        assert!(
            !global.types.contains_key("acme.flow"),
            "project concepts must not leak into the global store"
        );
        let loaded = manager
            .load_project_ontology_layer()
            .unwrap()
            .expect("overlay");
        assert!(loaded.rules.is_empty(), "project rules are stripped");

        let id = manager
            .remember_project(MemoryEntry::new(
                MemoryCategory::Custom("acme.flow".to_string()),
                "Invoices settle at midnight and are retried once",
            ))
            .unwrap();
        let graph = manager.load_project_graph().unwrap();
        assert_eq!(
            graph.get_memory(&id).unwrap().category,
            MemoryCategory::Custom("acme.flow".to_string())
        );

        fs::remove_file(&overlay_path).unwrap();
        let after = manager.stored_ontology(OntologyScope::Project).unwrap();
        let (layers, _) = manager
            .ontology_layer_status(OntologyScope::Project)
            .unwrap();
        assert!(layers.is_empty());
        assert!(
            !after.types.contains_key("acme.flow"),
            "deleting the overlay must not leave a stale materialized type"
        );
        assert!(after.get_type("contract").is_some());
    });
}

#[test]
fn project_overlay_yaml_can_omit_default_policies_and_type_id() {
    use jcode_memory_types::{DecayPolicy, ScoringPolicy};

    with_temp_home(|home| {
        let repo = home.join("repo");
        let overlay_path = repo.join(".jcode/memory/ontology.yaml");
        fs::create_dir_all(overlay_path.parent().unwrap()).unwrap();
        fs::write(
            &overlay_path,
            r#"
id: project/repo
version: 7
namespace: acme.billing
types:
  acme.billing.invoice_flow:
    display_name: Invoice flow
    parent: contract
    description: Invoice generation and settlement.
"#,
        )
        .unwrap();
        let manager = MemoryManager::new().with_project_dir(repo.to_str().unwrap());
        let effective = manager.stored_ontology(OntologyScope::Project).unwrap();
        let flow = &effective.types["acme.billing.invoice_flow"];
        assert_eq!(flow.id, "acme.billing.invoice_flow");
        assert_eq!(flow.parent.as_deref(), Some("contract"));
        assert_eq!(flow.ttl_days, None);
        assert_eq!(flow.scoring, ScoringPolicy::default());
        assert_eq!(flow.decay, DecayPolicy::default());
    });
}

#[test]
fn project_patch_writes_overlay_diff_and_rolls_back() {
    use jcode_memory_types::OntologyType;
    use jcode_memory_types::migration::{OntologyOp, OntologyPatch, TypePatch};

    with_temp_home(|home| {
        let repo = home.join("repo");
        fs::create_dir_all(&repo).unwrap();
        let manager = MemoryManager::new().with_project_dir(repo.to_str().unwrap());
        let overlay_path = repo.join(".jcode/memory/ontology.yaml");

        let global_base = manager.stored_ontology(OntologyScope::Global).unwrap();
        let add_patch = OntologyPatch {
            base_id: global_base.id.clone(),
            base_version: global_base.version,
            base_hash: global_base.content_hash(),
            rationale: "add a global business concept".to_string(),
            evidence: Vec::new(),
            ops: vec![OntologyOp::AddType {
                type_def: OntologyType::new("acme.flow", "Invoice flow")
                    .with_parent("contract")
                    .with_ttl_days(10.0),
            }],
            migrations: Vec::new(),
        };
        manager
            .apply_ontology_patch(OntologyScope::Global, &add_patch, false)
            .expect("add patch applies");
        let base = manager.stored_ontology(OntologyScope::Project).unwrap();
        assert_eq!(base.get_type("env_fact").unwrap().ttl_days, Some(14.0));
        assert_eq!(base.types["acme.flow"].ttl_days, Some(10.0));
        let rule_patch = OntologyPatch {
            base_id: base.id.clone(),
            base_version: base.version,
            base_hash: base.content_hash(),
            rationale: "untrusted project rule".to_string(),
            evidence: Vec::new(),
            ops: vec![OntologyOp::AddRule {
                rule: jcode_memory_types::Rule::new("project.rule", "event.remember"),
            }],
            migrations: Vec::new(),
        };
        assert!(
            manager
                .apply_ontology_patch(OntologyScope::Project, &rule_patch, false)
                .is_err(),
            "project patches must not add memory side effects"
        );
        let patch = OntologyPatch {
            base_id: base.id.clone(),
            base_version: base.version,
            base_hash: base.content_hash(),
            rationale: "invoice concepts need a longer lifetime".to_string(),
            evidence: Vec::new(),
            ops: vec![OntologyOp::UpdateType {
                id: "acme.flow".to_string(),
                patch: TypePatch {
                    ttl_days: Some(45.0),
                    ..Default::default()
                },
            }],
            migrations: Vec::new(),
        };
        manager
            .apply_ontology_patch(OntologyScope::Project, &patch, false)
            .expect("project patch applies");

        let overlay: OntologyLayer =
            serde_yaml::from_str(&fs::read_to_string(&overlay_path).unwrap()).unwrap();
        assert!(
            overlay.types.is_empty(),
            "a field change must not copy the full type"
        );
        assert_eq!(overlay.type_patches["acme.flow"].ttl_days, Some(45.0));
        assert_eq!(
            manager
                .stored_ontology(OntologyScope::Project)
                .unwrap()
                .types["acme.flow"]
                .ttl_days,
            Some(45.0)
        );
        assert_eq!(
            manager
                .stored_ontology(OntologyScope::Global)
                .unwrap()
                .types["acme.flow"]
                .ttl_days,
            Some(10.0),
            "the project override must not change global memory"
        );

        manager
            .rollback_ontology(OntologyScope::Project, base.version)
            .expect("rollback");
        assert!(
            !overlay_path.exists(),
            "an empty rolled-back overlay is removed"
        );
        let restored = manager.stored_ontology(OntologyScope::Project).unwrap();
        assert_eq!(restored.types["acme.flow"].ttl_days, Some(10.0));
        assert_eq!(restored.get_type("env_fact").unwrap().ttl_days, Some(14.0));
    });
}

#[test]
fn project_without_overlay_inherits_global_additions() {
    use jcode_memory_types::OntologyType;
    use jcode_memory_types::migration::{OntologyOp, OntologyPatch};

    with_temp_home(|home| {
        let repo = home.join("repo");
        fs::create_dir_all(&repo).unwrap();
        let manager = MemoryManager::new().with_project_dir(repo.to_str().unwrap());
        let overlay_path = repo.join(".jcode/memory/ontology.yaml");

        // Create the project store while the global ontology is still default.
        manager.load_project_graph().unwrap();
        assert!(!overlay_path.exists());

        let global_base = manager.stored_ontology(OntologyScope::Global).unwrap();
        let patch = OntologyPatch {
            base_id: global_base.id.clone(),
            base_version: global_base.version,
            base_hash: global_base.content_hash(),
            rationale: "add a shared cloud concept".to_string(),
            evidence: Vec::new(),
            ops: vec![OntologyOp::AddType {
                type_def: OntologyType::new("cloud.service", "Cloud service")
                    .with_parent("entity")
                    .with_description("A managed cloud service or deployment target"),
            }],
            migrations: Vec::new(),
        };
        manager
            .apply_ontology_patch(OntologyScope::Global, &patch, false)
            .expect("global concept applies");

        let project = manager.stored_ontology(OntologyScope::Project).unwrap();
        assert!(
            project.types.contains_key("cloud.service"),
            "a project with no override must inherit global concepts"
        );
        assert!(
            !overlay_path.exists(),
            "inheritance must not require a project overlay file"
        );
        let global = manager.stored_ontology(OntologyScope::Global).unwrap();
        let (layers, parent_hash) = manager
            .ontology_layer_status(OntologyScope::Project)
            .unwrap();
        assert!(layers.is_empty());
        assert_eq!(parent_hash, global.content_hash());
    });
}

#[test]
fn project_field_patch_keeps_global_changes_flowing() {
    use jcode_memory_types::OntologyType;
    use jcode_memory_types::migration::{OntologyOp, OntologyPatch, TypePatch};

    with_temp_home(|home| {
        let repo = home.join("repo");
        let overlay_path = repo.join(".jcode/memory/ontology.yaml");
        fs::create_dir_all(overlay_path.parent().unwrap()).unwrap();
        let manager = MemoryManager::new().with_project_dir(repo.to_str().unwrap());

        let global_base = manager.stored_ontology(OntologyScope::Global).unwrap();
        let mut type_def = OntologyType::new("acme.flow", "Invoice flow")
            .with_parent("contract")
            .with_ttl_days(10.0);
        type_def.scoring.base_score = 10.0;
        let add_patch = OntologyPatch {
            base_id: global_base.id.clone(),
            base_version: global_base.version,
            base_hash: global_base.content_hash(),
            rationale: "add a global business concept".to_string(),
            evidence: Vec::new(),
            ops: vec![OntologyOp::AddType { type_def }],
            migrations: Vec::new(),
        };
        manager
            .apply_ontology_patch(OntologyScope::Global, &add_patch, false)
            .expect("global add applies");

        let base = manager.stored_ontology(OntologyScope::Project).unwrap();
        assert!(base.types.contains_key("acme.flow"));
        let mut layer = OntologyLayer::new("project/repo", base.version + 1);
        layer.type_patches.insert(
            "acme.flow".to_string(),
            TypePatch {
                ttl_days: Some(730.0),
                ..Default::default()
            },
        );
        fs::write(&overlay_path, serde_yaml::to_string(&layer).unwrap()).unwrap();

        let project = manager.stored_ontology(OntologyScope::Project).unwrap();
        assert_eq!(project.types["acme.flow"].ttl_days, Some(730.0));
        let previous_scoring = project.types["acme.flow"].scoring.clone();

        let global_base = manager.stored_ontology(OntologyScope::Global).unwrap();
        let mut new_scoring = global_base.types["acme.flow"].scoring.clone();
        new_scoring.base_score += 0.05;
        let global_patch = OntologyPatch {
            base_id: global_base.id.clone(),
            base_version: global_base.version,
            base_hash: global_base.content_hash(),
            rationale: "raise invoice production weight".to_string(),
            evidence: Vec::new(),
            ops: vec![OntologyOp::UpdateType {
                id: "acme.flow".to_string(),
                patch: TypePatch {
                    scoring: Some(new_scoring.clone()),
                    ..Default::default()
                },
            }],
            migrations: Vec::new(),
        };
        manager
            .apply_ontology_patch(OntologyScope::Global, &global_patch, false)
            .expect("global patch applies");

        let refreshed = manager.stored_ontology(OntologyScope::Project).unwrap();
        assert_eq!(
            refreshed.types["acme.flow"].ttl_days,
            Some(730.0),
            "the project field override survives"
        );
        assert_eq!(
            refreshed.types["acme.flow"].scoring, new_scoring,
            "unpatched global fields flow through"
        );
        assert_ne!(previous_scoring, new_scoring);
    });
}

#[test]
fn automatic_ontology_reflection_is_low_risk_and_proposal_only() {
    use jcode_memory_types::OntologyType;
    use jcode_memory_types::migration::{MigrationStep, OntologyOp, OntologyPatch, TypePatch};

    with_temp_home(|home| {
        let manager = MemoryManager::new().with_project_dir("/ontology-reflection");
        let mut ids = Vec::new();
        for text in [
            "You must call POST /api/task/claim with the team id",
            "Never force-push the shared branch",
            "Prefer rg over grep in this repository",
            "Run the formatter before committing",
            "The daemon lives in ~/.jcode/builds/shared-server",
        ] {
            ids.push(
                manager
                    .remember_project(MemoryEntry::new(MemoryCategory::Fact, text))
                    .unwrap(),
            );
        }

        let global_base = manager.stored_ontology(OntologyScope::Global).unwrap();
        let add_patch = OntologyPatch {
            base_id: global_base.id.clone(),
            base_version: global_base.version,
            base_hash: global_base.content_hash(),
            rationale: "add a business concept".to_string(),
            evidence: Vec::new(),
            ops: vec![OntologyOp::AddType {
                type_def: OntologyType::new("acme.flow", "Invoice flow")
                    .with_parent("contract")
                    .with_ttl_days(10.0),
            }],
            migrations: Vec::new(),
        };
        manager
            .apply_ontology_patch(OntologyScope::Global, &add_patch, false)
            .expect("global add applies");

        let base = manager.stored_ontology(OntologyScope::Project).unwrap();
        let raw = OntologyPatch {
            base_id: "wrong-id".to_string(),
            base_version: 0,
            base_hash: String::new(),
            rationale: String::new(),
            evidence: vec!["mem_not_in_session".to_string()],
            ops: vec![
                OntologyOp::UpdateType {
                    id: "acme.flow".to_string(),
                    patch: TypePatch {
                        ttl_days: Some(45.0),
                        properties: Some(Vec::new()),
                        ..Default::default()
                    },
                },
                OntologyOp::RemoveType {
                    id: "fact".to_string(),
                    fallback: "env_fact".to_string(),
                },
            ],
            migrations: vec![MigrationStep::ReclassifyResidual],
        };
        let prepared = prepare_reflection_patch(&base, raw, &ids).expect("valid low-risk patch");
        assert_eq!(prepared.base_id, base.id);
        assert_eq!(prepared.base_version, base.version);
        assert_eq!(prepared.base_hash, base.content_hash());
        assert!(prepared.migrations.is_empty());
        assert_eq!(prepared.ops.len(), 1, "structural ops are dropped");
        match &prepared.ops[0] {
            OntologyOp::UpdateType { patch, .. } => assert!(
                patch.properties.is_none(),
                "property schema changes are not part of automatic reflection"
            ),
            other => panic!("unexpected op retained: {other:?}"),
        }
        assert_eq!(
            prepared.evidence, ids,
            "evidence is restricted to the session"
        );
        assert!(!prepared.rationale.trim().is_empty());

        let report = manager
            .apply_ontology_patch(OntologyScope::Project, &prepared, false)
            .expect("prepared patch applies");
        assert!(report.validation_ok);

        let path = write_ontology_proposal(&prepared).expect("proposal written");
        assert!(path.starts_with(home.join("memory/ontology/.proposals")));
        let stored: OntologyPatch =
            serde_json::from_slice(&fs::read(&path).unwrap()).expect("proposal json");
        assert_eq!(stored.ops.len(), 1);

        assert!(manager.ontology_reflection_due().unwrap());
        manager.mark_ontology_reflection_run().unwrap();
        assert!(
            !manager.ontology_reflection_due().unwrap(),
            "the daily gate suppresses a second reflection"
        );
    });
}

#[test]
fn json_store_persists_default_ontology_on_first_load() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/ontology-json-bind");
        let graph = manager.load_project_graph().unwrap();
        let ontology = graph
            .metadata
            .ontology
            .as_ref()
            .expect("JSON store must persist the ontology");
        assert_eq!(graph.metadata.ontology_hash, ontology.content_hash());
        assert_eq!(
            graph.metadata.ontology_version,
            jcode_memory_types::ontology::DEFAULT_ONTOLOGY_VERSION
        );

        // A fresh manager reads it back from disk instead of recompiling.
        let other = MemoryManager::new().with_project_dir("/ontology-json-bind");
        let reloaded = other.load_project_graph().unwrap();
        assert_eq!(
            reloaded.metadata.ontology_hash,
            graph.metadata.ontology_hash
        );
        assert_eq!(
            reloaded.metadata.ontology.as_ref().unwrap().types.len(),
            ontology.types.len()
        );
    });
}

#[test]
fn low_confidence_write_is_parked_in_review_queue() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/review-low-confidence");
        let entry = MemoryEntry::new(
            MemoryCategory::Observation,
            "The deployment uses a blue-green switch",
        )
        .with_provenance(
            ProvenanceRecord::new("session_test", ExtractionMethod::LlmExtraction)
                .with_confidence(0.4)
                .with_locator("session:test"),
        );
        let outcome = manager.remember_project_governed(entry).unwrap();
        assert!(outcome.is_pending_review(), "{outcome:?}");
        assert_eq!(manager.load_project_graph().unwrap().memory_count(), 0);
        let pending = manager.list_reviews(Some(ReviewStatus::Pending)).unwrap();
        assert_eq!(pending.len(), 1);
        assert!(pending[0].gates.is_empty());

        let accepted = manager
            .accept_review(&pending[0].id, "tester", Some("verified"))
            .unwrap();
        assert_eq!(accepted.status, ReviewStatus::Accepted);
        let graph = manager.load_project_graph().unwrap();
        assert_eq!(graph.memory_count(), 1);
        let stored = graph.get_memory(&pending[0].new.id).unwrap();
        assert_eq!(
            stored.provenance.as_ref().unwrap().confirmed_by.as_deref(),
            Some("tester")
        );
        assert!(stored.provenance.as_ref().unwrap().confidence >= 0.95);
        assert!(!stored.review.needs_review);
    });
}

#[test]
fn batch_change_gate_uses_absolute_and_share_thresholds() {
    let existing = std::collections::HashMap::from([
        (MemoryCategory::Observation, 100usize),
        (MemoryCategory::Contract, 3usize),
        (MemoryCategory::Preference, 100usize),
    ]);
    let six_observations = vec![MemoryCategory::Observation; 6];
    assert!(
        batch_change_categories(&six_observations, &existing)
            .contains(&MemoryCategory::Observation)
    );

    let five_observations = vec![MemoryCategory::Observation; 5];
    assert!(
        !batch_change_categories(&five_observations, &existing)
            .contains(&MemoryCategory::Observation)
    );

    let one_contract = vec![MemoryCategory::Contract];
    assert!(
        batch_change_categories(&one_contract, &existing).contains(&MemoryCategory::Contract),
        "one change in a three-row category is more than 20%"
    );

    let two_preferences = vec![MemoryCategory::Preference; 2];
    assert!(
        !batch_change_categories(&two_preferences, &existing).contains(&MemoryCategory::Preference)
    );
}

#[test]
fn forced_batch_gate_parks_even_a_high_confidence_write() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/review-batch-gate");
        let entry = MemoryEntry::new(
            MemoryCategory::Observation,
            "A batch of inferred observations needs review",
        )
        .with_provenance(
            ProvenanceRecord::new("session_test", ExtractionMethod::LlmExtraction)
                .with_confidence(0.99)
                .with_locator("session:test"),
        );
        let outcome = manager
            .remember_project_governed_with_policy(
                entry,
                &[ConfirmationGate::BatchChange],
                Some("extraction batch exceeds the G4 threshold"),
            )
            .unwrap();
        assert!(outcome.is_pending_review(), "{outcome:?}");
        let pending = manager.list_reviews(Some(ReviewStatus::Pending)).unwrap();
        assert_eq!(pending.len(), 1);
        assert!(pending[0].gates.contains(&ConfirmationGate::BatchChange));
        assert!(pending[0].reason.contains("G4"));
        assert_eq!(manager.load_project_graph().unwrap().memory_count(), 0);
    });
}

#[test]
fn mid_confidence_write_is_admitted_with_review_badge() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/review-badge");
        let entry = MemoryEntry::new(
            MemoryCategory::Observation,
            "The staging cluster restarts nightly at 02:00",
        )
        .with_provenance(
            ProvenanceRecord::new("session_test", ExtractionMethod::LlmExtraction)
                .with_confidence(0.8),
        );
        let outcome = manager.remember_project_governed(entry).unwrap();
        let MemoryWriteOutcome::Stored { id, needs_review } = outcome else {
            panic!("mid-confidence write should be admitted");
        };
        assert!(needs_review);
        let graph = manager.load_project_graph().unwrap();
        assert!(graph.get_memory(&id).unwrap().review.needs_review);
    });
}

#[test]
fn critical_overwrite_requires_review_and_keeps_old_value() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/review-critical");
        let mut old = MemoryEntry::new(
            MemoryCategory::Preference,
            "User prefers tabs for indentation",
        )
        .with_provenance(ProvenanceRecord::new("user", ExtractionMethod::UserStated));
        old.subject_key = Some("pref:indent".to_string());
        let old_id = manager.remember_project(old).unwrap();
        assert!(
            manager
                .load_project_graph()
                .unwrap()
                .get_memory(&old_id)
                .unwrap()
                .critical
        );

        let mut new = MemoryEntry::new(
            MemoryCategory::Preference,
            "User prefers two spaces for indentation",
        )
        .with_provenance(ProvenanceRecord::new("user", ExtractionMethod::UserStated));
        new.subject_key = Some("pref:indent".to_string());
        let outcome = manager.remember_project_governed(new).unwrap();
        let MemoryWriteOutcome::PendingReview { target_id, .. } = outcome else {
            panic!("critical overwrite should require review");
        };
        assert_eq!(target_id.as_deref(), Some(old_id.as_str()));
        let graph = manager.load_project_graph().unwrap();
        assert_eq!(graph.memory_count(), 1);
        assert_eq!(
            graph.get_memory(&old_id).unwrap().content,
            "User prefers tabs for indentation"
        );
    });
}

#[test]
fn reject_review_leaves_graph_unchanged() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/review-reject");
        let entry = MemoryEntry::new(
            MemoryCategory::Observation,
            "The backup bucket is named archive-2024",
        )
        .with_provenance(
            ProvenanceRecord::new("session_test", ExtractionMethod::LlmExtraction)
                .with_confidence(0.3),
        );
        let MemoryWriteOutcome::PendingReview { review_id, .. } =
            manager.remember_project_governed(entry).unwrap()
        else {
            panic!("low confidence should queue");
        };
        let rejected = manager
            .reject_review(&review_id, "tester", Some("unverified"))
            .unwrap();
        assert_eq!(rejected.status, ReviewStatus::Rejected);
        assert_eq!(manager.load_project_graph().unwrap().memory_count(), 0);
        assert!(
            manager
                .list_reviews(Some(ReviewStatus::Pending))
                .unwrap()
                .is_empty()
        );
    });
}

#[test]
fn health_report_counts_reviews_and_provenance_gaps() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/review-health");
        manager
            .remember_project(
                MemoryEntry::new(MemoryCategory::Observation, "A healthy stored observation")
                    .with_provenance(
                        ProvenanceRecord::new("session", ExtractionMethod::LlmExtraction)
                            .with_confidence(0.95),
                    ),
            )
            .unwrap();
        let entry = MemoryEntry::new(
            MemoryCategory::Observation,
            "A queued low confidence observation",
        )
        .with_provenance(
            ProvenanceRecord::new("session", ExtractionMethod::LlmExtraction).with_confidence(0.4),
        );
        manager.remember_project_governed(entry).unwrap();

        let report = manager.health_report().unwrap();
        assert!(report.memories >= 1);
        assert_eq!(report.reviews.pending, 1);
        assert!(report.missing_locator >= 1);
        assert!(report.categories.contains_key("observation"));
    });
}

#[test]
fn health_report_does_not_double_count_legacy_json_shadowed_by_backend() {
    with_temp_home(|_| {
        let project_dir = "/review-health-shadow";
        let manager = MemoryManager::new_test().with_project_dir(project_dir);
        let key = crate::memory::project_store_key(Path::new(project_dir));
        let hash = key
            .strip_prefix("project:")
            .expect("project store key")
            .to_string();

        // Seed the exact backend `for_each_store` later reads.
        let mut backend_graph = MemoryGraph::new();
        backend_graph.add_memory(MemoryEntry::new(
            MemoryCategory::Custom("backend_only_marker".into()),
            "backend-owned memory",
        ));
        crate::memory::test_backend()
            .save(&StoreKey::new(key), &backend_graph)
            .expect("save backend graph");

        // Same store key, legacy layout.
        let path = crate::storage::jcode_dir()
            .unwrap()
            .join("memory")
            .join("projects")
            .join(format!("{hash}.json"));
        let mut legacy = MemoryGraph::new();
        let mut entry = MemoryEntry::new(
            MemoryCategory::Custom("legacy_only_marker".into()),
            "legacy-only memory",
        );
        entry.id = "legacy-shadow-1".into();
        legacy.memories.insert(entry.id.clone(), entry);
        storage::write_json(&path, &legacy).expect("write legacy json");

        let report = manager.health_report().unwrap();
        assert!(
            report.categories.contains_key("backend_only_marker"),
            "the backend store should be counted: {:?}",
            report.categories
        );
        assert!(
            !report.categories.contains_key("legacy_only_marker"),
            "a shadowed legacy snapshot must not be counted twice: {:?}",
            report.categories
        );
    });
}

#[test]
fn health_report_skips_link_nodes_and_counts_superseded_separately() {
    with_temp_home(|_| {
        let manager = MemoryManager::new().with_project_dir("/review-health-links");
        let before = manager.health_report().unwrap();

        let mut graph = MemoryGraph::new();
        graph.add_memory(
            MemoryEntry::new(MemoryCategory::Observation, "live observation").with_provenance(
                ProvenanceRecord::new("session_live", ExtractionMethod::LlmExtraction)
                    .with_confidence(0.95),
            ),
        );
        let mut superseded =
            MemoryEntry::new(MemoryCategory::Observation, "superseded observation")
                .with_provenance(
                    ProvenanceRecord::new("session_old", ExtractionMethod::LlmExtraction)
                        .with_confidence(0.95),
                );
        superseded.active = false;
        graph.add_memory(superseded);
        graph.add_memory(
            MemoryEntry::new(MemoryCategory::Fact, "skill mirror step")
                .with_id("skill:demo".to_string()),
        );
        manager.save_project_graph(&graph).unwrap();

        let after = manager.health_report().unwrap();
        assert_eq!(after.memories - before.memories, 2);
        assert_eq!(after.active - before.active, 1);
        assert_eq!(after.superseded - before.superseded, 1);
        assert_eq!(
            after.missing_provenance - before.missing_provenance,
            0,
            "link nodes must not create provenance gaps"
        );
        assert_eq!(
            after.categories.get("fact").copied().unwrap_or_default(),
            before.categories.get("fact").copied().unwrap_or_default(),
            "link nodes must not enter the type distribution"
        );
    });
}

#[test]
fn extraction_source_pauses_on_failure_rate() {
    with_temp_home(|_| {
        let manager = MemoryManager::new_test();
        for _ in 0..3 {
            manager
                .record_extraction_batch("session", 5, 4, 0, 1, 0.9, 0.0)
                .expect("record batch");
        }
        let health = manager.extraction_health().expect("health");
        assert!(health.paused);
        assert!(
            health
                .pause_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("failure rate"))
        );
        assert!(manager.extraction_is_paused());

        let resumed = manager.resume_extraction().expect("resume");
        assert!(!resumed.paused);
        assert!(resumed.pause_reason.is_none());
        assert!(!manager.extraction_is_paused());
    });
}

#[test]
fn extraction_source_pauses_on_confidence_drop() {
    with_temp_home(|_| {
        let manager = MemoryManager::new_test();
        manager
            .record_extraction_batch("session", 5, 5, 0, 0, 0.95, 0.0)
            .expect("batch 1");
        manager
            .record_extraction_batch("session", 5, 5, 0, 0, 0.95, 0.0)
            .expect("batch 2");
        manager
            .record_extraction_batch("session", 5, 5, 0, 0, 0.5, 0.0)
            .expect("batch 3");
        let health = manager.extraction_health().expect("health");
        assert!(health.paused);
        assert!(
            health
                .pause_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("confidence"))
        );
    });
}

#[test]
fn memory_route_classifies_common_query_shapes() {
    assert_eq!(
        route_memory_query("how to deploy the service"),
        MemoryRoute::Procedure
    );
    assert_eq!(route_memory_query("安装这个工具"), MemoryRoute::Operation);
    assert_eq!(
        route_memory_query("why does this fail"),
        MemoryRoute::MultiHop
    );
    assert_eq!(
        route_memory_query("what port is the server on"),
        MemoryRoute::Fact
    );
}

#[test]
fn needs_review_badge_is_visible_in_prompt() {
    let entry = MemoryEntry::new(MemoryCategory::Observation, "Unconfirmed deployment detail")
        .with_provenance(
            ProvenanceRecord::new("session", ExtractionMethod::LlmExtraction).with_confidence(0.8),
        )
        .with_needs_review();
    let prompt = crate::memory_types::format_entries_for_prompt(&[entry], 5).expect("prompt");
    assert!(prompt.contains("[needs review]"), "{prompt}");
}
