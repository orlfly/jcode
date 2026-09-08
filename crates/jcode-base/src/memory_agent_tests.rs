use super::*;
use crate::memory::MemoryCategory;

#[test]
fn extraction_transcript_omits_internal_system_reminders() {
    let messages = vec![
        crate::message::Message::user(
            "<system-reminder>\n# Session Context\nHardware: private\n</system-reminder>",
        ),
        crate::message::Message::user("Remember that tests use a temporary database."),
        crate::message::Message::assistant_text("Understood."),
    ];

    let transcript = build_transcript_for_extraction(&messages);

    assert!(!transcript.contains("Session Context"));
    assert!(!transcript.contains("Hardware: private"));
    assert!(transcript.contains("tests use a temporary database"));
    assert!(transcript.contains("Understood"));
}

#[test]
fn infer_candidate_tag_uses_repeated_non_stopword() {
    let tag =
        infer_candidate_tag("scheduler retries failed jobs and scheduler metrics update dashboard");
    assert_eq!(tag.as_deref(), Some("scheduler"));
}

#[test]
fn apply_cluster_assignment_links_members() {
    let mut graph = MemoryGraph::new();
    let mut a = MemoryEntry::new(MemoryCategory::Fact, "A");
    a.embedding = Some(vec![1.0, 0.0]);
    let id_a = graph.add_memory(a);

    let mut b = MemoryEntry::new(MemoryCategory::Fact, "B");
    b.embedding = Some(vec![0.0, 1.0]);
    let id_b = graph.add_memory(b);

    let stats = apply_cluster_assignment(
        &mut graph,
        "project",
        &[id_a.clone(), id_b.clone()],
        Utc::now(),
    );

    assert_eq!(stats.clusters_touched, 1);
    assert_eq!(stats.member_links, 2);
    assert_eq!(graph.clusters.len(), 1);

    let cluster_id = graph
        .clusters
        .keys()
        .next()
        .expect("cluster id")
        .to_string();
    assert!(
        graph
            .get_edges(&id_a)
            .iter()
            .any(|e| e.target == cluster_id && matches!(e.kind, EdgeKind::InCluster))
    );
    assert!(
        graph
            .get_edges(&id_b)
            .iter()
            .any(|e| e.target == cluster_id && matches!(e.kind, EdgeKind::InCluster))
    );
}

#[test]
fn apply_confidence_updates_batches_boost_and_decay() {
    let _guard = crate::storage::lock_test_env();
    let old = std::env::var("JCODE_HOME").ok();
    let dir = std::env::temp_dir().join(format!(
        "jcode-conf-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    crate::env::set_var("JCODE_HOME", &dir);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let manager = crate::memory::MemoryManager::new().with_project_dir("/tmp/jcode-conf-batch");

        let mut keep_entry = MemoryEntry::new(MemoryCategory::Fact, "verified memory")
            .with_embedding(vec![1.0, 0.0]);
        keep_entry.confidence = 0.5; // below cap so a boost is observable
        let keep = manager.remember_project(keep_entry).unwrap();
        // A genuinely DORMANT memory: never accessed and older than the 30-day
        // dormancy window, so the conditional decay in apply_confidence_updates
        // applies to it. Backdate AFTER remember_project because its Touch
        // effect stamps updated_at = now.
        let stale = manager
            .remember_project(
                MemoryEntry::new(MemoryCategory::Fact, "rejected memory")
                    .with_embedding(vec![0.0, 1.0]),
            )
            .unwrap();
        {
            let mut graph = manager.load_project_graph().unwrap();
            let entry = graph.get_memory_mut(&stale).unwrap();
            entry.access_count = 0;
            entry.updated_at = chrono::Utc::now() - chrono::Duration::days(60);
            manager.save_project_graph(&graph).unwrap();
        }

        let conf_before = |id: &str| {
            manager
                .load_project_graph()
                .unwrap()
                .get_memory(id)
                .unwrap()
                .confidence
        };
        let keep_before = conf_before(&keep);
        let stale_before = conf_before(&stale);

        let (boosted, decayed) = apply_confidence_updates(
            &manager,
            std::slice::from_ref(&keep),
            std::slice::from_ref(&stale),
        );
        assert_eq!(boosted, 1, "one verified memory boosted");
        assert_eq!(decayed, 1, "one dormant rejected memory decayed");

        let keep_after = conf_before(&keep);
        let stale_after = conf_before(&stale);
        assert!(keep_after > keep_before, "verified confidence should rise");
        assert!(
            stale_after < stale_before,
            "rejected confidence should fall"
        );
    }));

    match old {
        Some(v) => crate::env::set_var("JCODE_HOME", v),
        None => crate::env::remove_var("JCODE_HOME"),
    }
    let _ = std::fs::remove_dir_all(&dir);
    if let Err(p) = result {
        std::panic::resume_unwind(p);
    }
}

#[test]
fn should_run_rerank_cadence_and_overrides() {
    // First rerank of a session always fires.
    assert!(should_run_rerank(0, None, 3, false));
    assert!(should_run_rerank(5, None, 3, false));

    // Topic change always fires, even mid-cadence.
    assert!(should_run_rerank(4, Some(3), 3, true));

    // Cadence floor: with cadence=3, must wait 3 turns since last rerank.
    assert!(!should_run_rerank(4, Some(3), 3, false)); // 1 turn since -> gated
    assert!(!should_run_rerank(5, Some(3), 3, false)); // 2 turns since -> gated
    assert!(should_run_rerank(6, Some(3), 3, false)); // 3 turns since -> fire
    assert!(should_run_rerank(10, Some(3), 3, false)); // well past -> fire

    // cadence <= 1 disables gating (every turn fires).
    assert!(should_run_rerank(4, Some(3), 1, false));
    assert!(should_run_rerank(4, Some(3), 0, false));
}

#[test]
fn hybrid_retrieval_uses_focused_query_with_empty_fallback() {
    let context = "old session context and tool output";

    assert_eq!(
        retrieval_query(context, "current user question"),
        "current user question"
    );
    assert_eq!(retrieval_query(context, "  \n"), context);
}

fn mem(content: &str) -> MemoryEntry {
    MemoryEntry::new(MemoryCategory::Fact, content)
}

#[test]
fn dynamic_gate_cuts_tail_at_score_gap() {
    // RRF-style descending scores with a sharp gap after the second item.
    let cands = vec![
        (mem("a"), 0.0163_f32),
        (mem("b"), 0.0161),
        (mem("c"), 0.0100), // big drop -> tail cut here
        (mem("d"), 0.0098),
        (mem("e"), 0.0097),
    ];
    let out = dynamic_gate_select(cands, 5);
    assert_eq!(out.len(), 2, "should keep only the two close-scoring items");
    assert_eq!(out[0].0.content, "a");
    assert_eq!(out[1].0.content, "b");
}

#[test]
fn dynamic_gate_keeps_top1_even_when_isolated() {
    // A lone strong candidate followed by far-weaker ones: keep exactly 1.
    let cands = vec![
        (mem("a"), 0.0200_f32),
        (mem("b"), 0.0100),
        (mem("c"), 0.0090),
    ];
    let out = dynamic_gate_select(cands, 5);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0.content, "a");
}

#[test]
fn dynamic_gate_respects_max_k_on_flat_scores() {
    // All scores ~equal: gate would keep all, but max_k caps the count.
    let cands: Vec<_> = (0..8)
        .map(|i| (mem(&format!("m{i}")), 0.0160_f32))
        .collect();
    let out = dynamic_gate_select(cands, 5);
    assert_eq!(out.len(), 5, "capped at max_k even when no gap appears");
}

#[test]
fn dynamic_gate_empty_input_returns_empty() {
    let out = dynamic_gate_select(Vec::new(), 5);
    assert!(out.is_empty());
}

// ============================================================================
// Acceptance tests for the strength-balance feature (E/B/A/C/D)
// ============================================================================

/// E/B acceptance: touch_entries on surfaced (verified) memories bumps
/// access_count and refreshes updated_at even when the judge did NOT run
/// this turn (the no-LLM cadence-carry path that previously skipped all
/// reinforcement).
#[test]
fn surfaced_memories_are_touched_even_without_judge() {
    let _guard = crate::storage::lock_test_env();
    let old = std::env::var("JCODE_HOME").ok();
    let dir = std::env::temp_dir().join(format!(
        "jcode-touch-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    crate::env::set_var("JCODE_HOME", &dir);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let manager = crate::memory::MemoryManager::new().with_project_dir("/tmp/jcode-touch");

        let id = manager
            .remember_project(
                crate::memory::MemoryEntry::new(MemoryCategory::Fact, "surfaced memory fact")
                    .with_embedding(vec![1.0, 0.0]),
            )
            .unwrap();

        // Backdate to simulate a memory untouched for a while.
        {
            let mut graph = manager.load_project_graph().unwrap();
            let entry = graph.get_memory_mut(&id).unwrap();
            entry.access_count = 0;
            entry.updated_at = chrono::Utc::now() - chrono::Duration::days(5);
            manager.save_project_graph(&graph).unwrap();
        }

        // Simulate the maintenance step the agent runs on a cadence-carry
        // turn (no judge verdict): touch the surfaced ids.
        manager.touch_entries(std::slice::from_ref(&id)).unwrap();

        let graph = manager.load_project_graph().unwrap();
        let entry = graph.get_memory(&id).unwrap();
        assert_eq!(
            entry.access_count, 1,
            "surfaced memory must gain access credit on a no-judge turn"
        );
        assert!(
            (chrono::Utc::now() - entry.updated_at).num_seconds() < 60,
            "surfaced memory updated_at must be refreshed to now"
        );
    }));

    match old {
        Some(v) => crate::env::set_var("JCODE_HOME", v),
        None => crate::env::remove_var("JCODE_HOME"),
    }
    let _ = std::fs::remove_dir_all(&dir);
    if let Err(payload) = result {
        std::panic::resume_unwind(payload);
    }
}

/// A acceptance: a judge-verified memory grows strength (not just confidence).
#[test]
fn judge_verified_memory_gains_strength() {
    let _guard = crate::storage::lock_test_env();
    let old = std::env::var("JCODE_HOME").ok();
    let dir = std::env::temp_dir().join(format!(
        "jcode-strength-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    crate::env::set_var("JCODE_HOME", &dir);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let manager = crate::memory::MemoryManager::new().with_project_dir("/tmp/jcode-strength");

        let id = manager
            .remember_project(
                crate::memory::MemoryEntry::new(MemoryCategory::Fact, "verified adoption fact")
                    .with_embedding(vec![1.0, 0.0]),
            )
            .unwrap();

        let strength_before = manager
            .load_project_graph()
            .unwrap()
            .get_memory(&id)
            .unwrap()
            .strength;

        let (boosted, _decayed) =
            apply_confidence_updates(&manager, std::slice::from_ref(&id), &[]);
        assert_eq!(boosted, 1);

        let graph = manager.load_project_graph().unwrap();
        let entry = graph.get_memory(&id).unwrap();
        assert_eq!(
            entry.strength,
            strength_before + 1,
            "judge-verified memory must gain +1 strength per verified turn"
        );
        assert_eq!(entry.reinforcements.len(), 1);
        assert_eq!(
            entry.reinforcements[0].session_id, "judge-verified",
            "reinforcement provenance should name the adoption channel"
        );
    }));

    match old {
        Some(v) => crate::env::set_var("JCODE_HOME", v),
        None => crate::env::remove_var("JCODE_HOME"),
    }
    let _ = std::fs::remove_dir_all(&dir);
    if let Err(payload) = result {
        std::panic::resume_unwind(payload);
    }
}

/// D acceptance: a rejected (judge-missed) memory that is FRESH or has been
/// accessed must NOT decay; only genuinely dormant memories (access_count == 0
/// and older than the dormancy window) decay.
#[test]
fn non_dormant_rejected_memories_are_not_decayed() {
    let _guard = crate::storage::lock_test_env();
    let old = std::env::var("JCODE_HOME").ok();
    let dir = std::env::temp_dir().join(format!(
        "jcode-nodecay-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    crate::env::set_var("JCODE_HOME", &dir);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let manager = crate::memory::MemoryManager::new().with_project_dir("/tmp/jcode-nodecay");

        // Fresh + never accessed (default state after remember_project).
        let fresh = manager
            .remember_project(
                crate::memory::MemoryEntry::new(MemoryCategory::Fact, "fresh near-miss fact")
                    .with_embedding(vec![1.0, 0.0]),
            )
            .unwrap();
        // Old but accessed (recently used).
        let used = manager
            .remember_project(
                crate::memory::MemoryEntry::new(MemoryCategory::Fact, "old but used fact")
                    .with_embedding(vec![0.0, 1.0]),
            )
            .unwrap();
        {
            let mut graph = manager.load_project_graph().unwrap();
            let entry = graph.get_memory_mut(&used).unwrap();
            entry.access_count = 3;
            entry.updated_at = chrono::Utc::now() - chrono::Duration::days(60);
            manager.save_project_graph(&graph).unwrap();
        }

        let conf = |id: &str| {
            manager
                .load_project_graph()
                .unwrap()
                .get_memory(id)
                .unwrap()
                .confidence
        };
        let fresh_before = conf(&fresh);
        let used_before = conf(&used);

        let (_boosted, decayed) = apply_confidence_updates(
            &manager,
            &[],
            &[fresh.clone(), used.clone()],
        );
        assert_eq!(decayed, 0, "neither fresh nor recently-used memory may decay");
        assert_eq!(conf(&fresh), fresh_before, "fresh memory confidence unchanged");
        assert_eq!(conf(&used), used_before, "accessed memory confidence unchanged");
    }));

    match old {
        Some(v) => crate::env::set_var("JCODE_HOME", v),
        None => crate::env::remove_var("JCODE_HOME"),
    }
    let _ = std::fs::remove_dir_all(&dir);
    if let Err(payload) = result {
        std::panic::resume_unwind(payload);
    }
}

/// C acceptance (tier boundary, LLM-free branch): pins the EXACT vs NEAR tier
/// boundary the dedup loop uses. Uses explicit embeddings so the test does not
/// depend on the local ONNX model (absent in sandboxed test homes); cosine over
/// explicit vectors exercises the exact same `score_and_filter` path that
/// `find_similar` drives at extraction time.
#[test]
fn tiered_dedup_exact_dup_clears_threshold_without_llm() {
    let _guard = crate::storage::lock_test_env();
    let old = std::env::var("JCODE_HOME").ok();
    let dir = std::env::temp_dir().join(format!(
        "jcode-tier-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    crate::env::set_var("JCODE_HOME", &dir);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        use crate::memory_agent::{EXACT_DUP_THRESHOLD, NEAR_DUP_THRESHOLD};
        let manager =
            crate::memory::MemoryManager::new().with_project_dir("/tmp/jcode-tiered-dedup");

        let id = manager
            .remember_project(
                crate::memory::MemoryEntry::new(
                    MemoryCategory::Fact,
                    "The scheduler retries failed jobs up to three times before alerting.",
                )
                .with_embedding(vec![1.0, 0.0]),
            )
            .unwrap();

        // Verbatim-level duplicate: cosine 1.0, far above EXACT (0.95), so the
        // extraction loop reinforces directly with NO LLM confirmation call.
        let exact_score = manager
            .find_similar_with_embedding(&[1.0, 0.0], NEAR_DUP_THRESHOLD, 3)
            .unwrap()
            .into_iter()
            .find(|(e, _)| e.id == id)
            .map(|(_, s)| s)
            .unwrap_or_else(|| panic!("verbatim duplicate must be retrieved at {NEAR_DUP_THRESHOLD}"));
        assert!(
            exact_score >= EXACT_DUP_THRESHOLD,
            "verbatim duplicate scored {exact_score}, must clear EXACT tier {EXACT_DUP_THRESHOLD} \
             so no LLM confirmation is needed"
        );

        // Rephrase-of-same-fact zone: cosine ~0.9 lands between NEAR (0.80)
        // and EXACT (0.95) -- retrieved as a near-dup candidate and handed to
        // the LLM confirm step rather than blindly reinforced.
        let rephrase_score = manager
            .find_similar_with_embedding(&[0.9, 0.436], NEAR_DUP_THRESHOLD, 3)
            .unwrap()
            .into_iter()
            .find(|(e, _)| e.id == id)
            .map(|(_, s)| s);
        if let Some(score) = rephrase_score {
            assert!(
                score >= NEAR_DUP_THRESHOLD && score < EXACT_DUP_THRESHOLD,
                "rephrase scored {score}, must land in the LLM-confirm tier \
                 [{NEAR_DUP_THRESHOLD}, {EXACT_DUP_THRESHOLD})"
            );
        }

        // Distinct fact: cosine ~0 < NEAR (0.80) so it is stored as a NEW
        // memory, never folded into the existing row.
        let distinct_score = manager
            .find_similar_with_embedding(&[0.0, 1.0], NEAR_DUP_THRESHOLD, 3)
            .unwrap()
            .into_iter()
            .find(|(e, _)| e.id == id)
            .map(|(_, s)| s);
        assert!(
            distinct_score.is_none(),
            "orthogonal memory scored {distinct_score:?}, must fall below NEAR tier"
        );
    }));

    match old {
        Some(v) => crate::env::set_var("JCODE_HOME", v),
        None => crate::env::remove_var("JCODE_HOME"),
    }
    let _ = std::fs::remove_dir_all(&dir);
    if let Err(payload) = result {
        std::panic::resume_unwind(payload);
    }
}
