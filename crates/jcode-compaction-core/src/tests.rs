use super::*;

#[test]
fn protected_checkpoint_lines_survive_incremental_merge() {
    let existing = r#"## Critical Context
- `[CONSTRAINT]` never force-push to master
- `[RETRACTED]` the cache is not the source of truth
- `[ERROR_FIX]` rebuild the bridge after changing the protocol
"#;
    let summary = r#"## Current Work
refactoring

## Critical Context
- `[ERROR_FIX]` rebuild the bridge after changing the protocol
"#;

    let (repaired, inserted) =
        preserve_protected_checkpoint_lines(Some(existing), summary.to_string());
    assert_eq!(inserted, 2);
    assert!(repaired.contains("- `[CONSTRAINT]` never force-push to master"));
    assert!(repaired.contains("- `[RETRACTED]` the cache is not the source of truth"));
    assert!(repaired.contains("- `[ERROR_FIX]` rebuild the bridge after changing the protocol"));
    assert_eq!(
        repaired.matches("[ERROR_FIX]").count(),
        1,
        "existing non-protected lines must not be duplicated: {repaired}"
    );
}

#[test]
fn protected_checkpoint_lines_are_inserted_before_graph_state() {
    let existing = "## Critical Context\n- `[CONSTRAINT]` keep the public API stable\n";
    let summary = "## Current Work\nediting\n\n## Graph Memory State\n- file:src/lib.rs\n";
    let (repaired, inserted) =
        preserve_protected_checkpoint_lines(Some(existing), summary.to_string());
    assert_eq!(inserted, 1);
    let constraint = repaired.find("[CONSTRAINT]").expect("constraint");
    let graph_state = repaired.find("## Graph Memory State").expect("graph state");
    assert!(
        constraint < graph_state,
        "contract line must stay inside the checkpoint body: {repaired}"
    );
}

#[test]
fn protected_checkpoint_lines_noop_without_existing_or_missing_lines() {
    let summary = "## Critical Context\n- `[CONSTRAINT]` keep it\n";
    let (unchanged, inserted) =
        preserve_protected_checkpoint_lines(None, summary.to_string());
    assert_eq!(inserted, 0);
    assert_eq!(unchanged, summary);

    let (unchanged, inserted) =
        preserve_protected_checkpoint_lines(Some(summary), summary.to_string());
    assert_eq!(inserted, 0);
    assert_eq!(unchanged, summary);
}

#[test]
fn emergency_truncation_noops_keep_shared_bodies_at_exact_limits() {
    let canonical = vec![Message::user("unchanged"), tool_image_msg(4096)];
    let mut request = canonical.clone();
    assert_eq!(emergency_truncate_tool_results(&mut request, 21), 0);
    assert_eq!(emergency_truncate_large_payloads(&mut request, 21, 4096), 0);
    for (original, current) in canonical.iter().zip(&request) {
        assert!(original.content.shares_storage_with(&current.content));
    }
    assert_eq!(
        serde_json::to_vec(&request).unwrap(),
        serde_json::to_vec(&canonical).unwrap()
    );
}

#[test]
fn emergency_tool_truncation_detaches_only_changed_messages() {
    let mut oversized = tool_image_msg(8);
    if let ContentBlock::ToolResult { content, .. } = &mut oversized.content[0] {
        *content = "é".repeat(100);
    }
    let canonical = vec![Message::user("unchanged"), tool_image_msg(8), oversized];
    let original = serde_json::to_vec(&canonical).unwrap();
    let mut request = canonical.clone();
    assert_eq!(emergency_truncate_tool_results(&mut request, 30), 1);
    for index in 0..2 {
        assert!(
            canonical[index]
                .content
                .shares_storage_with(&request[index].content)
        );
    }
    assert!(
        !canonical[2]
            .content
            .shares_storage_with(&request[2].content)
    );
    if let ContentBlock::ToolResult { content, .. } = &request[2].content[0] {
        assert_eq!(
            content,
            &emergency_truncated_tool_result(&"é".repeat(100), 30)
        );
    } else {
        panic!("expected tool result");
    }
    assert!(matches!(request[2].content[1], ContentBlock::Image { .. }));
    assert_eq!(serde_json::to_vec(&canonical).unwrap(), original);
}

#[test]
fn emergency_large_payload_truncation_detaches_only_changed_messages() {
    let mut oversized = tool_image_msg(4097);
    if let ContentBlock::ToolResult { content, .. } = &mut oversized.content[0] {
        *content = "é".repeat(100);
    }
    let canonical = vec![Message::user("unchanged"), tool_image_msg(4096), oversized];
    let original = serde_json::to_vec(&canonical).unwrap();
    let mut request = canonical.clone();
    assert_eq!(emergency_truncate_large_payloads(&mut request, 30, 4096), 2);
    for index in 0..2 {
        assert!(
            canonical[index]
                .content
                .shares_storage_with(&request[index].content)
        );
    }
    assert!(
        !canonical[2]
            .content
            .shares_storage_with(&request[2].content)
    );
    if let ContentBlock::ToolResult { content, .. } = &request[2].content[0] {
        assert_eq!(
            content,
            &emergency_truncated_tool_result(&"é".repeat(100), 30)
        );
    } else {
        panic!("expected tool result");
    }
    if let ContentBlock::Text {
        text,
        cache_control,
    } = &request[2].content[1]
    {
        assert_eq!(
            text,
            "[Image omitted during emergency context recovery: media_type=image/png, original_base64_chars=4097. Rely on adjacent browser/tool text, screenshots saved to disk, or re-open/re-screenshot if visual details are needed.]"
        );
        assert!(cache_control.is_none());
    } else {
        panic!("expected image omission marker");
    }
    assert_eq!(serde_json::to_vec(&canonical).unwrap(), original);
}

#[test]
fn noop_image_recovery_preserves_shared_content_allocations() {
    use jcode_message_types::SharedContent;

    let text: SharedContent = vec![ContentBlock::Text {
        text: "large tool output".repeat(32 * 1024),
        cache_control: None,
    }]
    .into();
    let mut text_request = text.clone();
    assert_eq!(strip_all_images_in_contents(&mut [&mut text_request]), 0);
    assert!(text.shares_storage_with(&text_request));

    let image: SharedContent = vec![
        ContentBlock::ToolResult {
            tool_use_id: "call".into(),
            content: "screenshot".into(),
            is_error: None,
        },
        ContentBlock::Image {
            media_type: "image/png".into(),
            data: "a".repeat(4096),
        },
    ]
    .into();
    let mut request = image.clone();
    assert_eq!(strip_large_images_in_contents(&mut [&mut request], 4096), 0);
    assert!(image.shares_storage_with(&request));
    assert_eq!(decay_old_tool_images_in_contents(&mut [&mut request]), 0);
    assert!(image.shares_storage_with(&request));

    assert_eq!(strip_all_images_in_contents(&mut [&mut request]), 1);
    assert!(!image.shares_storage_with(&request));
    assert!(matches!(image[1], ContentBlock::Image { .. }));
    assert!(matches!(request[1], ContentBlock::Text { .. }));
}

#[test]
fn reusable_cache_boundary_excludes_only_split_uncached_tail() {
    for provider in ["Claude", "anthropic", "unknown"] {
        assert_eq!(
            reusable_cache_tokens_from_usage(provider, 2_482, Some(615_629), Some(1_375)),
            617_004
        );
    }
    for provider in ["openai", "openai-api", "codex"] {
        assert_eq!(
            reusable_cache_tokens_from_usage(provider, 619_486, Some(615_629), Some(1_375)),
            619_486
        );
    }
    assert_eq!(
        reusable_cache_tokens_from_usage("Claude", 2_482, Some(0), Some(0)),
        0
    );
    assert_eq!(
        reusable_cache_tokens_from_usage("Claude", 0, Some(615_629), Some(1_375)),
        617_004
    );
}

#[test]
fn compaction_text_keeps_native_search_query_and_urls() {
    let messages = vec![Message {
        role: Role::Assistant,
        content: vec![
            ContentBlock::ProviderNative {
                provider: "anthropic".to_string(),
                item: serde_json::json!({"type": "server_tool_use", "id": "srvtoolu_1",
                    "name": "web_search", "input": {"query": "rust news"}}),
            },
            ContentBlock::ProviderNative {
                provider: "anthropic".to_string(),
                item: serde_json::json!({"type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1", "content": [{"type": "web_search_result",
                    "url": "https://blog.rust-lang.org/", "title": "Rust Blog",
                    "encrypted_content": "ENC"}]}),
            },
        ]
        .into(),
        timestamp: None,
        tool_duration_ms: None,
    }];
    let text = build_compaction_conversation_text(&messages, None);
    assert!(text.contains("rust news"), "{text}");
    assert!(text.contains("https://blog.rust-lang.org/"), "{text}");
    assert!(!text.contains("ENC"), "{text}");
}

#[test]
fn effective_context_split_accounting_adds_cache_counters() {
    // Anthropic-style: input is the uncached remainder.
    assert_eq!(
        effective_context_tokens_from_usage("anthropic", 10_000, Some(300_000), Some(5_000)),
        315_000
    );
    assert_eq!(
        effective_context_tokens_from_usage("Claude", 10_000, Some(300_000), None),
        310_000
    );
}

#[test]
fn effective_context_subset_accounting_does_not_double_count() {
    // OpenAI-style: prompt_tokens already includes cached tokens.
    assert_eq!(
        effective_context_tokens_from_usage("openai", 400_000, Some(390_000), None),
        400_000
    );
    assert_eq!(
        effective_context_tokens_from_usage("openai-api", 10_000, Some(6_000), Some(2_000)),
        10_000
    );
    // No cache info at all: pass through.
    assert_eq!(
        effective_context_tokens_from_usage("opencode-go", 396_000, None, None),
        396_000
    );
}

#[test]
fn effective_context_infers_split_accounting_from_counter_shape() {
    // cache_read > input implies input can't already contain it.
    assert_eq!(
        effective_context_tokens_from_usage("unknown", 10_000, Some(500_000), None),
        510_000
    );
    // Any cache_creation implies split accounting.
    assert_eq!(
        effective_context_tokens_from_usage("unknown", 10_000, Some(2_000), Some(1_000)),
        13_000
    );
}

#[test]
fn effective_context_zero_uncached_input_preserves_cached_prompt() {
    assert_eq!(
        effective_context_tokens_from_usage("anthropic", 0, Some(300_000), Some(5_000)),
        305_000
    );
}

#[test]
fn builds_compaction_prompt_with_summary_and_truncated_tool_result() {
    let summary = Summary {
        text: "prior work".to_string(),
        openai_encrypted_content: None,
        covers_up_to_turn: 1,
        original_turn_count: 1,
    };
    let message = Message::user("hello");
    let prompt = build_compaction_prompt(&[message], Some(&summary), 10_000);
    assert!(prompt.contains("## Previous Summary"));
    assert!(prompt.contains("prior work"));
    assert!(prompt.contains("**User:**"));
    assert!(prompt.contains(SUMMARY_PROMPT));
}

#[test]
fn compaction_prompt_requires_nine_section_checkpoint_contracts() {
    let prompt = build_compaction_prompt(&[Message::user("probe")], None, 10_000);
    for section in [
        "## Primary Request and Intent",
        "## Key Technical Concepts",
        "## Files and Code",
        "## Errors and Fixes",
        "## Pending Jobs",
        "## Current Work",
        "## Next Step",
        "## Critical Context",
    ] {
        assert!(prompt.contains(section), "missing section: {section}");
    }
    for tag in [
        "[CONSTRAINT]",
        "[RETRACTED]",
        "[ERROR_FIX]",
        "[DECISION]",
        "[TODO]",
        "authority=checkpoint",
    ] {
        assert!(prompt.contains(tag), "missing contract marker: {tag}");
    }
    assert!(
        prompt.contains("Do not add a `## Graph Memory State` section"),
        "prompt must leave deterministic graph anchors to the runtime"
    );
}

#[test]
fn truncates_on_utf8_boundary() {
    assert_eq!(truncate_str_boundary("éabc", 1), "");
    assert_eq!(truncate_str_boundary("éabc", 2), "é");
}

#[test]
fn mean_embedding_is_normalized() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let mean = mean_embedding(&[&a, &b], 2);
    let norm = (mean[0] * mean[0] + mean[1] * mean[1]).sqrt();
    assert!((norm - 1.0).abs() < 0.0001);
}

#[test]
fn safe_cutoff_keeps_tool_use_with_tool_result() {
    let tool_use = Message {
        role: Role::Assistant,
        content: vec![ContentBlock::ToolUse {
            id: "call_1".to_string(),
            name: "read".to_string(),
            input: serde_json::json!({"file":"src/lib.rs"}),
            thought_signature: None,
        }]
        .into(),
        timestamp: None,
        tool_duration_ms: None,
    };
    let tool_result = Message {
        role: Role::User,
        content: vec![ContentBlock::ToolResult {
            tool_use_id: "call_1".to_string(),
            content: "ok".to_string(),
            is_error: None,
        }]
        .into(),
        timestamp: None,
        tool_duration_ms: None,
    };
    let messages = vec![
        Message::user("old"),
        tool_use,
        tool_result,
        Message::user("new"),
    ];

    assert_eq!(safe_compaction_cutoff(&messages, 2), 1);
}

#[test]
fn estimates_tokens_with_large_budget_overhead() {
    let summary = Summary {
        text: "abcd".repeat(100),
        openai_encrypted_content: None,
        covers_up_to_turn: 1,
        original_turn_count: 1,
    };

    assert_eq!(estimate_compaction_tokens(Some(&summary), 0, 1000), 100);
    assert_eq!(
        estimate_compaction_tokens(Some(&summary), 0, DEFAULT_TOKEN_BUDGET),
        100 + SYSTEM_OVERHEAD_TOKENS
    );
}

#[test]
fn image_token_cost_is_bounded_not_base64_length() {
    // Regression: a large base64 image payload must not be counted as ~len/4
    // tokens. Doing so inflated the estimate ~100x and caused repeated
    // back-to-back compactions.
    let huge_base64 = "A".repeat(1_400_000);
    let mut image_msg = Message::user("");
    image_msg.content = vec![ContentBlock::Image {
        media_type: "image/png".to_string(),
        data: huge_base64.clone(),
    }]
    .into();

    let chars = message_char_count(&image_msg);
    // Flat per-image cost in char-equivalents, far below the raw payload.
    assert_eq!(chars, IMAGE_TOKEN_COST * CHARS_PER_TOKEN);
    assert!(
        chars < huge_base64.len() / 10,
        "image should not be charged anywhere near its base64 length"
    );

    // The token estimate for four such images stays small.
    let tokens = estimate_compaction_tokens_from_chars(chars * 4, DEFAULT_TOKEN_BUDGET);
    assert!(
        tokens < SYSTEM_OVERHEAD_TOKENS + 4 * IMAGE_TOKEN_COST + 10,
        "four images should cost ~{} tokens, got {}",
        SYSTEM_OVERHEAD_TOKENS + 4 * IMAGE_TOKEN_COST,
        tokens
    );
}

#[test]
fn summary_clone_shares_encrypted_body_when_text_changes() {
    let body: std::sync::Arc<str> = "encrypted-body-".repeat(128 * 1024).into();
    let summary = Summary {
        text: "original summary".into(),
        openai_encrypted_content: Some(body.clone()),
        covers_up_to_turn: 3,
        original_turn_count: 7,
    };
    let mut cloned = summary.clone();
    cloned.text.push_str(" changed");
    assert!(std::sync::Arc::ptr_eq(
        &body,
        cloned.openai_encrypted_content.as_ref().unwrap()
    ));
    assert_eq!(summary.text, "original summary");
    cloned.openai_encrypted_content = None;
    assert!(std::sync::Arc::ptr_eq(
        &body,
        summary.openai_encrypted_content.as_ref().unwrap()
    ));
}

#[test]
fn openai_native_compaction_cost_is_bounded_not_ciphertext_length() {
    // Regression: a 1.1M-char encrypted payload was estimated at ~285k
    // tokens, so compaction reported 294k -> 303k while the provider saw
    // 219k -> 34k.
    let summary = Summary {
        text: String::new(),
        openai_encrypted_content: Some("x".repeat(1_141_900).into()),
        covers_up_to_turn: 229,
        original_turn_count: 229,
    };
    assert_eq!(
        estimate_compaction_tokens(Some(&summary), 0, DEFAULT_TOKEN_BUDGET),
        OPENAI_COMPACTION_TOKEN_COST + SYSTEM_OVERHEAD_TOKENS
    );

    let mut msg = Message::user("");
    msg.content = vec![ContentBlock::OpenAICompaction {
        encrypted_content: "x".repeat(1_141_900),
    }]
    .into();
    assert_eq!(
        message_char_count(&msg),
        OPENAI_COMPACTION_TOKEN_COST * CHARS_PER_TOKEN
    );
}

#[test]
fn builds_semantic_text_from_relevant_content() {
    let message = Message {
        role: Role::User,
        content: vec![
            ContentBlock::Text {
                text: "hello world".to_string(),
                cache_control: None,
            },
            ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: "tool output".to_string(),
                is_error: None,
            },
        ]
        .into(),
        timestamp: None,
        tool_duration_ms: None,
    };

    assert_eq!(semantic_message_text(&message), "hello world");
    assert_eq!(semantic_goal_text(&[message]), "hello world tool output");
    assert_eq!(semantic_cache_key("stable"), semantic_cache_key("stable"));
}

#[test]
fn builds_emergency_summary_with_tools_and_files() {
    let messages = vec![
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "read".to_string(),
                input: serde_json::json!({"file":"src/lib.rs"}),
                thought_signature: None,
            }]
            .into(),
            timestamp: None,
            tool_duration_ms: None,
        },
        Message::user("Edited src/compaction.rs and Cargo.toml, ignored https://example.com"),
    ];

    let summary = build_emergency_summary_text(Some("previous"), 2, 201_000, 200_000, &messages);
    assert!(summary.contains("previous"));
    assert!(summary.contains("2 messages were dropped"));
    assert!(summary.contains("Tools used: read"));
    assert!(summary.contains("Files referenced: Cargo.toml, src/compaction.rs"));
    assert!(!summary.contains("https://example.com"));
}

#[test]
fn emergency_truncation_is_utf8_safe() {
    let original = format!("{}middle{}", "é".repeat(20), "尾".repeat(20));
    let truncated = emergency_truncated_tool_result(&original, 25);
    assert!(truncated.contains("chars truncated for context recovery"));
    assert!(truncated.is_char_boundary(truncated.len()));
}

#[test]
fn emergency_truncation_replaces_large_images_with_text_marker() {
    let mut messages = vec![Message {
        role: Role::User,
        content: vec![ContentBlock::Image {
            media_type: "image/png".to_string(),
            data: "a".repeat(2048),
        }]
        .into(),
        timestamp: None,
        tool_duration_ms: None,
    }];

    let truncated = emergency_truncate_large_payloads(&mut messages, 4000, 1024);
    assert_eq!(truncated, 1);
    match &messages[0].content[0] {
        ContentBlock::Text { text, .. } => {
            assert!(text.contains("Image omitted during emergency context recovery"));
            assert!(text.contains("original_base64_chars=2048"));
        }
        other => panic!("expected image to be replaced with text marker, got {other:?}"),
    }
}

#[test]
fn detects_request_payload_too_large_errors() {
    // Image rejections are a different class.
    assert!(is_image_rejection_error(
        "Anthropic API error (400 Bad Request): {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"messages.720.content.0.tool_result.content.1.image.source.base64.media_type: Input should be 'image/jpeg', 'image/png', 'image/gif' or 'image/webp'\"}}"
    ));
    assert!(is_image_rejection_error(
        "400 invalid_request_error: messages.3.content.1.image.source.base64: Could not process image"
    ));
    assert!(!is_image_rejection_error(
        "Anthropic API error (413 Payload Too Large): image too big request_too_large"
    ));
    assert!(!is_image_rejection_error(
        "400 invalid_request_error: messages.5: `tool_use` ids were found without `tool_result` blocks"
    ));
    assert!(!is_image_rejection_error("rate limit exceeded"));

    assert!(is_request_payload_too_large_error(
        "Anthropic API error (413 Payload Too Large): {\"error\":{\"type\":\"request_too_large\",\"message\":\"Request exceeds the maximum size\"}}"
    ));
    assert!(is_request_payload_too_large_error(
        "413 Request Entity Too Large"
    ));
    assert!(is_request_payload_too_large_error("request too large"));
    // Not a payload error — should not match.
    assert!(!is_request_payload_too_large_error(
        "rate limit exceeded, retry after 20s"
    ));
    // Embedded digits must not trip the standalone 413 check.
    assert!(!is_request_payload_too_large_error(
        "model version 4130 is unavailable"
    ));
}

fn image_msg(data_len: usize) -> Message {
    Message {
        role: Role::User,
        content: vec![ContentBlock::Image {
            media_type: "image/png".to_string(),
            data: "a".repeat(data_len),
        }]
        .into(),
        timestamp: None,
        tool_duration_ms: None,
    }
}

#[test]
fn strip_large_images_drops_oldest_until_under_budget() {
    // Four 1000-char images = 4000 total; budget 2500 should drop the two
    // oldest (leaving 2000 <= 2500), keeping the two most recent.
    let mut messages = vec![
        image_msg(1000),
        image_msg(1000),
        image_msg(1000),
        image_msg(1000),
    ];

    let stripped = emergency_strip_large_images(&mut messages, 2500);
    assert_eq!(stripped, 2);
    // Oldest two replaced with text markers.
    assert!(matches!(messages[0].content[0], ContentBlock::Text { .. }));
    assert!(matches!(messages[1].content[0], ContentBlock::Text { .. }));
    // Most recent two preserved as images.
    assert!(matches!(messages[2].content[0], ContentBlock::Image { .. }));
    assert!(matches!(messages[3].content[0], ContentBlock::Image { .. }));
    if let ContentBlock::Text { text, .. } = &messages[0].content[0] {
        assert!(text.contains("Image omitted during request-size recovery"));
        assert!(text.contains("original_base64_chars=1000"));
    }
}

fn tool_image_msg(data_len: usize) -> Message {
    Message {
        role: Role::User,
        content: vec![
            ContentBlock::ToolResult {
                tool_use_id: "call".to_string(),
                content: "Captured main display".to_string(),
                is_error: None,
            },
            ContentBlock::Image {
                media_type: "image/png".to_string(),
                data: "a".repeat(data_len),
            },
        ]
        .into(),
        timestamp: None,
        tool_duration_ms: None,
    }
}

fn decay(messages: &mut [Message]) -> usize {
    let mut contents: Vec<_> = messages.iter_mut().map(|m| &mut m.content).collect();
    decay_old_tool_images_in_contents(&mut contents)
}

fn is_image(message: &Message) -> bool {
    message
        .content
        .iter()
        .any(|block| matches!(block, ContentBlock::Image { .. }))
}

#[test]
fn tool_image_decay_waits_for_trigger_then_keeps_newest() {
    let mut messages: Vec<Message> = (0..TOOL_IMAGE_DECAY_TRIGGER_COUNT)
        .map(|_| tool_image_msg(10))
        .collect();
    assert_eq!(
        decay(&mut messages),
        0,
        "at the trigger count nothing decays"
    );

    messages.push(tool_image_msg(10));
    let total = messages.len();
    assert_eq!(decay(&mut messages), total - TOOL_IMAGE_DECAY_KEEP_COUNT);
    for (index, message) in messages.iter().enumerate() {
        assert_eq!(
            is_image(message),
            index >= total - TOOL_IMAGE_DECAY_KEEP_COUNT
        );
        // The tool_result itself is preserved for tool_use pairing.
        assert!(matches!(
            message.content[0],
            ContentBlock::ToolResult { .. }
        ));
    }
    let ContentBlock::Text { text, .. } = &messages[0].content[1] else {
        panic!("decayed image must become a text marker");
    };
    assert!(text.contains("Older tool screenshot omitted"));
    assert!(text.contains("original_base64_chars=10"));
    // Stable after decay: nothing more to do until the trigger is hit again.
    assert_eq!(decay(&mut messages), 0);
}

#[test]
fn tool_image_decay_bounds_bytes_but_keeps_newest_huge_image() {
    let huge = TOOL_IMAGE_DECAY_KEEP_CHARS + 1;
    let mut messages = vec![
        tool_image_msg(huge),
        tool_image_msg(huge),
        tool_image_msg(huge),
    ];
    assert_eq!(decay(&mut messages), 2);
    assert!(!is_image(&messages[0]));
    assert!(!is_image(&messages[1]));
    assert!(
        is_image(&messages[2]),
        "newest screenshot must still be sent"
    );
}

#[test]
fn tool_image_decay_ignores_user_pasted_images() {
    let mut messages = vec![image_msg(10)];
    messages.extend((0..TOOL_IMAGE_DECAY_TRIGGER_COUNT + 5).map(|_| tool_image_msg(10)));
    decay(&mut messages);
    assert!(
        is_image(&messages[0]),
        "user images are not tool screenshots"
    );
}

#[test]
fn strip_large_images_noop_when_under_budget() {
    let mut messages = vec![image_msg(500), image_msg(500)];
    let stripped = emergency_strip_large_images(&mut messages, 4000);
    assert_eq!(stripped, 0);
    assert!(matches!(messages[0].content[0], ContentBlock::Image { .. }));
    assert!(matches!(messages[1].content[0], ContentBlock::Image { .. }));
}

#[test]
fn strip_large_images_strips_all_when_single_image_exceeds_budget() {
    // Even a lone over-budget image is stripped (better than re-sending an
    // oversized request that the provider will reject again).
    let mut messages = vec![image_msg(8000)];
    let stripped = emergency_strip_large_images(&mut messages, 2000);
    assert_eq!(stripped, 1);
    assert!(matches!(messages[0].content[0], ContentBlock::Text { .. }));
}

#[test]
fn build_emergency_summary_strips_prior_emergency_markers() {
    // Simulate a summary that already contains one emergency marker from a
    // previous hard compact, plus a non-emergency prefix (e.g. an earlier
    // proactive summary) and trailing emergency noise (Tools used,
    // Files referenced, a second marker).
    let prior = "# Earlier proactive summary\n\n\
                 **[Emergency compaction]**: 5 messages were dropped to recover from context overflow.\n\
                 The conversation had ~150k tokens which exceeded the 200k limit.\n\n\
                 Tools used: read, bash\n\n\
                 Files referenced: src/lib.rs\n\n\
                 **[Emergency compaction]**: 3 messages were dropped to recover from context overflow.\n\
                 The conversation had ~180k tokens which exceeded the 200k limit.";

    let fresh = build_emergency_summary_text(Some(prior), 7, 220_000, 200_000, &[]);

    // Non-emergency prefix preserved.
    assert!(
        fresh.contains("# Earlier proactive summary"),
        "earlier non-emergency summary should be preserved, got: {fresh}"
    );
    // Exactly one emergency marker (the freshly built one) — no accumulation.
    assert_eq!(
        fresh.matches("[Emergency compaction]").count(),
        1,
        "prior emergency markers should be stripped, got: {fresh}"
    );
    assert!(
        fresh.contains("7 messages were dropped"),
        "fresh marker should describe this compaction, got: {fresh}"
    );
    assert!(
        !fresh.contains("5 messages were dropped") && !fresh.contains("3 messages were dropped"),
        "stale emergency details should be gone, got: {fresh}"
    );
}

#[test]
fn build_emergency_summary_no_existing_keeps_all_content() {
    // No existing summary — new emergency block should be built normally.
    let fresh = build_emergency_summary_text(None, 2, 250_000, 200_000, &[]);
    assert_eq!(fresh.matches("[Emergency compaction]").count(), 1);
    assert!(fresh.contains("2 messages were dropped"));
}

#[test]
fn build_emergency_summary_existing_without_marker_passes_through() {
    // Existing summary without any emergency marker (e.g. a normal proactive
    // summary) should be preserved verbatim and the new emergency block
    // appended on top.
    let prior = "# Proactive summary\nUser asked about X. Agent did Y.";
    let fresh = build_emergency_summary_text(Some(prior), 1, 210_000, 200_000, &[]);
    assert!(fresh.contains("# Proactive summary"));
    assert!(fresh.contains("User asked about X. Agent did Y."));
    assert!(fresh.contains("1 messages were dropped"));
}
