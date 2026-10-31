use super::*;
use std::path::Path;

#[test]
fn acp_tool_kind_maps_core_tools() {
    assert_eq!(tool_kind("read"), "read");
    assert_eq!(tool_kind("apply_patch"), "edit");
    assert_eq!(tool_kind("bash"), "execute");
    assert_eq!(tool_kind("agentgrep"), "search");
    assert_eq!(tool_kind("webfetch"), "fetch");
    assert_eq!(tool_kind("swarm"), "other");
}

#[test]
fn json_rpc_parse_errors_use_standard_codes() {
    let (code, _) = JsonRpcMessage::parse("not json").unwrap_err();
    assert_eq!(code, JSONRPC_PARSE_ERROR);

    let (code, message) = JsonRpcMessage::parse(r#"{"method":"initialize"}"#).unwrap_err();
    assert_eq!(code, JSONRPC_INVALID_REQUEST);
    assert!(message.contains("jsonrpc"));
}

#[test]
fn prompt_from_params_accepts_text_images_and_resources() {
    let params = json!({
        "sessionId": "s1",
        "prompt": [
            {"type": "text", "text": "hello"},
            {"type": "image", "mimeType": "image/png", "data": "abc"},
            {"type": "resource", "resource": {"uri": "file:///tmp/a.rs", "text": "fn main(){}"}},
            {"type": "resource_link", "uri": "file:///tmp/b.rs", "name": "b.rs"}
        ]
    });
    let (text, images) = prompt_from_params(&params).unwrap();
    assert!(text.contains("hello"));
    assert!(text.contains("Embedded resource: file:///tmp/a.rs"));
    assert!(text.contains("Resource link: b.rs"));
    assert_eq!(images, vec![("image/png".to_string(), "abc".to_string())]);
}

#[test]
fn prompt_usage_deduplicates_live_snapshots_and_resets_per_call() {
    let mut usage = TurnUsage::default();
    usage.observe_snapshot(10, 2, Some(4), None);
    usage.observe_snapshot(10, 3, Some(4), Some(5));
    usage.observe_snapshot(10, 3, Some(4), Some(5));
    // Stale or partial snapshots must not lower the per-call watermark.
    usage.observe_snapshot(8, 1, Some(2), None);
    usage.observe_snapshot(10, 3, Some(4), Some(5));
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 3);
    assert_eq!(usage.cached_read_tokens, Some(4));
    assert_eq!(usage.cached_write_tokens, Some(5));
    usage.begin_call();
    usage.observe_snapshot(20, 3, Some(6), Some(7));
    assert_eq!(usage.input_tokens, 30);
    assert_eq!(usage.output_tokens, 6);
    assert_eq!(usage.cached_read_tokens, Some(10));
    assert_eq!(usage.cached_write_tokens, Some(12));
}

#[test]
fn prompt_response_reports_usage_accumulated_across_the_turn() {
    let mut usage = TurnUsage::default();
    usage.add(10, 2, Some(4), Some(5));
    usage.add(20, 3, Some(6), Some(7));

    assert_eq!(
        prompt_response("end_turn", &usage),
        json!({
            "stopReason": "end_turn",
            "usage": {
                "totalTokens": 57,
                "inputTokens": 30,
                "outputTokens": 5,
                "cachedReadTokens": 10,
                "cachedWriteTokens": 12,
            }
        })
    );
}

#[test]
fn prompt_response_omits_unreported_usage_and_cache_fields() {
    assert_eq!(
        prompt_response("end_turn", &TurnUsage::default()),
        json!({ "stopReason": "end_turn" })
    );

    let mut usage = TurnUsage::default();
    usage.add(10, 2, None, None);
    assert_eq!(
        prompt_response("cancelled", &usage),
        json!({
            "stopReason": "cancelled",
            "usage": {
                "totalTokens": 12,
                "inputTokens": 10,
                "outputTokens": 2,
            }
        })
    );
}

#[test]
fn initialize_standard_omits_jcode_meta() {
    let result = initialize_result(&json!({"protocolVersion": 1}), AcpProfile::Standard);
    assert_eq!(result["protocolVersion"], 1);
    assert!(result["agentCapabilities"].get("_meta").is_none());
    assert_eq!(result["agentCapabilities"]["loadSession"], true);
}

#[test]
fn initialize_full_advertises_jcode_extension_meta() {
    let result = initialize_result(&json!({"protocolVersion": 1}), AcpProfile::Full);
    assert_eq!(
        result["agentCapabilities"]["_meta"]["jcode"]["profile"],
        "full"
    );
}

#[test]
fn event_mapper_maps_tool_lifecycle() {
    let mut mapper = EventMapper::new("session1".to_string(), AcpProfile::Standard);
    let start = mapper.map_event(ServerEvent::ToolStart {
        id: "tool1".to_string(),
        name: "bash".to_string(),
    });
    assert_eq!(start[0]["sessionUpdate"], "tool_call");
    assert_eq!(start[0]["kind"], "execute");

    let input = mapper.map_event(ServerEvent::ToolInput {
        id: None,
        delta: "{\"command\":\"true\"}".to_string(),
    });
    assert_eq!(input[0]["rawInput"]["command"], "true");

    let done = mapper.map_event(ServerEvent::ToolDone {
        id: "tool1".to_string(),
        name: "bash".to_string(),
        output: "ok".to_string(),
        error: None,
    });
    assert_eq!(done[0]["status"], "completed");
    assert_eq!(done[0]["content"][0]["content"]["text"], "ok");
}

#[test]
fn non_empty_mcp_servers_are_tolerated_until_session_scoped_mcp_is_supported() {
    let params = json!({"mcpServers": [{"name": "fs"}]});
    assert!(validate_acp_mcp_servers(&params).is_ok());

    let params = json!({"mcpServers": []});
    assert!(validate_acp_mcp_servers(&params).is_ok());
}

#[test]
fn advertised_commands_cover_all_acp_daemon_model_controls() {
    let commands = acp_available_commands();
    let names: Vec<&str> = commands
        .iter()
        .map(|command| command["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["model", "models", "effort"]);
    assert_eq!(commands[0]["input"]["hint"], "model id (optional)");
    assert!(commands[1].get("input").is_none());
    assert!(
        commands[2]["input"]["hint"]
            .as_str()
            .unwrap()
            .contains("high")
    );
}

#[test]
fn advertised_commands_parse_to_real_dispatch_variants() {
    assert_eq!(
        parse_acp_slash_command("/model claude-sonnet-4-5")
            .unwrap()
            .unwrap(),
        AcpSlashCommand::Model(Some("claude-sonnet-4-5".to_string()))
    );
    assert_eq!(
        parse_acp_slash_command("/model ").unwrap().unwrap(),
        AcpSlashCommand::Model(None)
    );
    assert_eq!(
        parse_acp_slash_command("/models").unwrap().unwrap(),
        AcpSlashCommand::Models
    );
    assert_eq!(
        parse_acp_slash_command("/effort xhigh").unwrap().unwrap(),
        AcpSlashCommand::Effort(Some("xhigh".to_string()))
    );
    assert!(parse_acp_slash_command("/models now").unwrap().is_err());
    assert!(parse_acp_slash_command("/not-advertised").is_none());
    assert!(parse_acp_slash_command(" /model literal").is_none());
    assert!(parse_acp_slash_command("ordinary prompt").is_none());
}

fn project_skills(names: &[&str]) -> (tempfile::TempDir, crate::skill::SkillRegistry) {
    let dir = tempfile::tempdir().unwrap();
    for name in names {
        let skill_dir = dir.path().join(".jcode/skills").join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: Test {name}\n---\nDo {name}.\n"),
        )
        .unwrap();
    }
    let registry = crate::skill::SkillRegistry::load_project_overlay(Some(dir.path())).unwrap();
    (dir, registry)
}

#[test]
fn slash_and_dollar_skill_invocations_resolve_with_trailing_prompt() {
    let (_dir, skills) = project_skills(&["interrogate"]);
    assert_eq!(
        resolve_acp_skill_invocation(&skills, "/interrogate"),
        Some(("interrogate".to_string(), None))
    );
    assert_eq!(
        resolve_acp_skill_invocation(&skills, "/interrogate the auth flow\n"),
        Some(("interrogate".to_string(), Some("the auth flow".to_string())))
    );
    assert_eq!(
        resolve_acp_skill_invocation(&skills, "$interrogate why"),
        Some(("interrogate".to_string(), Some("why".to_string())))
    );
}

#[test]
fn unknown_escaped_and_path_like_slash_text_is_untouched() {
    let (_dir, skills) = project_skills(&["interrogate"]);
    for text in [
        "/unknown do things",
        "/home/user/file.rs explain",
        " /interrogate escaped",
        "please /interrogate",
        "$HOME is set",
        "ordinary prompt",
    ] {
        assert_eq!(resolve_acp_skill_invocation(&skills, text), None, "{text}");
    }
}

#[test]
fn explicit_skill_metadata_is_read_from_prompt_meta() {
    assert_eq!(
        explicit_skill_from_params(&json!({"_meta": {"jcode": {"skill": "/interrogate"}}})),
        Some("interrogate".to_string())
    );
    assert_eq!(
        explicit_skill_from_params(&json!({"_meta": {"activeSkill": "$review"}})),
        Some("review".to_string())
    );
    assert_eq!(
        explicit_skill_from_params(&json!({"_meta": {"skill": " "}})),
        None
    );
    assert_eq!(explicit_skill_from_params(&json!({})), None);
}

#[test]
fn skill_commands_advertise_installed_skills_without_shadowing_builtins() {
    let (_dir, skills) = project_skills(&["interrogate", "model"]);
    let commands = acp_skill_commands(&skills);
    let names: Vec<&str> = commands
        .iter()
        .map(|command| command["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["interrogate"]);
    assert_eq!(commands[0]["description"], "Test interrogate");
}

#[test]
fn compatibility_methods_accept_host_field_names_and_aliases() {
    assert_eq!(
        compatibility_option_value(
            &json!({"modelId": "deepseek-v4-flash"}),
            &["modelId", "model"],
            "session/set_model"
        )
        .unwrap(),
        "deepseek-v4-flash"
    );
    assert_eq!(
        compatibility_option_value(
            &json!({"reasoningEffort": "high"}),
            &["effort", "reasoningEffort"],
            "session/set_reasoning_effort"
        )
        .unwrap(),
        "high"
    );
    assert!(
        compatibility_option_value(
            &json!({"effort": ""}),
            &["effort", "reasoningEffort"],
            "session/set_reasoning_effort"
        )
        .unwrap_err()
        .contains("non-empty")
    );
}

#[test]
fn cwd_must_be_absolute() {
    let params = json!({"cwd": "relative"});
    assert!(cwd_from_params(&params).is_err());
    let params = json!({"cwd": "/tmp"});
    assert_eq!(cwd_from_params(&params).unwrap(), Path::new("/tmp"));
}

#[test]
fn config_options_include_model_selector_and_effort_ladder() {
    let state = SessionUiState {
        provider_name: Some("openai".to_string()),
        model: Some("gpt-5.2".to_string()),
        available_models: vec!["gpt-5.2".to_string(), "gpt-5.2-codex".to_string()],
        reasoning_effort: Some("high".to_string()),
        ..SessionUiState::default()
    };
    let options = session_config_options(&state);
    assert_eq!(options.len(), 2);

    let model = &options[0];
    assert_eq!(model["id"], CONFIG_ID_MODEL);
    assert_eq!(model["category"], "model");
    assert_eq!(model["type"], "select");
    assert_eq!(model["currentValue"], "gpt-5.2");
    assert_eq!(model["options"].as_array().unwrap().len(), 2);

    let effort = &options[1];
    assert_eq!(effort["id"], CONFIG_ID_EFFORT);
    assert_eq!(effort["category"], "thought_level");
    assert_eq!(effort["currentValue"], "high");
    let effort_values: Vec<&str> = effort["options"]
        .as_array()
        .unwrap()
        .iter()
        .map(|option| option["value"].as_str().unwrap())
        .collect();
    assert!(effort_values.contains(&"medium"));
    assert!(
        !effort_values.iter().any(|value| value.starts_with("swarm")),
        "swarm sentinels are TUI-only and must not leak over ACP: {effort_values:?}"
    );
}

#[test]
fn config_options_current_model_prepended_when_not_listed() {
    let state = SessionUiState {
        provider_name: Some("anthropic".to_string()),
        model: Some("claude-opus-4-6".to_string()),
        available_models: vec!["claude-sonnet-4-5".to_string()],
        reasoning_effort: None,
        ..SessionUiState::default()
    };
    let options = session_config_options(&state);
    let model_values: Vec<&str> = options[0]["options"]
        .as_array()
        .unwrap()
        .iter()
        .map(|option| option["value"].as_str().unwrap())
        .collect();
    assert_eq!(model_values[0], "claude-opus-4-6");
    assert!(model_values.contains(&"claude-sonnet-4-5"));
}

#[test]
fn legacy_models_catalog_is_emitted_alongside_config_options() {
    let state = SessionUiState {
        provider_name: Some("deepseek".to_string()),
        model: Some("deepseek-v4-flash".to_string()),
        available_models: vec!["deepseek-v4-pro".to_string()],
        reasoning_effort: Some("high".to_string()),
        ..SessionUiState::default()
    };
    let mut result = json!({"sessionId": "s1"});
    insert_session_configuration(&mut result, &state);

    assert!(result["configOptions"].is_array());
    assert_eq!(result["models"]["currentModelId"], "deepseek-v4-flash");
    let ids: Vec<&str> = result["models"]["availableModels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|model| model["modelId"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["deepseek-v4-flash", "deepseek-v4-pro"]);
}

#[test]
fn config_options_empty_without_model_state() {
    let options = session_config_options(&SessionUiState::default());
    assert!(options.is_empty());
}

#[test]
fn context_limit_falls_back_to_default_for_unknown_models() {
    let state = SessionUiState {
        provider_name: Some("mystery".to_string()),
        model: Some("mystery-model-9000".to_string()),
        available_models: Vec::new(),
        reasoning_effort: None,
        ..SessionUiState::default()
    };
    assert_eq!(
        state.context_limit(),
        crate::provider::DEFAULT_CONTEXT_LIMIT as u64
    );
}

#[test]
fn model_route_specs_qualify_provider() {
    let route = |model: &str, provider: &str, api: &str| ModelRoute {
        model: model.to_string(),
        provider: provider.to_string(),
        api_method: api.to_string(),
        available: true,
        detail: String::new(),
        cheapness: None,
        usage: None,
    };
    let routes = vec![
        route(
            "glm-5.3-flash",
            "OpenCode Go",
            "openai-compatible:opencode-go",
        ),
        route("glm-5.3-flash", "Z.AI", "openai-compatible:zai"),
    ];
    let specs = model_route_specs_from_routes(&routes, None);
    // First-wins (no session preference): OpenCode Go claims the spec.
    assert_eq!(
        specs.get("glm-5.3-flash").map(String::as_str),
        Some("opencode-go:glm-5.3-flash")
    );
    // The spec must be a set_model-acceptable routing string that binds
    // the provider, not the bare model id.
    assert!(specs["glm-5.3-flash"].contains(':'));
}

#[test]
fn model_labels_prefer_session_provider() {
    let route = |model: &str, provider: &str| ModelRoute {
        model: model.to_string(),
        provider: provider.to_string(),
        api_method: String::new(),
        available: true,
        detail: String::new(),
        cheapness: None,
        usage: None,
    };
    let routes = vec![
        route("glm-5.3-flash", "OpenCode Go"),
        route("glm-5.3-flash", "Z.AI"),
        route("glm-4.6", "Z.AI"),
    ];
    let labels = model_provider_labels_from_routes(&routes, Some("Z.AI"));
    assert_eq!(
        labels.get("glm-5.3-flash").map(String::as_str),
        Some("Z.AI")
    );
    assert_eq!(labels.get("glm-4.6").map(String::as_str), Some("Z.AI"));
    // Without a matching session provider, first-wins still applies.
    let labels = model_provider_labels_from_routes(&routes, Some("Kimi"));
    assert_eq!(
        labels.get("glm-5.3-flash").map(String::as_str),
        Some("OpenCode Go")
    );
}

#[tokio::test]
async fn available_models_update_refreshes_ui_state_and_reports_change() {
    let (sock_a, _sock_b) = tokio::net::UnixStream::pair().unwrap();
    let (sock_c, _sock_d) = tokio::net::UnixStream::pair().unwrap();
    let reader = sock_a.into_split().0;
    let writer = sock_c.into_split().1;
    let session =
        DaemonSession::new(String::new(), reader, writer, 1).with_ui_state(SessionUiState {
            model: Some("stale-model".to_string()),
            available_models: vec!["stale-model".to_string()],
            ..SessionUiState::default()
        });
    let route = ModelRoute {
        model: "live-model".to_string(),
        provider: "openrouter".to_string(),
        api_method: String::new(),
        available: true,
        detail: String::new(),
        cheapness: None,
        usage: None,
    };
    let event = ServerEvent::AvailableModelsUpdated {
        provider_name: Some("OpenRouter".to_string()),
        provider_model: Some("live-model".to_string()),
        available_models: vec!["live-model".to_string()],
        available_model_routes: vec![route],
    };
    assert!(apply_available_models_update_inner(&session, &event).await);
    let state = session.ui_state.lock().await;
    assert_eq!(state.available_models, vec!["live-model".to_string()]);
    assert_eq!(state.model.as_deref(), Some("live-model"));
    assert_eq!(state.provider_name.as_deref(), Some("OpenRouter"));
    assert_eq!(
        state
            .model_provider_labels
            .get("live-model")
            .map(String::as_str),
        Some("openrouter")
    );
    // A repeated push with an unchanged catalog reports no change so
    // callers do not spam config_option_update notifications.
    drop(state);
    assert!(!apply_available_models_update_inner(&session, &event).await);
}

#[test]
fn model_options_keep_one_entry_per_provider_route() {
    // Both Z.AI and OpenCode Go serve `glm-5.3-flash`; the picker must
    // offer both routes instead of collapsing to the first one seen.
    let route = |model: &str, provider: &str, profile: &str| crate::provider::ModelRoute {
        model: model.to_string(),
        provider: provider.to_string(),
        api_method: format!("openai-compatible:{profile}"),
        available: true,
        detail: String::new(),
        cheapness: None,
        usage: None,
    };
    let routes = vec![
        route("glm-5.3-flash", "OpenCode Go", "opencode-go"),
        route("glm-5.3-flash", "Z.AI", "zai"),
    ];
    let state = SessionUiState {
        provider_name: Some("OpenCode Go".to_string()),
        model: Some("glm-5.3-flash".to_string()),
        available_models: vec!["glm-5.3-flash".to_string()],
        model_provider_labels: model_provider_labels_from_routes(&routes, Some("OpenCode Go")),
        model_route_specs: model_route_specs_from_routes(&routes, Some("OpenCode Go")),
        available_model_routes: routes,
        ..SessionUiState::default()
    };
    let options = session_config_options(&state);
    let model = &options[0];
    let values: Vec<&str> = model["options"]
        .as_array()
        .unwrap()
        .iter()
        .map(|option| option["value"].as_str().unwrap())
        .collect();
    assert!(
        values.contains(&"zai:glm-5.3-flash"),
        "Z.AI route must stay selectable: {values:?}"
    );
    assert!(
        values.contains(&"opencode-go:glm-5.3-flash"),
        "OpenCode Go route must stay selectable: {values:?}"
    );
    // The current model's spec must remain an option value so clients can
    // render the selection state.
    assert!(values.contains(&model["currentValue"].as_str().unwrap()));
}

#[test]
fn host_session_env_forwards_only_host_namespaces() {
    // SAFETY: single-threaded test process mutation of env vars for a
    // narrowly-scoped assertion; restored before returning.
    let original: Vec<(String, String)> = std::env::vars().collect();
    unsafe {
        std::env::set_var("KANEO_API_KEY", "probe-key");
        std::env::set_var("KANEO_API_URL", "http://localhost:1337");
        std::env::set_var("AIONUI_CONVERSATION_ID", "conv-1");
        std::env::set_var("UNRELATED_SECRET", "must-not-travel");
    }
    let forwarded = host_session_env();
    unsafe {
        std::env::remove_var("KANEO_API_KEY");
        std::env::remove_var("KANEO_API_URL");
        std::env::remove_var("AIONUI_CONVERSATION_ID");
        std::env::remove_var("UNRELATED_SECRET");
        for (key, value) in original {
            std::env::set_var(key, value);
        }
    }
    let keys: Vec<&str> = forwarded.iter().map(|(key, _)| key.as_str()).collect();
    assert!(keys.contains(&"KANEO_API_KEY"), "{keys:?}");
    assert!(keys.contains(&"KANEO_API_URL"), "{keys:?}");
    assert!(keys.contains(&"AIONUI_CONVERSATION_ID"), "{keys:?}");
    assert!(
        !keys.contains(&"UNRELATED_SECRET"),
        "only host namespaces may travel over the wire: {keys:?}"
    );
}

struct RestoreHome(Option<String>);
impl Drop for RestoreHome {
    fn drop(&mut self) {
        unsafe {
            match &self.0 {
                Some(value) => std::env::set_var("JCODE_HOME", value),
                None => std::env::remove_var("JCODE_HOME"),
            }
        }
    }
}

#[tokio::test]
async fn acp_skill_outcome_records_invocation_memory() {
    let _lock = crate::storage::lock_test_env();
    let home = tempfile::tempdir().unwrap();
    let _restore = RestoreHome(std::env::var("JCODE_HOME").ok());
    unsafe { std::env::set_var("JCODE_HOME", home.path()) };

    let project = home.path().join("project");
    std::fs::create_dir_all(&project).unwrap();
    let (sock_a, _sock_b) = tokio::net::UnixStream::pair().unwrap();
    let (sock_c, _sock_d) = tokio::net::UnixStream::pair().unwrap();
    let session = DaemonSession::new(
        String::new(),
        sock_a.into_split().0,
        sock_c.into_split().1,
        1,
    )
    .with_working_dir(project.clone());

    record_acp_skill_outcome(
        &session,
        Some("probe"),
        crate::memory_types::ActionOutcome::Succeeded,
    );

    let manager = crate::memory::MemoryManager::new().with_project_dir(&project);
    let graph = manager.load_project_graph().unwrap();
    let invocation = graph
        .get_memory("skill-invocation:probe")
        .expect("ACP outcome must persist an invocation row");
    assert!(invocation
        .tags
        .iter()
        .any(|tag| tag == "skill-outcome:succeeded"));
}
