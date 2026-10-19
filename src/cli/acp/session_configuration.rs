//! ACP model and reasoning effort configuration presentation.

use super::*;

pub(super) const CONFIG_ID_MODEL: &str = "model";
pub(super) const CONFIG_ID_EFFORT: &str = "reasoning_effort";

pub(super) fn acp_available_commands() -> Vec<Value> {
    vec![
        json!({
            "name": "model",
            "description": "Switch the model for this session, or show the current model",
            "input": { "hint": "model id (optional)" },
        }),
        json!({
            "name": "models",
            "description": "List models available from the active provider",
        }),
        json!({
            "name": "effort",
            "description": "Set reasoning effort, or show the current effort",
            "input": { "hint": "none|minimal|low|medium|high|xhigh|max (optional)" },
        }),
    ]
}

pub(super) fn insert_session_configuration(result: &mut Value, state: &SessionUiState) {
    let Some(object) = result.as_object_mut() else {
        return;
    };
    let config_options = session_config_options(state);
    if !config_options.is_empty() {
        object.insert("configOptions".to_string(), Value::Array(config_options));
    }
    if let Some(models) = session_models(state) {
        object.insert("models".to_string(), models);
    }
}

/// Display label for one model picker entry.
///
/// Rendered as `provider · model` when the catalog told us which provider
/// serves the name, matching the TUI picker's convention. Falls back to the
/// bare name so a route-less catalog (or a model outside the reported routes)
/// still renders. `value`/`modelId` deliberately stay unqualified.
fn model_option_label(state: &SessionUiState, model: &str) -> String {
    match state.model_provider_labels.get(model) {
        Some(provider) if !provider.is_empty() => format!("{provider} · {model}"),
        _ => model.to_string(),
    }
}

/// One picker entry per (provider, model) route.
///
/// Model names alone are not unique across providers (e.g. both Z.AI and
/// OpenCode Go serve `glm-5.3-flash`), so entries are keyed by the qualified
/// routing spec instead of the bare name. When no route catalog is available
/// the bare model names are used with fallback labels, matching the legacy
/// behavior. Returns `(value, label)` pairs; `value` is what the client sends
/// back on `set_config_option` (bare names are also accepted there).
fn model_picker_entries(state: &SessionUiState) -> Vec<(String, String)> {
    if state.available_model_routes.is_empty() {
        return state
            .available_models
            .iter()
            .map(|name| (name.clone(), model_option_label(state, name)))
            .collect();
    }
    let preferred = state.provider_name.as_deref();
    let mut entries: Vec<(String, String)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let push_route = |route: &ModelRoute,
                      entries: &mut Vec<(String, String)>,
                      seen: &mut std::collections::HashSet<String>| {
        if route.model.is_empty() || route.provider.trim().is_empty() || !route.available {
            return;
        }
        let spec = RouteSelection::from_model_route(route).routed_model_spec();
        if !seen.insert(spec.clone()) {
            return;
        }
        let display = cli_route_provider_display(&route.provider, &route.api_method);
        let label = if display.is_empty() {
            route.model.clone()
        } else {
            format!("{display} \u{b7} {}", route.model)
        };
        entries.push((spec, label));
    };
    // The session's current provider's routes first so they sort to the top.
    if let Some(preferred) = preferred.map(str::trim).filter(|p| !p.is_empty()) {
        for route in &state.available_model_routes {
            if &cli_route_provider_display(&route.provider, &route.api_method) == preferred {
                push_route(route, &mut entries, &mut seen);
            }
        }
    }
    for route in &state.available_model_routes {
        push_route(route, &mut entries, &mut seen);
    }
    entries
}

pub(super) fn session_models(state: &SessionUiState) -> Option<Value> {
    let current = state.model.as_deref()?;
    let mut models = state.available_models.clone();
    if !models.iter().any(|candidate| candidate == current) {
        models.insert(0, current.to_string());
    }
    Some(json!({
        "availableModels": models
            .into_iter()
            .map(|model| {
                let label = model_option_label(state, &model);
                json!({ "modelId": model, "name": label })
            })
            .collect::<Vec<_>>(),
        "currentModelId": current,
    }))
}

pub(super) fn available_efforts(state: &SessionUiState) -> Vec<&'static str> {
    crate::provider::inferred_reasoning_efforts(
        state.provider_name.as_deref(),
        state.model.as_deref(),
    )
    .into_iter()
    // `swarm`/`swarm-deep` are TUI sentinels, not provider effort levels.
    .filter(|effort| !effort.starts_with("swarm"))
    .collect()
}

/// Build the ACP `configOptions` array (model selector plus reasoning effort)
/// from the current session provider state. Empty when the daemon reported no
/// usable model state.
pub(super) fn session_config_options(state: &SessionUiState) -> Vec<Value> {
    let mut options = Vec::new();

    if let Some(model) = state.model.as_deref() {
        let mut entries = model_picker_entries(state);
        // The current value must match one of the option values so clients
        // can render the selection state.
        let current_value = state
            .model_route_specs
            .get(model)
            .cloned()
            .unwrap_or_else(|| model.to_string());
        if !entries.iter().any(|(value, _)| value == &current_value) {
            entries.insert(0, (current_value.clone(), model_option_label(state, model)));
        }
        let select_options: Vec<Value> = entries
            .into_iter()
            .map(|(value, name)| json!({ "value": value, "name": name }))
            .collect();
        options.push(json!({
            "type": "select",
            "id": CONFIG_ID_MODEL,
            "name": "Model",
            "category": "model",
            "currentValue": current_value,
            "options": select_options,
        }));
    }

    let efforts = available_efforts(state);
    if !efforts.is_empty() {
        let current = state
            .reasoning_effort
            .as_deref()
            .filter(|effort| efforts.contains(effort))
            .unwrap_or_else(|| {
                if efforts.contains(&"medium") {
                    "medium"
                } else {
                    efforts[0]
                }
            });
        let select_options: Vec<Value> = efforts
            .iter()
            .map(|name| json!({ "value": name, "name": name }))
            .collect();
        options.push(json!({
            "type": "select",
            "id": CONFIG_ID_EFFORT,
            "name": "Reasoning effort",
            "category": "thought_level",
            "currentValue": current,
            "options": select_options,
        }));
    }

    options
}
