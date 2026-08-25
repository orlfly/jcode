//! Parsing of OpenAI-compatible `/v1/models` catalog responses.
//!
//! Kept separate from the provider runtime so the shape-tolerance rules for the
//! many gateways jcode speaks to (vLLM, llama.cpp, Ollama, LM Studio, vendor
//! clouds) live in one small, testable place.

use anyhow::{Context, Result};
use jcode_provider_openrouter::{ModelInfo, ModelPricing};
use serde_json::Value;

pub(crate) fn parse_openai_compatible_models_response(raw_body: &str) -> Result<Vec<ModelInfo>> {
    let value: Value = serde_json::from_str(raw_body)?;
    let items = match &value {
        Value::Array(items) => items,
        Value::Object(object) => object
            .get("data")
            .or_else(|| object.get("models"))
            .and_then(Value::as_array)
            .context("missing model array")?,
        _ => anyhow::bail!("model catalog response must be an object or array"),
    };

    let mut models = Vec::new();
    for item in items {
        if let Some(model) = parse_model_info_value(item) {
            models.push(model);
        }
    }

    if models.is_empty() {
        anyhow::bail!("model catalog response did not contain any valid model objects");
    }

    Ok(models)
}

pub(crate) fn parse_model_info_value(value: &Value) -> Option<ModelInfo> {
    let object = value.as_object()?;
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| object.get("name").and_then(Value::as_str))?
        .to_string();
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| object.get("display_name").and_then(Value::as_str))
        .or_else(|| object.get("displayName").and_then(Value::as_str))
        .unwrap_or("")
        .to_string();

    Some(ModelInfo {
        id,
        name,
        context_length: first_u64_field(
            object,
            &[
                "context_length",
                "contextLength",
                "max_context_length",
                "maxModelLength",
                "max_model_len",
                "trainingContextLength",
            ],
        )
        // llama.cpp's /v1/models reports the serving context only inside
        // `meta` (`n_ctx`, with `n_ctx_train` as the trained maximum). Without
        // this, local llama.cpp models fall back to the generic 200K default
        // and the context gauge overstates the real window (issue #447).
        .or_else(|| {
            object
                .get("meta")
                .and_then(Value::as_object)
                .and_then(|meta| first_u64_field(meta, &["n_ctx", "n_ctx_train"]))
        }),
        pricing: parse_model_pricing(object.get("pricing")),
        created: object.get("created").and_then(value_as_u64),
        supports_image_input: parse_supports_image_input(object),
    })
}

/// Resolve image-input support from a catalog model object.
///
/// Some OpenAI-compatible gateways advertise it as a boolean `vision` /
/// `supports_vision` / `image_input` field, or as a `capabilities` array that
/// contains `"vision"` / `"image"`. `None` means the catalog did not say, and
/// callers fall back to provider-level heuristics.
fn parse_supports_image_input(object: &serde_json::Map<String, Value>) -> Option<bool> {
    for key in ["vision", "supports_vision", "image_input", "supports_image_input"] {
        if let Some(value) = object.get(key) {
            if let Some(b) = value.as_bool() {
                return Some(b);
            }
        }
    }
    if let Some(caps) = object.get("capabilities").and_then(Value::as_array) {
        let has_vision = caps.iter().any(|c| {
            c.as_str()
                .is_some_and(|s| s.eq_ignore_ascii_case("vision") || s.eq_ignore_ascii_case("image"))
        });
        return Some(has_vision);
    }
    None
}

pub(crate) fn first_u64_field(
    object: &serde_json::Map<String, Value>,
    keys: &[&str],
) -> Option<u64> {
    keys.iter()
        .find_map(|key| object.get(*key).and_then(value_as_u64))
}

pub(crate) fn value_as_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse::<u64>().ok(),
        _ => None,
    }
}

pub(crate) fn value_as_pricing_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

pub(crate) fn parse_model_pricing(value: Option<&Value>) -> ModelPricing {
    let Some(Value::Object(object)) = value else {
        return ModelPricing::default();
    };

    ModelPricing {
        prompt: object
            .get("prompt")
            .or_else(|| object.get("input"))
            .and_then(value_as_pricing_string),
        completion: object
            .get("completion")
            .or_else(|| object.get("output"))
            .and_then(value_as_pricing_string),
        input_cache_read: object
            .get("input_cache_read")
            .or_else(|| object.get("cached_input"))
            .and_then(value_as_pricing_string),
        input_cache_write: object
            .get("input_cache_write")
            .and_then(value_as_pricing_string),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_vision_from_capabilities_array() {
        let v: Value = serde_json::from_str(
            r#"{"id":"qwen3.5:397b-cloud","capabilities":["completion","thinking","tools","vision"]}"#,
        )
        .unwrap();
        let model = parse_model_info_value(&v).unwrap();
        assert_eq!(model.supports_image_input, Some(true));
    }

    #[test]
    fn parses_no_vision_from_capabilities_without_vision() {
        let v: Value = serde_json::from_str(
            r#"{"id":"deepseek-v4-flash:cloud","capabilities":["completion","thinking","tools"]}"#,
        )
        .unwrap();
        let model = parse_model_info_value(&v).unwrap();
        assert_eq!(model.supports_image_input, Some(false));
    }

    #[test]
    fn parses_boolean_vision_field() {
        let v: Value = serde_json::from_str(r#"{"id":"m","vision":true}"#).unwrap();
        let model = parse_model_info_value(&v).unwrap();
        assert_eq!(model.supports_image_input, Some(true));
    }

    #[test]
    fn leaves_image_support_unknown_when_absent() {
        let v: Value = serde_json::from_str(r#"{"id":"m"}"#).unwrap();
        let model = parse_model_info_value(&v).unwrap();
        assert_eq!(model.supports_image_input, None);
    }
}
