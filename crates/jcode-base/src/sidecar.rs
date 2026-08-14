//! Lightweight sidecar client for fast, cheap model calls.
//!
//! Used for memory relevance verification and other quick tasks that don't
//! need the full Agent SDK infrastructure.
//!
//! Automatically selects the best available backend:
//! - OpenAI (gpt-5.6-luna, reasoning=none) if Codex credentials are available
//! - Claude (claude-haiku-4-5-20241022) if Claude credentials are available

use crate::auth;
use anyhow::{Context, Result};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Fast/cheap OpenAI model used when Codex credentials are available.
pub const SIDECAR_OPENAI_MODEL: &str = "gpt-5.6-luna";
const SIDECAR_OPENAI_REASONING: &str = "none";
const SIDECAR_OPENAI_OAUTH_FALLBACK_MODEL: &str = "gpt-5.4";
const SIDECAR_OPENAI_OAUTH_FALLBACK_REASONING: &str = "low";

/// Fast/cheap Claude model used when only Claude credentials are available.
const SIDECAR_CLAUDE_MODEL: &str = "claude-haiku-4-5-20251001";

/// OpenAI Responses API
const OPENAI_API_BASE: &str = "https://api.openai.com/v1";
const CHATGPT_API_BASE: &str = "https://chatgpt.com/backend-api/codex";
const OPENAI_RESPONSES_PATH: &str = "responses";
const OPENAI_ORIGINATOR: &str = "codex_cli_rs";

/// Claude Messages API endpoint (with beta=true for OAuth)
const CLAUDE_API_URL: &str = "https://api.anthropic.com/v1/messages?beta=true";

/// Claude Messages API endpoint for direct API-key access (no OAuth beta flag).
const CLAUDE_API_KEY_URL: &str = "https://api.anthropic.com/v1/messages";

/// User-Agent for OAuth requests (must match Claude CLI format)
const CLAUDE_CLI_USER_AGENT: &str = "claude-cli/1.0.0";

/// Beta headers required for OAuth
const OAUTH_BETA_HEADERS: &str = "oauth-2025-04-20,claude-code-20250219";

/// Claude Code identity block required for OAuth direct API access
const CLAUDE_CODE_IDENTITY: &str = "You are Claude Code, Anthropic's official CLI for Claude.";
const CLAUDE_CODE_JCODE_NOTICE: &str = "You are jcode, powered by Claude Code. You are a third-party CLI, not the official Claude Code CLI.";

/// Maximum tokens for sidecar responses (keep small for speed/cost)
const DEFAULT_MAX_TOKENS: u32 = 1024;

/// OpenRouter-style model names carry a `vendor/family` namespace (e.g.
/// `anthropic/claude-sonnet-4`, `openai/gpt-5.5`). Direct OpenAI-compatible
/// providers (DeepSeek, Moonshot, MiniMax, etc.) expect their own bare model
/// ids (`deepseek-chat`, `moonshot-v1-8k`). When the sidecar forks a provider
/// whose endpoint is a direct API but whose model is in the OpenRouter
/// namespace, we MUST rewrite the model to a known-compatible default instead
/// of trusting the stored model — otherwise the direct API returns HTTP 400
/// "Model Not Exist" and the memory rerank falls into `all_judges_failed`.
fn safe_model_for_provider(provider: &dyn crate::provider::Provider) -> String {
    let raw = provider.model();
    if !is_namespaced_model(&raw) {
        return raw;
    }

    // A provider's runtime label (e.g. "deepseek", "openai-compatible:deepseek",
    // "openrouter-compatible") is the single best signal we have for the
    // endpoint shape. If the runtime points at a direct OpenAI-compatible
    // service, the stored model must not use OpenRouter's vendor/family
    // namespace.
    let name = provider.name().to_ascii_lowercase();
    if !is_direct_openai_compatible_runtime(&name) {
        return raw;
    }

    let default = provider_default_model_for(&name);
    match default {
        Some(default) => {
            crate::logging::warn(&format!(
                "Sidecar: provider '{}' carries model '{}' which is incompatible with \
                 a direct OpenAI-compatible endpoint; falling back to default '{}' to \
                 avoid memory rerank degradation (all_judges_failed).",
                name, raw, default
            ));
            default.to_string()
        }
        None => raw,
    }
}

/// Direct OpenAI-compatible runtime labels that do NOT understand the
/// `vendor/family` namespace. Matched by substring so the varied
/// `openai-compatible:<profile>` shapes and dashed profile ids all work.
fn is_direct_openai_compatible_runtime(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
    const KNOWN_RUNTIMES: &[&str] = &[
        "deepseek",
        "moonshot",
        "kimi",
        "minimax",
        "minimaxi",
        "bigmodel",
        "zhipu",
        "zhipuai",
        "cerebras",
        "groq",
        "fireworks",
        "together",
        "deepinfra",
        "mistral",
        "perplexity",
        "xai",
        "nvidia",
        "nvidia-nim",
        "coda",
        "siliconflow",
        "lingyiwanwu",
        "stepfun",
        "baichuan",
        "sensenova",
        "cohere",
        "chutes",
        "lmstudio",
        "ollama",
        "vllm",
        "llamacpp",
    ];
    if KNOWN_RUNTIMES.iter().any(|k| normalized == *k || normalized.contains(k)) {
        return true;
    }
    // The openai-compatible generic namespace is always direct.
    normalized.starts_with("openai-compatible:") || normalized == "openai-compatible"
}

/// OpenRouter-style model (`<vendor>/<family>`). Don't be too restrictive —
/// the slash is the spiciest signal. We let the endpoint check decide
/// whether the namespace actually breaks things.
fn is_namespaced_model(model: &str) -> bool {
    if let Some((vendor, _rest)) = model.split_once('/') {
        !vendor.is_empty()
            && vendor.len() <= 64
            && !vendor.contains('\\')
            && !vendor.contains('?')
            && !vendor.contains('#')
    } else {
        false
    }
}

/// Per-runtime default model that the sidecar can use when the stored
/// model is incompatible with the endpoint. These mirror the defaults the
/// catalogs ship with so we always have a model that the endpoint accepts.
///
/// Update these whenever a new direct OpenAI-compatible profile is added.
fn provider_default_model_for(name: &str) -> Option<&'static str> {
    if name.contains("deepseek") {
        return Some("deepseek-chat");
    }
    if name.contains("moonshot") || name.contains("kimi") {
        return Some("moonshot-v1-8k");
    }
    if name.contains("minimaxi") || name == "minimax" || name.contains("minimax") {
        return Some("MiniMax-M3");
    }
    if name.contains("bigmodel") || name.contains("zhipu") {
        return Some("glm-4.6");
    }
    if name.contains("cerebras") {
        return Some("llama-3.3-70b");
    }
    if name.contains("groq") {
        return Some("llama-3.3-70b-versatile");
    }
    if name.contains("fireworks") {
        return Some("accounts/fireworks/models/llama-v3p3-70b-instruct");
    }
    if name.contains("together") {
        return Some("meta-llama/Llama-3.3-70B-Instruct-Turbo");
    }
    if name.contains("deepinfra") {
        return Some("meta-llama/Llama-3.3-70B-Instruct");
    }
    if name.contains("mistral") {
        return Some("mistral-large-latest");
    }
    if name.contains("perplexity") {
        return Some("sonar");
    }
    if name.contains("xai") {
        return Some("grok-3-mini");
    }
    if name.contains("nvidia") {
        return Some("meta/llama-3.3-70b-instruct");
    }
    if name.contains("cohere") {
        return Some("command-r-plus");
    }
    if name.contains("siliconflow") {
        return Some("Qwen/Qwen2.5-72B-Instruct");
    }
    if name.contains("lingyiwanwu") {
        return Some("yi-large");
    }
    if name.contains("stepfun") {
        return Some("step-1v-32k");
    }
    if name.contains("baichuan") {
        return Some("Baichuan4");
    }
    if name.contains("sensenova") {
        return Some("SenseChat-5");
    }
    if name.contains("chutes") {
        return Some("deepseek-ai/DeepSeek-V3");
    }
    if name == "lmstudio" || name.contains("lmstudio") {
        return Some("local-model");
    }
    if name == "ollama" || name.contains("ollama") {
        return Some("llama3.3");
    }
    if name == "vllm" || name.contains("vllm") {
        return Some("local-model");
    }
    if name == "llamacpp" || name.contains("llamacpp") {
        return Some("local-model");
    }
    if name.starts_with("openai-compatible:") || name == "openai-compatible" {
        // Generic OpenAI-compatible: we don't know what models the endpoint
        // exposes, so we can't suggest a safe default. The caller will fall
        // through to the raw model and the request will still fail loudly
        // if the endpoint rejects it — no silent degradation.
        return None;
    }
    None
}

/// Whether retrying a failed sidecar request can reasonably succeed without a
/// configuration or credential change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidecarErrorKind {
    Transient,
    Permanent,
}

#[derive(Debug)]
struct SidecarHttpError {
    provider: &'static str,
    status: StatusCode,
    body: String,
}

impl fmt::Display for SidecarHttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} API error ({}): {}",
            self.provider, self.status, self.body
        )
    }
}

impl std::error::Error for SidecarHttpError {}

/// Classify a sidecar failure for retry policy. HTTP client/auth/request errors
/// are permanent; throttling, server failures, and transport failures are
/// transient. Unknown provider errors retain the conservative retry behavior.
pub fn classify_error(error: &anyhow::Error) -> SidecarErrorKind {
    if let Some(error) = error.downcast_ref::<SidecarHttpError>() {
        return classify_http_status(error.status);
    }
    for cause in error.chain() {
        if let Some(error) = cause.downcast_ref::<reqwest::Error>() {
            if let Some(status) = error.status() {
                return classify_http_status(status);
            }
            return SidecarErrorKind::Transient;
        }
    }

    // Provider-backed sidecars may not expose a typed HTTP error yet.
    let message = error.to_string().to_ascii_lowercase();
    if [
        "400",
        "401",
        "403",
        "404",
        "bad request",
        "unauthorized",
        "forbidden",
        "not_found_error",
    ]
    .iter()
    .any(|needle| message.contains(needle))
    {
        SidecarErrorKind::Permanent
    } else {
        SidecarErrorKind::Transient
    }
}

fn classify_http_status(status: StatusCode) -> SidecarErrorKind {
    if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
        SidecarErrorKind::Transient
    } else if status.is_client_error() {
        SidecarErrorKind::Permanent
    } else {
        SidecarErrorKind::Transient
    }
}

/// Which backend the sidecar is using
#[derive(Debug, Clone, Copy, PartialEq)]
enum SidecarBackend {
    OpenAI,
    Claude,
    /// Dispatch through the live agent provider (`crate::provider::active_provider_fork`).
    /// Used when neither OpenAI nor Claude OAuth credentials are present but the
    /// user is running on another provider (Copilot, Antigravity, Gemini,
    /// Cursor, Bedrock, OpenRouter). This is what makes the memory sidecar work
    /// on ALL providers instead of only the two with dedicated HTTP clients.
    Provider,
}

/// Lightweight client for fast sidecar calls
#[derive(Clone)]
pub struct Sidecar {
    client: reqwest::Client,
    model: String,
    max_tokens: u32,
    backend: SidecarBackend,
    /// Optional explicit reasoning effort override (OpenAI Responses API).
    /// When `Some`, this effort is always sent; when `None`, the default
    /// per-model behavior applies. Used by the memory benchmark to pin
    /// GPT-5.5 with no thinking.
    reasoning_override: Option<String>,
}

impl Sidecar {
    /// Create a new sidecar client, auto-selecting the best available backend.
    /// Prefers OpenAI (GPT-5.6 Luna with no reasoning) if creds exist, falls back to Claude.
    pub fn new() -> Self {
        let configured_model = crate::config::config().agents.memory_model.clone();
        Self::with_configured_model(configured_model)
    }

    fn with_configured_model(configured_model: Option<String>) -> Self {
        let (backend, model) = if let Some(model) = configured_model {
            match crate::provider::provider_for_model(&model) {
                Some("openai") => (SidecarBackend::OpenAI, model),
                Some("claude") => (SidecarBackend::Claude, model),
                _ => {
                    crate::logging::warn(&format!(
                        "Ignoring unsupported memory sidecar model override '{}'; expected an OpenAI or Claude model",
                        model
                    ));
                    Self::auto_select_backend()
                }
            }
        } else {
            Self::auto_select_backend()
        };

        Self {
            client: crate::provider::shared_http_client(),
            model,
            max_tokens: DEFAULT_MAX_TOKENS,
            backend,
            reasoning_override: None,
        }
    }

    /// Pick the best available sidecar backend.
    ///
    /// Preference order:
    /// 1. OpenAI GPT-5.6 Luna at reasoning=none if Codex creds exist.
    /// 2. Claude haiku (dedicated fast/cheap OAuth path) if Claude creds exist.
    /// 3. The live agent provider (works for EVERY provider jcode supports:
    ///    Copilot, Antigravity, Gemini, Cursor, Bedrock, OpenRouter, and even
    ///    OpenAI/Claude API-key setups), dispatched via `complete_simple`.
    ///
    /// Only when no provider is registered at all do we fall back to Claude,
    /// which then fails on use with a clear credentials error.
    fn auto_select_backend() -> (SidecarBackend, String) {
        if auth::codex::load_credentials().is_ok() {
            (SidecarBackend::OpenAI, SIDECAR_OPENAI_MODEL.to_string())
        } else if auth::claude::load_credentials().is_ok() {
            (SidecarBackend::Claude, SIDECAR_CLAUDE_MODEL.to_string())
        } else if let Some(provider) = crate::provider::active_provider_fork() {
            // Dispatch through whatever provider the user is running on. The
            // model string is informational here; the provider already has the
            // user's selected model and routes accordingly.
            //
            // SAFETY: if the provider is a direct OpenAI-compatible profile
            // (e.g. DeepSeek, Moonshot, OpenRouter-compatible), the model name
            // it carries may belong to a different namespace (e.g. the
            // OpenRouter-style `anthropic/claude-sonnet-4` on a DeepSeek direct
            // endpoint). Sending that to the direct API returns HTTP 400
            // "Model Not Exist" and cascades into `all_judges_failed` for the
            // memory rerank. To prevent this, prefer the provider's *default*
            // model (the one it was constructed with) when it carries a
            // namespace identifier that the endpoint does not understand.
            let model = safe_model_for_provider(provider.as_ref());
            (SidecarBackend::Provider, model)
        } else {
            // No credentials and no live provider: default to Claude so the
            // eventual error message is actionable.
            (SidecarBackend::Claude, SIDECAR_CLAUDE_MODEL.to_string())
        }
    }

    /// Whether the provider-backed sidecar can confidently use the configured
    /// model. When the active provider is a direct API (DeepSeek, Moonshot,
    /// etc.) and its model name carries an OpenRouter-style namespace
    /// (e.g. `anthropic/claude-sonnet-4`), the sidecar will silently re-route
    /// to the provider's default model. Callers that need to know whether the
    /// sidecar is operating in degraded mode can use this to log a warning.
    pub fn provider_model_is_compatible(provider: &dyn crate::provider::Provider) -> bool {
        let raw = provider.model();
        if !is_namespaced_model(&raw) {
            return true;
        }
        let name = provider.name().to_ascii_lowercase();
        !is_direct_openai_compatible_runtime(&name)
    }

    /// Whether a usable LLM backend is actually reachable for the sidecar right
    /// now. Unlike [`Sidecar::auto_select_backend`] this does NOT fall back to a
    /// Claude placeholder when nothing is logged in: it returns `true` only when
    /// real Codex/Claude credentials exist or a live agent provider is
    /// registered.
    ///
    /// Re-evaluated live (reads credentials/provider state on each call) so that
    /// adding or removing a login is reflected without a restart. This is the
    /// signal the memory system uses to decide whether the LLM precision judge
    /// can run; if it returns `false`, memory's sidecar mode is treated as
    /// unavailable rather than silently degrading to the no-LLM path.
    pub fn llm_backend_available() -> bool {
        auth::codex::load_credentials().is_ok()
            || auth::claude::load_credentials().is_ok()
            || crate::provider::active_provider_fork().is_some()
    }

    /// Return the currently selected sidecar model name.
    pub fn model_name(&self) -> &str {
        &self.model
    }

    /// Construct a sidecar pinned to a specific Claude model (used by the
    /// memory recall benchmark judge so the relevance labels come from a strong,
    /// fixed model regardless of the user's configured memory model).
    pub fn with_claude_model(model: impl Into<String>) -> Self {
        Self {
            client: crate::provider::shared_http_client(),
            model: model.into(),
            max_tokens: DEFAULT_MAX_TOKENS,
            backend: SidecarBackend::Claude,
            reasoning_override: None,
        }
    }

    /// Construct a sidecar pinned to a specific OpenAI model via Codex/OpenAI
    /// OAuth, with an optional explicit reasoning effort (e.g. "none"/"minimal"
    /// for no-thinking). Used by the memory recall benchmark judge.
    pub fn with_openai_model(model: impl Into<String>, reasoning_effort: Option<String>) -> Self {
        Self {
            client: crate::provider::shared_http_client(),
            model: model.into(),
            max_tokens: DEFAULT_MAX_TOKENS,
            backend: SidecarBackend::OpenAI,
            reasoning_override: reasoning_effort,
        }
    }

    /// Return the currently selected backend label.
    pub fn backend_name(&self) -> &'static str {
        match self.backend {
            SidecarBackend::OpenAI => "openai",
            SidecarBackend::Claude => "claude",
            SidecarBackend::Provider => "provider",
        }
    }

    /// Simple completion - send a prompt, get a response.
    /// Routes to the correct API based on the detected backend.
    pub async fn complete(&self, system: &str, user_message: &str) -> Result<String> {
        match self.backend {
            SidecarBackend::OpenAI => self.complete_openai(system, user_message).await,
            SidecarBackend::Claude => self.complete_claude(system, user_message).await,
            SidecarBackend::Provider => self.complete_via_provider(system, user_message).await,
        }
    }

    /// Complete via the live agent provider (`complete_simple`).
    ///
    /// This is the universal path: it works for every provider jcode supports,
    /// because `complete_simple` is a default method on the `Provider` trait that
    /// collects the streamed `TextDelta`s into a single string. The provider was
    /// forked at construction time, so it carries the user's selected model.
    async fn complete_via_provider(&self, system: &str, user_message: &str) -> Result<String> {
        let provider = crate::provider::active_provider_fork().context(
            "No active provider registered for sidecar; memory features require a logged-in provider",
        )?;

        // If the stored model name is incompatible with the provider's
        // endpoint (e.g. `anthropic/claude-sonnet-4` on a DeepSeek direct
        // API endpoint), switch the provider to its safe default for the
        // lifetime of this call. Doing it on the forked provider keeps the
        // main agent's selection untouched.
        let safe_model = safe_model_for_provider(provider.as_ref());
        if safe_model != self.model && safe_model != provider.model() {
            if let Err(err) = provider.set_model(&safe_model) {
                crate::logging::warn(&format!(
                    "Sidecar: failed to switch provider to safe model '{}': {}",
                    safe_model, err
                ));
            }
        }

        provider
            .complete_simple(user_message, system)
            .await
            .context("Sidecar completion via active provider failed")
    }

    /// Complete via OpenAI Responses API.
    ///
    /// - Direct API key mode: non-streaming, simple JSON response.
    /// - ChatGPT OAuth mode: streaming SSE (required by chatgpt.com endpoint).
    ///   Prefer codex-spark there too, but fall back to GPT-5.4 with low
    ///   reasoning if spark is unavailable for the current account.
    async fn complete_openai(&self, system: &str, user_message: &str) -> Result<String> {
        let creds = auth::codex::load_credentials()
            .context("Failed to load OpenAI/Codex credentials for sidecar")?;

        let is_chatgpt_mode = !creds.refresh_token.is_empty() || creds.id_token.is_some();
        let base = if is_chatgpt_mode {
            CHATGPT_API_BASE
        } else {
            OPENAI_API_BASE
        };
        let url = format!("{}/{}", base.trim_end_matches('/'), OPENAI_RESPONSES_PATH);

        let (primary_model, resolved_reasoning) =
            resolve_openai_request_model(&self.model, is_chatgpt_mode);
        // An explicit reasoning override (e.g. benchmark judge pinning GPT-5.5
        // to no-thinking) always wins over the per-model default.
        let primary_reasoning: Option<&str> =
            self.reasoning_override.as_deref().or(resolved_reasoning);

        match self
            .complete_openai_with_model(
                &url,
                creds.access_token.as_str(),
                creds.account_id.as_deref(),
                is_chatgpt_mode,
                system,
                user_message,
                primary_model,
                primary_reasoning,
            )
            .await
        {
            Ok(text) => {
                crate::provider::clear_model_unavailable_for_account(primary_model);
                Ok(text)
            }
            Err(OpenAiSidecarError::Api { status, body })
                if is_chatgpt_mode
                    && primary_model == SIDECAR_OPENAI_MODEL
                    && is_openai_model_unavailable(status, &body) =>
            {
                let reason = classify_openai_model_unavailable(status, &body)
                    .unwrap_or_else(|| format!("model denied by OpenAI API (status {})", status));
                crate::provider::record_model_unavailable_for_account(primary_model, &reason);
                crate::logging::info(&format!(
                    "Sidecar fallback: {} unavailable in ChatGPT OAuth mode; retrying {} with reasoning={} ({})",
                    primary_model,
                    SIDECAR_OPENAI_OAUTH_FALLBACK_MODEL,
                    SIDECAR_OPENAI_OAUTH_FALLBACK_REASONING,
                    reason
                ));

                let fallback = self
                    .complete_openai_with_model(
                        &url,
                        creds.access_token.as_str(),
                        creds.account_id.as_deref(),
                        is_chatgpt_mode,
                        system,
                        user_message,
                        SIDECAR_OPENAI_OAUTH_FALLBACK_MODEL,
                        Some(SIDECAR_OPENAI_OAUTH_FALLBACK_REASONING),
                    )
                    .await;

                match fallback {
                    Ok(text) => {
                        crate::provider::clear_model_unavailable_for_account(
                            SIDECAR_OPENAI_OAUTH_FALLBACK_MODEL,
                        );
                        Ok(text)
                    }
                    Err(OpenAiSidecarError::Api { status, body })
                        if is_openai_model_unavailable(status, &body)
                            && auth::claude::load_credentials().is_ok() =>
                    {
                        // Both GPT-5.6 Luna and the gpt-5.4 OAuth
                        // fallback are denied for this ChatGPT account. Rather
                        // than dead-end the sidecar, fall back to Claude haiku
                        // when Claude credentials are available.
                        let reason = classify_openai_model_unavailable(status, &body)
                            .unwrap_or_else(|| {
                                format!("model denied by OpenAI API (status {})", status)
                            });
                        crate::provider::record_model_unavailable_for_account(
                            SIDECAR_OPENAI_OAUTH_FALLBACK_MODEL,
                            &reason,
                        );
                        crate::logging::info(&format!(
                            "Sidecar fallback: {} also unavailable in ChatGPT OAuth mode; falling back to Claude {} ({})",
                            SIDECAR_OPENAI_OAUTH_FALLBACK_MODEL, SIDECAR_CLAUDE_MODEL, reason
                        ));
                        let claude = Self {
                            client: self.client.clone(),
                            model: SIDECAR_CLAUDE_MODEL.to_string(),
                            max_tokens: self.max_tokens,
                            backend: SidecarBackend::Claude,
                            reasoning_override: None,
                        };
                        claude.complete_claude(system, user_message).await
                    }
                    Err(err) => Err(err.into_anyhow()),
                }
            }
            Err(err) => Err(err.into_anyhow()),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "OpenAI sidecar call needs endpoint, auth, account, mode, prompts, model, and reasoning effort"
    )]
    async fn complete_openai_with_model(
        &self,
        url: &str,
        access_token: &str,
        account_id: Option<&str>,
        is_chatgpt_mode: bool,
        system: &str,
        user_message: &str,
        model: &str,
        reasoning_effort: Option<&str>,
    ) -> std::result::Result<String, OpenAiSidecarError> {
        let request = build_openai_request(
            model,
            system,
            user_message,
            is_chatgpt_mode,
            reasoning_effort,
        );

        let mut builder = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", access_token))
            .header("Content-Type", "application/json");

        if is_chatgpt_mode {
            builder = builder.header("originator", OPENAI_ORIGINATOR);
            if let Some(account_id) = account_id {
                builder = builder.header("chatgpt-account-id", account_id);
            }
        }

        let response = builder
            .json(&request)
            .send()
            .await
            .context("Failed to send request to OpenAI API")
            .map_err(OpenAiSidecarError::other)?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(OpenAiSidecarError::Api { status, body });
        }

        if is_chatgpt_mode {
            collect_openai_sse_text(response)
                .await
                .map_err(OpenAiSidecarError::other)
        } else {
            let result: serde_json::Value = response
                .json()
                .await
                .context("Failed to parse OpenAI API response")
                .map_err(OpenAiSidecarError::other)?;
            extract_openai_response_text(&result).map_err(OpenAiSidecarError::other)
        }
    }

    /// Complete via Claude Messages API
    async fn complete_claude(&self, system: &str, user_message: &str) -> Result<String> {
        // Respect the runtime's pinned Anthropic credential mode. The main agent
        // may be running in API-key mode (`claude-api`), where the org forbids
        // OAuth and Anthropic returns a 403 "OAuth authentication is currently
        // not allowed for this organization." The sidecar previously hardcoded
        // the OAuth path, so memory calls (consensus judge, extraction) failed
        // even though the main agent worked fine on the API key. Mirror the main
        // provider's resolution: use the direct API key when API-key mode is
        // pinned (or when no OAuth credentials exist but a key does), and fall
        // back to the API key if an OAuth request is rejected as forbidden.
        if anthropic_sidecar_prefers_api_key()
            && let Ok(key) = crate::provider::anthropic::load_anthropic_api_key()
        {
            return self
                .complete_claude_api_key(system, user_message, &key)
                .await;
        }

        match self.complete_claude_oauth(system, user_message).await {
            Ok(text) => Ok(text),
            Err(err) if is_anthropic_oauth_forbidden(&err) => {
                match crate::provider::anthropic::load_anthropic_api_key() {
                    Ok(key) => {
                        crate::logging::info(
                            "Sidecar Claude: OAuth forbidden for organization; falling back to API key",
                        );
                        self.complete_claude_api_key(system, user_message, &key)
                            .await
                    }
                    Err(_) => Err(err),
                }
            }
            Err(err) => Err(err),
        }
    }

    /// OAuth (Claude subscription) completion path.
    async fn complete_claude_oauth(&self, system: &str, user_message: &str) -> Result<String> {
        let creds = auth::claude::load_credentials()
            .context("Failed to load Claude credentials for sidecar")?;

        let request = ClaudeMessagesRequest {
            model: &self.model,
            max_tokens: self.max_tokens,
            system: build_claude_system_param(system),
            messages: vec![ClaudeMessage {
                role: "user",
                content: user_message,
            }],
        };

        let response = crate::provider::anthropic::apply_oauth_attribution_headers(
            self.client
                .post(CLAUDE_API_URL)
                .header("Authorization", format!("Bearer {}", creds.access_token))
                .header("User-Agent", CLAUDE_CLI_USER_AGENT)
                .header("anthropic-version", "2023-06-01")
                .header("anthropic-beta", OAUTH_BETA_HEADERS)
                .header("content-type", "application/json")
                .json(&request),
            &crate::provider::anthropic::new_oauth_request_id(),
        )
        .send()
        .await
        .context("Failed to send request to Claude API")?;

        Self::parse_claude_response(response).await
    }

    /// Direct API-key completion path (`x-api-key`).
    ///
    /// Unlike the OAuth path this must NOT inject the "You are Claude Code"
    /// identity spoof: that block is only valid for the OAuth/subscription
    /// endpoint and a direct API key talks to the standard Messages API.
    async fn complete_claude_api_key(
        &self,
        system: &str,
        user_message: &str,
        api_key: &str,
    ) -> Result<String> {
        let request = ClaudeMessagesRequest {
            model: &self.model,
            max_tokens: self.max_tokens,
            system: build_claude_api_key_system_param(system),
            messages: vec![ClaudeMessage {
                role: "user",
                content: user_message,
            }],
        };

        let response = self
            .client
            .post(CLAUDE_API_KEY_URL)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", "prompt-caching-2024-07-31")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Claude API")?;

        Self::parse_claude_response(response).await
    }

    /// Shared response parsing for both Claude credential paths.
    async fn parse_claude_response(response: reqwest::Response) -> Result<String> {
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(SidecarHttpError {
                provider: "Claude",
                status,
                body: error_text,
            }
            .into());
        }

        let result: ClaudeMessagesResponse = response
            .json()
            .await
            .context("Failed to parse Claude API response")?;

        let text = result
            .content
            .into_iter()
            .filter_map(|block| {
                if let ClaudeContentBlock::Text { text } = block {
                    Some(text)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        Ok(text)
    }

    /// Check if a memory is relevant to the current context
    /// Returns (is_relevant, explanation)
    pub async fn check_relevance(
        &self,
        memory_content: &str,
        current_context: &str,
    ) -> Result<(bool, String)> {
        let system = r#"You are a memory relevance checker. Your job is to determine if a stored memory is relevant to the current context.

Respond in this exact format:
RELEVANT: yes/no
REASON: <brief explanation>

Be conservative - only say "yes" if the memory would actually be useful for the current task."#;

        let prompt = format!(
            "## Stored Memory\n{}\n\n## Current Context\n{}\n\nIs this memory relevant to the current context?",
            memory_content, current_context
        );

        let response = self.complete(system, &prompt).await?;

        // Parse response
        let mut is_relevant = false;
        for line in response.lines() {
            let line = line.trim();
            if line.len() >= 9 && line[..9].eq_ignore_ascii_case("relevant:") {
                let value = line[9..].trim();
                is_relevant = value.eq_ignore_ascii_case("yes") || value.starts_with("yes");
                break;
            }
        }
        let reason = response
            .lines()
            .find(|line| line.to_lowercase().starts_with("reason:"))
            .map(|line| line.trim_start_matches(|c: char| !c.is_alphabetic()).trim())
            .unwrap_or(&response)
            .to_string();

        Ok((is_relevant, reason))
    }

    /// Check if new information contradicts existing information
    /// Returns true if the two statements are contradictory
    pub async fn check_contradiction(
        &self,
        new_content: &str,
        existing_content: &str,
    ) -> Result<bool> {
        let system = "You are a contradiction detector. Given two statements, determine if the new information directly contradicts the existing information. Reply with exactly YES or NO.";

        let prompt = format!(
            "## Existing Information\n{}\n\n## New Information\n{}\n\nDoes the new information contradict the existing information?",
            existing_content, new_content
        );

        let response = self.complete(system, &prompt).await?;
        let trimmed = response.trim().to_uppercase();
        Ok(trimmed.starts_with("YES"))
    }

    /// Extract memories from a session transcript
    pub async fn extract_memories(&self, transcript: &str) -> Result<Vec<ExtractedMemory>> {
        self.extract_memories_with_existing(transcript, &[]).await
    }

    /// Extract memories from a session transcript, aware of what's already stored.
    pub async fn extract_memories_with_existing(
        &self,
        transcript: &str,
        existing: &[String],
    ) -> Result<Vec<ExtractedMemory>> {
        let mut system = String::from(
            r#"You are a memory extraction assistant. Extract important NEW learnings from the conversation that should be remembered for future sessions.

Categories (use EXACTLY one of these):
- fact: Technical facts about the codebase, architecture, patterns, dependencies, tools, environment
- preference: User preferences, workflow habits, UX expectations, coding style, conventions, how they want the assistant to behave
- correction: Mistakes that were corrected, bugs found and fixed, wrong assumptions, things the user corrected
- entity: Named entities worth tracking - people, projects, services, repos, teams

Categorization rules:
- If it describes what the USER WANTS or HOW THEY LIKE THINGS, it is "preference", not "fact"
- If it describes a BUG FIX or MISTAKE, it is "correction", not "fact"
- "fact" is for objective technical information about code/systems, not user behavior

IMPORTANT - Do NOT extract:
- Transient debugging details, compile errors, or intermediate build steps
- Specific commit hashes, git operations, or "changes were committed/pushed" details
- Line-by-line code changes like "X was updated to Y in file Z" - these belong in git history, not memory
- Self-evident project context (e.g., the project name, repo URL, language) that is already in the system prompt
- Redundant variations of information already known (check the "Already known" list carefully)

Quality bar: Only extract information that would ACTUALLY BE USEFUL if recalled in a future session on a different topic. Ask: "Would a developer benefit from knowing this weeks from now?"

For each memory, output in this format (one per line):
CATEGORY|CONTENT|TRUST

Where:
- CATEGORY is one of: fact, preference, correction, entity
- CONTENT is a concise statement (1-2 sentences max, under 200 characters preferred)
- TRUST is one of: high (user stated), medium (observed), low (inferred)

Output ONLY the formatted lines, no other text. If no NEW memories worth extracting, output nothing."#,
        );

        if !existing.is_empty() {
            system.push_str("\n\nAlready known (do NOT re-extract these or close paraphrases):\n");
            for mem in existing.iter().take(80) {
                system.push_str("- ");
                system.push_str(crate::util::truncate_str(mem, 150));
                system.push('\n');
            }
        }

        let response = self.complete(&system, transcript).await?;

        let memories = response
            .lines()
            .filter(|line| line.contains('|'))
            .filter_map(|line| {
                let parts: Vec<&str> = line.split('|').collect();
                if parts.len() >= 3 {
                    Some(ExtractedMemory {
                        category: parts[0].trim().to_lowercase(),
                        content: parts[1].trim().to_string(),
                        trust: parts[2].trim().to_lowercase(),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(memories)
    }
}

impl Default for Sidecar {
    fn default() -> Self {
        Self::new()
    }
}

/// The public model constant for backward compatibility in tests.
#[cfg(test)]
pub const SIDECAR_FAST_MODEL: &str = SIDECAR_OPENAI_MODEL;

fn resolve_openai_request_model(
    preferred_model: &str,
    is_chatgpt_mode: bool,
) -> (&str, Option<&'static str>) {
    if preferred_model != SIDECAR_OPENAI_MODEL {
        return (preferred_model, None);
    }

    match (
        is_chatgpt_mode,
        crate::provider::is_model_available_for_account(SIDECAR_OPENAI_MODEL),
    ) {
        (true, Some(false)) => (
            SIDECAR_OPENAI_OAUTH_FALLBACK_MODEL,
            Some(SIDECAR_OPENAI_OAUTH_FALLBACK_REASONING),
        ),
        _ => (SIDECAR_OPENAI_MODEL, Some(SIDECAR_OPENAI_REASONING)),
    }
}

fn build_openai_request(
    model: &str,
    system: &str,
    user_message: &str,
    stream: bool,
    reasoning_effort: Option<&str>,
) -> serde_json::Value {
    let mut instructions = String::new();
    if !system.is_empty() {
        instructions.push_str(system);
    }

    let mut request = serde_json::json!({
        "model": model,
        "instructions": instructions,
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": user_message,
            }],
        }],
        "stream": stream,
        "store": false,
    });

    if let Some(effort) = reasoning_effort {
        request["reasoning"] = serde_json::json!({ "effort": effort });
    }

    request
}

fn classify_openai_model_unavailable(status: StatusCode, body: &str) -> Option<String> {
    let lower = body.to_ascii_lowercase();
    let mentions_model = lower.contains("model")
        || lower.contains("slug")
        || lower.contains("engine")
        || lower.contains("deployment");
    let unavailable = lower.contains("not available")
        || lower.contains("unavailable")
        || lower.contains("does not have access")
        || lower.contains("not enabled")
        || lower.contains("not found")
        || lower.contains("unknown model")
        || lower.contains("unsupported model")
        || lower.contains("invalid model");

    if !mentions_model || !unavailable {
        return None;
    }

    if matches!(
        status,
        StatusCode::NOT_FOUND
            | StatusCode::FORBIDDEN
            | StatusCode::BAD_REQUEST
            | StatusCode::UNPROCESSABLE_ENTITY
    ) {
        let trimmed = body.trim();
        return Some(if trimmed.is_empty() {
            format!("model denied by OpenAI API (status {})", status)
        } else {
            format!(
                "model denied by OpenAI API (status {}): {}",
                status, trimmed
            )
        });
    }

    None
}

fn is_openai_model_unavailable(status: StatusCode, body: &str) -> bool {
    classify_openai_model_unavailable(status, body).is_some()
}

enum OpenAiSidecarError {
    Api { status: StatusCode, body: String },
    Other(anyhow::Error),
}

impl OpenAiSidecarError {
    fn other(err: anyhow::Error) -> Self {
        Self::Other(err)
    }

    fn into_anyhow(self) -> anyhow::Error {
        match self {
            Self::Api { status, body } => SidecarHttpError {
                provider: "OpenAI",
                status,
                body,
            }
            .into(),
            Self::Other(err) => err,
        }
    }
}

/// A memory extracted by the sidecar
#[derive(Debug, Clone)]
pub struct ExtractedMemory {
    pub category: String,
    pub content: String,
    pub trust: String,
}

/// Collect text from an OpenAI Responses API SSE stream.
///
/// Parses `data: <json>` lines and accumulates text deltas from
/// `response.output_text.delta` events, stopping on completion/done.
async fn collect_openai_sse_text(response: reqwest::Response) -> Result<String> {
    use futures::StreamExt;
    let mut stream = response.bytes_stream();
    let mut text = String::new();
    let mut buf = String::new();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.context("Error reading SSE stream")?;
        buf.push_str(&String::from_utf8_lossy(&bytes));

        // Process all complete lines in the buffer
        while let Some(newline_pos) = buf.find('\n') {
            let line = buf[..newline_pos].trim_end_matches('\r').to_string();
            buf = buf[newline_pos + 1..].to_string();

            if let Some(data) = crate::util::sse_data_line(&line) {
                if data == "[DONE]" {
                    return Ok(text);
                }
                if let Ok(event) = serde_json::from_str::<SseEvent>(data) {
                    match event.kind.as_str() {
                        "response.output_text.delta" => {
                            if let Some(delta) = event.delta {
                                text.push_str(&delta);
                            }
                        }
                        "response.completed" | "response.incomplete" => {
                            return Ok(text);
                        }
                        "response.failed" | "error" => {
                            let msg = event
                                .error
                                .as_ref()
                                .and_then(|e| e.as_str())
                                .unwrap_or("unknown error");
                            anyhow::bail!("OpenAI SSE error: {}", msg);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    Ok(text)
}

/// Extract text from a non-streaming OpenAI Responses API JSON response.
fn extract_openai_response_text(result: &serde_json::Value) -> Result<String> {
    let mut text = String::new();
    if let Some(output) = result.get("output").and_then(|v| v.as_array()) {
        for item in output {
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if item_type == "message"
                && let Some(content) = item.get("content").and_then(|v| v.as_array())
            {
                for block in content {
                    let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if (block_type == "output_text" || block_type == "text")
                        && let Some(t) = block.get("text").and_then(|v| v.as_str())
                    {
                        text.push_str(t);
                    }
                }
            }
        }
    }
    Ok(text)
}

#[derive(Deserialize)]
struct SseEvent {
    #[serde(rename = "type")]
    kind: String,
    delta: Option<String>,
    error: Option<serde_json::Value>,
}

// Claude API types

#[derive(Serialize)]
struct ClaudeMessagesRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<ClaudeApiSystem<'a>>,
    messages: Vec<ClaudeMessage<'a>>,
}

#[derive(Serialize)]
struct ClaudeMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ClaudeApiSystem<'a> {
    Blocks(Vec<ClaudeApiSystemBlock<'a>>),
}

#[derive(Serialize)]
struct ClaudeApiSystemBlock<'a> {
    #[serde(rename = "type")]
    block_type: &'static str,
    text: &'a str,
}

fn build_claude_system_param(system: &str) -> Option<ClaudeApiSystem<'_>> {
    let mut blocks = Vec::new();
    blocks.push(ClaudeApiSystemBlock {
        block_type: "text",
        text: CLAUDE_CODE_IDENTITY,
    });
    blocks.push(ClaudeApiSystemBlock {
        block_type: "text",
        text: CLAUDE_CODE_JCODE_NOTICE,
    });
    if !system.is_empty() {
        blocks.push(ClaudeApiSystemBlock {
            block_type: "text",
            text: system,
        });
    }
    Some(ClaudeApiSystem::Blocks(blocks))
}

/// Build the system param for the direct API-key path.
///
/// The "You are Claude Code" identity spoof and jcode notice are only valid
/// for the OAuth/subscription endpoint; a direct API key talks to the standard
/// Messages API and must not impersonate the official CLI. So this only carries
/// the caller's own system prompt (if any).
fn build_claude_api_key_system_param(system: &str) -> Option<ClaudeApiSystem<'_>> {
    if system.is_empty() {
        return None;
    }
    Some(ClaudeApiSystem::Blocks(vec![ClaudeApiSystemBlock {
        block_type: "text",
        text: system,
    }]))
}

/// Whether the sidecar's Claude backend should use the direct API key rather
/// than OAuth. True when the runtime is pinned to Anthropic API-key mode
/// (`claude-api`), or when no OAuth credentials are present at all. Mirrors the
/// main provider's resolution so memory features authenticate the same way the
/// agent does.
fn anthropic_sidecar_prefers_api_key() -> bool {
    match jcode_provider_core::runtime_env_pinned_mode(
        jcode_provider_core::DualAuthProvider::Anthropic,
    ) {
        Some(jcode_provider_core::AuthMode::ApiKey) => true,
        Some(jcode_provider_core::AuthMode::Oauth) => false,
        None => auth::claude::load_credentials().is_err(),
    }
}

/// Recognize the Anthropic "OAuth not allowed for this organization" 403 so the
/// sidecar can transparently fall back to the API key.
fn is_anthropic_oauth_forbidden(err: &anyhow::Error) -> bool {
    let msg = err.to_string();
    msg.contains("403")
        && (msg.contains("OAuth authentication is currently not allowed")
            || msg.contains("permission_error"))
}

#[derive(Deserialize)]
struct ClaudeMessagesResponse {
    content: Vec<ClaudeContentBlock>,
    #[serde(rename = "usage")]
    _usage: Option<ClaudeUsage>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ClaudeContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct ClaudeUsage {
    #[serde(rename = "input_tokens")]
    _input_tokens: u32,
    #[serde(rename = "output_tokens")]
    _output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::codex;
    use std::ffi::OsString;

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set_path(key: &'static str, value: &std::path::Path) -> Self {
            let previous = std::env::var_os(key);
            crate::env::set_var(key, value);
            Self { key, previous }
        }

        fn unset(key: &'static str) -> Self {
            let previous = std::env::var_os(key);
            crate::env::remove_var(key);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(previous) = &self.previous {
                crate::env::set_var(self.key, previous);
            } else {
                crate::env::remove_var(self.key);
            }
        }
    }

    #[test]
    fn test_sidecar_fast_model() {
        assert_eq!(SIDECAR_FAST_MODEL, "gpt-5.6-luna");
        assert_eq!(SIDECAR_CLAUDE_MODEL, "claude-haiku-4-5-20251001");
    }

    #[test]
    fn is_namespaced_model_recognises_openrouter_style() {
        // OpenRouter shape: `vendor/family`
        assert!(is_namespaced_model("anthropic/claude-sonnet-4"));
        assert!(is_namespaced_model("openai/gpt-5.5"));
        assert!(is_namespaced_model("google/gemini-3-flash"));
        assert!(is_namespaced_model("meta-llama/llama-3.3-70b-instruct"));

        // Direct endpoints' bare ids must NOT be considered namespaced, even
        // when they have dashes or dots.
        assert!(!is_namespaced_model("deepseek-v4-flash"));
        assert!(!is_namespaced_model("MiniMax-M3"));
        assert!(!is_namespaced_model("moonshot-v1-8k"));
        assert!(!is_namespaced_model("claude-haiku-4-5-20251001"));

        // Some direct endpoints use slashes for their own naming (Kimi).
        // We don't reject these because the endpoint check will.
        assert!(is_namespaced_model("moonshotai/kimi-k2.5"));

        // Malformed inputs
        assert!(!is_namespaced_model(""));
        assert!(!is_namespaced_model("/no-vendor"));
        assert!(!is_namespaced_model("has\\backslash/model"));
    }

    #[test]
    fn is_direct_openai_compatible_runtime_matches_known_providers() {
        // Direct OpenAI-compatible providers.
        assert!(is_direct_openai_compatible_runtime("deepseek"));
        assert!(is_direct_openai_compatible_runtime("openai-compatible:deepseek"));
        assert!(is_direct_openai_compatible_runtime("DEEPSEEK"));
        assert!(is_direct_openai_compatible_runtime("moonshotai"));
        assert!(is_direct_openai_compatible_runtime("openai-compatible:moonshot-v1-8k"));
        assert!(is_direct_openai_compatible_runtime("kimi"));
        assert!(is_direct_openai_compatible_runtime("openai-compatible"));
        assert!(is_direct_openai_compatible_runtime("openai-compatible:my-custom-thing"));

        // OpenRouter-shaped providers (NOT direct)
        assert!(!is_direct_openai_compatible_runtime("openrouter"));
        assert!(!is_direct_openai_compatible_runtime("anthropic"));
        assert!(!is_direct_openai_compatible_runtime("openai")); // openai proper is OAuth/API-key, not direct
        assert!(!is_direct_openai_compatible_runtime("claude"));
        assert!(!is_direct_openai_compatible_runtime("copilot"));
        assert!(!is_direct_openai_compatible_runtime("gemini"));
        assert!(!is_direct_openai_compatible_runtime("bedrock"));
        assert!(!is_direct_openai_compatible_runtime("cursor"));
    }

    #[test]
    fn provider_default_model_for_returns_known_defaults() {
        assert_eq!(provider_default_model_for("deepseek"), Some("deepseek-chat"));
        assert_eq!(
            provider_default_model_for("openai-compatible:deepseek"),
            Some("deepseek-chat")
        );
        assert_eq!(
            provider_default_model_for("moonshot"),
            Some("moonshot-v1-8k")
        );
        assert_eq!(provider_default_model_for("kimi"), Some("moonshot-v1-8k"));
        assert_eq!(
            provider_default_model_for("minimaxi"),
            Some("MiniMax-M3")
        );
        assert_eq!(
            provider_default_model_for("openai-compatible:my-random-thing"),
            None,
            "generic openai-compatible runs don't know which model works"
        );
        assert_eq!(provider_default_model_for("openrouter"), None);
        assert_eq!(provider_default_model_for("anthropic"), None);
    }

    /// Regression test for the 2026-08-14 memory incident: the sidecar was
    /// dispatching to a DeepSeek direct endpoint with an OpenRouter-style
    /// model name (`anthropic/claude-sonnet-4`), causing HTTP 400 "Model Not
    /// Exist" and `all_judges_failed` for every memory rerank.
    #[test]
    fn safe_model_for_provider_rewrites_namespaced_model_on_direct_endpoint() {
        struct StubProvider {
            name: &'static str,
            model: &'static str,
        }
        #[async_trait::async_trait]
        impl crate::provider::Provider for StubProvider {
            async fn complete(
                &self,
                _messages: &[crate::message::Message],
                _tools: &[crate::message::ToolDefinition],
                _system: &str,
                _resume_session_id: Option<&str>,
            ) -> Result<crate::provider::EventStream> {
                unreachable!("safe_model_for_provider does not call complete")
            }
            fn name(&self) -> &str {
                self.name
            }
            fn model(&self) -> String {
                self.model.to_string()
            }
            fn fork(&self) -> std::sync::Arc<dyn crate::provider::Provider> {
                std::sync::Arc::new(StubProvider {
                    name: self.name,
                    model: self.model,
                })
            }
        }

        // Bug case: DeepSeek direct provider with OpenRouter-style model.
        let buggy = StubProvider {
            name: "deepseek",
            model: "anthropic/claude-sonnet-4",
        };
        let safe = safe_model_for_provider(&buggy);
        assert_eq!(
            safe, "deepseek-chat",
            "sidecar must rewrite the OpenRouter-style model to a DeepSeek-safe default"
        );

        // Already-correct case: bare model on a direct endpoint.
        let ok = StubProvider {
            name: "deepseek",
            model: "deepseek-v4-flash",
        };
        assert_eq!(safe_model_for_provider(&ok), "deepseek-v4-flash");

        // OpenRouter-shaped namespaced model on OpenRouter — must stay.
        let or = StubProvider {
            name: "openrouter",
            model: "anthropic/claude-sonnet-4",
        };
        assert_eq!(safe_model_for_provider(&or), "anthropic/claude-sonnet-4");

        // Bare model on a non-direct provider — must stay.
        let claude = StubProvider {
            name: "claude",
            model: "claude-haiku-4-5-20251001",
        };
        assert_eq!(safe_model_for_provider(&claude), "claude-haiku-4-5-20251001");
    }

    #[test]
    fn provider_model_is_compatible_returns_false_for_buggy_pairs() {
        struct StubProvider {
            name: &'static str,
            model: &'static str,
        }
        #[async_trait::async_trait]
        impl crate::provider::Provider for StubProvider {
            async fn complete(
                &self,
                _messages: &[crate::message::Message],
                _tools: &[crate::message::ToolDefinition],
                _system: &str,
                _resume_session_id: Option<&str>,
            ) -> Result<crate::provider::EventStream> {
                unreachable!()
            }
            fn name(&self) -> &str {
                self.name
            }
            fn model(&self) -> String {
                self.model.to_string()
            }
            fn fork(&self) -> std::sync::Arc<dyn crate::provider::Provider> {
                std::sync::Arc::new(StubProvider {
                    name: self.name,
                    model: self.model,
                })
            }
        }

        let buggy = StubProvider {
            name: "deepseek",
            model: "anthropic/claude-sonnet-4",
        };
        assert!(!Sidecar::provider_model_is_compatible(&buggy));

        let ok = StubProvider {
            name: "deepseek",
            model: "deepseek-v4-flash",
        };
        assert!(Sidecar::provider_model_is_compatible(&ok));
    }

    #[test]
    fn sidecar_http_error_classifies_permanent_client_failures() {
        for status in [
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::NOT_FOUND,
        ] {
            let error: anyhow::Error = SidecarHttpError {
                provider: "test",
                status,
                body: "failure".to_string(),
            }
            .into();
            assert_eq!(classify_error(&error), SidecarErrorKind::Permanent);
        }
    }

    #[test]
    fn sidecar_http_error_classifies_retryable_failures() {
        for status in [StatusCode::TOO_MANY_REQUESTS, StatusCode::BAD_GATEWAY] {
            let error: anyhow::Error = SidecarHttpError {
                provider: "test",
                status,
                body: "failure".to_string(),
            }
            .into();
            assert_eq!(classify_error(&error), SidecarErrorKind::Transient);
        }
        assert_eq!(
            classify_error(&anyhow::anyhow!("connection reset")),
            SidecarErrorKind::Transient
        );
    }

    #[test]
    fn test_backend_selection_prefers_openai() {
        // Make backend selection deterministic by isolating credentials.
        let _guard = crate::storage::lock_test_env();
        let temp = tempfile::TempDir::new().expect("create temp jcode home");
        let _home = EnvVarGuard::set_path("JCODE_HOME", temp.path());
        let _openai = EnvVarGuard::unset("OPENAI_API_KEY");

        codex::upsert_account_from_tokens("openai-1", "sk-test-key-123", "", None, None)
            .expect("write OpenAI test auth");
        crate::auth::claude::upsert_account(crate::auth::claude::AnthropicAccount {
            label: "claude-1".to_string(),
            access: "claude-access".to_string(),
            refresh: "claude-refresh".to_string(),
            expires: 4_102_444_800_000,
            email: None,
            scopes: Vec::new(),
            subscription_type: None,
        })
        .expect("write Claude test auth");

        let sidecar = Sidecar::with_configured_model(None);
        assert_eq!(sidecar.backend, SidecarBackend::OpenAI);
        assert_eq!(sidecar.model, SIDECAR_OPENAI_MODEL);
        codex::set_active_account_override(None);
        crate::auth::claude::set_active_account_override(None);
    }

    #[test]
    fn test_chatgpt_oauth_uses_luna_with_no_reasoning_when_available() {
        let _guard = crate::storage::lock_test_env();
        let temp = tempfile::TempDir::new().expect("create temp jcode home");
        let _home = EnvVarGuard::set_path("JCODE_HOME", temp.path());
        codex::set_active_account_override(Some("openai-1".to_string()));
        crate::provider::clear_all_model_unavailability_for_account();
        crate::provider::populate_account_models(vec![
            SIDECAR_OPENAI_MODEL.to_string(),
            SIDECAR_OPENAI_OAUTH_FALLBACK_MODEL.to_string(),
        ]);

        let (model, reasoning) = resolve_openai_request_model(SIDECAR_OPENAI_MODEL, true);
        assert_eq!(model, SIDECAR_OPENAI_MODEL);
        assert_eq!(reasoning, Some(SIDECAR_OPENAI_REASONING));

        codex::set_active_account_override(None);
    }

    #[test]
    fn test_chatgpt_oauth_falls_back_to_gpt_5_4_low_when_luna_unavailable() {
        let _guard = crate::storage::lock_test_env();
        let temp = tempfile::TempDir::new().expect("create temp jcode home");
        let _home = EnvVarGuard::set_path("JCODE_HOME", temp.path());
        codex::set_active_account_override(Some("openai-1".to_string()));
        crate::provider::clear_all_model_unavailability_for_account();
        crate::provider::populate_account_models(vec![
            SIDECAR_OPENAI_OAUTH_FALLBACK_MODEL.to_string(),
        ]);

        let (model, reasoning) = resolve_openai_request_model(SIDECAR_OPENAI_MODEL, true);
        assert_eq!(model, SIDECAR_OPENAI_OAUTH_FALLBACK_MODEL);
        assert_eq!(reasoning, Some(SIDECAR_OPENAI_OAUTH_FALLBACK_REASONING));

        codex::set_active_account_override(None);
    }

    #[test]
    fn test_build_openai_request_uses_configured_default_and_fallback_reasoning() {
        let request = build_openai_request(
            SIDECAR_OPENAI_OAUTH_FALLBACK_MODEL,
            "system",
            "hello",
            true,
            Some(SIDECAR_OPENAI_OAUTH_FALLBACK_REASONING),
        );
        assert_eq!(request["model"], SIDECAR_OPENAI_OAUTH_FALLBACK_MODEL);
        assert_eq!(
            request["reasoning"],
            serde_json::json!({"effort": SIDECAR_OPENAI_OAUTH_FALLBACK_REASONING})
        );

        let luna_request = build_openai_request(
            SIDECAR_OPENAI_MODEL,
            "system",
            "hello",
            true,
            Some(SIDECAR_OPENAI_REASONING),
        );
        assert_eq!(luna_request["model"], SIDECAR_OPENAI_MODEL);
        assert_eq!(
            luna_request["reasoning"],
            serde_json::json!({"effort": SIDECAR_OPENAI_REASONING})
        );
    }

    #[test]
    fn test_openai_api_key_mode_uses_luna_with_no_reasoning() {
        let (model, reasoning) = resolve_openai_request_model(SIDECAR_OPENAI_MODEL, false);
        assert_eq!(model, SIDECAR_OPENAI_MODEL);
        assert_eq!(reasoning, Some(SIDECAR_OPENAI_REASONING));
    }

    // ---- Provider-backed sidecar (works on ALL providers) -------------------

    /// Minimal provider stub that echoes a fixed reply for `complete`, so the
    /// default `complete_simple` path the sidecar uses can be exercised without
    /// network access. Stands in for any of the 8 real providers.
    struct StubProvider {
        name: &'static str,
        reply: String,
    }

    #[async_trait::async_trait]
    impl crate::provider::Provider for StubProvider {
        async fn complete(
            &self,
            _messages: &[crate::message::Message],
            _tools: &[crate::message::ToolDefinition],
            _system: &str,
            _resume_session_id: Option<&str>,
        ) -> Result<crate::provider::EventStream> {
            let reply = self.reply.clone();
            let stream = futures::stream::once(async move {
                Ok(jcode_message_types::StreamEvent::TextDelta(reply))
            });
            Ok(Box::pin(stream))
        }

        fn name(&self) -> &str {
            self.name
        }

        fn model(&self) -> String {
            format!("{}-model", self.name)
        }

        fn fork(&self) -> std::sync::Arc<dyn crate::provider::Provider> {
            std::sync::Arc::new(StubProvider {
                name: self.name,
                reply: self.reply.clone(),
            })
        }
    }

    /// With NO OpenAI/Claude credentials, the sidecar must select the live
    /// agent provider (the universal path) instead of failing. This is the core
    /// guarantee that memory features work on every provider, not just two.
    #[test]
    fn sidecar_uses_active_provider_when_no_oauth_creds() {
        let _guard = crate::storage::lock_test_env();
        let temp = tempfile::TempDir::new().expect("create temp jcode home");
        let _home = EnvVarGuard::set_path("JCODE_HOME", temp.path());
        let _openai = EnvVarGuard::unset("OPENAI_API_KEY");

        // Simulate running on a non-OpenAI/Claude provider (e.g. Gemini).
        crate::provider::set_active_provider(std::sync::Arc::new(StubProvider {
            name: "gemini",
            reply: "[2,1]".to_string(),
        }));

        let sidecar = Sidecar::with_configured_model(None);
        assert_eq!(
            sidecar.backend_name(),
            "provider",
            "with no OAuth creds, the sidecar must route through the active provider"
        );

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let out = rt
            .block_on(sidecar.complete("rank these", "1. a\n2. b"))
            .expect("provider-backed completion should succeed");
        assert_eq!(out, "[2,1]", "sidecar must return the provider's text");
    }

    /// Every provider jcode supports should drive the sidecar end-to-end via the
    /// universal `complete_simple` path. We iterate over each provider label to
    /// make the "works for ALL providers" guarantee explicit and regression-proof.
    #[test]
    fn sidecar_provider_path_works_for_all_providers() {
        let _guard = crate::storage::lock_test_env();
        let temp = tempfile::TempDir::new().expect("create temp jcode home");
        let _home = EnvVarGuard::set_path("JCODE_HOME", temp.path());
        let _openai = EnvVarGuard::unset("OPENAI_API_KEY");

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        for provider in [
            "claude",
            "openai",
            "copilot",
            "antigravity",
            "gemini",
            "cursor",
            "bedrock",
            "openrouter",
        ] {
            crate::provider::set_active_provider(std::sync::Arc::new(StubProvider {
                name: provider,
                reply: "[1]".to_string(),
            }));
            let sidecar = Sidecar::with_configured_model(None);
            assert_eq!(
                sidecar.backend_name(),
                "provider",
                "{provider}: sidecar should use the provider path with no OAuth creds"
            );
            let out = rt
                .block_on(sidecar.complete("sys", "user"))
                .unwrap_or_else(|e| panic!("{provider}: provider-backed completion failed: {e}"));
            assert_eq!(out, "[1]", "{provider}: sidecar must echo provider output");
        }
    }

    #[test]
    fn test_is_anthropic_oauth_forbidden() {
        // The exact error string the sidecar surfaces from a forbidden OAuth org.
        let forbidden = anyhow::anyhow!(
            "Claude API error (403 Forbidden): {{\"type\":\"error\",\"error\":{{\"type\":\"permission_error\",\"message\":\"OAuth authentication is currently not allowed for this organization.\"}}}}"
        );
        assert!(is_anthropic_oauth_forbidden(&forbidden));

        // Unrelated failures must NOT trigger the API-key fallback.
        assert!(!is_anthropic_oauth_forbidden(&anyhow::anyhow!(
            "Claude API error (401 Unauthorized): bad token"
        )));
        assert!(!is_anthropic_oauth_forbidden(&anyhow::anyhow!(
            "Failed to send request to Claude API"
        )));
        // A 403 from a permission_error (the organization gate) still counts even
        // if the human-readable message phrasing changes slightly.
        assert!(is_anthropic_oauth_forbidden(&anyhow::anyhow!(
            "Claude API error (403 Forbidden): {{\"error\":{{\"type\":\"permission_error\"}}}}"
        )));
    }

    #[test]
    fn test_build_claude_api_key_system_param_omits_identity_spoof() {
        // API-key path must NOT impersonate the official Claude Code CLI.
        let none = build_claude_api_key_system_param("");
        assert!(none.is_none(), "empty system => no system param");

        let ClaudeApiSystem::Blocks(blocks) =
            build_claude_api_key_system_param("be terse").expect("system present");
        assert_eq!(blocks.len(), 1, "only the caller's system prompt is sent");
        assert_eq!(blocks[0].text, "be terse");

        // The OAuth builder, by contrast, injects the Claude Code identity spoof.
        let ClaudeApiSystem::Blocks(oauth_blocks) =
            build_claude_system_param("be terse").expect("oauth system present");
        assert!(
            oauth_blocks.iter().any(|b| b.text == CLAUDE_CODE_IDENTITY),
            "oauth path keeps the identity block"
        );
    }

    #[test]
    fn test_anthropic_sidecar_prefers_api_key_respects_pinned_mode() {
        // Pinning the runtime to API-key mode must make the sidecar prefer the key.
        let _g =
            EnvVarGuard::set_path("JCODE_RUNTIME_PROVIDER", std::path::Path::new("claude-api"));
        assert!(
            anthropic_sidecar_prefers_api_key(),
            "claude-api runtime => prefer API key"
        );

        // Pinning to OAuth mode must NOT prefer the key.
        let _g2 = EnvVarGuard::set_path("JCODE_RUNTIME_PROVIDER", std::path::Path::new("claude"));
        assert!(
            !anthropic_sidecar_prefers_api_key(),
            "claude (oauth) runtime => do not force API key"
        );
    }
}
