//! Chrome DevTools Protocol (CDP) browser backend.
//!
//! This provider drives Google Chrome (or a Chromium-family browser) through the
//! Chrome DevTools Protocol over a WebSocket. It supports two operation modes:
//!
//! - **headless** (`mode = "headless"`): Chrome runs with `--headless=new` and
//!   no visible window. Best for unattended automation.
//! - **visible** (`mode = "visible"`): Chrome runs with a real window on the
//!   user's display so the user can watch, interact with, and collaborate on the
//!   browser alongside the agent.
//!
//! The provider launches its own Chrome instance with a per-session
//! user-data-dir and `--remote-debugging-port`, then talks to it over CDP. This
//! mirrors the Firefox Agent Bridge backend but requires no Firefox extension or
//! native messaging host.

use super::*;
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD;
use futures_util::{SinkExt, StreamExt};
use serde_json::{Map, Value, json};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

/// Static provider for Chrome via CDP.
pub struct ChromeCdpProvider;

#[async_trait]
impl BrowserProvider for ChromeCdpProvider {
    fn id(&self) -> &'static str {
        "chrome_cdp"
    }

    fn supported_browsers(&self) -> &'static [&'static str] {
        &["auto", "chrome"]
    }

    async fn status(&self, ctx: &ToolContext) -> Result<ToolOutput> {
        chrome_status(self, ctx).await
    }

    async fn setup(&self) -> Result<ToolOutput> {
        Ok(attach_browser_metadata(
            chrome_setup().await?,
            self.id(),
            "chrome",
        ))
    }

    async fn ensure_ready(&self) -> Result<Option<String>> {
        ensure_chrome_ready().await
    }

    async fn execute(
        &self,
        action: &str,
        input: &BrowserInput,
        ctx: &ToolContext,
    ) -> Result<ToolOutput> {
        Ok(attach_browser_metadata(
            execute_chrome_action(self, action, input, ctx).await?,
            self.id(),
            "chrome",
        ))
    }
}

static NEXT_CDP_ID: AtomicU64 = AtomicU64::new(1);

/// Resolve which Chrome/Chromium binary to use.
fn chrome_binary() -> Result<PathBuf> {
    for candidate in chrome_binary_candidates() {
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    // Fall back to `chrome` / `google-chrome` on PATH.
    for name in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"] {
        if let Some(found) = which_on_path(name) {
            return Ok(found);
        }
    }
    bail!(
        "Chrome/Chromium binary not found. Install Google Chrome or Chromium, or set JCODE_CHROME_PATH."
    )
}

fn chrome_binary_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var("JCODE_CHROME_PATH") {
        out.push(PathBuf::from(p));
    }
    #[cfg(target_os = "linux")]
    {
        for p in [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/opt/google/chrome/chrome",
        ] {
            out.push(PathBuf::from(p));
        }
    }
    #[cfg(target_os = "macos")]
    {
        out.push(PathBuf::from(
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        ));
        out.push(PathBuf::from(
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ));
    }
    #[cfg(target_os = "windows")]
    {
        for p in [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ] {
            out.push(PathBuf::from(p));
        }
    }
    out
}

fn which_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let cand = dir.join(name);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

/// Random-ish free port for the CDP debugger.
fn pick_debug_port() -> u16 {
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").ok();
    match listener {
        Some(l) => l.local_addr().map(|a| a.port()).unwrap_or(9333),
        None => 9333,
    }
}

/// Map a jcode session id to a stable Chrome launch identity.
fn chrome_session_id(session_id: &str) -> String {
    let sanitized: String = session_id
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .take(64)
        .collect();
    if sanitized.is_empty() {
        "default".to_string()
    } else {
        sanitized
    }
}

fn chrome_pid_path(session_id: &str) -> PathBuf {
    crate::storage::runtime_dir().join(format!("chrome-session-{}.pid", chrome_session_id(session_id)))
}

fn chrome_marker_path(session_id: &str) -> PathBuf {
    crate::storage::runtime_dir().join(format!("chrome-session-{}.json", chrome_session_id(session_id)))
}

fn is_chrome_session_alive(session_id: &str) -> bool {
    let pid_path = chrome_pid_path(session_id);
    if let Ok(pid_str) = std::fs::read_to_string(&pid_path) {
        if let Ok(pid) = pid_str.trim().parse::<u32>() {
            return crate::platform::is_process_running(pid);
        }
    }
    false
}

fn write_session_marker(session_id: &str, port: u16, user_data_dir: &PathBuf) {
    let marker = chrome_marker_path(session_id);
    let _ = std::fs::create_dir_all(marker.parent().unwrap_or(std::path::Path::new(".")));
    let body = json!({
        "port": port,
        "userDataDir": user_data_dir.to_string_lossy().to_string(),
        "launchedAt": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
    });
    let _ = std::fs::write(&marker, serde_json::to_string(&body).unwrap_or_default());
}

fn write_pid(session_id: &str, pid: u32) {
    let pid_path = chrome_pid_path(session_id);
    let _ = std::fs::create_dir_all(pid_path.parent().unwrap_or(std::path::Path::new(".")));
    let _ = std::fs::write(&pid_path, pid.to_string());
}

/// Determine effective launch mode: "headless" or "visible".
fn effective_mode(input: &BrowserInput) -> String {
    match input.mode.as_deref() {
        Some(m) if m.eq_ignore_ascii_case("visible") => "visible".to_string(),
        Some(m) if m.eq_ignore_ascii_case("headless") => "headless".to_string(),
        _ => "headless".to_string(),
    }
}

/// Launch (or reuse) a Chrome instance for the session, returning its debug port.
async fn ensure_chrome_launched(session_id: &str, mode: &str) -> Result<u16> {
    let sid = chrome_session_id(session_id);

    // Reuse an already-running session.
    if is_chrome_session_alive(session_id)
        && let Some(port) = launch_session_marker(&sid)
    {
        return Ok(port);
    }

    let bin = chrome_binary()?;
    let user_data_dir = crate::storage::runtime_dir().join(format!("chrome-profile-{}", sid));
    std::fs::create_dir_all(&user_data_dir)?;

    let port = pick_debug_port();
    let mut cmd = tokio::process::Command::new(&bin);
    cmd.arg(format!("--remote-debugging-port={}", port))
        .arg("--remote-allow-origins=*")
        .arg(format!("--user-data-dir={}", user_data_dir.display()))
        .arg("--no-first-run")
        .arg("--no-default-browser-check")
        .arg("about:blank");

    if mode == "headless" {
        cmd.arg("--headless=new").arg("--disable-gpu");
    } else {
        // Visible mode: let the OS display the window. Prefer an existing DISPLAY
        // on Linux; rely on the platform's windowing by default.
        #[cfg(target_os = "linux")]
        if std::env::var_os("DISPLAY").is_none() {
            cmd.arg("--headless=new").arg("--disable-gpu");
        }
    }

    let mut child = cmd
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("Failed to launch Chrome: {}", bin.display()))?;

    // Wait for the debugger endpoint to come up.
    let started = std::time::Instant::now();
    let deadline = std::time::Duration::from_secs(15);
    let mut ready = false;
    while started.elapsed() < deadline {
        if cdp_version(port).await.is_some() {
            ready = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    if !ready {
        let _ = child.kill().await;
        bail!("Chrome did not start its DevTools endpoint on port {} within 15s", port);
    }

    if let Some(pid) = child.id() {
        write_pid(session_id, pid);
    }
    write_session_marker(session_id, port, &user_data_dir);

    Ok(port)
}

fn launch_session_marker(session_id: &str) -> Option<u16> {
    let marker = chrome_marker_path(session_id);
    let raw = std::fs::read_to_string(&marker).ok()?;
    let v: Value = serde_json::from_str(&raw).ok()?;
    v.get("port").and_then(|p| p.as_u64()).map(|p| p as u16)
}

async fn cdp_http_json(port: u16, path: &str) -> Option<Value> {
    let url = format!("http://127.0.0.1:{}/{}", port, path.trim_start_matches('/'));
    let resp = reqwest::Client::new().get(&url).send().await.ok()?;
    resp.json::<Value>().await.ok()
}

async fn cdp_version(port: u16) -> Option<Value> {
    cdp_http_json(port, "/json/version").await
}

/// Connect a CDP WebSocket to a page target (or the browser target).
async fn connect_page(port: u16, create_if_missing: bool) -> Result<WebSocketStream<MaybeTlsStream<TcpStream>>> {
    // Find an existing page target.
    if let Some(targets) = cdp_http_json(port, "json").await
        && let Some(list) = targets.as_array()
    {
        for t in list {
            if t.get("type").and_then(|x| x.as_str()) == Some("page") {
                if let Some(ws) = t.get("webSocketDebuggerUrl").and_then(|x| x.as_str()) {
                    let (stream, _) = connect_async(ws).await.ok().context("connect CDP page websocket")?;
                    return Ok(stream);
                }
            }
        }
    }

    if create_if_missing {
        // Open a new tab via the browser-level /json/new endpoint.
        if let Some(created) = cdp_http_json(port, "/json/new?about:blank").await
            && let Some(ws) = created.get("webSocketDebuggerUrl").and_then(|x| x.as_str())
        {
            let (stream, _) = connect_async(ws).await.ok().context("connect CDP new tab websocket")?;
            return Ok(stream);
        }
    }

    bail!("Could not find or create a Chrome page target")
}

/// Send a CDP command and await its response.
async fn send_cdp(
    ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>,
    method: &str,
    params: Value,
) -> Result<Value> {
    let id = NEXT_CDP_ID.fetch_add(1, Ordering::Relaxed);
    let req = json!({ "id": id, "method": method, "params": params });
    ws.send(WsMessage::Text(req.to_string().into())).await?;

    loop {
        match ws.next().await {
            Some(Ok(WsMessage::Text(text))) => {
                let msg: Value = serde_json::from_str(&text)?;
                if msg.get("id").and_then(|x| x.as_u64()) == Some(id) {
                    if let Some(err) = msg.get("error") {
                        bail!("CDP {} error: {}", method, err);
                    }
                    return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
                }
            }
            Some(Ok(_)) => continue,
            Some(Err(e)) => bail!("CDP websocket error: {}", e),
            None => bail!("CDP websocket closed while awaiting {}", method),
        }
    }
}

// ---- Provider trait impl ----

async fn chrome_status(_provider: &ChromeCdpProvider, _ctx: &ToolContext) -> Result<ToolOutput> {
    let metadata = json!({
        "backend": "chrome_cdp",
        "browser": "chrome",
        "ready": true,
        "mode": "headless/visible via Chrome CDP",
        "setup_complete": true,
        "binary_installed": chrome_binary().is_ok(),
        "responding": true,
    });
    Ok(ToolOutput::new("Chrome CDP backend is available.")
        .with_title("browser status")
        .with_metadata(metadata))
}

async fn chrome_setup() -> Result<ToolOutput> {
    let bin = chrome_binary()?;
    Ok(ToolOutput::new(format!(
        "Chrome CDP backend is ready. Using Chrome at:\n  {}\n\nLaunch mode is selected per call via the 'mode' parameter (headless or visible).",
        bin.display()
    ))
    .with_title("browser setup")
    .with_metadata(json!({
        "backend": "chrome_cdp",
        "browser": "chrome",
        "binary": bin.to_string_lossy().to_string(),
    })))
}

async fn ensure_chrome_ready() -> Result<Option<String>> {
    // Chrome requires no extension or native host; readiness is just the binary.
    if chrome_binary().is_ok() {
        Ok(None)
    } else {
        bail!(
            "Chrome/Chromium is not installed. Install Chrome and set JCODE_CHROME_PATH if needed."
        )
    }
}

async fn execute_chrome_action(
    _provider: &ChromeCdpProvider,
    action: &str,
    input: &BrowserInput,
    ctx: &ToolContext,
) -> Result<ToolOutput> {
    let mode = effective_mode(input);
    let port = ensure_chrome_launched(&ctx.session_id, &mode).await?;

    match action {
        "open" => {
            let url = input
                .url
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("url is required for open"))?;
            let result = chrome_navigate(port, url).await?;
            Ok(render_chrome_action("open", result))
        }
        "snapshot" => {
            let result = chrome_snapshot(port).await?;
            Ok(render_chrome_action("snapshot", result))
        }
        "get_content" => {
            let format = input.format.as_deref().unwrap_or("text");
            let result = chrome_get_content(port, format).await?;
            Ok(render_chrome_action("get_content", result))
        }
        "eval" => {
            let script = input
                .script
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("script is required for eval"))?;
            let result = chrome_eval(port, script).await?;
            Ok(render_chrome_action("eval", result))
        }
        "screenshot" => {
            chrome_screenshot(port).await
        }
        "list_tabs" => {
            let result = chrome_list_tabs(port).await?;
            Ok(render_chrome_action("list_tabs", result))
        }
        "new_tab" => {
            let url = input.url.as_deref().unwrap_or("about:blank");
            let result = chrome_new_tab(port, url).await?;
            Ok(render_chrome_action("new_tab", result))
        }
        "click" => {
            let selector = input
                .selector
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("click requires selector"))?;
            let result = chrome_click(port, selector).await?;
            Ok(render_chrome_action("click", result))
        }
        "type" => {
            let text = input
                .text
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("text is required for type"))?;
            let selector = input.selector.as_deref();
            let result = chrome_type(port, selector, text, input.clear.unwrap_or(false), input.submit.unwrap_or(false)).await?;
            Ok(render_chrome_action("type", result))
        }
        "wait" => {
            let selector = input.selector.as_deref();
            let result = chrome_wait(port, selector, input.timeout_ms.unwrap_or(5000)).await?;
            Ok(render_chrome_action("wait", result))
        }
        "provider_command" => {
            let method = input.provider_action.as_deref().ok_or_else(|| {
                anyhow::anyhow!("provider_action is required when action='provider_command'")
            })?;
            let params = input.params.clone().unwrap_or(Value::Object(Map::new()));
            let mut ws = connect_page(port, false).await?;
            let result = send_cdp(&mut ws, method, params).await?;
            Ok(render_chrome_action("provider_command", result))
        }
        other => bail!("Unsupported chrome browser action: {}", other),
    }
}

// ---- CDP action helpers ----

async fn cdp_session(port: u16) -> Result<WebSocketStream<MaybeTlsStream<TcpStream>>> {
    connect_page(port, true).await
}

async fn chrome_navigate(port: u16, url: &str) -> Result<Value> {
    let mut ws = cdp_session(port).await?;
    // Ensure the Page domain is enabled for navigation events.
    let _ = send_cdp(&mut ws, "Page.enable", json!({})).await;
    let result = send_cdp(&mut ws, "Page.navigate", json!({ "url": url })).await?;
    Ok(json!({ "url": url, "frameId": result.get("frameId").cloned().unwrap_or(Value::Null) }))
}

async fn chrome_snapshot(port: u16) -> Result<Value> {
    let mut ws = cdp_session(port).await?;
    let title = send_cdp(&mut ws, "Runtime.evaluate", json!({
        "expression": "document.title", "returnByValue": true
    })).await?;
    let text = send_cdp(&mut ws, "Runtime.evaluate", json!({
        "expression": "document.body ? document.body.innerText : ''", "returnByValue": true
    })).await?;
    let url = send_cdp(&mut ws, "Runtime.evaluate", json!({
        "expression": "location.href", "returnByValue": true
    })).await?;
    Ok(json!({
        "content": text.get("result").and_then(|v| v.get("value")).and_then(|v| v.as_str()).unwrap_or(""),
        "title": title.get("result").and_then(|v| v.get("value")).and_then(|v| v.as_str()).unwrap_or(""),
        "url": url.get("result").and_then(|v| v.get("value")).and_then(|v| v.as_str()).unwrap_or(""),
    }))
}

async fn chrome_get_content(port: u16, format: &str) -> Result<Value> {
    let mut ws = cdp_session(port).await?;
    match format {
        "html" => {
            let html = send_cdp(&mut ws, "Runtime.evaluate", json!({
                "expression": "document.documentElement.outerHTML", "returnByValue": true
            })).await?;
            Ok(json!({ "html": html.get("result").and_then(|v| v.get("value")).and_then(|v| v.as_str()).unwrap_or("") }))
        }
        _ => {
            let title = send_cdp(&mut ws, "Runtime.evaluate", json!({
                "expression": "document.title", "returnByValue": true
            })).await?;
            let text = send_cdp(&mut ws, "Runtime.evaluate", json!({
                "expression": "document.body ? document.body.innerText : ''", "returnByValue": true
            })).await?;
            let url = send_cdp(&mut ws, "Runtime.evaluate", json!({
                "expression": "location.href", "returnByValue": true
            })).await?;
            Ok(json!({
                "title": title.get("result").and_then(|v| v.get("value")).and_then(|v| v.as_str()).unwrap_or(""),
                "url": url.get("result").and_then(|v| v.get("value")).and_then(|v| v.as_str()).unwrap_or(""),
                "text": text.get("result").and_then(|v| v.get("value")).and_then(|v| v.as_str()).unwrap_or(""),
            }))
        }
    }
}

async fn chrome_eval(port: u16, script: &str) -> Result<Value> {
    let mut ws = cdp_session(port).await?;
    let result = send_cdp(&mut ws, "Runtime.evaluate", json!({
        "expression": script, "returnByValue": true, "awaitPromise": true
    })).await?;
    Ok(result.get("result").cloned().unwrap_or(Value::Null))
}

async fn chrome_screenshot(port: u16) -> Result<ToolOutput> {
    let mut ws = cdp_session(port).await?;
    let _ = send_cdp(&mut ws, "Page.enable", json!({})).await;
    let shot = send_cdp(&mut ws, "Page.captureScreenshot", json!({ "format": "png" })).await?;
    let data = shot.get("data").and_then(|v| v.as_str()).unwrap_or("");

    if data.is_empty() {
        return Ok(ToolOutput::new("Chrome returned an empty screenshot.")
            .with_title("browser screenshot"));
    }

    // Decode base64 back to bytes so with_labeled_image handles it consistently.
    let bytes = STANDARD.decode(data).unwrap_or_default();
    let mut output = ToolOutput::new("Captured browser screenshot.".to_string())
        .with_title("browser screenshot");
    if !bytes.is_empty() {
        output = output.with_labeled_image("image/png", STANDARD.encode(&bytes), "browser screenshot");
    }
    Ok(output)
}

async fn chrome_list_tabs(port: u16) -> Result<Value> {
    let targets = cdp_http_json(port, "/json").await.unwrap_or(Value::Array(vec![]));
    let tabs: Vec<Value> = targets
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter(|t| t.get("type").and_then(|x| x.as_str()) == Some("page"))
        .map(|t| {
            json!({
                "id": t.get("id").cloned().unwrap_or(Value::Null),
                "title": t.get("title").cloned().unwrap_or(Value::Null),
                "url": t.get("url").cloned().unwrap_or(Value::Null),
            })
        })
        .collect();
    Ok(json!({ "tabs": tabs }))
}

async fn chrome_new_tab(port: u16, url: &str) -> Result<Value> {
    let url_param = urlencoding::encode(url).to_string();
    let target = cdp_http_json(port, &format!("/json/new?{}", url_param)).await
        .unwrap_or(Value::Null);
    Ok(json!({ "target": target }))
}

async fn chrome_click(port: u16, selector: &str) -> Result<Value> {
    let script = format!(
        r#"(function() {{
            const el = document.querySelector({sel});
            if (!el) return {{ clicked: false, error: "Element not found: {seltxt}" }};
            el.click();
            return {{ clicked: true, tag: el.tagName }};
        }})()"#,
        sel = serde_json::to_string(selector).unwrap_or_else(|_| format!("\"{}\"", selector)),
        seltxt = selector.replace('"', "\\\""),
    );
    let mut ws = cdp_session(port).await?;
    let result = send_cdp(&mut ws, "Runtime.evaluate", json!({
        "expression": script, "returnByValue": true
    })).await?;
    Ok(result.get("result").cloned().unwrap_or(Value::Null))
}

async fn chrome_type(port: u16, selector: Option<&str>, text: &str, clear: bool, _submit: bool) -> Result<Value> {
    let mut ws = cdp_session(port).await?;
    let selector_lit = selector.map(|s| serde_json::to_string(s).unwrap_or_else(|_| format!("\"{}\"", s)));
    let target_lit = match &selector_lit {
        Some(sel) => format!("document.querySelector({sel})"),
        None => "document.activeElement".to_string(),
    };
    let script = format!(
        r#"(function() {{
            const target = {target};
            if (!target) return {{ typed: false, error: 'No target' }};
            target.focus();
            const value = {text};
            if ({clear}) target.value = '';
            if (value !== null) {{
                // Set value and dispatch input/change events for frameworks.
                target.value = value;
                target.dispatchEvent(new Event('input', {{ bubbles: true }}));
                target.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }}
            return {{ typed: true, tag: target.tagName, value: target.value }};
        }})()"#,
        target = target_lit,
        text = serde_json::to_string(text).unwrap_or_else(|_| "\"\"".to_string()),
        clear = clear,
    );
    let result = send_cdp(&mut ws, "Runtime.evaluate", json!({
        "expression": script, "returnByValue": true
    })).await?;
    Ok(result.get("result").cloned().unwrap_or(Value::Null))
}

async fn chrome_wait(port: u16, selector: Option<&str>, timeout_ms: u64) -> Result<Value> {
    let mut ws = cdp_session(port).await?;
    let selector_lit = selector.map(|s| serde_json::to_string(s).unwrap_or_else(|_| format!("\"{}\"", s)));
    let selector_expr = selector_lit.unwrap_or("null".to_string());
    let script = format!(
        r#"(async function() {{
            const sel = {sel};
            const deadline = Date.now() + {timeout};
            while (Date.now() < deadline) {{
                const el = sel ? document.querySelector(sel) : document.body;
                if (el) return {{ satisfied: true, found: !!sel }};
                await new Promise(r => setTimeout(r, 100));
            }}
            return {{ satisfied: false, found: false }};
        }})()"#,
        sel = selector_expr,
        timeout = timeout_ms,
    );
    let result = send_cdp(&mut ws, "Runtime.evaluate", json!({
        "expression": script, "returnByValue": true, "awaitPromise": true
    })).await?;
    Ok(result.get("result").cloned().unwrap_or(Value::Null))
}

// ---- Output helpers ----

fn render_chrome_action(action: &str, result: Value) -> ToolOutput {
    let body = match action {
        "snapshot" => result
            .get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| serde_json::to_string_pretty(&result).unwrap_or_default()),
        "get_content" => chrome_format_content(&result),
        "eval" => {
            let rendered = if let Some(s) = result.as_str() {
                s.to_string()
            } else {
                serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string())
            };
            rendered
        }
        _ => serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string()),
    };
    ToolOutput::new(body)
        .with_title(format!("browser {}", action))
        .with_metadata(result)
}

fn chrome_format_content(result: &Value) -> String {
    if let Some(text) = result.get("text").and_then(|v| v.as_str()) {
        return text.to_string();
    }
    if let Some(html) = result.get("html").and_then(|v| v.as_str()) {
        return html.to_string();
    }
    if let Some(title) = result.get("title").and_then(|v| v.as_str()) {
        if let Some(url) = result.get("url").and_then(|v| v.as_str()) {
            return format!("{}\n{}", title, url);
        }
        return title.to_string();
    }
    serde_json::to_string_pretty(result).unwrap_or_default()
}

#[cfg(test)]
#[path = "chrome_tests.rs"]
mod chrome_tests;
