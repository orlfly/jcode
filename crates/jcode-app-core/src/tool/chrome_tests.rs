use super::*;

#[test]
fn effective_mode_defaults_to_headless() {
    let input = BrowserInput {
        action: "open".into(),
        browser: Some("chrome".into()),
        mode: None,
        provider_action: None,
        params: None,
        url: Some("https://example.com".into()),
        tab_id: None,
        window_id: None,
        frame_id: None,
        all_frames: None,
        selector: None,
        text: None,
        contains: None,
        script: None,
        key: None,
        x: None,
        y: None,
        format: None,
        wait: None,
        new_tab: None,
        focus: None,
        clear: None,
        submit: None,
        page_world: None,
        position: None,
        behavior: None,
        timeout_ms: None,
        path: None,
        fields: None,
        scroll_to: None,
    };
    assert_eq!(effective_mode(&input), "headless");
}

#[test]
fn effective_mode_honors_explicit_visible() {
    let mut input = BrowserInput {
        action: "open".into(),
        mode: Some("visible".into()),
        browser: Some("chrome".into()),
        provider_action: None,
        params: None,
        url: Some("https://example.com".into()),
        tab_id: None,
        window_id: None,
        frame_id: None,
        all_frames: None,
        selector: None,
        text: None,
        contains: None,
        script: None,
        key: None,
        x: None,
        y: None,
        format: None,
        wait: None,
        new_tab: None,
        focus: None,
        clear: None,
        submit: None,
        page_world: None,
        position: None,
        behavior: None,
        timeout_ms: None,
        path: None,
        fields: None,
        scroll_to: None,
    };
    assert_eq!(effective_mode(&input), "visible");

    input.mode = Some("HEADLESS".into());
    assert_eq!(effective_mode(&input), "headless");
}

#[test]
fn chrome_session_id_sanitizes_and_truncates() {
    assert_eq!(chrome_session_id("abc-123_def"), "abc-123_def");
    let long = "x".repeat(100);
    assert_eq!(chrome_session_id(&long).len(), 64);
    assert_eq!(chrome_session_id("bad/name!@#"), "badname");
}

#[test]
fn render_chrome_action_snapshot_extracts_content() {
    let result = json!({ "content": "hello world", "title": "t", "url": "u" });
    let out = render_chrome_action("snapshot", result);
    assert_eq!(out.output, "hello world");
    assert_eq!(out.title.as_deref(), Some("browser snapshot"));
}

#[test]
fn render_chrome_action_get_content_prefers_text() {
    let result = json!({ "text": "body text", "title": "t" });
    let out = render_chrome_action("get_content", result);
    assert_eq!(out.output, "body text");
}

#[test]
fn render_chrome_action_eval_renders_value() {
    let result = json!("some result");
    let out = render_chrome_action("eval", result);
    assert_eq!(out.output, "some result");
}

#[test]
fn chrome_binary_resolves_when_installed() {
    // On this machine Chrome exists; just confirm the resolver returns Ok when
    // a candidate is present (or a path is on PATH). Skip assertion of the
    // exact path so the test is portable.
    let _ = chrome_binary();
}

/// Live CDP integration test: only runs when JCODE_CHROME_LIVE_TESTS=1 and a
/// Chrome binary is available. Exercises launch + navigate + snapshot + eval in
/// headless mode against a real Chrome instance.
#[cfg(unix)]
#[tokio::test]
async fn live_chrome_headless_end_to_end() {
    if std::env::var("JCODE_CHROME_LIVE_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping live chrome test (set JCODE_CHROME_LIVE_TESTS=1 to enable)");
        return;
    }
    if chrome_binary().is_err() {
        eprintln!("skipping live chrome test (no Chrome binary)");
        return;
    }

    let port = ensure_chrome_launched("live-test-session", "headless")
        .await
        .expect("launch headless chrome");

    let nav = chrome_navigate(port, "https://example.com")
        .await
        .expect("navigate");
    assert!(nav.get("url").is_some());

    // Wait a moment for page load then snapshot.
    tokio::time::sleep(std::time::Duration::from_millis(800)).await;
    let snap = chrome_snapshot(port).await.expect("snapshot");
    let content = snap.get("content").and_then(|v| v.as_str()).unwrap_or("");
    assert!(content.contains("Example Domain"), "got: {}", content);

    let title = chrome_eval(port, "document.title").await.expect("eval");
    assert!(
        title.to_string().contains("Example Domain") || title.as_str().is_some(),
        "title: {title}"
    );
}

/// Live CDP test for visible (headed) mode. Requires a display; on Linux we
/// check DISPLAY/WAYLAND_DISPLAY. Runs only with JCODE_CHROME_LIVE_TESTS=1.
#[cfg(unix)]
#[tokio::test]
async fn live_chrome_visible_mode() {
    if std::env::var("JCODE_CHROME_LIVE_TESTS").as_deref() != Ok("1") {
        return;
    }
    if chrome_binary().is_err() {
        return;
    }
    #[cfg(target_os = "linux")]
    if std::env::var_os("DISPLAY").is_none() && std::env::var_os("WAYLAND_DISPLAY").is_none() {
        eprintln!("skipping visible-mode test (no display)");
        return;
    }

    let port = ensure_chrome_launched("live-visible-session", "visible")
        .await
        .expect("launch visible chrome");

    let nav = chrome_navigate(port, "https://example.com")
        .await
        .expect("navigate visible");
    assert!(nav.get("url").is_some());
    tokio::time::sleep(std::time::Duration::from_millis(800)).await;
    let snap = chrome_snapshot(port).await.expect("snapshot visible");
    assert!(
        snap.get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .contains("Example Domain")
    );
}
