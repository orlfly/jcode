#[cfg(any(windows, unix))]
use super::build_shell_command;
use super::format_command_output;

#[test]
fn format_command_output_truncates_on_utf8_boundary() {
    let input = format!("{}é", "a".repeat(29_999));
    let output = format_command_output(input, None);
    assert!(output.ends_with("\n... (output truncated)"));
    assert!(output.starts_with(&"a".repeat(29_999)));
}

#[cfg(windows)]
#[tokio::test]
async fn build_shell_command_uses_cmd_and_executes_command() {
    let output = build_shell_command("echo hello-from-cmd")
        .output()
        .await
        .expect("run cmd command");
    assert!(output.status.success(), "cmd command should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.to_ascii_lowercase().contains("hello-from-cmd"),
        "unexpected stdout: {}",
        stdout
    );

    let probe_path = std::env::temp_dir().join(format!(
        "jcode-cmd-quoting-probe-{}.cmd",
        std::process::id()
    ));
    std::fs::write(
        &probe_path,
        concat!(
            "@echo off\r\n",
            "if \"%~1\"==\"text with spaces\" if \"%~2\"==\"\" (\r\n",
            "  echo quoted-argument-ok\r\n",
            "  exit /b 0\r\n",
            ")\r\n",
            "echo first=[%~1] second=[%~2]\r\n",
            "exit /b 1\r\n",
        ),
    )
    .expect("write cmd quoting probe");

    let quoted_command = format!("call \"{}\" \"text with spaces\"", probe_path.display());
    let quoted_output = build_shell_command(&quoted_command)
        .output()
        .await
        .expect("run cmd quoting probe");
    let _ = std::fs::remove_file(&probe_path);
    let quoted_stdout = String::from_utf8_lossy(&quoted_output.stdout);
    let quoted_stderr = String::from_utf8_lossy(&quoted_output.stderr);
    assert!(
        quoted_output.status.success(),
        "quoted argument should remain one child-process argument; stdout={quoted_stdout:?} stderr={quoted_stderr:?}"
    );
    assert!(
        quoted_stdout.contains("quoted-argument-ok"),
        "unexpected quoted-command stdout: {quoted_stdout}"
    );
}

#[cfg(unix)]
#[test]
fn build_shell_command_uses_disk_backed_scratch_directory() {
    // Keep JCODE_HOME stable until the final directory assertion. Running
    // synchronously avoids holding the environment MutexGuard across await.
    let _env_lock = crate::storage::lock_test_env();
    let expected = super::tool_scratch_dir().expect("jcode scratch directory");
    let output = build_shell_command("printf '%s\\n%s\\n' \"$TMPDIR\" \"$JCODE_SCRATCH_DIR\"")
        .as_std_mut()
        .output()
        .expect("run bash command");
    assert!(output.status.success(), "bash command should succeed");
    let stdout = String::from_utf8(output.stdout).expect("utf-8 scratch paths");
    let paths = stdout.lines().collect::<Vec<_>>();
    let expected = expected.to_string_lossy().into_owned();
    assert_eq!(paths, vec![expected.as_str(), expected.as_str()]);
    assert!(std::path::Path::new(&expected).is_dir());
}

/// The daemon runs tool shells from its own process environment, so
/// host-provided vars (AionUi injects `KANEO_API_KEY` into the connecting
/// ACP process) only reach the agent's command if the session env is
/// projected onto the child. Regression guard for the devops agent
/// reporting "missing Kaneo API key".
#[cfg(unix)]
#[tokio::test]
async fn build_shell_command_projects_session_env_to_children() {
    let sentinel = "probe-sentinel-abc123";
    let session_id = "session_env_projection_test";
    jcode_base::session_env::set_session_env(
        session_id,
        vec![
            ("KANEO_API_KEY".to_string(), sentinel.to_string()),
            (
                "KANEO_API_URL".to_string(),
                "http://localhost:1337".to_string(),
            ),
        ],
    );
    let mut command = build_shell_command("printf '%s\\n' \"$KANEO_API_KEY\" \"$KANEO_API_URL\"");
    super::apply_session_env(&mut command, session_id);
    let output = command.output().await.expect("run shell command");
    jcode_base::session_env::clear_session_env(session_id);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(sentinel),
        "session env must reach the tool shell: {stdout:?}"
    );
    assert!(
        stdout.contains("http://localhost:1337"),
        "session env must reach the tool shell: {stdout:?}"
    );
}

/// An attaching client's env must win over whatever the session was created
/// with: the turn can run in the session's original context.
#[cfg(unix)]
#[tokio::test]
async fn apply_session_env_uses_the_latest_registered_value() {
    let session_id = "session_env_attach_test";
    jcode_base::session_env::set_session_env(
        session_id,
        vec![("KANEO_API_KEY".to_string(), "stale".to_string())],
    );
    jcode_base::session_env::set_session_env(
        session_id,
        vec![("KANEO_API_KEY".to_string(), "rotated".to_string())],
    );
    let mut command = build_shell_command("printf '%s' \"$KANEO_API_KEY\"");
    super::apply_session_env(&mut command, session_id);
    let output = command.output().await.expect("run shell command");
    jcode_base::session_env::clear_session_env(session_id);
    assert_eq!(String::from_utf8_lossy(&output.stdout), "rotated");
}

#[cfg(unix)]
#[tokio::test]
async fn build_shell_command_omits_session_env_outside_the_scope() {
    let output = build_shell_command("printf 'value=%s\\n' \"${KANEO_API_KEY:-unset}\"")
        .output()
        .await
        .expect("run shell command");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("probe-sentinel-abc123"), "{stdout:?}");
}
