//! Host-provided environment variables for a session's tool child processes.
//!
//! Frontends (the ACP adapter bridging a desktop host, for example) may inject
//! integration credentials — `KANEO_API_URL`, `KANEO_API_KEY`, ... — into the
//! process they spawn. Tools do not run in that process: the shared daemon
//! executes them, so the variables must be carried across the process boundary
//! and re-applied where the shell is actually spawned.
//!
//! Keying by session id (rather than a per-turn task-local) matters because a
//! client can attach to an already-running session: the turn and its tools are
//! then executed in the session's original context, which would otherwise keep
//! the environment captured when the session was first created. Recording on
//! every subscribe means the most recent client's environment wins for that
//! session's subsequent command spawns, regardless of who owns the session.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

type EnvList = Vec<(String, String)>;

fn registry() -> &'static RwLock<HashMap<String, EnvList>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, EnvList>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Record the host-provided env for `session_id`.
///
/// Last write wins, so an attach (or a re-subscribe after credentials rotate)
/// replaces the previous values. An empty list clears the entry.
pub fn set_session_env(session_id: &str, env: EnvList) {
    if session_id.trim().is_empty() {
        return;
    }
    let mut registry = registry()
        .write()
        .unwrap_or_else(|error| error.into_inner());
    if env.is_empty() {
        registry.remove(session_id);
    } else {
        registry.insert(session_id.to_string(), env);
    }
}

/// Drop any recorded env for `session_id` (e.g. when its client disconnects).
pub fn clear_session_env(session_id: &str) {
    let mut registry = registry()
        .write()
        .unwrap_or_else(|error| error.into_inner());
    registry.remove(session_id);
}

/// The host-provided env for `session_id`; empty when none was recorded.
pub fn session_env_for(session_id: &str) -> EnvList {
    registry()
        .read()
        .unwrap_or_else(|error| error.into_inner())
        .get(session_id)
        .cloned()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorded_env_is_scoped_per_session_and_last_writer_wins() {
        set_session_env("ses_one", vec![("KANEO_API_KEY".into(), "one".into())]);
        set_session_env("ses_two", vec![("KANEO_API_KEY".into(), "two".into())]);

        assert_eq!(
            session_env_for("ses_one"),
            vec![("KANEO_API_KEY".to_string(), "one".to_string())]
        );
        assert_eq!(
            session_env_for("ses_two"),
            vec![("KANEO_API_KEY".to_string(), "two".to_string())]
        );
        assert!(session_env_for("ses_absent").is_empty());

        // Attaching with fresh credentials replaces the session's entry.
        set_session_env("ses_one", vec![("KANEO_API_KEY".into(), "rotated".into())]);
        assert_eq!(
            session_env_for("ses_one"),
            vec![("KANEO_API_KEY".to_string(), "rotated".to_string())]
        );

        clear_session_env("ses_one");
        assert!(session_env_for("ses_one").is_empty());
        assert_eq!(session_env_for("ses_two").len(), 1);
    }
}
