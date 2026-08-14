//! Background maintenance for the on-disk session store.
//!
//! Session transcripts (`<id>.json`) are kept forever, but the atomic-write
//! layer also leaves a single rolling `<id>.bak` next to each file as a
//! crash-recovery copy (see `jcode_storage::write_bytes_inner`). That backup is
//! only ever consulted when the primary `.json` is found to be corrupt on the
//! very next read. For sessions that have not been touched in weeks the primary
//! is stable, so the stale `.bak` is pure disk overhead (these accumulate into
//! gigabytes over time).
//!
//! Two retention rules are enforced here:
//!   1. **Age window**: any `.bak` whose mtime is older than
//!      [`BACKUP_RETENTION_DAYS`] is removed unconditionally.
//!   2. **Per-session cap**: any session_id is allowed at most
//!      [`MAX_BACKUPS_PER_SESSION`] `.bak` files; the oldest spillover is
//!      pruned even within the age window.
//!
//! This module never touches the `.json` transcripts themselves, so no session
//! data is lost; at worst a very old, already-stable session loses its
//! redundant recovery copy.

use crate::storage;
use chrono::{DateTime, Duration, Local};
use std::collections::HashMap;
use std::path::Path;

/// Backups older than this are considered safe to remove. Chosen conservatively
/// so any realistic "crashed mid-write, reopened later" scenario still has its
/// recovery copy.
const BACKUP_RETENTION_DAYS: i64 = 7;

/// Per-session cap on the number of `.bak` files retained. The atomic-write
/// layer normally leaves exactly one `<id>.bak`, but `pre-wipe-<ts>.bak`
/// recovery copies (see `persistence::Session::pre_wipe_backup_path`) can stack
/// up for sessions that go through repeated shrink/save cycles. Capping at 5
/// keeps disk usage bounded without losing recent recovery history.
const MAX_BACKUPS_PER_SESSION: usize = 5;

/// Retention for the `node-compile-cache` directory inside `~/.jcode/scratch`.
/// Node writes here on every command run, so it grows fast but each entry is
/// invalidated by content hash — keeping a rolling 7-day window is enough
/// to amortize the cost of cold-start compilation while bounding the worst
/// case. Set conservatively; this only governs cleanup, never generation.
const NODE_COMPILE_CACHE_TTL_DAYS: i64 = 7;

/// How often the scratch cleanup pass may run per machine. Co-ordinated via
/// the same `scratch-prune.stamp` marker pattern used for the `.bak` prune so
/// concurrent jcode processes do not race on the directory walk.
const SCRATCH_PRUNE_INTERVAL_SECS: u64 = 24 * 60 * 60;

/// Minimum interval between prune passes across all jcode processes.
///
/// The prune walks the entire sessions directory (easily 100k+ entries on a
/// long-lived install), which profiles as the single largest CPU cost of TUI
/// startup when it runs unconditionally. Backups only need to be reclaimed
/// eventually, so one pass per interval per machine is plenty; a marker file's
/// mtime coordinates that across concurrently spawned processes.
const BACKUP_PRUNE_INTERVAL_SECS: u64 = 24 * 60 * 60;

/// Remove stale `<id>.bak` files from the sessions directory.
///
/// Best-effort: any I/O error is ignored so this can run on a background thread
/// at startup without ever affecting launch. Skips cheaply (one stat) unless
/// the machine-wide prune interval has elapsed, so spawning many jcode
/// processes at once does not trigger many full directory walks.
pub fn prune_old_session_backups() {
    if let Ok(base) = storage::jcode_dir() {
        let sessions_dir = base.join("sessions");
        if !claim_prune_slot(&base, "sessions-bak-prune.stamp", BACKUP_PRUNE_INTERVAL_SECS) {
            return;
        }
        prune_old_session_backups_in(&sessions_dir, Local::now());
    }
}

/// Prune the `~/.jcode/scratch/` directory.
///
/// Two rules apply (see [`SCRATCH_PRUNE_INTERVAL_SECS`]):
///   1. Any top-level entry whose name starts with `jcode-session-test-` is
///      removed unconditionally. These are per-test transient tempdirs and
///      have no value after the test that created them completes.
///   2. The `node-compile-cache` subdirectory is removed when its mtime is
///      older than [`NODE_COMPILE_CACHE_TTL_DAYS`]; the contents are content-
///      addressed, so node will rebuild lazily without loss.
///
/// Best-effort and rate-limited like [`prune_old_session_backups`] — any I/O
/// error is swallowed and a single process per interval actually walks the
/// tree.
pub fn prune_scratch_dir() {
    let Ok(base) = storage::jcode_dir() else {
        return;
    };
    let scratch = base.join("scratch");
    if !scratch.exists() {
        return;
    }
    if !claim_prune_slot(
        &base,
        "scratch-prune.stamp",
        SCRATCH_PRUNE_INTERVAL_SECS,
    ) {
        return;
    }
    prune_scratch_dir_in(&scratch, Local::now());
}

/// Clear session-scoped scratch entries belonging to a single session.
///
/// Called from the session-close path (see
/// `server::client_disconnect_cleanup`). Unlike [`prune_scratch_dir`], this
/// runs immediately on every close — there is no rate limiter — and only
/// touches entries that look session-scoped, leaving `node-compile-cache` and
/// any unrelated content untouched.
///
/// The current bash tool (`tool::bash`) sets `TMPDIR` to the global scratch
/// directory for every command, so per-session isolation is by convention:
/// entries created under names matching `jcode-session-test-<id>` or
/// `node-compile-cache-<id>` (defensive) are removed. Anything else is left
/// alone so unrelated concurrent sessions are not disturbed.
pub fn clear_session_scratch(session_id: &str) {
    let Ok(base) = storage::jcode_dir() else {
        return;
    };
    let scratch = base.join("scratch");
    if !scratch.exists() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(&scratch) else {
        return;
    };
    let id = session_id;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() && !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // Only match names that are clearly tied to this session; leave
        // anything else (including the shared `node-compile-cache`) alone.
        let scoped = name.starts_with("jcode-session-test-") && name.contains(id)
            || name.starts_with(&format!("node-compile-cache-{id}"));
        if scoped {
            let _ = if path.is_dir() {
                std::fs::remove_dir_all(&path)
            } else {
                std::fs::remove_file(&path)
            };
        }
    }
}

/// Returns true when this process should run the prune pass now, updating the
/// marker so other processes (and future spawns) skip until the next interval.
///
/// The marker touch happens before the walk, so a burst of simultaneous spawns
/// resolves to at most a couple of walkers (racing between the stat and the
/// touch) instead of one per process, and steady-state spawns do a single stat.
fn claim_prune_slot(base: &Path, marker_name: &str, interval_secs: u64) -> bool {
    let marker = base.join(marker_name);
    if let Ok(metadata) = std::fs::metadata(&marker)
        && let Ok(modified) = metadata.modified()
        && let Ok(age) = std::time::SystemTime::now().duration_since(modified)
        && age.as_secs() < interval_secs
    {
        return false;
    }
    // Touch (create or refresh) the marker to claim the slot.
    std::fs::write(&marker, b"").is_ok()
}

/// Core of [`prune_scratch_dir`], parameterized on the directory and "now"
/// for unit testing.
fn prune_scratch_dir_in(scratch: &Path, now: DateTime<Local>) {
    let Ok(entries) = std::fs::read_dir(scratch) else {
        return;
    };
    let cutoff = now - Duration::days(NODE_COMPILE_CACHE_TTL_DAYS);
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.starts_with("jcode-session-test-") {
            let _ = if path.is_dir() {
                std::fs::remove_dir_all(&path)
            } else {
                std::fs::remove_file(&path)
            };
            continue;
        }
        if name == "node-compile-cache" {
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            let Ok(modified) = metadata.modified() else {
                continue;
            };
            let modified: DateTime<Local> = modified.into();
            if modified < cutoff {
                let _ = std::fs::remove_dir_all(&path);
            }
        }
    }
}

/// Core of [`prune_old_session_backups`], parameterized on the directory and
/// "now" for unit testing.
///
/// Two retention rules apply (see module docs):
///   * Age window: any `.bak` older than [`BACKUP_RETENTION_DAYS`] is removed.
///   * Per-session cap: a session_id may keep at most
///     [`MAX_BACKUPS_PER_SESSION`] `.bak` files; the oldest spillover is
///     pruned even when within the age window.
fn prune_old_session_backups_in(sessions_dir: &Path, now: DateTime<Local>) {
    let Ok(entries) = std::fs::read_dir(sessions_dir) else {
        return;
    };
    let cutoff = now - Duration::days(BACKUP_RETENTION_DAYS);

    // Group surviving .bak files by session_id, sorted newest-first within
    // each group. Files that fail to stat are skipped silently.
    let mut groups: HashMap<String, Vec<(DateTime<Local>, std::path::PathBuf)>> =
        HashMap::new();
    for entry in entries.flatten() {
        let path = entry.path();
        // Only prune the atomic-write backup files; never the .json transcripts
        // or anything else (journals, tmp files, etc.).
        let is_bak = path.extension().map(|e| e == "bak").unwrap_or(false);
        if !is_bak {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        let modified: DateTime<Local> = modified.into();
        if let Some(session_id) = extract_session_id(&path) {
            groups
                .entry(session_id)
                .or_default()
                .push((modified, path));
        }
    }

    for (_session_id, mut backups) in groups {
        // Newest first.
        backups.sort_by(|a, b| b.0.cmp(&a.0));
        for (idx, (modified, path)) in backups.into_iter().enumerate() {
            // Rule 1: outside the age window.
            // Rule 2: beyond the per-session cap (idx counts from 0, so idx >= MAX).
            if modified < cutoff || idx >= MAX_BACKUPS_PER_SESSION {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}

/// Extract the session_id from a `.bak` filename. Both naming conventions
/// observed in the codebase are supported:
///   * `<session_id>.bak`            (atomic-write recovery copy)
///   * `<session_id>.pre-wipe-<ts>.bak` (shrink guard, persistence.rs)
///
/// Returns `None` for any other `.bak` filename so the caller skips it.
fn extract_session_id(path: &Path) -> Option<String> {
    let stem = path.file_name()?.to_str()?;
    let session_id = stem
        .strip_suffix(".bak")?
        // Strip the `.pre-wipe-<timestamp>` suffix when present, keeping the
        // session_id prefix that comes before it.
        .split(".pre-wipe-")
        .next()?;
    if session_id.is_empty() {
        return None;
    }
    Some(session_id.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use std::time::{Duration as StdDuration, SystemTime};

    #[test]
    fn claim_prune_slot_rate_limits_within_interval_and_reclaims_after() {
        let dir = std::env::temp_dir().join(format!(
            "jcode-bak-claim-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&dir).expect("create temp dir");

        // First claim wins and creates the marker.
        assert!(
            claim_prune_slot(&dir, "sessions-bak-prune.stamp", BACKUP_PRUNE_INTERVAL_SECS),
            "first claim should win"
        );
        let marker = dir.join("sessions-bak-prune.stamp");
        assert!(marker.exists(), "marker should be created");

        // A concurrent/subsequent spawn within the interval is rejected.
        assert!(
            !claim_prune_slot(&dir, "sessions-bak-prune.stamp", BACKUP_PRUNE_INTERVAL_SECS),
            "second claim within interval should be skipped"
        );

        // Once the marker is older than the interval the slot opens again.
        let old = SystemTime::now() - StdDuration::from_secs(BACKUP_PRUNE_INTERVAL_SECS + 60);
        File::options()
            .write(true)
            .open(&marker)
            .and_then(|f| f.set_modified(old))
            .expect("age the marker");
        assert!(
            claim_prune_slot(&dir, "sessions-bak-prune.stamp", BACKUP_PRUNE_INTERVAL_SECS),
            "claim should succeed after the interval elapses"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn prunes_only_old_bak_files() {
        let dir = std::env::temp_dir().join(format!(
            "jcode-bak-prune-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&dir).expect("create temp dir");

        let write = |name: &str, age_days: u64| {
            let path = dir.join(name);
            let mut f = File::create(&path).expect("create");
            f.write_all(b"{}").ok();
            if age_days > 0 {
                let mtime = SystemTime::now() - StdDuration::from_secs(age_days * 24 * 60 * 60);
                f.set_modified(mtime).expect("set mtime");
            }
            path
        };

        // 60-day-old backup: should be pruned (rule 1: age window).
        let old_bak = write("session_old.bak", 60);
        // 5-day-old backup: within window, should survive.
        let recent_bak = write("session_recent.bak", 5);
        // Transcripts must never be removed, regardless of age.
        let old_json = write("session_old.json", 60);
        let recent_json = write("session_recent.json", 0);
        // Other artifacts must be left alone.
        let journal = write("session_old.journal.jsonl", 60);

        prune_old_session_backups_in(&dir, Local::now());

        assert!(!old_bak.exists(), "old .bak should be pruned");
        assert!(recent_bak.exists(), "recent .bak must survive");
        assert!(
            old_json.exists(),
            "old .json transcript must never be removed"
        );
        assert!(recent_json.exists(), "recent .json transcript must survive");
        assert!(journal.exists(), "journals are out of scope");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn caps_per_session_backup_count() {
        let dir = std::env::temp_dir().join(format!(
            "jcode-bak-cap-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&dir).expect("create temp dir");

        let session_id = "session_noisy";
        // 7 pre-wipe backups, all within the 7-day age window but more than
        // MAX_BACKUPS_PER_SESSION (5). The 2 oldest must be pruned.
        let mut paths = Vec::new();
        for i in 0..7 {
            // Distinct second offsets so the order is unambiguous.
            let age_secs = (i as u64) * 60;
            let p = dir.join(format!("{session_id}.pre-wipe-{i}.bak"));
            let mut f = File::create(&p).expect("create");
            f.write_all(b"{}").ok();
            f.set_modified(SystemTime::now() - StdDuration::from_secs(age_secs))
                .expect("set mtime");
            paths.push(p);
        }

        prune_old_session_backups_in(&dir, Local::now());

        // The 5 newest (i=0..4) must survive; i=5,6 (oldest) must be gone.
        for (i, p) in paths.iter().enumerate() {
            if i < 5 {
                assert!(p.exists(), "backup #{i} should be retained (within cap)");
            } else {
                assert!(!p.exists(), "backup #{i} should be pruned (over cap)");
            }
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn extract_session_id_handles_both_naming_conventions() {
        assert_eq!(
            extract_session_id(Path::new("session_abc_123_xyz.bak")),
            Some("session_abc_123_xyz".to_string())
        );
        assert_eq!(
            extract_session_id(Path::new("session_abc_123_xyz.pre-wipe-1700000000.bak")),
            Some("session_abc_123_xyz".to_string())
        );
        assert_eq!(extract_session_id(Path::new("not-a-session.txt")), None);
        assert_eq!(extract_session_id(Path::new(".bak")), None);
    }

    fn make_dir_unique(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "jcode-{tag}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn prune_scratch_dir_in_removes_session_test_entries_and_old_node_cache() {
        let dir = make_dir_unique("scratch-prune");

        // Three transient entries — all must be removed unconditionally.
        let fresh_test = dir.join("jcode-session-test-fresh");
        fs::create_dir_all(&fresh_test).unwrap();
        File::create(fresh_test.join("payload")).unwrap();

        // node-compile-cache older than the TTL — must be removed.
        let old_cache = dir.join("node-compile-cache");
        fs::create_dir_all(&old_cache).unwrap();
        File::create(old_cache.join("x")).unwrap();
        let old = SystemTime::now() - StdDuration::from_secs(NODE_COMPILE_CACHE_TTL_DAYS * 24 * 60 * 60 + 3600);
        File::options()
            .write(true)
            .open(&old_cache)
            .and_then(|f| f.set_modified(old))
            .expect("age node-compile-cache");

        // node-compile-cache inside the TTL — must survive.
        let new_cache = dir.join("node-compile-cache-fresh");
        fs::create_dir_all(&new_cache).unwrap();

        // Unrelated top-level content — must survive.
        let other = dir.join("user-uploaded.txt");
        File::create(&other).unwrap();

        prune_scratch_dir_in(&dir, Local::now());

        assert!(!fresh_test.exists(), "session-test entry must be removed");
        assert!(!old_cache.exists(), "old node-compile-cache must be removed");
        assert!(new_cache.exists(), "fresh non-matching dir must survive");
        assert!(other.exists(), "unrelated content must survive");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn clear_session_scratch_only_removes_matching_entries() {
        let dir = make_dir_unique("scratch-clear");

        let sid = "session_abc_123_xyz";
        // Should be removed: name contains the session id.
        let mine_dir = dir.join(format!("jcode-session-test-{sid}"));
        fs::create_dir_all(&mine_dir).unwrap();
        File::create(mine_dir.join("payload")).unwrap();

        // Should be removed: node-compile-cache-<id>.
        let mine_cache = dir.join(format!("node-compile-cache-{sid}"));
        fs::create_dir_all(&mine_cache).unwrap();

        // Should survive: a different session's scratch entry.
        let theirs_dir = dir.join("jcode-session-test-session_other_999");
        fs::create_dir_all(&theirs_dir).unwrap();

        // Should survive: the shared node-compile-cache.
        let shared_cache = dir.join("node-compile-cache");
        fs::create_dir_all(&shared_cache).unwrap();

        // Should survive: an unrelated file.
        let other = dir.join("keep-me.txt");
        File::create(&other).unwrap();

        clear_session_scratch(sid);

        assert!(!mine_dir.exists(), "session's own scratch must be cleared");
        assert!(!mine_cache.exists(), "session's own compile cache must be cleared");
        assert!(theirs_dir.exists(), "other session's scratch must be untouched");
        assert!(shared_cache.exists(), "shared node-compile-cache must be untouched");
        assert!(other.exists(), "unrelated file must be untouched");

        fs::remove_dir_all(&dir).ok();
    }
}
