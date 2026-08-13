//! Pluggable memory graph storage backends.
//!
//! [`GraphBackend`] is implemented by:
//!
//! - [`SqliteGvecBackend`]: the SQLite backend backed by the
//!   `sqlite-gvec` graph extension (vendored under `third_party/`).
//!   This is the default backend selected by `active_backend_name()`.
//!   Uses gvec's Cypher subset to perform row-level inserts/updates
//!   inside a single transaction, which avoids the read-modify-write
//!   amplification of the JSON backend.
//!
//! - [`JsonBackend`]: the legacy JSON-snapshot backend, kept as a
//!   zero-dependency fallback when the `sqlite-gvec` feature is off or
//!   the SQLite backend cannot be opened, and as a one-time migration
//!   source for existing JSON memory when sqlite becomes the default.

use crate::storage;
use anyhow::{Context, Result};
use jcode_memory_types::{GraphBackend, GraphMutation, MemoryGraph, StoreKey, apply_mutations_in_place};
use std::fmt::Debug;
use std::path::{Path, PathBuf};

#[cfg(feature = "sqlite-gvec")]
mod sqlite_gvec;
#[cfg(feature = "sqlite-gvec")]
pub use sqlite_gvec::SqliteGvecBackend;

/// JSON-file backend. The graph is serialized as a single
/// `MemoryGraph` document and written atomically with `write_json`.
///
/// File layout (per `StoreKey`):
///   `<jcode_dir>/memory/backend-json/<key>.json`
///
/// The legacy path layout
/// (`<jcode_dir>/memory/projects/<project_hash>.json`) continues to be
/// the canonical store; this backend writes to a separate directory
/// when constructed directly via [`JsonBackend::with_root_dir`]. The
/// existing memory code paths still call `storage::write_json`
/// directly on the legacy path, so this backend is primarily useful
/// for tests and as a reference implementation.
pub struct JsonBackend {
    root: PathBuf,
}

impl Debug for JsonBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonBackend")
            .field("root", &self.root)
            .finish()
    }
}

impl JsonBackend {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Default root: `<jcode_dir>/memory/backend-json/`.
    pub fn default_root() -> Result<PathBuf> {
        let dir = storage::jcode_dir()?.join("memory").join("backend-json");
        Ok(dir)
    }

    fn path_for(&self, key: &StoreKey) -> PathBuf {
        self.root.join(format!("{}.json", sanitize(key.as_str())))
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' })
        .collect()
}

impl GraphBackend for JsonBackend {
    fn name(&self) -> &str {
        "json"
    }

    fn load(&self, key: &StoreKey) -> Result<MemoryGraph> {
        let path = self.path_for(key);
        if !path.exists() {
            return Ok(MemoryGraph::new());
        }
        match storage::read_json::<MemoryGraph>(&path) {
            Ok(graph) => Ok(graph),
            Err(e) => {
                crate::logging::warn(&format!(
                    "JsonBackend: failed to read {}: {e}; falling back to empty graph",
                    path.display()
                ));
                Ok(MemoryGraph::new())
            }
        }
    }

    fn save(&self, key: &StoreKey, graph: &MemoryGraph) -> Result<()> {
        std::fs::create_dir_all(&self.root)
            .with_context(|| format!("create JsonBackend root {}", self.root.display()))?;
        let path = self.path_for(key);
        storage::write_json(&path, graph)
            .with_context(|| format!("JsonBackend: write {}", path.display()))?;
        Ok(())
    }

    fn apply_mutations(
        &self,
        key: &StoreKey,
        mutations: &[GraphMutation],
    ) -> Result<MemoryGraph> {
        let mut graph = self.load(key)?;
        apply_mutations_in_place(&mut graph, mutations)?;
        self.save(key, &graph)?;
        Ok(graph)
    }
}

/// Convert a JSON-file backend path to its SQLite backend equivalent,
/// so callers can migrate from one to the other in place.
pub fn json_path_to_sqlite_path(json_path: &Path) -> PathBuf {
    json_path.with_extension("sqlite")
}

#[cfg(test)]
mod tests {
    use super::*;
    use jcode_memory_types::{EdgeKind, MemoryCategory, MemoryEntry, MemoryScope};
    use tempfile::tempdir;

    #[test]
    fn json_backend_round_trip() {
        let dir = tempdir().unwrap();
        let backend = JsonBackend::new(dir.path());

        let key = StoreKey::new("test-project-1234");
        let empty = backend.load(&key).unwrap();
        assert_eq!(empty.memory_count(), 0);

        let mut entry = MemoryEntry::new(
            MemoryCategory::Fact,
            "the rust programming language",
        );
        entry.id = "m1".into();
        entry.tags = vec!["alpha".into()];

        let upsert = GraphMutation::UpsertMemory {
            id: "m1".into(),
            json: serde_json::to_string(&entry).unwrap(),
        };
        let edge = GraphMutation::UpsertEdge {
            from: "m1".into(),
            to: "tag:alpha".into(),
            kind_json: serde_json::to_string(&EdgeKind::HasTag).unwrap(),
        };
        backend.apply_mutations(&key, &[upsert, edge]).unwrap();

        let loaded = backend.load(&key).unwrap();
        assert_eq!(loaded.memory_count(), 1);
        assert_eq!(loaded.tags.len(), 0); // tag node not upserted; only edge present
        assert!(loaded.edges.contains_key("m1"));
    }

    #[test]
    fn sanitize_replaces_unsafe_chars() {
        assert_eq!(sanitize("a/b\\c d"), "a_b_c_d");
        assert_eq!(sanitize("safe-name_123.ok"), "safe-name_123.ok");
    }
}