use crate::memory_graph::MemoryGraph;
use jcode_memory_types::StoreKey;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

// === Graph Cache ===

const BACKEND_CACHE_LIMIT: usize = 8;

struct GraphCacheEntry {
    graph: MemoryGraph,
    /// Path-mode cache uses `modified` (mtime) to invalidate; backend
    /// cache uses `version` as a monotonic counter bumped by the
    /// backend's save path.
    modified: Option<SystemTime>,
    // Only read by cached_graph_for_backend; path-mode entries carry 0.
    version: u64,
}

struct GraphCache {
    /// Legacy entries keyed by JSON path; mtime-validated.
    entries: HashMap<PathBuf, GraphCacheEntry>,
    /// New entries keyed by `(backend_name, store_key)`; only valid
    /// while `version` matches the backend's current generation.
    backend_entries: HashMap<(String, String), GraphCacheEntry>,
    /// FIFO insertion order for `backend_entries`, bounded so a
    /// maintenance sweep over many stores cannot pin every graph in
    /// memory for the rest of the process.
    backend_order: VecDeque<(String, String)>,
}

impl GraphCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            backend_entries: HashMap::new(),
            backend_order: VecDeque::new(),
        }
    }
}

static GRAPH_CACHE: OnceLock<Mutex<GraphCache>> = OnceLock::new();

fn graph_cache() -> &'static Mutex<GraphCache> {
    GRAPH_CACHE.get_or_init(|| Mutex::new(GraphCache::new()))
}

fn graph_mtime(path: &PathBuf) -> Option<SystemTime> {
    std::fs::metadata(path).ok().and_then(|m| m.modified().ok())
}

pub(super) fn cached_graph(path: &PathBuf) -> Option<MemoryGraph> {
    let modified = graph_mtime(path);
    let cache = graph_cache().lock().ok()?;
    let entry = cache.entries.get(path)?;
    if entry.modified == modified {
        Some(entry.graph.clone())
    } else {
        None
    }
}

pub(super) fn cache_graph(path: PathBuf, graph: &MemoryGraph) {
    let modified = graph_mtime(&path);
    if let Ok(mut cache) = graph_cache().lock() {
        cache.entries.insert(
            path,
            GraphCacheEntry {
                graph: graph.clone(),
                modified,
                version: 0,
            },
        );
    }
}

/// Drop the path-mode cache entry so the next load re-reads and re-normalizes
/// the file. Used by tests that simulate a store written by an older version.
pub(super) fn forget_cached_graph(path: &PathBuf) {
    if let Ok(mut cache) = graph_cache().lock() {
        cache.entries.remove(path);
    }
}

/// Cache a graph loaded from a non-file backend (sqlite-gvec, etc.).
/// `version` is bumped by the caller on every write; cache hits
/// only while the backend reports the same version.
pub fn cache_graph_for_backend(
    backend_name: &str,
    key: &StoreKey,
    graph: &MemoryGraph,
    version: u64,
) {
    if let Ok(mut cache) = graph_cache().lock() {
        let cache_key = (backend_name.to_string(), key.as_str().to_string());
        if cache
            .backend_entries
            .insert(
                cache_key.clone(),
                GraphCacheEntry {
                    graph: graph.clone(),
                    modified: None,
                    version,
                },
            )
            .is_none()
        {
            cache.backend_order.push_back(cache_key);
        }
        while cache.backend_order.len() > BACKEND_CACHE_LIMIT {
            if let Some(expired) = cache.backend_order.pop_front() {
                cache.backend_entries.remove(&expired);
            }
        }
    }
}

pub fn cached_graph_for_backend(
    backend_name: &str,
    key: &StoreKey,
    version: u64,
) -> Option<MemoryGraph> {
    let cache = graph_cache().lock().ok()?;
    let entry = cache
        .backend_entries
        .get(&(backend_name.to_string(), key.as_str().to_string()))?;
    if entry.version == version {
        Some(entry.graph.clone())
    } else {
        None
    }
}
