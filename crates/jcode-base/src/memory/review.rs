//! Durable review queue and memory health report types.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use jcode_memory_types::{MemoryReviewItem, ReviewStatus};

use crate::storage;

/// Append-only queue file. One JSON object per line.
pub(super) fn queue_path(test_mode: bool) -> Result<PathBuf> {
    let base = if test_mode {
        storage::jcode_dir()?.join("memory").join("test")
    } else {
        storage::jcode_dir()?.join("memory")
    };
    Ok(base.join("reviews").join("pending.jsonl"))
}

pub(super) fn load_items(path: &Path) -> Result<Vec<MemoryReviewItem>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = std::fs::read_to_string(path)?;
    let mut items = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<MemoryReviewItem>(line) {
            Ok(item) => items.push(item),
            Err(error) => crate::logging::warn(&format!(
                "Skipping malformed review item in {}: {}",
                path.display(),
                error
            )),
        }
    }
    Ok(items)
}

pub(super) fn append_item(path: &Path, item: &MemoryReviewItem) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let line = serde_json::to_string(item)?;
    writeln!(file, "{line}")?;
    file.flush()?;
    Ok(())
}

pub(super) fn save_items(path: &Path, items: &[MemoryReviewItem]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("jsonl.tmp");
    let mut out = String::new();
    for item in items {
        out.push_str(&serde_json::to_string(item)?);
        out.push('\n');
    }
    std::fs::write(&tmp, out)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub(super) fn find_item<'a>(
    items: &'a [MemoryReviewItem],
    id: &str,
) -> Option<&'a MemoryReviewItem> {
    items.iter().find(|item| item.id == id)
}

pub(super) fn update_status(
    items: &mut [MemoryReviewItem],
    id: &str,
    status: ReviewStatus,
    reviewer: Option<&str>,
    note: Option<&str>,
) -> Result<MemoryReviewItem> {
    let now = chrono::Utc::now();
    let item = items
        .iter_mut()
        .find(|item| item.id == id)
        .ok_or_else(|| anyhow::anyhow!("Review item not found: {id}"))?;
    if item.status != ReviewStatus::Pending && status == ReviewStatus::Accepted {
        anyhow::bail!(
            "Review item {} is already {}; cannot accept it again",
            id,
            item.status.as_str()
        );
    }
    item.status = status;
    item.confirmed_by = reviewer.map(str::to_string);
    item.confirmed_at = Some(now);
    if let Some(note) = note {
        item.note = Some(note.to_string());
    }
    Ok(item.clone())
}

/// Aggregate health counters for one or more memory stores.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MemoryHealthReport {
    pub stores: usize,
    pub memories: usize,
    pub active: usize,
    pub archived: usize,
    pub expired: usize,
    pub disputed: usize,
    pub critical: usize,
    pub needs_review: usize,
    pub low_confidence: usize,
    pub missing_provenance: usize,
    pub missing_locator: usize,
    pub contradictions: usize,
    pub duplicate_subjects: usize,
    pub categories: BTreeMap<String, usize>,
    pub reviews: ReviewHealth,
}

impl MemoryHealthReport {
    pub fn summary(&self) -> String {
        format!(
            "{} store(s), {} memories ({} active), {} needs review, {} critical, {} contradiction(s), {} pending review(s), oldest {} day(s)",
            self.stores,
            self.memories,
            self.active,
            self.needs_review,
            self.critical,
            self.contradictions,
            self.reviews.pending,
            self.reviews.oldest_pending_days.unwrap_or_default()
        )
    }
}

/// Review backlog counters.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ReviewHealth {
    pub pending: usize,
    pub deferred: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub oldest_pending_days: Option<i64>,
    pub by_gate: BTreeMap<String, usize>,
}

impl ReviewHealth {
    pub fn observe(&mut self, item: &MemoryReviewItem, now: chrono::DateTime<chrono::Utc>) {
        match item.status {
            ReviewStatus::Pending => {
                self.pending += 1;
                let age = (now - item.created_at).num_days().max(0);
                self.oldest_pending_days =
                    Some(self.oldest_pending_days.map_or(age, |old| old.max(age)));
                for gate in &item.gates {
                    *self.by_gate.entry(format!("{gate:?}")).or_default() += 1;
                }
            }
            ReviewStatus::Deferred => self.deferred += 1,
            ReviewStatus::Accepted => self.accepted += 1,
            ReviewStatus::Rejected => self.rejected += 1,
        }
    }
}
