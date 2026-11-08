//! Extraction source health: pause automatic extraction when a source degrades.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::storage;

pub(super) const MAX_FAILURE_RATE: f32 = 0.05;
pub(super) const MAX_CONFIDENCE_DROP: f32 = 0.10;
pub(super) const MIN_CANDIDATES_FOR_PAUSE: usize = 10;
pub(super) const HEALTH_WINDOW: usize = 10;

/// One extraction batch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtractionBatch {
    pub at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub session_id: String,
    pub candidates: usize,
    pub admitted: usize,
    pub queued: usize,
    pub failed: usize,
    pub avg_confidence: f32,
    pub duplicate_ratio: f32,
}

/// Durable extraction health state.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ExtractionHealth {
    #[serde(default)]
    pub paused: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pause_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub batches: Vec<ExtractionBatch>,
}

impl ExtractionHealth {
    pub fn failure_rate(&self) -> f32 {
        let candidates: usize = self.batches.iter().map(|batch| batch.candidates).sum();
        let failed: usize = self.batches.iter().map(|batch| batch.failed).sum();
        let total = candidates.max(failed);
        if total == 0 {
            0.0
        } else {
            failed as f32 / total as f32
        }
    }

    pub fn recent_batches(&self) -> &[ExtractionBatch] {
        let start = self.batches.len().saturating_sub(HEALTH_WINDOW);
        &self.batches[start..]
    }

    pub fn summary(&self) -> String {
        let recent = self.recent_batches();
        if recent.is_empty() {
            return "no extraction batches recorded".to_string();
        }
        let candidates: usize = recent.iter().map(|batch| batch.candidates).sum();
        let failed: usize = recent.iter().map(|batch| batch.failed).sum();
        let avg =
            recent.iter().map(|batch| batch.avg_confidence).sum::<f32>() / recent.len() as f32;
        let duplicate = recent
            .iter()
            .map(|batch| batch.duplicate_ratio)
            .sum::<f32>()
            / recent.len() as f32;
        format!(
            "{} batch(es), {} candidate(s), {} failure(s), avg confidence {:.2}, duplicate ratio {:.2}{}",
            recent.len(),
            candidates,
            failed,
            avg,
            duplicate,
            self.pause_reason
                .as_deref()
                .map(|reason| format!(", PAUSED: {reason}"))
                .unwrap_or_default()
        )
    }
}

pub(super) fn health_path(test_mode: bool) -> Result<PathBuf> {
    let base = if test_mode {
        storage::jcode_dir()?.join("memory").join("test")
    } else {
        storage::jcode_dir()?.join("memory")
    };
    Ok(base.join("extraction_health.json"))
}

pub(super) fn load(path: &std::path::Path) -> Result<ExtractionHealth> {
    if !path.exists() {
        return Ok(ExtractionHealth::default());
    }
    let raw = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&raw).unwrap_or_default())
}

pub(super) fn save(path: &std::path::Path, health: &ExtractionHealth) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(health)?)?;
    Ok(())
}
