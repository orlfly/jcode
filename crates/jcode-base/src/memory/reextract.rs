//! Re-extract memories from a historical session transcript.
//!
//! Automatic extraction runs once at session finalize. This module provides an
//! explicit, dry-run-first path for replaying an old session through the same
//! extractor and governed admission pipeline. It is intentionally separate
//! from [`super::MemoryManager::extract_with_sidecar`] so a manual backfill
//! never records source-health batches or pauses automatic extraction.

use super::{
    ExtractionMethod, MemoryCategory, MemoryEntry, MemoryManager, MemoryWriteOutcome,
    ProvenanceRecord, TrustLevel, batch_change_categories, memory_sidecar_enabled,
};
use crate::memory_agent::build_transcript_for_extraction;
use crate::session::Session;
use crate::sidecar::Sidecar;
use anyhow::{Context, Result};
use jcode_memory_types::normalize_memory_search_text;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Resolve session ids from explicit flags and/or a manifest whose rows carry
/// `source` ids and optional `tiers`.
pub fn resolve_reextract_sessions(
    sessions: &[String],
    manifest: Option<&Path>,
    tier: Option<&str>,
    limit: usize,
) -> Result<Vec<String>> {
    let mut ids = Vec::new();
    let mut seen = HashSet::new();

    let mut push = |id: &str| {
        if !id.is_empty() && seen.insert(id.to_string()) {
            ids.push(id.to_string());
        }
    };
    for id in sessions {
        push(id);
    }

    if let Some(path) = manifest {
        let text = std::fs::read_to_string(path).map_err(|error| {
            anyhow::anyhow!("read re-extract manifest {}: {error}", path.display())
        })?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
            anyhow::anyhow!("parse re-extract manifest {}: {error}", path.display())
        })?;
        let rows = value
            .get("rows")
            .and_then(|rows| rows.as_array())
            .ok_or_else(|| {
                anyhow::anyhow!("re-extract manifest {} has no rows array", path.display())
            })?;
        for row in rows {
            if let Some(tier) = tier {
                let matches = row
                    .get("tiers")
                    .and_then(|tiers| tiers.as_array())
                    .is_some_and(|tiers| tiers.iter().any(|value| value.as_str() == Some(tier)));
                if !matches {
                    continue;
                }
            }
            if let Some(source) = row.get("source").and_then(|source| source.as_str())
                && source.starts_with("session_")
            {
                push(source);
            }
        }
    }

    ids.truncate(limit);
    if ids.is_empty() {
        anyhow::bail!(
            "no sessions selected; pass --session <id> or --manifest <path> [--tier <name>]"
        );
    }
    Ok(ids)
}

/// Split messages into roughly `max_chars` transcript chunks.
///
/// Tool results are already truncated by
/// [`build_transcript_for_extraction`], so the estimate uses the same formatter
/// per message. `max_chunks == 0` means unlimited. Returns `(chunks, truncated)`
/// where `truncated` means more chunks existed than the limit allowed.
fn split_message_chunks(
    messages: &[crate::message::Message],
    max_chars: usize,
    max_chunks: usize,
) -> (Vec<Vec<crate::message::Message>>, bool) {
    if messages.is_empty() {
        return (Vec::new(), false);
    }
    if max_chars == 0 {
        return (vec![messages.to_vec()], false);
    }

    let mut chunks = Vec::new();
    let mut current = Vec::new();
    let mut current_chars = 0usize;
    for message in messages {
        let message_chars = build_transcript_for_extraction(std::slice::from_ref(message))
            .chars()
            .count();
        if !current.is_empty() && current_chars + message_chars > max_chars {
            chunks.push(std::mem::take(&mut current));
            current_chars = 0;
        }
        current.push(message.clone());
        current_chars += message_chars;
    }
    if !current.is_empty() {
        chunks.push(current);
    }

    let limit = if max_chunks == 0 {
        usize::MAX
    } else {
        max_chunks
    };
    let truncated = chunks.len() > limit;
    if truncated {
        chunks.truncate(limit);
    }
    (chunks, truncated)
}

/// One candidate produced while replaying a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReextractItem {
    pub category: String,
    pub content: String,
    pub trust: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locator: Option<String>,
    /// Candidate duplicates an existing active memory or an earlier candidate.
    pub duplicate: bool,
    /// Category crossed the G4 batch-change threshold.
    pub batch_gated: bool,
    /// `preview`, `duplicate`, `stored`, `stored_needs_review`,
    /// `pending_review`, or `failed`.
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stored_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub review_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Summary of one session replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReextractReport {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub working_dir: Option<String>,
    pub messages: usize,
    pub transcript_chars: usize,
    pub chunks_processed: usize,
    pub chunks_total: usize,
    pub truncated: bool,
    pub candidates: usize,
    pub duplicates: usize,
    pub admitted: usize,
    pub queued: usize,
    pub failed: usize,
    pub applied: bool,
    pub items: Vec<ReextractItem>,
}

impl MemoryManager {
    /// Replay `session` through the memory extractor.
    ///
    /// `apply == false` is a pure preview: the extractor still runs, but no
    /// memory or review row is written. `force` bypasses the automatic
    /// extraction source-health pause; manual replays otherwise respect it.
    pub async fn reextract_session(
        &self,
        session: &Session,
        apply: bool,
        force: bool,
        chunk_chars: usize,
        max_chunks: usize,
    ) -> Result<ReextractReport> {
        if !memory_sidecar_enabled() {
            anyhow::bail!(
                "memory extraction sidecar is disabled; set agents.memory_sidecar_enabled = true"
            );
        }
        if !Sidecar::llm_backend_available() {
            anyhow::bail!(
                "memory extraction model is unavailable; configure the memory extraction model or provider"
            );
        }
        if !force && self.extraction_is_paused() {
            anyhow::bail!(
                "automatic memory extraction source is paused; run `jcode memory source resume` or pass --force"
            );
        }

        let messages = session.messages_for_provider_uncached();
        if messages.is_empty() {
            anyhow::bail!("session {} has no messages", session.id);
        }
        let transcript = build_transcript_for_extraction(&messages);
        if transcript.trim().is_empty() {
            anyhow::bail!("session {} produced an empty transcript", session.id);
        }

        let graph = self.load_project_graph().ok();
        let ontology = graph.as_ref().map(|graph| self.ontology_for_graph(graph));
        let existing_counts = graph
            .as_ref()
            .map(|graph| {
                let mut counts = HashMap::new();
                for entry in graph.active_memories() {
                    *counts.entry(entry.category.clone()).or_insert(0usize) += 1;
                }
                counts
            })
            .unwrap_or_default();
        let existing_contents: Vec<String> = self
            .list_all()
            .unwrap_or_default()
            .into_iter()
            .filter(|entry| entry.active)
            .map(|entry| entry.content)
            .collect();

        let sidecar = Sidecar::for_session(session);
        let (chunks, truncated) = split_message_chunks(&messages, chunk_chars, max_chunks);
        let chunks_total = chunks.len();
        let mut extracted = Vec::new();
        let mut transcript_chars = 0usize;
        let mut chunks_processed = 0usize;
        for (index, chunk) in chunks.iter().enumerate() {
            let chunk_transcript = build_transcript_for_extraction(chunk);
            if chunk_transcript.trim().is_empty() {
                continue;
            }
            transcript_chars += chunk_transcript.chars().count();
            let candidates = sidecar
                .extract_memories_with_ontology(
                    &chunk_transcript,
                    &existing_contents,
                    ontology.as_deref(),
                )
                .await
                .with_context(|| {
                    format!(
                        "extract session {} chunk {}/{}",
                        session.id,
                        index + 1,
                        chunks_total
                    )
                })?;
            chunks_processed += 1;
            extracted.extend(candidates);
        }

        let mut report = self.process_extracted_candidates(
            &session.id,
            extracted,
            &existing_contents,
            &existing_counts,
            apply,
        );
        report.working_dir = session.working_dir.clone();
        report.messages = messages.len();
        report.transcript_chars = transcript_chars;
        report.chunks_processed = chunks_processed;
        report.chunks_total = chunks_total;
        report.truncated = truncated;
        Ok(report)
    }

    /// Classify and optionally persist extractor output. Kept synchronous so
    /// tests can exercise duplicate and admission behavior without an LLM.
    pub(crate) fn process_extracted_candidates(
        &self,
        session_id: &str,
        extracted: Vec<crate::sidecar::ExtractedMemory>,
        existing_contents: &[String],
        existing_counts: &HashMap<MemoryCategory, usize>,
        apply: bool,
    ) -> ReextractReport {
        let candidates = extracted.len();
        let candidate_categories: Vec<MemoryCategory> = extracted
            .iter()
            .map(|memory| memory.category.parse().unwrap_or(MemoryCategory::Fact))
            .collect();
        let batch_gated_categories =
            batch_change_categories(&candidate_categories, existing_counts);

        let mut existing_norm: HashSet<String> = existing_contents
            .iter()
            .map(|content| normalize_memory_search_text(content, &[]))
            .collect();
        let mut items = Vec::with_capacity(candidates);
        let mut duplicates = 0usize;
        let mut admitted = 0usize;
        let mut queued = 0usize;
        let mut failed = 0usize;

        for (memory, category) in extracted.into_iter().zip(candidate_categories) {
            let trust = match memory.trust.as_str() {
                "high" => TrustLevel::High,
                "low" => TrustLevel::Low,
                _ => TrustLevel::Medium,
            };
            let confidence = trust.provenance_confidence();
            let method = if trust == TrustLevel::High {
                ExtractionMethod::UserStated
            } else {
                ExtractionMethod::LlmExtraction
            };
            let locator = memory
                .locator
                .clone()
                .unwrap_or_else(|| format!("session:{session_id}"));
            let normalized = normalize_memory_search_text(&memory.content, &[]);
            let duplicate = !existing_norm.insert(normalized);
            let batch_gated = batch_gated_categories.contains(&category);
            let mut item = ReextractItem {
                category: category.to_string(),
                content: memory.content.clone(),
                trust: memory.trust.clone(),
                locator: Some(locator.clone()),
                duplicate,
                batch_gated,
                status: "preview".to_string(),
                stored_id: None,
                review_id: None,
                error: None,
            };

            if duplicate {
                duplicates += 1;
                item.status = "duplicate".to_string();
                items.push(item);
                continue;
            }
            if !apply {
                items.push(item);
                continue;
            }

            let mut entry = MemoryEntry::new(category, memory.content)
                .with_source(session_id)
                .with_trust(trust)
                .with_provenance(
                    ProvenanceRecord::new(session_id, method)
                        .with_confidence(confidence)
                        .with_locator(locator),
                );
            for (key, value) in memory.properties {
                entry = entry.with_property(key, value);
            }

            let outcome = if batch_gated {
                self.remember_project_governed_with_policy(
                    entry,
                    &[jcode_memory_types::ConfirmationGate::BatchChange],
                    Some("re-extract batch exceeds the G4 bulk-change threshold for this category"),
                )
            } else {
                self.remember_project_governed(entry)
            };
            match outcome {
                Ok(MemoryWriteOutcome::Stored { id, needs_review }) => {
                    admitted += 1;
                    item.status = "stored".to_string();
                    item.stored_id = Some(id);
                    if needs_review {
                        item.status = "stored_needs_review".to_string();
                    }
                }
                Ok(MemoryWriteOutcome::PendingReview { review_id, .. }) => {
                    queued += 1;
                    item.status = "pending_review".to_string();
                    item.review_id = Some(review_id);
                }
                Err(error) => {
                    failed += 1;
                    item.status = "failed".to_string();
                    item.error = Some(error.to_string());
                }
            }
            items.push(item);
        }

        ReextractReport {
            session_id: session_id.to_string(),
            working_dir: None,
            messages: 0,
            transcript_chars: 0,
            chunks_processed: 0,
            chunks_total: 0,
            truncated: false,
            candidates,
            duplicates,
            admitted,
            queued,
            failed,
            applied: apply,
            items,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_temp_home<T>(f: impl FnOnce() -> T) -> T {
        let _guard = crate::storage::lock_test_env();
        let temp = tempfile::tempdir().expect("tempdir");
        let previous = std::env::var("JCODE_HOME").ok();
        crate::env::set_var("JCODE_HOME", temp.path());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        match previous {
            Some(value) => crate::env::set_var("JCODE_HOME", value),
            None => crate::env::remove_var("JCODE_HOME"),
        }
        match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    fn extracted(content: &str, category: &str) -> crate::sidecar::ExtractedMemory {
        crate::sidecar::ExtractedMemory {
            category: category.to_string(),
            content: content.to_string(),
            trust: "high".to_string(),
            locator: None,
            properties: Default::default(),
        }
    }

    #[test]
    fn preview_marks_duplicates_without_writing() {
        let manager = MemoryManager::new_test();
        let report = manager.process_extracted_candidates(
            "session_test",
            vec![
                extracted("same fact", "env_fact"),
                extracted("same fact", "env_fact"),
                extracted("new fact", "observation"),
            ],
            &["same fact".to_string()],
            &HashMap::new(),
            false,
        );

        assert_eq!(report.candidates, 3);
        assert_eq!(report.duplicates, 2);
        assert_eq!(report.admitted, 0);
        assert_eq!(report.queued, 0);
        assert_eq!(report.failed, 0);
        assert!(!report.applied);
        assert_eq!(report.items[0].status, "duplicate");
        assert_eq!(report.items[1].status, "duplicate");
        assert_eq!(report.items[2].status, "preview");
    }

    #[test]
    fn manifest_selection_filters_tiers_and_dedupes_sessions() {
        let temp = tempfile::tempdir().expect("tempdir");
        let manifest = temp.path().join("manifest.json");
        std::fs::write(
            &manifest,
            serde_json::json!({
                "rows": [
                    {"source": "session_a", "tiers": ["tier1_safe"]},
                    {"source": "session_b", "tiers": ["tier3_full"]},
                    {"source": "skill_registry", "tiers": ["tier1_safe"]},
                    {"source": "session_a", "tiers": ["tier1_safe"]}
                ]
            })
            .to_string(),
        )
        .expect("write manifest");

        let ids = resolve_reextract_sessions(
            &["session_c".to_string(), "session_c".to_string()],
            Some(&manifest),
            Some("tier1_safe"),
            10,
        )
        .expect("resolve");
        assert_eq!(ids, vec!["session_c", "session_a"]);

        let limited = resolve_reextract_sessions(
            &["session_c".to_string()],
            Some(&manifest),
            Some("tier1_safe"),
            1,
        )
        .expect("resolve limited");
        assert_eq!(limited, vec!["session_c"]);
    }

    #[test]
    fn apply_writes_non_duplicate_candidates_through_governed_path() {
        with_temp_home(|| {
            let manager = MemoryManager::new_test();
            let report = manager.process_extracted_candidates(
                "session_test",
                vec![extracted("a durable observation", "observation")],
                &[],
                &HashMap::new(),
                true,
            );
            assert_eq!(report.candidates, 1);
            assert_eq!(report.duplicates, 0);
            assert_eq!(report.admitted, 1);
            assert_eq!(report.queued, 0);
            assert_eq!(report.failed, 0);
            assert_eq!(report.items[0].status, "stored");
            assert!(report.items[0].stored_id.is_some());

            let stored = manager.list_all().expect("list");
            assert!(
                stored
                    .iter()
                    .any(|entry| entry.content == "a durable observation"),
                "re-extracted memory must be retrievable"
            );
        });
    }

    #[test]
    fn chunking_respects_budget_and_max_chunks() {
        let messages: Vec<crate::message::Message> = (0..8)
            .map(|index| crate::message::Message::user(&format!("message {index} with some text")))
            .collect();

        let (chunks, truncated) = split_message_chunks(&messages, 80, 0);
        assert!(chunks.len() > 1, "expected multiple chunks");
        assert!(!truncated);
        assert_eq!(chunks.iter().map(Vec::len).sum::<usize>(), messages.len());

        let (limited, truncated) = split_message_chunks(&messages, 80, 2);
        assert_eq!(limited.len(), 2);
        assert!(truncated);

        let (single, truncated) = split_message_chunks(&messages, 0, 2);
        assert_eq!(single.len(), 1);
        assert!(!truncated);
    }
}
