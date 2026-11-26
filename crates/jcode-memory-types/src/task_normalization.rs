//! Stage S0 task text normalization (design 4.3).
//!
//! Deterministic half only: split compound tasks on explicit conjunctions,
//! collapse whitespace, unify a small closed set of time units, and resolve
//! pronouns against caller-supplied context mentions. Every edit keeps the
//! original byte span so the LLM stage and audits can trace it back.
//!
//! Not implemented here: ellipsis reconstruction, time/date resolution,
//! project-id canonicalization and language translation. Those need LLM or
//! belief state and are reported as explicit `not implemented` warnings.

use serde::{Deserialize, Serialize};

/// Maximum sub-tasks kept after splitting; extra segments are reported.
pub const MAX_NORMALIZED_SEGMENTS: usize = 5;

/// Separators that indicate a compound task. Deliberately excludes `，`/`,`,
/// which usually join clauses inside one task.
const COMPOUND_SEPARATORS: [&str; 8] = [
    "；",
    ";",
    "然后",
    "并且",
    "同时",
    "以及",
    "另外",
    " and then ",
];

/// Closed unit table, longest keys first so `毫秒` wins over `秒` and
/// `个工作日` wins over `日`.
const UNIT_TABLE: [(&str, &str); 6] = [
    ("个工作日", "workdays"),
    ("毫秒", "ms"),
    ("秒钟", "s"),
    ("分钟", "min"),
    ("小时", "h"),
    ("秒", "s"),
];

const PRONOUNS: [&str; 10] = [
    "这个模块",
    "该模块",
    "这个文件",
    "该文件",
    "这个",
    "该",
    "它",
    "it",
    "this",
    "they",
];

/// One normalized sub-task with its span in the original text (bytes).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedSegment {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

/// One pronoun resolution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Resolution {
    pub surface: String,
    pub resolved: String,
    pub start: usize,
    pub end: usize,
    pub source: String,
}

/// One unit substitution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Substitution {
    pub original: String,
    pub normalized: String,
    pub start: usize,
    pub end: usize,
}

/// S0 output: normalized text plus traceable edits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedTask {
    pub original: String,
    /// Normalized text. Segments are joined with `; `, so offsets here are not
    /// the original offsets; use `segments` for traceability.
    pub text: String,
    pub segments: Vec<NormalizedSegment>,
    pub resolutions: Vec<Resolution>,
    pub substitutions: Vec<Substitution>,
    pub language: String,
    pub warnings: Vec<String>,
}

/// Normalize task text. `context_mentions` supplies antecedent mentions in
/// recency order (oldest first); the last one wins.
pub fn normalize_task_text(text: &str, context_mentions: &[String]) -> NormalizedTask {
    let mut warnings = Vec::new();
    let language = detect_language(text).to_string();
    if language == "unknown" {
        warnings.push("language could not be determined from script".to_string());
    }
    warnings.push(
        "S0 is deterministic: ellipsis reconstruction, date resolution, project-id canonicalization \
         and translation are not implemented"
            .to_string(),
    );
    let antecedent = context_mentions
        .iter()
        .rev()
        .find(|mention| !mention.trim().is_empty())
        .cloned();

    let mut raw_segments = split_compound(text);
    if raw_segments.len() > MAX_NORMALIZED_SEGMENTS {
        warnings.push(format!(
            "split produced {} segments; kept the first {MAX_NORMALIZED_SEGMENTS}",
            raw_segments.len()
        ));
        raw_segments.truncate(MAX_NORMALIZED_SEGMENTS);
    }
    if raw_segments.is_empty() {
        raw_segments.push((0, text.len()));
    }

    let mut segments = Vec::new();
    let mut resolutions = Vec::new();
    let mut substitutions = Vec::new();
    let mut normalized_parts = Vec::new();

    for (start, end) in raw_segments {
        let raw = &text[start..end];
        let mut segment = String::new();
        let mut index = 0;
        while index < raw.len() {
            let rest = &raw[index..];
            if let Some((original, normalized)) = UNIT_TABLE
                .iter()
                .find(|(original, _)| rest.starts_with(original))
            {
                segment.push_str(normalized);
                substitutions.push(Substitution {
                    original: (*original).to_string(),
                    normalized: (*normalized).to_string(),
                    start: start + index,
                    end: start + index + original.len(),
                });
                index += original.len();
                continue;
            }
            let ch = rest.chars().next().expect("non-empty rest");
            index += ch.len_utf8();
            if ch.is_whitespace() {
                if segment.ends_with(' ') {
                    continue;
                }
                segment.push(' ');
                continue;
            }
            segment.push(ch);
        }

        let mut normalized_segment = segment.trim().to_string();
        if let Some(antecedent) = &antecedent {
            // Replace at most one pronoun per segment: later occurrences keep
            // their surface text so the unresolved-pronoun warning still fires
            // and no index goes stale after the replacement changes length.
            for pronoun in PRONOUNS {
                let raw_offset = find_bounded(&raw.to_lowercase(), pronoun);
                let normalized_offset = find_bounded(&normalized_segment.to_lowercase(), pronoun);
                let (Some(raw_offset), Some(normalized_offset)) = (raw_offset, normalized_offset)
                else {
                    continue;
                };
                let surface = normalized_segment
                    [normalized_offset..normalized_offset + pronoun.len()]
                    .to_string();
                normalized_segment.replace_range(
                    normalized_offset..normalized_offset + pronoun.len(),
                    antecedent,
                );
                resolutions.push(Resolution {
                    surface,
                    resolved: antecedent.clone(),
                    start: start + raw_offset,
                    end: start + raw_offset + pronoun.len(),
                    source: format!("context mention `{antecedent}`"),
                });
                break;
            }
        }

        normalized_parts.push(normalized_segment.clone());
        segments.push(NormalizedSegment {
            text: normalized_segment,
            start,
            end,
        });
    }

    if antecedent.is_none()
        && segments.iter().any(|segment| {
            PRONOUNS
                .iter()
                .any(|pronoun| segment.text.contains(pronoun))
        })
    {
        warnings.push("pronoun left unresolved: no context mentions supplied".to_string());
    }

    NormalizedTask {
        original: text.to_string(),
        text: normalized_parts.join("; "),
        segments,
        resolutions,
        substitutions,
        language,
        warnings,
    }
}

fn find_bounded(haystack: &str, needle: &str) -> Option<usize> {
    let mut search_from = 0;
    while let Some(offset) = haystack[search_from..].find(needle) {
        let absolute = search_from + offset;
        let before_ok = absolute == 0
            || !haystack[..absolute]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_alphanumeric());
        let after = absolute + needle.len();
        let after_ok = after == haystack.len()
            || !haystack[after..]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphanumeric());
        if before_ok && after_ok {
            return Some(absolute);
        }
        search_from = absolute + needle.len();
    }
    None
}

fn split_compound(text: &str) -> Vec<(usize, usize)> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut index = 0;
    while index < text.len() {
        let rest = &text[index..];
        match COMPOUND_SEPARATORS
            .iter()
            .find(|separator| rest.starts_with(*separator))
        {
            Some(separator) => {
                if let Some((trimmed_start, trimmed_end)) = trim_span(text, start, index) {
                    parts.push((trimmed_start, trimmed_end));
                }
                index += separator.len();
                start = index;
            }
            None => {
                let ch = rest.chars().next().expect("non-empty rest");
                index += ch.len_utf8();
            }
        }
    }
    if let Some((trimmed_start, trimmed_end)) = trim_span(text, start, text.len()) {
        parts.push((trimmed_start, trimmed_end));
    }
    parts
}

fn trim_span(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    let slice = &text[start..end];
    let leading = slice.len() - slice.trim_start().len();
    let trailing = slice.len() - slice.trim_end().len();
    let trimmed_start = start + leading;
    let trimmed_end = end.saturating_sub(trailing);
    (trimmed_start < trimmed_end).then_some((trimmed_start, trimmed_end))
}

fn detect_language(text: &str) -> &'static str {
    if text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)) {
        "zh"
    } else if text.chars().any(|c| c.is_ascii_alphabetic()) {
        "en"
    } else {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_compound_tasks_and_keeps_original_spans() {
        let text = "定位 A 项目的芯片供应；然后评估对交付任务的影响";
        let normalized = normalize_task_text(text, &[]);
        assert_eq!(normalized.segments.len(), 2);
        for segment in &normalized.segments {
            assert_eq!(&text[segment.start..segment.end], segment.text);
        }
        assert_eq!(normalized.segments[0].text, "定位 A 项目的芯片供应");
        assert_eq!(normalized.segments[1].text, "评估对交付任务的影响");
        assert_eq!(
            normalized.text,
            "定位 A 项目的芯片供应; 评估对交付任务的影响"
        );
        assert_eq!(normalized.language, "zh");
    }

    #[test]
    fn keeps_single_task_whole_and_collapses_whitespace() {
        let text = "  查找   File\t到 API 的路径\u{3000} ";
        let normalized = normalize_task_text(text, &[]);
        assert_eq!(normalized.segments.len(), 1);
        assert_eq!(normalized.text, "查找 File 到 API 的路径");
        assert_eq!(normalized.language, "zh");
    }

    #[test]
    fn resolves_pronouns_from_the_most_recent_context_mention() {
        let text = "它是否依赖交付任务";
        let normalized = normalize_task_text(text, &["A 项目".to_string(), "芯片供应".to_string()]);
        assert_eq!(normalized.segments[0].text, "芯片供应是否依赖交付任务");
        assert_eq!(normalized.resolutions.len(), 1);
        assert_eq!(normalized.resolutions[0].resolved, "芯片供应");
        assert!(!normalized.warnings.iter().any(|w| w.contains("unresolved")));
    }

    #[test]
    fn unresolved_pronoun_is_reported_not_silently_kept() {
        let normalized = normalize_task_text("它依赖什么", &[]);
        assert_eq!(normalized.resolutions.len(), 0);
        assert!(
            normalized
                .warnings
                .iter()
                .any(|warning| warning.contains("pronoun left unresolved")),
            "{:?}",
            normalized.warnings
        );
    }

    #[test]
    fn unifies_units_with_original_substitution_spans() {
        let text = "修复耗时 300 毫秒，目标是 2 个工作日";
        let normalized = normalize_task_text(text, &[]);
        assert!(normalized.text.contains("300 ms"), "{}", normalized.text);
        assert!(
            normalized.text.contains("2 workdays"),
            "{}",
            normalized.text
        );
        assert_eq!(normalized.substitutions.len(), 2);
        assert_eq!(
            &text[normalized.substitutions[0].start..normalized.substitutions[0].end],
            "毫秒"
        );
        assert_eq!(normalized.substitutions[0].normalized, "ms");
    }

    #[test]
    fn truncates_excess_segments_and_warns() {
        let text = "一；二；三；四；五；六；七";
        let normalized = normalize_task_text(text, &[]);
        assert_eq!(normalized.segments.len(), MAX_NORMALIZED_SEGMENTS);
        assert!(
            normalized
                .warnings
                .iter()
                .any(|warning| warning.contains("kept the first")),
            "{:?}",
            normalized.warnings
        );
    }

    #[test]
    fn detects_english_and_unknown_scripts() {
        assert_eq!(normalize_task_text("find path", &[]).language, "en");
        assert_eq!(normalize_task_text("12345", &[]).language, "unknown");
    }
}
