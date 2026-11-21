//! Deterministic extraction of explicit domain facts from user prose.
//!
//! This runs before compaction, when the original sentences are still
//! available. It is intentionally conservative: only declarative user
//! statements with an explicit subject, predicate and object become typed
//! facts. Questions, negations and assistant claims are ignored.

use std::sync::LazyLock;

use jcode_memory_types::DomainFact;
use regex::Regex;

use crate::message::{ContentBlock, Message, Role};

pub const MAX_DOMAIN_FACTS: usize = 32;

/// Result of one extraction pass.
#[derive(Debug, Clone, Default)]
pub struct DomainFactExtraction {
    pub facts: Vec<DomainFact>,
    pub warnings: Vec<String>,
}

static ENGLISH_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^(?P<subject>[\p{L}\p{N}][\p{L}\p{N} _./'-]{0,79}?)\s+(?P<predicate>refers to|depends on|means|requires|contains|uses|owns|is|are)\s+(?P<object>[^.!?\n]{2,160})$",
    )
    .expect("english domain fact regex")
});

static CHINESE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(?P<subject>[\p{Han}A-Za-z0-9_ .-]{1,60}?)\s*(?P<predicate>不属于|不是|属于|使用|依赖|需要|负责|包含|是)\s*(?P<object>[^。！？\n]{1,120})$",
    )
    .expect("chinese domain fact regex")
});

/// Extract explicit facts from user text blocks only.
pub fn extract_domain_facts(session_id: &str, messages: &[Message]) -> DomainFactExtraction {
    let mut extraction = DomainFactExtraction::default();
    for (message_index, message) in messages.iter().enumerate() {
        if message.role != Role::User {
            continue;
        }
        for block in message.content.iter() {
            let ContentBlock::Text { text, .. } = block else {
                continue;
            };
            for (sentence_index, sentence) in split_sentences(text).into_iter().enumerate() {
                if extraction.facts.len() >= MAX_DOMAIN_FACTS {
                    return extraction;
                }
                let Some((subject, predicate, object)) = parse_sentence(&sentence) else {
                    continue;
                };
                let id = fact_id(&subject, &predicate, &object);
                if extraction.facts.iter().any(|fact| fact.id == id) {
                    continue;
                }
                extraction.facts.push(DomainFact {
                    id,
                    subject,
                    predicate,
                    object,
                    statement: sentence.chars().take(300).collect(),
                    source_session: session_id.to_string(),
                    source_role: "user".to_string(),
                    source_locator: format!("message:{message_index}#sentence:{sentence_index}"),
                    confidence: 0.75,
                    created_at: chrono::Utc::now(),
                });
            }
        }
    }
    extraction
}

fn split_sentences(text: &str) -> Vec<String> {
    text.split(|ch: char| matches!(ch, '.' | '!' | '?' | '。' | '！' | '？' | '\n'))
        .map(str::trim)
        .filter(|sentence| sentence.chars().count() >= 4)
        .map(|sentence| sentence.chars().take(400).collect())
        .collect()
}

fn parse_sentence(sentence: &str) -> Option<(String, String, String)> {
    let trimmed = sentence.trim().trim_matches(|ch: char| {
        matches!(
            ch,
            '-' | '*' | '`' | '"' | '\'' | ':' | ';' | ',' | '(' | ')' | '[' | ']'
        )
    });
    if trimmed.is_empty() {
        return None;
    }
    let lower = trimmed.to_lowercase();
    if lower.contains(" not ") || lower.contains("n't ") {
        return None;
    }
    if trimmed.contains("不是")
        || trimmed.contains("不属于")
        || trimmed.contains("并非")
        || trimmed.contains("没有")
    {
        return None;
    }

    if let Some(captures) = ENGLISH_PATTERN.captures(trimmed) {
        let subject = clean_subject(&captures["subject"])?;
        let predicate = canonical_predicate(&captures["predicate"])?.to_string();
        let object = clean_object(&captures["object"])?;
        return Some((subject, predicate, object));
    }
    if let Some(captures) = CHINESE_PATTERN.captures(trimmed) {
        let subject = clean_subject(&captures["subject"])?;
        let predicate = canonical_predicate(&captures["predicate"])?.to_string();
        let object = clean_object(&captures["object"])?;
        return Some((subject, predicate, object));
    }
    None
}

fn canonical_predicate(raw: &str) -> Option<&'static str> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "is" | "are" | "means" | "refers to" | "是" | "属于" => Some("is_a"),
        "uses" | "使用" => Some("uses"),
        "depends on" | "依赖" => Some("depends_on"),
        "owns" | "负责" => Some("owns"),
        "requires" | "需要" => Some("requires"),
        "contains" | "包含" => Some("contains"),
        _ => None,
    }
}

fn clean_subject(raw: &str) -> Option<String> {
    let value = clean_value(raw);
    if value.chars().count() < 2 || value.chars().count() > 80 {
        return None;
    }
    if matches!(
        value.to_ascii_lowercase().as_str(),
        "it" | "this"
            | "that"
            | "they"
            | "them"
            | "we"
            | "he"
            | "she"
            | "there"
            | "these"
            | "those"
            | "其"
            | "它"
            | "这"
            | "那"
    ) {
        return None;
    }
    Some(value)
}

fn clean_object(raw: &str) -> Option<String> {
    let value = clean_value(raw);
    if value.chars().count() < 2 || value.chars().count() > 160 {
        return None;
    }
    let lower = value.to_ascii_lowercase();
    if lower.starts_with("not ") || lower.starts_with("no ") {
        return None;
    }
    Some(value)
}

fn clean_value(raw: &str) -> String {
    raw.trim()
        .trim_matches(|ch: char| {
            matches!(
                ch,
                '`' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | ':' | '-' | '*'
            )
        })
        .trim()
        .to_string()
}

fn fact_id(subject: &str, predicate: &str, object: &str) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for part in [
        subject.to_ascii_lowercase(),
        predicate.to_ascii_lowercase(),
        object.to_ascii_lowercase(),
    ] {
        for byte in part.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("domain:{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_message(text: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
                cache_control: None,
            }]
            .into(),
            timestamp: None,
            tool_duration_ms: None,
        }
    }

    fn assistant_message(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
                cache_control: None,
            }]
            .into(),
            timestamp: None,
            tool_duration_ms: None,
        }
    }

    #[test]
    fn extracts_explicit_english_facts() {
        let extraction = extract_domain_facts(
            "s1",
            &[user_message(
                "Billing service uses Stripe. The parser depends on tree-sitter.",
            )],
        );
        assert_eq!(extraction.facts.len(), 2, "{extraction:?}");
        assert!(extraction.facts.iter().any(|fact| {
            fact.subject == "Billing service" && fact.predicate == "uses" && fact.object == "Stripe"
        }));
        assert!(
            extraction
                .facts
                .iter()
                .any(|fact| { fact.subject == "The parser" && fact.predicate == "depends_on" })
        );
    }

    #[test]
    fn extracts_explicit_chinese_facts() {
        let extraction = extract_domain_facts(
            "s2",
            &[user_message("计费服务依赖 Stripe。搜索模块使用 Elastic。")],
        );
        assert_eq!(extraction.facts.len(), 2, "{extraction:?}");
        assert!(extraction.facts.iter().any(|fact| {
            fact.subject == "计费服务" && fact.predicate == "depends_on" && fact.object == "Stripe"
        }));
    }

    #[test]
    fn ignores_questions_negations_and_assistant_claims() {
        let extraction = extract_domain_facts(
            "s3",
            &[
                user_message("Does billing use Stripe?"),
                user_message("Billing is not Stripe."),
                user_message("计费服务不是 Stripe。"),
                assistant_message("Billing uses Stripe."),
            ],
        );
        assert!(extraction.facts.is_empty(), "{extraction:?}");
    }

    #[test]
    fn deduplicates_the_same_triple_and_keeps_provenance() {
        let extraction = extract_domain_facts(
            "s4",
            &[
                user_message("Billing uses Stripe."),
                user_message("Billing uses Stripe."),
            ],
        );
        assert_eq!(extraction.facts.len(), 1, "{extraction:?}");
        assert_eq!(extraction.facts[0].source_role, "user");
        assert!(extraction.facts[0].source_locator.starts_with("message:"));
    }

    #[test]
    fn rejects_pronoun_subjects_and_oversized_input() {
        let extraction = extract_domain_facts(
            "s5",
            &[
                user_message("It uses Stripe."),
                user_message(&format!("{} uses Stripe.", "x".repeat(100))),
            ],
        );
        assert!(extraction.facts.is_empty(), "{extraction:?}");
    }
}
