//! Deterministic index nodes for a skill's Markdown body and sidecar scripts.
//!
//! `SKILL.md` remains the source of truth. This module only derives bounded,
//! retrieval-sized nodes from it: sections, steps, fenced code blocks and
//! script metadata. No model call is involved, and no source text is rewritten
//! except for length caps.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;

use crate::memory::MemoryEntry;
use crate::memory_graph::MemoryGraph;
use crate::memory_types::MemoryCategory;

use super::Skill;

const MAX_PARTS: usize = 96;
const MAX_SECTION_CHARS: usize = 4_000;
const MAX_STEP_CHARS: usize = 1_200;
const MAX_CODE_CHARS: usize = 4_000;
const MAX_SCRIPT_CHARS: usize = 1_200;
const MAX_SCRIPT_FILES: usize = 32;
const MAX_SCRIPT_BYTES: u64 = 512 * 1024;
const MAX_SCRIPT_SYMBOLS: usize = 12;
const MIN_STEP_CHARS: usize = 12;

static PYTHON_SYMBOL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)").unwrap()
});
static JS_SYMBOL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class)\s+([A-Za-z_$][A-Za-z0-9_$]*)",
    )
    .unwrap()
});
static SHELL_SYMBOL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{").unwrap());

#[derive(Default)]
struct SectionDraft {
    level: usize,
    title: String,
    slug: String,
    prose: Vec<String>,
    steps: Vec<String>,
}

fn overview_section() -> SectionDraft {
    SectionDraft {
        level: 0,
        title: "Overview".to_string(),
        slug: "overview".to_string(),
        prose: Vec::new(),
        steps: Vec::new(),
    }
}

/// Reconcile the derived index nodes for one skill.
///
/// Returns `(mirrored, pruned)`. The caller owns the store save.
pub(super) fn sync_skill_parts(graph: &mut MemoryGraph, skill: &Skill) -> (usize, usize) {
    let root = format!("skill:{}", skill.name);
    let parts = build_skill_parts(skill);
    let desired: HashSet<String> = parts.iter().map(|part| part.id.clone()).collect();

    let mut mirrored = 0;
    for part in parts {
        if upsert_part(graph, &root, part) {
            mirrored += 1;
        }
    }

    let prefix = format!("{root}::");
    let stale: Vec<String> = graph
        .memories
        .keys()
        .filter(|id| id.starts_with(&prefix) && !desired.contains(*id))
        .cloned()
        .collect();
    let mut pruned = 0;
    for id in stale {
        if graph.remove_memory(&id).is_some() {
            pruned += 1;
        }
    }
    (mirrored, pruned)
}

/// Remove every derived part for a skill, for example when a global skill's
/// project mirror must not duplicate the canonical global part nodes.
pub(super) fn prune_skill_parts(graph: &mut MemoryGraph, skill_name: &str) -> usize {
    let prefix = format!("skill:{skill_name}::");
    let stale: Vec<String> = graph
        .memories
        .keys()
        .filter(|id| id.starts_with(&prefix))
        .cloned()
        .collect();
    let mut pruned = 0;
    for id in stale {
        if graph.remove_memory(&id).is_some() {
            pruned += 1;
        }
    }
    pruned
}

/// Remove derived parts under a root id, used when the root mirror itself is
/// pruned because the skill disappeared.
pub(super) fn prune_skill_parts_under(graph: &mut MemoryGraph, root_id: &str) -> usize {
    let prefix = format!("{root_id}::");
    let stale: Vec<String> = graph
        .memories
        .keys()
        .filter(|id| id.starts_with(&prefix))
        .cloned()
        .collect();
    let mut pruned = 0;
    for id in stale {
        if graph.remove_memory(&id).is_some() {
            pruned += 1;
        }
    }
    pruned
}

#[cfg(test)]
pub(super) fn is_skill_part_id(id: &str) -> bool {
    id.starts_with("skill:") && id.contains("::")
}

fn upsert_part(graph: &mut MemoryGraph, root: &str, entry: MemoryEntry) -> bool {
    let id = entry.id.clone();
    if let Some(existing) = graph.get_memory(&id) {
        let old_tags = existing.tags.clone();
        let changed = existing.content != entry.content
            || existing.tags != entry.tags
            || existing.category != entry.category
            || existing.subject_key != entry.subject_key
            || existing.search_text != entry.search_text
            || existing.source != entry.source
            || existing.trust != entry.trust;
        let next_updated_at = entry.updated_at;
        if let Some(existing) = graph.get_memory_mut(&id) {
            existing.category = entry.category;
            existing.content = entry.content;
            existing.tags = entry.tags;
            existing.source = entry.source;
            existing.trust = entry.trust;
            existing.subject_key = entry.subject_key;
            existing.search_text = entry.search_text;
            existing.active = true;
            existing.superseded_by = None;
            existing.artifact = None;
            if changed {
                existing.updated_at = next_updated_at;
                existing.embedding = None;
                existing.embedding_model = None;
            }
        }
        let new_tags = graph
            .get_memory(&id)
            .map(|entry| entry.tags.clone())
            .unwrap_or_default();
        for tag in old_tags.iter().filter(|tag| !new_tags.contains(tag)) {
            graph.untag_memory(&id, tag);
        }
        for tag in new_tags.iter().filter(|tag| !old_tags.contains(tag)) {
            graph.tag_memory(&id, tag);
        }
        let had_parent_edge = graph.get_edges(&id).iter().any(|edge| {
            edge.target == root && matches!(edge.kind, crate::memory_types::EdgeKind::DerivedFrom)
        });
        graph.add_edge(&id, root, crate::memory_types::EdgeKind::DerivedFrom);
        changed || !had_parent_edge
    } else {
        graph.add_memory(entry);
        graph.add_edge(&id, root, crate::memory_types::EdgeKind::DerivedFrom);
        true
    }
}

/// Build the derived nodes without touching a graph. Exposed for tests.
pub(super) fn build_skill_parts(skill: &Skill) -> Vec<MemoryEntry> {
    let mut parts = Vec::new();
    let mut section = overview_section();
    let mut section_counts: HashMap<String, usize> = HashMap::new();
    let mut pending_step: Option<String> = None;
    let mut step_order = 0usize;

    let lines: Vec<&str> = skill.content.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];

        if let Some((fence_char, fence_len, lang)) = code_fence_start(line) {
            flush_step(
                &mut section,
                &mut pending_step,
                &mut step_order,
                skill,
                &mut parts,
            );
            let mut code_lines = Vec::new();
            i += 1;
            while i < lines.len() {
                if code_fence_end(lines[i], fence_char, fence_len) {
                    i += 1;
                    break;
                }
                code_lines.push(lines[i]);
                i += 1;
            }
            push_code_part(skill, &section, &lang, &code_lines, &mut parts);
            continue;
        }

        if let Some((level, title)) = parse_heading(line) {
            flush_step(
                &mut section,
                &mut pending_step,
                &mut step_order,
                skill,
                &mut parts,
            );
            flush_section(skill, &mut section, &mut parts);
            let slug = unique_slug(title, &mut section_counts);
            section = SectionDraft {
                level,
                title: title.to_string(),
                slug,
                prose: Vec::new(),
                steps: Vec::new(),
            };
            i += 1;
            continue;
        }

        if let Some(item) = list_item_text(line) {
            flush_step(
                &mut section,
                &mut pending_step,
                &mut step_order,
                skill,
                &mut parts,
            );
            if section_has_no_index(&section.title) {
                // Evidence ids and unchecked TODO markers are metadata, not
                // executable procedure steps.
            } else if section_suppresses_steps(&section.title) {
                section.prose.push(format!("- {item}"));
            } else if is_metadata_list_item(&item) || is_checklist_item(&item) {
                // Skip generated metadata bullets such as `subject:` or
                // `weight (corrections count double):`.
            } else if item.chars().count() >= MIN_STEP_CHARS {
                pending_step = Some(item);
            }
            i += 1;
            continue;
        }

        if let Some(step) = pending_step.as_mut() {
            if line.trim().is_empty() {
                flush_step(
                    &mut section,
                    &mut pending_step,
                    &mut step_order,
                    skill,
                    &mut parts,
                );
                i += 1;
                continue;
            }
            if line.starts_with("  ") || line.starts_with('\t') {
                step.push(' ');
                step.push_str(line.trim());
                i += 1;
                continue;
            }
            flush_step(
                &mut section,
                &mut pending_step,
                &mut step_order,
                skill,
                &mut parts,
            );
        }

        if !line.trim().is_empty() {
            section.prose.push(line.trim_end().to_string());
        } else if !section.prose.is_empty() {
            section.prose.push(String::new());
        }
        i += 1;
    }

    flush_step(
        &mut section,
        &mut pending_step,
        &mut step_order,
        skill,
        &mut parts,
    );
    flush_section(skill, &mut section, &mut parts);
    let scripts = script_parts(skill);
    let script_budget = scripts.len().min(16);
    if parts.len() + scripts.len() > MAX_PARTS {
        parts.truncate(MAX_PARTS.saturating_sub(script_budget));
    }
    parts.extend(scripts.into_iter().take(script_budget));

    let mut seen = HashSet::new();
    parts.retain(|part| seen.insert(part.id.clone()));
    parts.truncate(MAX_PARTS);
    parts
}

fn push_code_part(
    skill: &Skill,
    section: &SectionDraft,
    lang: &str,
    code_lines: &[&str],
    parts: &mut Vec<MemoryEntry>,
) {
    let code = code_lines.join("\n");
    if code.trim().is_empty() {
        return;
    }
    let lang = if lang.trim().is_empty() {
        "text".to_string()
    } else {
        compact_key(lang, 24)
    };
    let body = format!(
        "Code block ({lang}) in skill `{}` / {}:\n\n{}",
        skill.name,
        section.title,
        truncate_chars(&code, MAX_CODE_CHARS)
    );
    let mut tags = part_tags(skill, "code");
    tags.push(format!("lang:{lang}"));
    tags.push(format!("skill-section:{}", section.slug));
    let category = MemoryCategory::Observation;
    parts.push(part_entry(
        skill,
        "code",
        &format!("{lang}-{}", &section.slug),
        category,
        body,
        tags,
    ));
}

fn flush_step(
    section: &mut SectionDraft,
    pending_step: &mut Option<String>,
    step_order: &mut usize,
    skill: &Skill,
    parts: &mut Vec<MemoryEntry>,
) {
    let Some(text) = pending_step.take() else {
        return;
    };
    let text = text.trim();
    if text.chars().count() < MIN_STEP_CHARS {
        return;
    }
    *step_order += 1;
    section.steps.push(text.to_string());
    let mut tags = part_tags(skill, "step");
    tags.push(format!("skill-section:{}", section.slug));
    tags.push(format!("step-order:{}", *step_order));
    let content = format!(
        "Step {} in skill `{}` / {}:\n{}",
        *step_order,
        skill.name,
        section.title,
        truncate_chars(text, MAX_STEP_CHARS)
    );
    parts.push(part_entry(
        skill,
        "step",
        &format!("{}-{}", *step_order, section.slug),
        MemoryCategory::ProcessRule,
        content,
        tags,
    ));
}

fn flush_section(skill: &Skill, section: &mut SectionDraft, parts: &mut Vec<MemoryEntry>) {
    if section.level == 0 && section.prose.is_empty() && section.steps.is_empty() {
        *section = overview_section();
        return;
    }
    // Generated skills carry a long "Evidence" list that is already represented
    // by artifact.declared_evidence/resolved_evidence and derived_from edges.
    // Indexing those memory ids again as prose only adds noise.
    if section_has_no_index(&section.title) {
        *section = overview_section();
        return;
    }
    let mut content = if section.level == 0 {
        format!("# {} / {}", skill.name, section.title)
    } else {
        format!("{} {}", "#".repeat(section.level), section.title)
    };
    if !section.prose.is_empty() {
        content.push_str("\n\n");
        content.push_str(section.prose.join("\n").trim());
    }
    if !section.steps.is_empty() {
        content.push_str(&format!(
            "\n\nThis section contains {} indexed step(s).",
            section.steps.len()
        ));
    }
    let mut tags = part_tags(skill, "section");
    tags.push(format!("skill-section:{}", section.slug));
    parts.push(part_entry(
        skill,
        "section",
        &section.slug,
        MemoryCategory::ProcessRule,
        truncate_chars(&content, MAX_SECTION_CHARS),
        tags,
    ));
    *section = overview_section();
}

fn part_entry(
    skill: &Skill,
    kind: &str,
    key: &str,
    category: MemoryCategory,
    content: String,
    tags: Vec<String>,
) -> MemoryEntry {
    let key = compact_key(key, 64);
    let hash = short_hash(&content);
    let id = format!("skill:{}::{kind}:{key}:{hash}", skill.name);
    let subject_key = format!("skill:{}::{kind}:{key}", skill.name);
    let mut entry = MemoryEntry::new(category, content).with_id(id);
    entry.tags = tags;
    entry.source = Some("skill_registry".to_string());
    entry.trust = crate::memory::TrustLevel::High;
    entry.subject_key = Some(subject_key);
    entry.search_text =
        crate::memory_types::normalize_memory_search_text(&entry.content, &entry.tags);
    entry
}

fn part_tags(skill: &Skill, kind: &str) -> Vec<String> {
    vec![
        format!("skill:{}", skill.name),
        format!("skill-part:{kind}"),
    ]
}

fn script_parts(skill: &Skill) -> Vec<MemoryEntry> {
    let Some(dir) = skill.path.parent() else {
        return Vec::new();
    };
    let mut files = Vec::new();
    collect_script_files(dir, dir, 0, &mut files);
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files.truncate(MAX_SCRIPT_FILES);

    let mut parts = Vec::new();
    for (relative, path) in files {
        let Ok(metadata) = std::fs::metadata(&path) else {
            continue;
        };
        if metadata.len() > MAX_SCRIPT_BYTES {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let lang = script_language(&path).unwrap_or("text");
        let hash = short_hash(&bytes);
        let text = String::from_utf8_lossy(&bytes);
        let symbols = script_symbols(lang, &text);
        let first_line = text
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .unwrap_or("")
            .chars()
            .take(160)
            .collect::<String>();
        let mut content = format!(
            "Script `{relative}` in skill `{}` ({lang}, {} bytes, sha256 {hash})",
            skill.name,
            metadata.len()
        );
        if !symbols.is_empty() {
            content.push_str(&format!("\nSymbols: {}", symbols.join(", ")));
        }
        if !first_line.is_empty() {
            content.push_str(&format!("\nFirst line: {first_line}"));
        }
        let mut tags = part_tags(skill, "script");
        tags.push(format!("lang:{lang}"));
        tags.push(format!("script:{}", compact_key(&relative, 48)));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if metadata.permissions().mode() & 0o111 != 0 {
                tags.push("script-executable".to_string());
            }
        }
        let key = format!("{}-{}", compact_key(&relative, 48), hash);
        let mut entry = part_entry(
            skill,
            "script",
            &key,
            MemoryCategory::Observation,
            truncate_chars(&content, MAX_SCRIPT_CHARS),
            tags,
        );
        let search_text = format!("{content}\n{}", truncate_chars(&text, MAX_CODE_CHARS));
        entry.search_text =
            crate::memory_types::normalize_memory_search_text(&search_text, &entry.tags);
        parts.push(entry);
    }
    parts
}

fn collect_script_files(root: &Path, dir: &Path, depth: usize, out: &mut Vec<(String, PathBuf)>) {
    if depth > 3 || out.len() >= MAX_SCRIPT_FILES {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if out.len() >= MAX_SCRIPT_FILES {
            break;
        }
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.')
            || matches!(
                name.as_str(),
                "node_modules" | "target" | "__pycache__" | "dist" | "build"
            )
        {
            continue;
        }
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            collect_script_files(root, &path, depth + 1, out);
            continue;
        }
        if !file_type.is_file() || name == "SKILL.md" {
            continue;
        }
        if script_language(&path).is_none() {
            continue;
        }
        let Ok(relative) = path.strip_prefix(root) else {
            continue;
        };
        out.push((relative.to_string_lossy().replace('\\', "/"), path));
    }
}

fn script_language(path: &Path) -> Option<&'static str> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    Some(match extension.as_str() {
        "py" => "python",
        "sh" | "bash" => "bash",
        "zsh" => "zsh",
        "js" | "mjs" | "cjs" => "javascript",
        "ts" => "typescript",
        "tsx" => "tsx",
        "rb" => "ruby",
        "go" => "go",
        "rs" => "rust",
        "pl" => "perl",
        "ps1" => "powershell",
        _ => return None,
    })
}

fn script_symbols(lang: &str, text: &str) -> Vec<String> {
    let regex = match lang {
        "python" => &*PYTHON_SYMBOL,
        "javascript" | "typescript" | "tsx" => &*JS_SYMBOL,
        "bash" | "zsh" => &*SHELL_SYMBOL,
        _ => return Vec::new(),
    };
    let mut seen = HashSet::new();
    let mut symbols = Vec::new();
    for capture in regex.captures_iter(text) {
        let Some(symbol) = capture.get(1).map(|m| m.as_str().to_string()) else {
            continue;
        };
        if seen.insert(symbol.clone()) {
            symbols.push(symbol);
            if symbols.len() >= MAX_SCRIPT_SYMBOLS {
                break;
            }
        }
    }
    symbols
}

fn code_fence_start(line: &str) -> Option<(char, usize, String)> {
    let trimmed = line.trim_start();
    let fence_char = trimmed.chars().next()?;
    if fence_char != '`' && fence_char != '~' {
        return None;
    }
    let fence_len = trimmed.chars().take_while(|c| *c == fence_char).count();
    if fence_len < 3 {
        return None;
    }
    let rest = trimmed[fence_len..].trim();
    let lang = rest.split_whitespace().next().unwrap_or("").to_string();
    Some((fence_char, fence_len, lang))
}

fn code_fence_end(line: &str, fence_char: char, fence_len: usize) -> bool {
    let trimmed = line.trim();
    trimmed.chars().take_while(|c| *c == fence_char).count() >= fence_len
        && trimmed.chars().all(|c| c == fence_char)
}

fn parse_heading(line: &str) -> Option<(usize, &str)> {
    let trimmed = line.trim_start();
    let level = trimmed.chars().take_while(|c| *c == '#').count();
    if !(1..=6).contains(&level) {
        return None;
    }
    let rest = &trimmed[level..];
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    let title = rest.trim();
    (!title.is_empty()).then_some((level, title))
}

fn list_item_text(line: &str) -> Option<String> {
    let indent = line.chars().take_while(|c| *c == ' ' || *c == '\t').count();
    if indent > 6 {
        return None;
    }
    let trimmed = line.trim_start();
    let rest = if let Some(rest) = trimmed.strip_prefix("- ") {
        rest
    } else if let Some(rest) = trimmed.strip_prefix("* ") {
        rest
    } else if let Some(rest) = trimmed.strip_prefix("+ ") {
        rest
    } else {
        let digits = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits == 0 {
            return None;
        }
        let after = &trimmed[digits..];
        if let Some(rest) = after.strip_prefix(". ") {
            rest
        } else if let Some(rest) = after.strip_prefix(") ") {
            rest
        } else {
            return None;
        }
    };
    let text = rest.trim();
    (!text.is_empty()).then(|| text.to_string())
}

fn section_suppresses_steps(title: &str) -> bool {
    let title = title.to_ascii_lowercase();
    title.contains("evidence")
        || title.contains("reference")
        || title.contains("sources")
        || title.contains("see also")
}

fn section_has_no_index(title: &str) -> bool {
    let title = title.trim().to_ascii_lowercase();
    title == "evidence"
        || title == "evidence list"
        || title == "evidence ids"
        || title.starts_with("evidence (")
        || title == "todo"
        || title == "todos"
}

fn is_metadata_list_item(item: &str) -> bool {
    let Some((key, _)) = item.split_once(':') else {
        return false;
    };
    let key = key.trim().to_ascii_lowercase();
    key.len() <= 48
        && (matches!(
            key.as_str(),
            "subject"
                | "evidence"
                | "stores"
                | "sessions"
                | "name"
                | "description"
                | "path"
                | "scope"
                | "tags"
                | "confidence"
                | "source"
                | "status"
        ) || key.starts_with("weight"))
}

fn is_checklist_item(item: &str) -> bool {
    let item = item.trim_start();
    item.starts_with("[ ]") || item.starts_with("[x]") || item.starts_with("[X]")
}

fn unique_slug(title: &str, counts: &mut HashMap<String, usize>) -> String {
    let base = compact_key(title, 48);
    let count = counts.entry(base.clone()).or_insert(0);
    *count += 1;
    if *count == 1 {
        base
    } else {
        format!("{base}-{count}")
    }
}

fn compact_key(value: &str, max: usize) -> String {
    let mut out = String::new();
    let mut pending_dash = false;
    for ch in value.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            if pending_dash && !out.is_empty() {
                out.push('-');
            }
            pending_dash = false;
            out.push(ch);
        } else {
            pending_dash = true;
        }
        if out.len() >= max {
            break;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        "part".to_string()
    } else {
        out
    }
}

fn truncate_chars(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    let mut out: String = value.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn short_hash(value: impl AsRef<[u8]>) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(value.as_ref());
    digest
        .iter()
        .take(4)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
#[path = "skill_index_tests.rs"]
mod tests;
