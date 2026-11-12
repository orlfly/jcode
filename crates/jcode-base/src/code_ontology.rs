//! Design-time bridge between runtime memory ontology types and code classes.
//!
//! The memory ontology describes semantic instances. This module records how a
//! codebase chooses to represent those semantics as Rust types, invariants and
//! lifecycle transitions. It is intentionally a validation layer, not a code
//! generator: a concept may be explicitly unmapped, and source checks are
//! shallow declaration scans rather than a Rust parser.

use anyhow::{Context, Result};
use jcode_memory_types::{Ontology, migration::ontology_content_hash};
use serde::{Deserialize, Serialize};
use std::path::Path;

fn default_version() -> u32 {
    1
}

/// A project-level design map.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeOntology {
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default)]
    pub concepts: Vec<CodeConcept>,
}

impl Default for CodeOntology {
    fn default() -> Self {
        Self {
            version: default_version(),
            concepts: Vec::new(),
        }
    }
}

/// How the code represents a semantic concept.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum CodeConceptKind {
    Entity,
    ValueObject,
    Aggregate,
    Service,
    Policy,
    Infrastructure,
}

/// Pinned ontology identity so a design decision can be reproduced later.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OntologyPin {
    pub id: String,
    pub version: u32,
    pub hash: String,
}

/// State machine declared by a code concept.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeLifecycle {
    #[serde(default)]
    pub states: Vec<String>,
    #[serde(default)]
    pub transitions: Vec<CodeTransition>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeTransition {
    pub from: String,
    pub to: String,
}

/// One code type and its relation to an ontology type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeConcept {
    pub id: String,
    pub kind: CodeConceptKind,
    /// Declaration name inside `source`, for example `Contract`.
    pub rust_type: String,
    /// Repository-relative source file containing the declaration.
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ontology_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unmapped_reason: Option<String>,
    /// Code fields/values that carry ontology properties. Accept either
    /// `property_id` or `properties.property_id`.
    #[serde(default)]
    pub fields: Vec<String>,
    #[serde(default)]
    pub invariants: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifecycle: Option<CodeLifecycle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ontology: Option<OntologyPin>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CodeOntologySeverity {
    Error,
    Warning,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeOntologyIssue {
    pub code: String,
    pub severity: CodeOntologySeverity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concept: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeOntologyReport {
    pub checked: usize,
    pub mapped: usize,
    pub unmapped: usize,
    pub issues: Vec<CodeOntologyIssue>,
}

impl CodeOntologyReport {
    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.severity == CodeOntologySeverity::Error)
            .count()
    }

    pub fn is_valid(&self) -> bool {
        self.error_count() == 0
    }

    pub fn summary(&self) -> String {
        format!(
            "{} concept(s): {} mapped, {} unmapped, {} error(s), {} warning(s)",
            self.checked,
            self.mapped,
            self.unmapped,
            self.error_count(),
            self.issues.len() - self.error_count()
        )
    }

    fn error(&mut self, code: &str, concept: Option<&str>, message: impl Into<String>) {
        self.issues.push(CodeOntologyIssue {
            code: code.to_string(),
            severity: CodeOntologySeverity::Error,
            concept: concept.map(str::to_string),
            message: message.into(),
        });
    }

    fn warning(&mut self, code: &str, concept: Option<&str>, message: impl Into<String>) {
        self.issues.push(CodeOntologyIssue {
            code: code.to_string(),
            severity: CodeOntologySeverity::Warning,
            concept: concept.map(str::to_string),
            message: message.into(),
        });
    }
}

/// Load a YAML (or JSON) design map.
pub fn load_code_ontology(path: &Path) -> Result<CodeOntology> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read code ontology {}", path.display()))?;
    let document: CodeOntology = serde_yaml::from_str(&raw)
        .with_context(|| format!("parse code ontology {}", path.display()))?;
    anyhow::ensure!(
        document.version == 1,
        "unsupported code ontology version {} (expected 1)",
        document.version
    );
    Ok(document)
}

/// Validate a design map against an ontology and a repository root.
pub fn validate_code_ontology(
    document: &CodeOntology,
    ontology: &Ontology,
    root: &Path,
) -> CodeOntologyReport {
    let mut report = CodeOntologyReport::default();
    let ontology_hash = ontology_content_hash(ontology);
    let mut ids = std::collections::HashSet::new();
    let mut rust_types = std::collections::HashMap::new();

    for concept in &document.concepts {
        report.checked += 1;
        if !ids.insert(concept.id.clone()) {
            report.error(
                "C001",
                Some(&concept.id),
                format!("duplicate concept id '{}'", concept.id),
            );
        }
        if let Some(previous) = rust_types.insert(concept.rust_type.clone(), concept.id.clone()) {
            report.warning(
                "C002",
                Some(&concept.id),
                format!(
                    "rust type '{}' is also declared by concept '{}'",
                    concept.rust_type, previous
                ),
            );
        }
        check_source(concept, root, &mut report);

        let Some(type_id) = concept.ontology_type.as_deref() else {
            report.unmapped += 1;
            if concept
                .unmapped_reason
                .as_deref()
                .map(str::trim)
                .is_none_or(str::is_empty)
            {
                report.error(
                    "C003",
                    Some(&concept.id),
                    "unmapped concept needs a non-empty unmapped_reason",
                );
            }
            continue;
        };
        report.mapped += 1;

        let Some(type_def) = ontology.get_type(type_id) else {
            report.error(
                "C004",
                Some(&concept.id),
                format!("ontology type '{}' does not exist", type_id),
            );
            continue;
        };

        match &concept.ontology {
            Some(pin) => {
                if pin.id != ontology.id {
                    report.error(
                        "C005",
                        Some(&concept.id),
                        format!(
                            "ontology pin id '{}' does not match '{}'",
                            pin.id, ontology.id
                        ),
                    );
                }
                if pin.version != ontology.version {
                    report.error(
                        "C006",
                        Some(&concept.id),
                        format!(
                            "ontology pin version {} does not match {}",
                            pin.version, ontology.version
                        ),
                    );
                }
                if pin.hash != ontology_hash {
                    report.error(
                        "C007",
                        Some(&concept.id),
                        format!(
                            "ontology pin hash '{}' does not match '{}'",
                            pin.hash, ontology_hash
                        ),
                    );
                }
            }
            None => report.warning(
                "C008",
                Some(&concept.id),
                "concept has no ontology id/version/hash pin",
            ),
        }

        for property in type_def
            .properties
            .iter()
            .filter(|property| property.required)
        {
            let mapped = concept.fields.iter().any(|field| {
                field == &property.id
                    || field
                        .strip_prefix("properties.")
                        .is_some_and(|field| field == property.id)
            });
            if !mapped {
                report.error(
                    "C009",
                    Some(&concept.id),
                    format!(
                        "required ontology property '{}' has no mapped code field",
                        property.id
                    ),
                );
            }
        }

        if let Some(lifecycle) = &concept.lifecycle {
            for state in &lifecycle.states {
                let known = type_def
                    .lifecycle
                    .states
                    .iter()
                    .any(|candidate| candidate.as_str() == state);
                if !known {
                    report.error(
                        "C010",
                        Some(&concept.id),
                        format!(
                            "lifecycle state '{}' is not declared by type '{}'",
                            state, type_id
                        ),
                    );
                }
            }
            for transition in &lifecycle.transitions {
                let allowed = type_def.lifecycle.transitions.iter().any(|candidate| {
                    candidate.from.as_str() == transition.from
                        && candidate.to.as_str() == transition.to
                });
                if !allowed {
                    report.error(
                        "C011",
                        Some(&concept.id),
                        format!(
                            "lifecycle transition {} -> {} is not allowed by type '{}'",
                            transition.from, transition.to, type_id
                        ),
                    );
                }
            }
        }
    }

    report
}

fn check_source(concept: &CodeConcept, root: &Path, report: &mut CodeOntologyReport) {
    if concept.rust_type.trim().is_empty() {
        report.error("C012", Some(&concept.id), "rust_type is empty");
        return;
    }
    if concept.source.trim().is_empty() {
        report.error("C013", Some(&concept.id), "source is empty");
        return;
    }
    let path = root.join(&concept.source);
    let source = match std::fs::read_to_string(&path) {
        Ok(source) => source,
        Err(error) => {
            report.error(
                "C014",
                Some(&concept.id),
                format!("cannot read source {}: {error}", path.display()),
            );
            return;
        }
    };
    let declared = ["struct", "enum", "trait", "type"]
        .iter()
        .any(|keyword| source.contains(&format!("{keyword} {}", concept.rust_type)));
    if !declared {
        report.error(
            "C015",
            Some(&concept.id),
            format!(
                "source {} does not declare `{}` with struct/enum/trait/type",
                concept.source, concept.rust_type
            ),
        );
    }
}

#[cfg(test)]
mod tests;
