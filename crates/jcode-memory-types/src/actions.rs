//! Action types and confirmation gates for memory operations.
//!
//! Mirrors the enterprise-ontology `action_types` layer: each operation on the
//! memory graph declares its target, preconditions, effects and whether it needs
//! human confirmation. Confirmation gates (G1-G4) are derived from the
//! enterprise-ontology diff_instances design and applied to memory mutations.

use serde::{Deserialize, Serialize};

use crate::instance::{ExtractionMethod, MemoryStatus};
use crate::{MemoryEntry, MemoryGraph};

/// A permission role that may execute an action.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRole {
    #[default]
    Agent,
    User,
    System,
}

/// Declared action type for a memory operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryActionSpec {
    pub id: String,
    pub target: MemoryActionTarget,
    /// Human-readable preconditions checked before execution.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub preconditions: Vec<String>,
    /// Declared effects of the action.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<MemoryEffect>,
    /// Roles allowed to execute this action.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_roles: Vec<MemoryRole>,
    /// Whether the action requires explicit confirmation before it is applied.
    #[serde(default)]
    pub requires_confirmation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MemoryActionTarget {
    Memory,
    Link,
    Tag,
    Cluster,
    Graph,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MemoryEffect {
    SetProperty { property: String },
    CreateRelation { relation: String },
    EndRelation { relation: String },
    ArchiveInstance,
    SupersedeInstance,
}

impl MemoryActionSpec {
    /// Built-in "remember" action: writes a memory instance.
    pub fn remember() -> Self {
        Self {
            id: "remember".to_string(),
            target: MemoryActionTarget::Memory,
            preconditions: vec![
                "max_5_tags".to_string(),
                "correction_has_target_tag".to_string(),
                "provenance_admissible".to_string(),
            ],
            effects: vec![MemoryEffect::SetProperty {
                property: "content".to_string(),
            }],
            allowed_roles: vec![MemoryRole::Agent, MemoryRole::User],
            requires_confirmation: false,
        }
    }

    /// Built-in "link" action: creates a semantic relation between memories.
    pub fn link() -> Self {
        Self {
            id: "link".to_string(),
            target: MemoryActionTarget::Link,
            preconditions: vec!["source_exists".to_string(), "target_exists".to_string()],
            effects: vec![MemoryEffect::CreateRelation {
                relation: "relates_to".to_string(),
            }],
            allowed_roles: vec![MemoryRole::Agent, MemoryRole::User],
            requires_confirmation: false,
        }
    }

    /// Built-in "forget" action: archives or deactivates a memory.
    pub fn forget() -> Self {
        Self {
            id: "forget".to_string(),
            target: MemoryActionTarget::Memory,
            preconditions: vec!["instance_exists".to_string()],
            effects: vec![MemoryEffect::ArchiveInstance],
            allowed_roles: vec![MemoryRole::Agent, MemoryRole::User],
            requires_confirmation: true,
        }
    }

    /// Built-in "supersede" action: replaces an older memory with a newer one.
    pub fn supersede() -> Self {
        Self {
            id: "supersede".to_string(),
            target: MemoryActionTarget::Memory,
            preconditions: vec![
                "newer_instance_exists".to_string(),
                "older_instance_exists".to_string(),
            ],
            effects: vec![MemoryEffect::SupersedeInstance],
            allowed_roles: vec![MemoryRole::Agent, MemoryRole::User],
            requires_confirmation: true,
        }
    }

    /// Built-in "tag" action: adds a tag to a memory.
    pub fn tag() -> Self {
        Self {
            id: "tag".to_string(),
            target: MemoryActionTarget::Tag,
            preconditions: vec!["instance_exists".to_string()],
            effects: vec![MemoryEffect::SetProperty {
                property: "tags".to_string(),
            }],
            allowed_roles: vec![MemoryRole::Agent, MemoryRole::User],
            requires_confirmation: false,
        }
    }

    /// Built-in "validate" action: runs the graph validation suite.
    pub fn validate() -> Self {
        Self {
            id: "validate".to_string(),
            target: MemoryActionTarget::Graph,
            preconditions: vec!["graph_loadable".to_string()],
            effects: vec![MemoryEffect::SetProperty {
                property: "validity".to_string(),
            }],
            allowed_roles: vec![MemoryRole::Agent, MemoryRole::User, MemoryRole::System],
            requires_confirmation: false,
        }
    }

    pub fn allows_role(&self, role: MemoryRole) -> bool {
        self.allowed_roles.is_empty() || self.allowed_roles.contains(&role)
    }
}

/// Confirmation gate triggered by a proposed change.
///
/// These map directly to the G1-G4 gates from enterprise-ontology diff_instances.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ConfirmationGate {
    /// G1: core / critical data is touched.
    CriticalData,
    /// G2: an authoritative source value is being overwritten.
    AuthoritativeOverwrite,
    /// G3: a status transition (active -> archived/expired) or disappearance.
    StatusTransition,
    /// G4: a batch change affecting more than 20% of memories of a category.
    BatchChange,
}

impl ConfirmationGate {
    /// Human-readable description of why confirmation is required.
    pub fn description(&self) -> &'static str {
        match self {
            ConfirmationGate::CriticalData => {
                "Critical memory data is involved; human confirmation is required."
            }
            ConfirmationGate::AuthoritativeOverwrite => {
                "An authoritative source value would be overwritten."
            }
            ConfirmationGate::StatusTransition => {
                "A memory status transition or deletion requires confirmation."
            }
            ConfirmationGate::BatchChange => {
                "More than 20% of memories in a category are affected."
            }
        }
    }
}

/// A proposed change to evaluate against confirmation gates.
#[derive(Debug, Clone, Default)]
pub struct ChangeProposal<'a> {
    pub action: Option<&'a MemoryActionSpec>,
    pub target: Option<&'a MemoryEntry>,
    pub new_entry: Option<&'a MemoryEntry>,
    pub category_count: Option<usize>,
    pub total_in_category: Option<usize>,
}

impl<'a> ChangeProposal<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_action(mut self, action: &'a MemoryActionSpec) -> Self {
        self.action = Some(action);
        self
    }

    pub fn with_target(mut self, target: &'a MemoryEntry) -> Self {
        self.target = Some(target);
        self
    }

    pub fn with_new_entry(mut self, new: &'a MemoryEntry) -> Self {
        self.new_entry = Some(new);
        self
    }

    pub fn with_category_counts(mut self, count: usize, total: usize) -> Self {
        self.category_count = Some(count);
        self.total_in_category = Some(total);
        self
    }
}

/// Evaluate which confirmation gates would be triggered by a proposed change.
pub fn evaluate_gates(proposal: ChangeProposal<'_>) -> Vec<ConfirmationGate> {
    let mut gates = Vec::new();

    // G1: critical data touched.
    let critical_old = proposal.target.map(|e| e.critical).unwrap_or(false);
    let critical_new = proposal.new_entry.map(|e| e.critical).unwrap_or(critical_old);
    if critical_old || critical_new {
        gates.push(ConfirmationGate::CriticalData);
    }

    // G2: overwriting an authoritative source value.
    if let (Some(old), Some(new)) = (proposal.target, proposal.new_entry) {
        let old_authoritative = old
            .provenance
            .as_ref()
            .map(|p| p.method.authority_rank() >= ExtractionMethod::StructuredMapping.authority_rank())
            .unwrap_or(false);
        let new_value_different = old.content != new.content || old.tags != new.tags;
        if old_authoritative && new_value_different {
            gates.push(ConfirmationGate::AuthoritativeOverwrite);
        }
    }

    // G3: status transition or deletion.
    let old_status = proposal
        .target
        .map(|e| e.lifecycle.status)
        .unwrap_or(MemoryStatus::Active);
    let new_status = proposal
        .new_entry
        .map(|e| e.lifecycle.status)
        .unwrap_or(old_status);
    if old_status != new_status
        || proposal
            .action
            .map(|a| matches!(a.id.as_str(), "forget" | "archive" | "delete"))
            .unwrap_or(false)
    {
        gates.push(ConfirmationGate::StatusTransition);
    }

    // G4: batch change affecting >20% of a category.
    if let (Some(count), Some(total)) = (proposal.category_count, proposal.total_in_category) {
        if total > 0 && count * 5 > total {
            gates.push(ConfirmationGate::BatchChange);
        }
    }

    gates
}

/// Check whether a proposed change is allowed without confirmation, given a role
/// and the current graph state.
pub fn requires_confirmation(
    _graph: &MemoryGraph,
    proposal: ChangeProposal<'_>,
    role: MemoryRole,
) -> bool {
    if let Some(action) = proposal.action {
        if !action.allows_role(role) {
            return true; // not allowed at all, treat as requiring confirmation
        }
        if action.requires_confirmation {
            return true;
        }
    }

    let gates = evaluate_gates(proposal);

    // System role may bypass non-critical gates, but never critical data.
    if role == MemoryRole::System {
        return gates.iter().any(|g| matches!(g, ConfirmationGate::CriticalData));
    }

    !gates.is_empty()
}

/// Check lightweight preconditions for a built-in action against a graph.
pub fn check_preconditions(
    graph: &MemoryGraph,
    action: &MemoryActionSpec,
    source_id: Option<&str>,
    target_id: Option<&str>,
) -> Result<(), Vec<String>> {
    let mut issues = Vec::new();
    for pre in &action.preconditions {
        match pre.as_str() {
            "instance_exists" | "source_exists" | "newer_instance_exists" => {
                if source_id.is_none_or(|id| !graph.memories.contains_key(id)) {
                    issues.push(format!("precondition failed: {pre}"));
                }
            }
            "target_exists" | "older_instance_exists" => {
                if target_id.is_none_or(|id| !graph.memories.contains_key(id)) {
                    issues.push(format!("precondition failed: {pre}"));
                }
            }
            "graph_loadable" => {
                // Graph is already loaded; treat as satisfied.
            }
            "provenance_admissible" => {
                // Checked at write time on the entry, not on the graph.
            }
            _ => {}
        }
    }
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::{ExtractionMethod, MemoryStatus, ProvenanceRecord};
    use crate::{MemoryCategory, MemoryEntry, MemoryGraph};

    fn make_entry(content: &str) -> MemoryEntry {
        MemoryEntry::new(MemoryCategory::Fact, content)
    }

    fn make_authoritative_entry(content: &str) -> MemoryEntry {
        MemoryEntry::new(MemoryCategory::Fact, content)
            .with_provenance(ProvenanceRecord::new("user", ExtractionMethod::UserStated))
    }

    fn make_critical_entry(content: &str) -> MemoryEntry {
        MemoryEntry::new(MemoryCategory::Fact, content)
            .with_critical(true)
            .with_provenance(ProvenanceRecord::new("user", ExtractionMethod::UserStated))
    }

    #[test]
    fn remember_action_allows_agent_without_confirmation() {
        let graph = MemoryGraph::new();
        let action = MemoryActionSpec::remember();
        let proposal = ChangeProposal::new().with_action(&action);
        assert!(!requires_confirmation(&graph, proposal, MemoryRole::Agent));
    }

    #[test]
    fn forget_action_requires_confirmation() {
        let graph = MemoryGraph::new();
        let action = MemoryActionSpec::forget();
        let proposal = ChangeProposal::new().with_action(&action);
        assert!(requires_confirmation(&graph, proposal, MemoryRole::Agent));
    }

    #[test]
    fn critical_data_triggers_g1() {
        let mut graph = MemoryGraph::new();
        let id = graph.add_memory(make_critical_entry("core password policy"));
        let target = graph.get_memory(&id).unwrap();
        let proposal = ChangeProposal::new().with_target(target);
        let gates = evaluate_gates(proposal);
        assert!(gates.contains(&ConfirmationGate::CriticalData));
    }

    #[test]
    fn authoritative_overwrite_triggers_g2() {
        let mut graph = MemoryGraph::new();
        let id = graph.add_memory(make_authoritative_entry("old value"));
        let target = graph.get_memory(&id).unwrap().clone();
        let new = make_entry("new value")
            .with_id(&target.id)
            .with_provenance(ProvenanceRecord::new("extraction", ExtractionMethod::LlmExtraction));
        let proposal = ChangeProposal::new()
            .with_target(&target)
            .with_new_entry(&new);
        let gates = evaluate_gates(proposal);
        assert!(gates.contains(&ConfirmationGate::AuthoritativeOverwrite));
    }

    #[test]
    fn status_transition_triggers_g3() {
        let mut graph = MemoryGraph::new();
        let id = graph.add_memory(make_entry("fact"));
        let target = graph.get_memory(&id).unwrap().clone();
        let mut new = target.clone();
        new.lifecycle.status = MemoryStatus::Archived;
        let proposal = ChangeProposal::new()
            .with_target(&target)
            .with_new_entry(&new);
        let gates = evaluate_gates(proposal);
        assert!(gates.contains(&ConfirmationGate::StatusTransition));
    }

    #[test]
    fn batch_change_triggers_g4() {
        let _graph = MemoryGraph::new();
        let proposal = ChangeProposal::new().with_category_counts(3, 10);
        let gates = evaluate_gates(proposal);
        assert!(gates.contains(&ConfirmationGate::BatchChange));

        let small = ChangeProposal::new().with_category_counts(1, 10);
        let gates_small = evaluate_gates(small);
        assert!(!gates_small.contains(&ConfirmationGate::BatchChange));
    }

    #[test]
    fn system_role_bypasses_non_critical_gates() {
        let mut graph = MemoryGraph::new();
        let id = graph.add_memory(make_entry("fact"));
        let target = graph.get_memory(&id).unwrap().clone();
        let mut new = target.clone();
        new.lifecycle.status = MemoryStatus::Archived;
        let proposal = ChangeProposal::new()
            .with_target(&target)
            .with_new_entry(&new);
        assert!(requires_confirmation(&graph, proposal.clone(), MemoryRole::Agent));
        assert!(!requires_confirmation(&graph, proposal, MemoryRole::System));
    }

    #[test]
    fn system_role_cannot_bypass_critical_gate() {
        let mut graph = MemoryGraph::new();
        let id = graph.add_memory(make_critical_entry("core"));
        let target = graph.get_memory(&id).unwrap().clone();
        let new = make_entry("changed").with_id(&target.id);
        let proposal = ChangeProposal::new()
            .with_target(&target)
            .with_new_entry(&new);
        assert!(requires_confirmation(&graph, proposal, MemoryRole::System));
    }

    #[test]
    fn check_preconditions_fails_for_missing_source() {
        let graph = MemoryGraph::new();
        let action = MemoryActionSpec::forget();
        let err = check_preconditions(&graph, &action, Some("missing"), None).unwrap_err();
        assert!(!err.is_empty());
    }
}
