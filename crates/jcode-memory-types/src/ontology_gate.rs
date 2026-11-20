//! Structured routing for the six-question ontology admission gate.
//!
//! The questions in the design document are not machine-decidable. This module
//! only routes already-collected evidence into the three documented exits:
//! `auto_pass`, `clarify`, and `ticket`. It never claims that a definition was
//! judged automatically.

use serde::{Deserialize, Serialize};

/// Responses to Q1-Q5 plus evidence that needs a human or a clarification.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SixQuestionSignals {
    /// Q1: appears on a validation, pruning, or audit code path.
    #[serde(default)]
    pub validation_path: bool,
    /// Q2: consumed by a query compiler, linker, or plan generator.
    #[serde(default)]
    pub machine_consumed: bool,
    /// Q3: a wrong judgment pollutes the main graph or creates compliance risk.
    #[serde(default)]
    pub compliance_risk: bool,
    /// Q4: requires one definition across requests, users, or projects.
    #[serde(default)]
    pub cross_scope_consistency: bool,
    /// Q5: closed positive and negative examples can be written down.
    #[serde(default)]
    pub closed_examples: bool,
    /// Questions whose answers are still unknown. Non-empty routes to clarify.
    #[serde(default)]
    pub unresolved_questions: Vec<String>,
    /// Evidence conflicts with the proposed definition. Routes to a human.
    #[serde(default)]
    pub conflicting_evidence: bool,
}

/// One of the three documented gate exits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OntologyGateExit {
    AutoPass,
    Clarify,
    Ticket,
    /// Q6: all structured questions are false; leave the decision to the model
    /// and keep a runtime trace instead of admitting the definition.
    LlmOnly,
}

impl OntologyGateExit {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AutoPass => "auto_pass",
            Self::Clarify => "clarify",
            Self::Ticket => "ticket",
            Self::LlmOnly => "llm_only",
        }
    }
}

/// Routed gate outcome with the questions that justified it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OntologyGateDecision {
    pub exit: OntologyGateExit,
    pub matched_questions: Vec<String>,
    pub rule_layer_required: bool,
    pub reasons: Vec<String>,
}

/// Route structured six-question evidence. Hit order is Q1 through Q5; Q6 is the
/// all-false fallback. Unresolved questions clarify first, because a ticket is
/// more expensive than an automatic follow-up.
pub fn route_ontology_definition(signals: &SixQuestionSignals) -> OntologyGateDecision {
    if signals.conflicting_evidence {
        return OntologyGateDecision {
            exit: OntologyGateExit::Ticket,
            matched_questions: Vec::new(),
            rule_layer_required: false,
            reasons: vec!["conflicting evidence needs human arbitration".to_string()],
        };
    }
    if !signals.unresolved_questions.is_empty() {
        let mut questions = signals.unresolved_questions.clone();
        questions.sort();
        questions.dedup();
        return OntologyGateDecision {
            exit: OntologyGateExit::Clarify,
            matched_questions: questions.clone(),
            rule_layer_required: false,
            reasons: vec![format!("unresolved question(s): {}", questions.join(", "))],
        };
    }

    let mut matched = Vec::new();
    let mut reasons = Vec::new();
    let mut rule_layer_required = false;
    if signals.validation_path {
        matched.push("Q1".to_string());
        reasons.push("Q1: definition is used by validation, pruning, or audit".to_string());
    }
    if signals.machine_consumed {
        matched.push("Q2".to_string());
        reasons.push("Q2: definition is consumed by a query compiler or planner".to_string());
    }
    if signals.compliance_risk {
        matched.push("Q3".to_string());
        rule_layer_required = true;
        reasons.push("Q3: wrong judgment risks graph pollution or compliance".to_string());
    }
    if signals.cross_scope_consistency {
        matched.push("Q4".to_string());
        reasons.push("Q4: definition must stay consistent across scopes".to_string());
    }
    if signals.closed_examples {
        matched.push("Q5".to_string());
        rule_layer_required = true;
        reasons.push("Q5: closed examples can be encoded as rules".to_string());
    }

    if matched.is_empty() {
        return OntologyGateDecision {
            exit: OntologyGateExit::LlmOnly,
            matched_questions: Vec::new(),
            rule_layer_required: false,
            reasons: vec![
                "Q6: no structured signal; leave to the model with a runtime trace".to_string(),
            ],
        };
    }

    OntologyGateDecision {
        exit: OntologyGateExit::AutoPass,
        matched_questions: matched,
        rule_layer_required,
        reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_hits_auto_pass_and_flag_rule_layer() {
        let decision = route_ontology_definition(&SixQuestionSignals {
            validation_path: true,
            compliance_risk: true,
            ..SixQuestionSignals::default()
        });
        assert_eq!(decision.exit, OntologyGateExit::AutoPass);
        assert_eq!(decision.matched_questions, vec!["Q1", "Q3"]);
        assert!(decision.rule_layer_required);
    }

    #[test]
    fn all_false_is_q6_not_auto_pass() {
        let decision = route_ontology_definition(&SixQuestionSignals::default());
        assert_eq!(decision.exit, OntologyGateExit::LlmOnly);
        assert!(decision.reasons[0].starts_with("Q6:"));
    }

    #[test]
    fn unresolved_and_conflicting_evidence_route_before_auto_pass() {
        let clarify = route_ontology_definition(&SixQuestionSignals {
            validation_path: true,
            unresolved_questions: vec!["Q2".to_string()],
            ..SixQuestionSignals::default()
        });
        assert_eq!(clarify.exit, OntologyGateExit::Clarify);
        assert_eq!(clarify.matched_questions, vec!["Q2"]);
        assert!(!clarify.reasons.is_empty());

        let ticket = route_ontology_definition(&SixQuestionSignals {
            validation_path: true,
            conflicting_evidence: true,
            ..SixQuestionSignals::default()
        });
        assert_eq!(ticket.exit, OntologyGateExit::Ticket);
    }
}
