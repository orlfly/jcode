//! Stage S7 plan consistency validation and targeted repair (design 4.3).
//!
//! Validation checks structural drift between a compiled [`PhysicalPlan`] and
//! the [`PathQuery`] it came from: anchors, relation whitelist fidelity, depth
//! bounds and fusion thresholds. Repairs are deterministic and capped at
//! [`MAX_REPAIR_ATTEMPTS`]; anything still failing is reported as an
//! unrepairable finding instead of being silently dropped or retried whole.

use serde::{Deserialize, Serialize};

use crate::path_query::PathQuery;
use crate::physical_plan::{PhysicalPlan, PlanStage, StageKind};

/// A query may carry up to 8 hops (matches `PathQuery` validation).
pub const MAX_PHYSICAL_DEPTH: usize = 8;
/// Repairs are attempted at most twice, as specified by 4.3 S7.
pub const MAX_REPAIR_ATTEMPTS: usize = 2;

/// One validation finding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanFinding {
    /// Stable code: `P01` anchors, `P02` relations, `P03` depth, `P04` fusion,
    /// `P05` ontology probe endpoints.
    pub code: String,
    pub stage: StageKind,
    pub message: String,
    pub repairable: bool,
}

/// One applied repair, for the audit trail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanRepair {
    pub attempt: usize,
    pub code: String,
    pub action: String,
}

/// Validation plus repair audit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanAudit {
    pub valid: bool,
    pub findings: Vec<PlanFinding>,
    pub repairs: Vec<PlanRepair>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub remaining: Vec<PlanFinding>,
}

fn anchors_of(query: &PathQuery) -> Vec<String> {
    let mut anchors = Vec::new();
    if let Some(start) = query.start.first() {
        anchors.push(start.clone());
    }
    anchors.extend(query.waypoints.iter().cloned());
    if let Some(goal) = query.goal.first() {
        anchors.push(goal.clone());
    }
    anchors
}

fn search_stage(plan: &PhysicalPlan) -> Option<&PlanStage> {
    plan.stage(StageKind::InstanceSearch)
}

fn mutable_stage(plan: &mut PhysicalPlan, kind: StageKind) -> Option<&mut PlanStage> {
    plan.stages.iter_mut().find(|stage| stage.kind == kind)
}

/// Validate one plan against its query without mutating it.
pub fn validate_physical_plan(plan: &PhysicalPlan, query: &PathQuery) -> Vec<PlanFinding> {
    let mut findings = Vec::new();
    let expected_anchors = anchors_of(query);

    let probe = plan.stage(StageKind::OntologyProbe);
    let search = search_stage(plan);

    let Some(search) = search else {
        findings.push(PlanFinding {
            code: "P01".to_string(),
            stage: StageKind::InstanceSearch,
            message: "plan has no instance_search stage".to_string(),
            repairable: true,
        });
        return findings;
    };

    let stage_anchors: Vec<String> = search
        .parameters
        .get("anchors")
        .and_then(|value| serde_json::from_value(value.clone()).ok())
        .unwrap_or_default();
    if !expected_anchors.is_empty() && stage_anchors != expected_anchors {
        findings.push(PlanFinding {
            code: "P01".to_string(),
            stage: StageKind::InstanceSearch,
            message: format!(
                "anchors {:?} do not match query anchors {:?}",
                stage_anchors, expected_anchors
            ),
            repairable: true,
        });
    }

    let stage_relations: Vec<String> = search
        .parameters
        .get("relations")
        .and_then(|value| serde_json::from_value(value.clone()).ok())
        .unwrap_or_default();
    if stage_relations != query.constraints.relations {
        findings.push(PlanFinding {
            code: "P02".to_string(),
            stage: StageKind::InstanceSearch,
            message: format!(
                "relations {:?} drifted from query constraints {:?}",
                stage_relations, query.constraints.relations
            ),
            repairable: true,
        });
    }

    let depth = search
        .parameters
        .get("depth")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize;
    if !(1..=MAX_PHYSICAL_DEPTH).contains(&depth) {
        findings.push(PlanFinding {
            code: "P03".to_string(),
            stage: StageKind::InstanceSearch,
            message: format!("depth {depth} outside 1..={MAX_PHYSICAL_DEPTH}"),
            repairable: true,
        });
    }

    if let Some(probe) = probe {
        let concepts: Vec<String> = probe
            .parameters
            .get("concepts")
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default();
        let endpoints_match = concepts.first() == query.start.first()
            && concepts.last() == query.goal.first()
            && concepts.len() >= 2;
        if !endpoints_match {
            findings.push(PlanFinding {
                code: "P05".to_string(),
                stage: StageKind::OntologyProbe,
                message: format!(
                    "concept path {:?} does not span query endpoints {:?} -> {:?}",
                    concepts, query.start, query.goal
                ),
                repairable: true,
            });
        }
    }

    if let Some(fusion) = plan.stage(StageKind::Fusion) {
        let limit = fusion
            .parameters
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let min_confidence = fusion
            .parameters
            .get("min_confidence")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let query_confidence = query.constraints.min_confidence.unwrap_or(0.0) as f64;
        let confidence_drift = (min_confidence - query_confidence).abs() > 1e-6;
        if limit == 0 || !(0.0..=1.0).contains(&min_confidence) || confidence_drift {
            findings.push(PlanFinding {
                code: "P04".to_string(),
                stage: StageKind::Fusion,
                message: format!(
                    "fusion limit={limit} min_confidence={min_confidence} invalid or drifted from query ({query_confidence})"
                ),
                repairable: true,
            });
        }
    }

    findings
}

fn repair_once(
    plan: &mut PhysicalPlan,
    query: &PathQuery,
    findings: &[PlanFinding],
    attempt: usize,
) -> Vec<PlanRepair> {
    let mut repairs = Vec::new();
    if findings
        .iter()
        .any(|finding| finding.code == "P01" || finding.code == "P05")
    {
        let expected = anchors_of(query);
        if let Some(search) = mutable_stage(plan, StageKind::InstanceSearch) {
            search
                .parameters
                .insert("anchors".to_string(), serde_json::json!(expected));
            repairs.push(PlanRepair {
                attempt,
                code: if findings.iter().any(|f| f.code == "P01") {
                    "P01"
                } else {
                    "P05"
                }
                .to_string(),
                action: "reset instance anchors to query start/waypoints/goal".to_string(),
            });
        }
        if let Some(probe) = mutable_stage(plan, StageKind::OntologyProbe) {
            let mut concepts = expected.clone();
            if concepts.len() < 2 {
                concepts.push(query.goal.first().cloned().unwrap_or_default());
            }
            probe
                .parameters
                .insert("concepts".to_string(), serde_json::json!(concepts));
        }
    }
    if findings.iter().any(|finding| finding.code == "P02") {
        if let Some(search) = mutable_stage(plan, StageKind::InstanceSearch) {
            search.parameters.insert(
                "relations".to_string(),
                serde_json::json!(query.constraints.relations),
            );
            repairs.push(PlanRepair {
                attempt,
                code: "P02".to_string(),
                action: "resync relation whitelist from query constraints".to_string(),
            });
        }
    }
    if findings.iter().any(|finding| finding.code == "P03") {
        let requested = query.constraints.depth.unwrap_or(4);
        let clamped = requested.clamp(1, MAX_PHYSICAL_DEPTH);
        if let Some(search) = mutable_stage(plan, StageKind::InstanceSearch) {
            search
                .parameters
                .insert("depth".to_string(), serde_json::json!(clamped));
            repairs.push(PlanRepair {
                attempt,
                code: "P03".to_string(),
                action: format!("clamp depth to {clamped}"),
            });
        }
    }
    if findings.iter().any(|finding| finding.code == "P04") {
        if let Some(fusion) = mutable_stage(plan, StageKind::Fusion) {
            let limit = query.constraints.limit.unwrap_or(3).max(1);
            let confidence = query
                .constraints
                .min_confidence
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            fusion
                .parameters
                .insert("limit".to_string(), serde_json::json!(limit));
            fusion
                .parameters
                .insert("min_confidence".to_string(), serde_json::json!(confidence));
            repairs.push(PlanRepair {
                attempt,
                code: "P04".to_string(),
                action: format!("reset fusion limit={limit} min_confidence={confidence}"),
            });
        }
    }
    repairs
}

/// Validate, repair at most [`MAX_REPAIR_ATTEMPTS`] times, and report.
pub fn validate_and_repair(plan: &PhysicalPlan, query: &PathQuery) -> (PhysicalPlan, PlanAudit) {
    let mut plan = plan.clone();
    let findings = validate_physical_plan(&plan, query);
    let mut repairs = Vec::new();
    if findings.is_empty() {
        return (
            plan,
            PlanAudit {
                valid: true,
                findings,
                repairs,
                remaining: Vec::new(),
            },
        );
    }

    let mut pending = findings.clone();
    let mut attempt = 0;
    while attempt < MAX_REPAIR_ATTEMPTS && !pending.is_empty() {
        attempt += 1;
        let applied = repair_once(&mut plan, query, &pending, attempt);
        if applied.is_empty() {
            break;
        }
        repairs.extend(applied);
        pending = validate_physical_plan(&plan, query);
    }

    let valid = pending.is_empty();
    (
        plan,
        PlanAudit {
            valid,
            findings,
            repairs,
            remaining: pending,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path_query::{PathQuery, PathQueryConstraints};
    use crate::physical_plan::compile_physical_plan;

    fn query() -> PathQuery {
        PathQuery {
            start: vec!["File".to_string()],
            goal: vec!["API".to_string()],
            waypoints: vec!["Symbol".to_string()],
            graph: None,
            constraints: PathQueryConstraints {
                relations: vec!["defines".to_string()],
                depth: Some(3),
                limit: Some(5),
                min_confidence: Some(0.4),
                ..PathQueryConstraints::default()
            },
        }
    }

    #[test]
    fn clean_plan_has_no_findings_or_repairs() {
        let query = query();
        let plan = compile_physical_plan(&query, &["File".into(), "Symbol".into(), "API".into()]);
        let (_, audit) = validate_and_repair(&plan, &query);
        assert!(audit.valid, "{audit:?}");
        assert!(audit.findings.is_empty());
        assert!(audit.repairs.is_empty());
    }

    #[test]
    fn resyncs_drifted_anchors_and_concept_endpoints() {
        let query = query();
        let mut plan = compile_physical_plan(&query, &["File".into(), "API".into()]);
        let search = plan
            .stages
            .iter_mut()
            .find(|stage| stage.kind == StageKind::InstanceSearch)
            .unwrap();
        search
            .parameters
            .insert("anchors".to_string(), serde_json::json!(["Wrong", "API"]));
        let probe = plan
            .stages
            .iter_mut()
            .find(|stage| stage.kind == StageKind::OntologyProbe)
            .unwrap();
        probe
            .parameters
            .insert("concepts".to_string(), serde_json::json!(["Other", "API"]));

        let (repaired, audit) = validate_and_repair(&plan, &query);
        assert!(audit.valid, "{audit:?}");
        let codes: Vec<&str> = audit.findings.iter().map(|f| f.code.as_str()).collect();
        assert!(codes.contains(&"P01"), "{codes:?}");
        assert!(codes.contains(&"P05"), "{codes:?}");
        let search = repaired.stage(StageKind::InstanceSearch).unwrap();
        assert_eq!(
            search.parameters["anchors"],
            serde_json::json!(["File", "Symbol", "API"])
        );
        let probe = repaired.stage(StageKind::OntologyProbe).unwrap();
        assert_eq!(
            probe.parameters["concepts"],
            serde_json::json!(["File", "Symbol", "API"])
        );
        assert!(!audit.repairs.is_empty());
    }

    #[test]
    fn clamps_depth_and_resets_fusion_thresholds() {
        let query = query();
        let mut plan = compile_physical_plan(&query, &[]);
        for stage in plan.stages.iter_mut() {
            match stage.kind {
                StageKind::InstanceSearch => {
                    stage
                        .parameters
                        .insert("depth".to_string(), serde_json::json!(99));
                }
                StageKind::Fusion => {
                    stage
                        .parameters
                        .insert("limit".to_string(), serde_json::json!(0));
                    stage
                        .parameters
                        .insert("min_confidence".to_string(), serde_json::json!(2.5));
                }
                _ => {}
            }
        }
        let (repaired, audit) = validate_and_repair(&plan, &query);
        assert!(audit.valid, "{audit:?}");
        let search = repaired.stage(StageKind::InstanceSearch).unwrap();
        assert_eq!(search.parameters["depth"], serde_json::json!(3));
        let fusion = repaired.stage(StageKind::Fusion).unwrap();
        assert_eq!(fusion.parameters["limit"], serde_json::json!(5));
        let confidence = fusion.parameters["min_confidence"]
            .as_f64()
            .expect("confidence");
        assert!((confidence - 0.4).abs() < 1e-6, "{confidence}");
        assert_eq!(audit.repairs.len(), 2);
    }

    #[test]
    fn reports_a_missing_search_stage_as_repairable() {
        let query = query();
        let mut plan = compile_physical_plan(&query, &[]);
        plan.stages
            .retain(|stage| stage.kind != StageKind::InstanceSearch);
        let findings = validate_physical_plan(&plan, &query);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].code, "P01");
        assert!(findings[0].repairable);
    }

    #[test]
    fn stops_after_two_attempts_and_reports_remaining_findings() {
        let query = query();
        let mut plan = compile_physical_plan(&query, &[]);
        plan.stages
            .retain(|stage| stage.kind != StageKind::InstanceSearch);
        let (_, audit) = validate_and_repair(&plan, &query);
        assert!(!audit.valid);
        assert!(audit.repairs.is_empty());
        assert_eq!(audit.remaining.len(), 1);
        assert_eq!(audit.remaining[0].code, "P01");
    }
}
