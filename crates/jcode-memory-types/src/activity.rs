//! Activity scheduler driven by the ontology.
//!
//! The runtime collects "events" (turn tick, topic change, session finalize,
//! periodic timer) and asks the scheduler which [`Activity`]s should fire.
//! The scheduler walks the ontology's activities, filters by trigger kind,
//! and produces a [`ScheduledActivity`] list that the caller (e.g. the
//! background `MemoryAgent`) translates into real work.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ontology::{ActivityStep, ActivityTrigger, Ontology};

/// A coarse event that the runtime can dispatch into the scheduler.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScheduleEvent {
    /// Periodic turn tick (e.g. end of a user turn).
    TurnTick {
        session_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        working_dir: Option<String>,
    },
    /// Topic-change detection between consecutive context embeddings.
    TopicChange {
        session_id: String,
        similarity: f32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        working_dir: Option<String>,
    },
    /// Session finalization.
    Finalize {
        session_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        working_dir: Option<String>,
    },
    /// Manual tick for periodic activities (used by maintenance loops).
    PeriodicTick {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        last_tick: Option<DateTime<Utc>>,
    },
}

impl ScheduleEvent {
    pub fn event_name(&self) -> &'static str {
        match self {
            ScheduleEvent::TurnTick { .. } => crate::ontology::EVENT_TURN_TICK,
            ScheduleEvent::TopicChange { .. } => crate::ontology::EVENT_TOPIC_CHANGE,
            ScheduleEvent::Finalize { .. } => crate::ontology::EVENT_FINALIZE,
            ScheduleEvent::PeriodicTick { .. } => "event.periodic_tick",
        }
    }

    pub fn session_id(&self) -> Option<&str> {
        match self {
            ScheduleEvent::TurnTick { session_id, .. }
            | ScheduleEvent::TopicChange { session_id, .. }
            | ScheduleEvent::Finalize { session_id, .. } => Some(session_id),
            _ => None,
        }
    }

    pub fn working_dir(&self) -> Option<&str> {
        match self {
            ScheduleEvent::TurnTick { working_dir, .. }
            | ScheduleEvent::TopicChange { working_dir, .. }
            | ScheduleEvent::Finalize { working_dir, .. } => working_dir.as_deref(),
            _ => None,
        }
    }
}

/// A planned activity ready to be executed by the runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledActivity {
    pub activity_id: String,
    pub steps: Vec<ActivityStep>,
    pub session_state: Option<String>,
    pub reason: String,
}

/// Determine which activities should fire for an event.
pub fn schedule(event: &ScheduleEvent, ontology: &Ontology) -> Vec<ScheduledActivity> {
    let mut out = Vec::new();
    match event {
        ScheduleEvent::TurnTick { session_id, .. } => {
            for activity in ontology.activities_for_event(event.event_name()) {
                if activity.session_states.is_empty()
                    || activity
                        .session_states
                        .iter()
                        .any(|s| s == "fresh_user_turn")
                {
                    out.push(ScheduledActivity {
                        activity_id: activity.id.clone(),
                        steps: activity.steps.clone(),
                        session_state: Some("fresh_user_turn".to_string()),
                        reason: format!("turn_tick session={}", session_id),
                    });
                }
            }
        }
        ScheduleEvent::TopicChange {
            session_id,
            similarity,
            ..
        } => {
            for activity in &ontology.activities {
                if let ActivityTrigger::OnTopicChange { threshold } = &activity.trigger
                    && *similarity <= *threshold
                {
                    out.push(ScheduledActivity {
                        activity_id: activity.id.clone(),
                        steps: activity.steps.clone(),
                        session_state: Some("topic_change".to_string()),
                        reason: format!(
                            "topic_change sim={:.3} session={}",
                            similarity, session_id
                        ),
                    });
                }
            }
        }
        ScheduleEvent::Finalize { session_id, .. } => {
            for activity in ontology.activities_for_event(event.event_name()) {
                out.push(ScheduledActivity {
                    activity_id: activity.id.clone(),
                    steps: activity.steps.clone(),
                    session_state: Some("finalize".to_string()),
                    reason: format!("finalize session={}", session_id),
                });
            }
        }
        ScheduleEvent::PeriodicTick { last_tick } => {
            let last = last_tick.unwrap_or_else(|| Utc::now());
            for activity in &ontology.activities {
                if let ActivityTrigger::Periodic { interval_turns, .. } = &activity.trigger {
                    let secs = (Utc::now() - last).num_seconds().max(0) as u64;
                    if secs >= u64::from(*interval_turns) * 60 {
                        out.push(ScheduledActivity {
                            activity_id: activity.id.clone(),
                            steps: activity.steps.clone(),
                            session_state: Some("periodic".to_string()),
                            reason: format!(
                                "periodic interval_turns={} elapsed_secs={}",
                                interval_turns, secs
                            ),
                        });
                    }
                }
            }
        }
    }
    out
}

/// Return the high-level reason for a scheduled step.  Useful for logging.
pub fn describe_step(step: &ActivityStep) -> &'static str {
    match step {
        ActivityStep::ComputeRelevance { .. } => "compute_relevance",
        ActivityStep::ExtractMemories { .. } => "extract_memories",
        ActivityStep::Summarize { .. } => "summarize",
        ActivityStep::GcArchived { .. } => "gc_archived",
        ActivityStep::CheckContradictions { .. } => "check_contradictions",
        ActivityStep::ReconcileOnWrite { .. } => "reconcile_on_write",
        ActivityStep::DistillSkillEvidence { .. } => "distill_skill_evidence",
        ActivityStep::Emit { .. } => "emit",
        ActivityStep::TouchSurfaced => "touch_surfaced",
    }
}

/// Filter step by step kind.  Useful when the runtime wants to evaluate a
/// specific kind without materializing the whole plan.
pub fn steps_of_kind<'a>(plan: &'a [ScheduledActivity], kind: &str) -> Vec<&'a ActivityStep> {
    plan.iter()
        .flat_map(|s| s.steps.iter())
        .filter(|s| describe_step(s) == kind)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::{
        ACTIVITY_FINAL_EXTRACT, ACTIVITY_PERIODIC_GC, ACTIVITY_SKILL_DISTILL,
        ACTIVITY_WRITE_RECONCILE, default_ontology,
    };

    #[test]
    fn finalize_schedules_extract_gc_and_distill() {
        let onto = default_ontology();
        let plan = schedule(
            &ScheduleEvent::Finalize {
                session_id: "s1".to_string(),
                working_dir: None,
            },
            &onto,
        );
        for expected in [
            ACTIVITY_FINAL_EXTRACT,
            ACTIVITY_PERIODIC_GC,
            ACTIVITY_SKILL_DISTILL,
        ] {
            assert!(
                plan.iter().any(|p| p.activity_id == expected),
                "finalize must schedule {expected}"
            );
        }
    }

    #[test]
    fn turn_tick_no_longer_schedules_llm_relevance() {
        // v2 moved recall off the scheduled-activity surface: the runtime
        // serves it locally (L1) and the JEV judge is an optional reranker.
        let onto = default_ontology();
        let plan = schedule(
            &ScheduleEvent::TurnTick {
                session_id: "s1".to_string(),
                working_dir: None,
            },
            &onto,
        );
        assert!(plan.is_empty());
        assert!(
            onto.activities
                .iter()
                .any(|a| a.id == ACTIVITY_WRITE_RECONCILE && !a.steps.is_empty())
        );
    }

    #[test]
    fn describe_step_covers_v2_kinds() {
        assert_eq!(
            describe_step(&ActivityStep::ReconcileOnWrite {
                reinforce_at: 0.85,
                supersede_at: 0.5,
                relate_at: 0.3,
            }),
            "reconcile_on_write"
        );
        assert_eq!(
            describe_step(&ActivityStep::DistillSkillEvidence {
                min_evidence: 3,
                min_stores: 2,
            }),
            "distill_skill_evidence"
        );
        assert_eq!(
            describe_step(&ActivityStep::TouchSurfaced),
            "touch_surfaced"
        );
    }
}
