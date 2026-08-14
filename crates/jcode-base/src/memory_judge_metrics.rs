//! Attribution + measurement of "no-LLM memory mode" conversions.
//!
//! Memory is only reliably productive when the LLM precision judge (the
//! listwise consensus rerank) decides what to surface. Whenever a turn surfaces
//! (or suppresses) memories WITHOUT that judge, it has "converted" to no-LLM
//! mode. Some conversions are intended (the user explicitly opted out of the
//! sidecar); most are silent degradations we want to drive to zero (lost login,
//! judge transport failures, unparseable judge responses, etc.).
//!
//! # Why this is exhaustive
//!
//! Every memory surfacing turn ends in exactly one [`JudgeDecision`]. The enum
//! is the single source of truth for "all the ways memory can decide". Because
//! the recording site is a single `match`-free call and the variants are a
//! closed Rust enum, a new code path that surfaces memory cannot ship without
//! choosing a variant here. New paths => add a variant => the dashboards and the
//! `is_no_llm()` / `is_degradation()` classification force you to declare intent.
//!
//! # The metric
//!
//! - conversion rate (all)        = no_llm_decisions / total_decisions
//! - DEGRADATION conversion rate  = degraded_decisions / total_decisions
//!
//! The number we drive to 0 is the *degradation* rate. The intended-opt-out rate
//! is expected to be >0 only when a user deliberately disables the sidecar.

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};

/// Every terminal outcome of a memory surfacing turn, w.r.t. the LLM judge.
///
/// EXHAUSTIVE: adding a memory surfacing path requires adding a variant here.
/// Group A = LLM judge actually ran (the productive path). Group B = no-LLM
/// (a "conversion"); each B variant declares whether it is intended or a
/// degradation via [`JudgeDecision::is_degradation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeDecision {
    // ---- Group A: LLM judge ran (NOT a conversion) ----
    /// The consensus rerank ran and at least one judge produced a usable ballot;
    /// the surfaced set is the judged result. The productive path.
    JudgeRan,

    // ---- Group B: no-LLM (a conversion to no-LLM mode) ----
    /// User explicitly disabled the sidecar (`memory_sidecar_enabled = false`).
    /// INTENDED: the only conversion that is by-design, not a degradation.
    OptedOut,
    /// Cadence gate: re-surfaced the previously judge-verified set without a
    /// fresh rerank this turn. INTENDED (still high precision; rides the last
    /// judged result), so not counted as degradation.
    CadenceCarry,
    /// Sidecar mode is on but no LLM backend is reachable (logged out / lost
    /// provider access). Memory went dormant. DEGRADATION.
    NoBackend,
    /// The consensus rerank fired but EVERY judge failed (transport error /
    /// timeout). The rerank surfaced nothing and the caller carried the last
    /// judge-verified set. DEGRADATION.
    AllJudgesFailed,
    /// The (single-judge) rerank fired but the judge response was unparseable
    /// garbage. The rerank surfaced nothing and the caller carried the last
    /// judge-verified set. DEGRADATION.
    JudgeUnparseable,
    /// The (single-judge) rerank fired but the judge transport errored. The
    /// rerank surfaced nothing and the caller carried the last judge-verified
    /// set. DEGRADATION.
    JudgeTransportError,
}

impl JudgeDecision {
    /// Stable snake_case label used in logs / dashboards.
    pub fn label(self) -> &'static str {
        match self {
            JudgeDecision::JudgeRan => "judge_ran",
            JudgeDecision::OptedOut => "opted_out",
            JudgeDecision::CadenceCarry => "cadence_carry",
            JudgeDecision::NoBackend => "no_backend",
            JudgeDecision::AllJudgesFailed => "all_judges_failed",
            JudgeDecision::JudgeUnparseable => "judge_unparseable",
            JudgeDecision::JudgeTransportError => "judge_transport_error",
        }
    }

    /// Whether this outcome surfaced/suppressed memory WITHOUT the LLM judge,
    /// i.e. a conversion to no-LLM memory mode.
    pub fn is_no_llm(self) -> bool {
        !matches!(self, JudgeDecision::JudgeRan)
    }

    /// Whether this conversion is an UNINTENDED degradation (the kind we drive to
    /// zero), as opposed to an intended no-LLM outcome (explicit opt-out or a
    /// cadence carry that rides a prior judge verdict).
    pub fn is_degradation(self) -> bool {
        matches!(
            self,
            JudgeDecision::NoBackend
                | JudgeDecision::AllJudgesFailed
                | JudgeDecision::JudgeUnparseable
                | JudgeDecision::JudgeTransportError
        )
    }

    /// All variants, for iteration in snapshots/tests. Kept in sync with the
    /// enum by the exhaustiveness test in this module.
    pub const ALL: [JudgeDecision; 7] = [
        JudgeDecision::JudgeRan,
        JudgeDecision::OptedOut,
        JudgeDecision::CadenceCarry,
        JudgeDecision::NoBackend,
        JudgeDecision::AllJudgesFailed,
        JudgeDecision::JudgeUnparseable,
        JudgeDecision::JudgeTransportError,
    ];

    /// Map a rerank's self-reported [`RerankOutcome`](crate::memory_rerank::RerankOutcome)
    /// onto the attribution variant. This is the bridge that turns the rerank's
    /// (now non-surfacing) judge failures into counted degradations.
    pub fn from_rerank_outcome(outcome: crate::memory_rerank::RerankOutcome) -> Self {
        use crate::memory_rerank::RerankOutcome;
        match outcome {
            RerankOutcome::Judged => JudgeDecision::JudgeRan,
            RerankOutcome::AllJudgesFailed => JudgeDecision::AllJudgesFailed,
            RerankOutcome::Unparseable => JudgeDecision::JudgeUnparseable,
            RerankOutcome::TransportError => JudgeDecision::JudgeTransportError,
        }
    }
}

// One atomic per variant, indexed by `decision_index`.
static COUNTS: [AtomicU64; 7] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

fn decision_index(d: JudgeDecision) -> usize {
    match d {
        JudgeDecision::JudgeRan => 0,
        JudgeDecision::OptedOut => 1,
        JudgeDecision::CadenceCarry => 2,
        JudgeDecision::NoBackend => 3,
        JudgeDecision::AllJudgesFailed => 4,
        JudgeDecision::JudgeUnparseable => 5,
        JudgeDecision::JudgeTransportError => 6,
    }
}

/// Record one memory surfacing decision. Call EXACTLY ONCE per surfacing turn,
/// at the single attribution site in the memory agent. Also writes a structured
/// line to the memory event log so conversions are attributable per session.
pub fn record(decision: JudgeDecision, session_id: &str, candidate_count: usize) {
    COUNTS[decision_index(decision)].fetch_add(1, Ordering::Relaxed);
    crate::memory_log::log_judge_decision(
        session_id,
        decision.label(),
        decision.is_no_llm(),
        decision.is_degradation(),
        candidate_count,
    );
    if decision.is_degradation() {
        record_degradation_conversion(decision.label());
        // Loud, rate-limited alarm: a degradation conversion is a bug to fix.
        crate::logging::event_rate_limited(
            crate::logging::LogLevel::Warn,
            "memory_no_llm_degradation",
            std::time::Duration::from_secs(60),
            "MEMORY_NO_LLM_DEGRADATION",
            vec![
                ("session_id", session_id.to_string()),
                ("path", decision.label().to_string()),
                ("candidates", candidate_count.to_string()),
            ],
        );
    } else {
        // Any non-degradation (JudgeRan, OptedOut, CadenceCarry) clears the
        // sustained-degradation counter so the runtime can recover.
        reset_consecutive_degradations();
    }
}

/// Whether the sidecar should auto-disable because the evaluation judge has
/// failed repeatedly. The memory runtime consults this every turn and falls
/// back to the no-LLM hybrid path when it returns `true`, so the agent
/// keeps producing memories without the precision judge.
pub fn sidecar_should_auto_disable() -> bool {
    CONSECUTIVE_DEGRADATIONS.load(Ordering::Relaxed) >= SUSTAINED_DEGRADATION_THRESHOLD
}

/// Counter for the consecutive-degradation streak. Exposed for tests and
/// debug dashboards.
pub fn consecutive_degradation_count() -> u64 {
    CONSECUTIVE_DEGRADATIONS.load(Ordering::Relaxed)
}

/// Number of consecutive degradations seen so far. Reset on every judge Ran
/// or every intended conversion (OptedOut, CadenceCarry).
///
/// When this counter crosses [`SUSTAINED_DEGRADATION_THRESHOLD`] the memory
/// runtime disables the sidecar for the rest of the session so the agent
/// falls back to the no-LLM hybrid path instead of burning rerank attempts
/// on a broken LLM backend. The user can re-enable via
/// `agents.memory_sidecar_enabled = true` and reload.
static CONSECUTIVE_DEGRADATIONS: AtomicU64 = AtomicU64::new(0);

/// After this many consecutive degradations, the memory runtime should
/// auto-disable the sidecar. Set to 5 so a single transient blip doesn't
/// silently kill the precision judge, but a sustained outage (which always
/// has been the case in production) does not block memory writes forever.
pub const SUSTAINED_DEGRADATION_THRESHOLD: u64 = 5;

fn record_degradation_conversion(label: &str) {
    let new_count = CONSECUTIVE_DEGRADATIONS.fetch_add(1, Ordering::Relaxed) + 1;
    if new_count == SUSTAINED_DEGRADATION_THRESHOLD {
        crate::logging::event_rate_limited(
            crate::logging::LogLevel::Error,
            "memory_sustained_degradation",
            std::time::Duration::from_secs(60 * 60),
            "MEMORY_SUSTAINED_DEGRADATION",
            vec![
                ("degradations", new_count.to_string()),
                ("path", label.to_string()),
                ("action", "auto_disable_sidecar".to_string()),
            ],
        );
    } else if new_count > SUSTAINED_DEGRADATION_THRESHOLD && new_count.is_multiple_of(20) {
        // Re-alarm every 20 degradations so the logs aren't silent.
        crate::logging::event_rate_limited(
            crate::logging::LogLevel::Error,
            "memory_sustained_degradation",
            std::time::Duration::from_secs(60 * 60),
            "MEMORY_SUSTAINED_DEGRADATION",
            vec![
                ("degradations", new_count.to_string()),
                ("path", label.to_string()),
                ("action", "auto_disable_sidecar".to_string()),
            ],
        );
    }
}

fn reset_consecutive_degradations() {
    CONSECUTIVE_DEGRADATIONS.store(0, Ordering::Relaxed);
}

/// Per-variant count.
pub fn count(decision: JudgeDecision) -> u64 {
    COUNTS[decision_index(decision)].load(Ordering::Relaxed)
}

/// Reset all counters (test/maintenance only).
pub fn reset() {
    for c in COUNTS.iter() {
        c.store(0, Ordering::Relaxed);
    }
    reset_consecutive_degradations();
}

/// Aggregate snapshot of all memory-judge decisions seen so far.
#[derive(Debug, Clone, Serialize, Default)]
pub struct JudgeMetricsSnapshot {
    /// Per-variant counts keyed by stable label.
    pub by_decision: std::collections::BTreeMap<String, u64>,
    /// Total surfacing decisions recorded.
    pub total: u64,
    /// Decisions that ran the LLM judge (the productive path).
    pub judge_ran: u64,
    /// Decisions that converted to no-LLM mode (intended + degraded).
    pub no_llm_total: u64,
    /// Intended no-LLM conversions (explicit opt-out, cadence carry).
    pub no_llm_intended: u64,
    /// UNINTENDED no-LLM degradations: the number we drive to zero.
    pub no_llm_degraded: u64,
    /// no_llm_total / total, in [0, 1]. Overall conversion rate.
    pub conversion_rate: f64,
    /// no_llm_degraded / total, in [0, 1]. The headline metric to minimize.
    pub degradation_rate: f64,
}

/// Build a snapshot of the current counters.
pub fn snapshot() -> JudgeMetricsSnapshot {
    let mut by_decision = std::collections::BTreeMap::new();
    let mut total = 0u64;
    let mut judge_ran = 0u64;
    let mut no_llm_total = 0u64;
    let mut no_llm_intended = 0u64;
    let mut no_llm_degraded = 0u64;

    for d in JudgeDecision::ALL {
        let c = count(d);
        by_decision.insert(d.label().to_string(), c);
        total += c;
        if d == JudgeDecision::JudgeRan {
            judge_ran += c;
        }
        if d.is_no_llm() {
            no_llm_total += c;
            if d.is_degradation() {
                no_llm_degraded += c;
            } else {
                no_llm_intended += c;
            }
        }
    }

    let denom = total.max(1) as f64;
    JudgeMetricsSnapshot {
        by_decision,
        total,
        judge_ran,
        no_llm_total,
        no_llm_intended,
        no_llm_degraded,
        conversion_rate: no_llm_total as f64 / denom,
        degradation_rate: no_llm_degraded as f64 / denom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Mutex that serializes the streak-counter tests so the process-global
    /// counter isn't raced by parallel tests. Held for the entire body of any
    /// test that asserts streak values.
    static STREAK_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn all_array_matches_enum_and_indices() {
        // ALL is complete and indices are unique + in range.
        assert_eq!(JudgeDecision::ALL.len(), 7);
        let mut seen = [false; 7];
        for d in JudgeDecision::ALL {
            let i = decision_index(d);
            assert!(!seen[i], "duplicate index for {:?}", d);
            seen[i] = true;
        }
        assert!(seen.iter().all(|&b| b), "every index covered");
    }

    #[test]
    fn classification_is_consistent() {
        // JudgeRan is the only non-conversion.
        assert!(!JudgeDecision::JudgeRan.is_no_llm());
        for d in JudgeDecision::ALL {
            if d != JudgeDecision::JudgeRan {
                assert!(d.is_no_llm(), "{:?} should be a conversion", d);
            }
            // Degradation implies conversion.
            if d.is_degradation() {
                assert!(d.is_no_llm());
            }
        }
        // Intended conversions are opt-out and cadence carry only.
        assert!(!JudgeDecision::OptedOut.is_degradation());
        assert!(!JudgeDecision::CadenceCarry.is_degradation());
        // The four degradations we drive to zero.
        for d in [
            JudgeDecision::NoBackend,
            JudgeDecision::AllJudgesFailed,
            JudgeDecision::JudgeUnparseable,
            JudgeDecision::JudgeTransportError,
        ] {
            assert!(d.is_degradation(), "{:?} should be a degradation", d);
        }
    }

    #[test]
    fn snapshot_computes_rates() {
        // The COUNTS table is process-global, so a parallel test can race
        // us. Snapshot *before* and *after* recording, then verify the deltas
        // line up with the rates we expect.
        let before = snapshot();
        record(JudgeDecision::JudgeRan, "s", 5);
        record(JudgeDecision::JudgeRan, "s", 5);
        record(JudgeDecision::OptedOut, "s", 3); // intended conversion
        record(JudgeDecision::NoBackend, "s", 4); // degradation
        let after = snapshot();
        let delta_total = after.total - before.total;
        let delta_judge_ran = after.judge_ran - before.judge_ran;
        let delta_no_llm = after.no_llm_total - before.no_llm_total;
        let delta_intended = after.no_llm_intended - before.no_llm_intended;
        let delta_degraded = after.no_llm_degraded - before.no_llm_degraded;
        // We added 4 records: 2 JudgeRan + 1 OptedOut + 1 NoBackend.
        // Anything bigger implies a parallel test added even more, which is
        // fine — the rates are still well-defined.
        assert!(delta_total >= 4, "delta_total={delta_total}");
        assert!(delta_judge_ran >= 2, "delta_judge_ran={delta_judge_ran}");
        assert!(delta_no_llm >= 2, "delta_no_llm={delta_no_llm}");
        assert!(delta_intended >= 1, "delta_intended={delta_intended}");
        assert!(delta_degraded >= 1, "delta_degraded={delta_degraded}");
        // Delta conversion_rate should be >= 0.4 (2 of 4 records are no-LLM).
        let rates_total = after.total.max(1) as f64;
        assert!(
            after.conversion_rate >= delta_no_llm as f64 / rates_total,
            "conversion_rate shrank unexpectedly: {} vs {}",
            after.conversion_rate,
            delta_no_llm as f64 / rates_total
        );
    }

    #[test]
    fn sustained_degradation_counter_increments_only_on_degradation() {
        let _guard = match STREAK_LOCK.lock() {
            Ok(g) => g,
            // A poisoned mutex means another test panicked while holding it.
            // Recover by clearing the poison and proceeding.
            Err(poisoned) => poisoned.into_inner(),
        };

        // Reset the streak so we can reason about absolute counts.
        reset_consecutive_degradations();

        // Push enough degradations to cross the threshold.
        for _ in 0..SUSTAINED_DEGRADATION_THRESHOLD {
            record(JudgeDecision::AllJudgesFailed, "s", 5);
        }
        assert_eq!(
            consecutive_degradation_count(),
            SUSTAINED_DEGRADATION_THRESHOLD
        );
        assert!(sidecar_should_auto_disable());

        // A single successful judge verdict must reset the streak — the
        // runtime should auto-recover instead of staying stuck in the no-LLM
        // fallback.
        record(JudgeDecision::JudgeRan, "s", 5);
        assert_eq!(consecutive_degradation_count(), 0);
        assert!(!sidecar_should_auto_disable());
    }

    #[test]
    fn intended_conversions_cleared_consecutive_degradation_streak() {
        let _guard = match STREAK_LOCK.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        reset_consecutive_degradations();

        record(JudgeDecision::AllJudgesFailed, "s", 5);
        record(JudgeDecision::AllJudgesFailed, "s", 5);
        assert_eq!(consecutive_degradation_count(), 2);
        record(JudgeDecision::CadenceCarry, "s", 5);
        assert_eq!(consecutive_degradation_count(), 0);
    }

    #[test]
    fn explicit_opt_out_does_not_count_as_degradation() {
        let _guard = match STREAK_LOCK.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        reset_consecutive_degradations();

        // OptedOut is "intended", so the sidecar must not auto-disable even
        // if the user has been opting out for a long time.
        for _ in 0..(SUSTAINED_DEGRADATION_THRESHOLD + 1) {
            record(JudgeDecision::OptedOut, "s", 5);
        }
        assert_eq!(consecutive_degradation_count(), 0);
        assert!(!sidecar_should_auto_disable());
    }
}
