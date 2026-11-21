//! Local quality flags attached to one returned path.
//!
//! These flags are runtime decision signals, not a global quality report. They
//! let a planner insert verification, request disambiguation, or avoid a path
//! with weak provenance without re-reading the whole graph.

use serde::{Deserialize, Serialize};

/// One local path-quality concern from design table 4.6.9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathQualityFlag {
    StaleEdge,
    HypothesizedEdge,
    HighDegreeHub,
    AmbiguousEntity,
    LowProvenance,
    StitchedPath,
    RelaxedQuery,
}

impl PathQualityFlag {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StaleEdge => "stale_edge",
            Self::HypothesizedEdge => "hypothesized_edge",
            Self::HighDegreeHub => "high_degree_hub",
            Self::AmbiguousEntity => "ambiguous_entity",
            Self::LowProvenance => "low_provenance",
            Self::StitchedPath => "stitched_path",
            Self::RelaxedQuery => "relaxed_query",
        }
    }
}

/// Quality flags plus human-readable reasons for one path.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathQuality {
    pub flags: Vec<PathQualityFlag>,
    pub reasons: Vec<String>,
}

impl PathQuality {
    pub fn is_empty(&self) -> bool {
        self.flags.is_empty()
    }

    pub fn has(&self, flag: PathQualityFlag) -> bool {
        self.flags.contains(&flag)
    }

    pub fn push(&mut self, flag: PathQualityFlag, reason: impl Into<String>) {
        if !self.flags.contains(&flag) {
            self.flags.push(flag);
        }
        let reason = reason.into();
        if !self.reasons.iter().any(|seen| seen == &reason) {
            self.reasons.push(reason);
        }
    }

    /// Stable space-separated flag list for compact tool output.
    pub fn label(&self) -> String {
        self.flags
            .iter()
            .map(|flag| flag.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_are_deduplicated_and_keep_one_reason() {
        let mut quality = PathQuality::default();
        quality.push(PathQualityFlag::StitchedPath, "crosses a bridge");
        quality.push(PathQualityFlag::StitchedPath, "crosses a bridge");
        assert_eq!(quality.flags, vec![PathQualityFlag::StitchedPath]);
        assert_eq!(quality.reasons, vec!["crosses a bridge"]);
        assert_eq!(quality.label(), "stitched_path");
    }
}
