//! Stage S4 waypoint generation over ontology relation signatures (4.3.2).
//!
//! Deterministic half only: enumerate simple concept paths through relation
//! signatures, score them by path length, selectivity, and hub penalty, and
//! return a small beam. No LLM reranking and no property filters yet.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::engineering_ontology::{CLASS_WILDCARD, EngineeringOntology};

/// Maximum intermediate concepts per candidate path.
pub const MAX_WAYPOINTS: usize = 3;
/// Number of candidates kept after scoring.
pub const BEAM_WIDTH: usize = 3;
const MAX_EXPLORED_PATHS: usize = 2000;

/// One scored concept path between two anchored concepts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaypointCandidate {
    /// Full concept chain including start and goal.
    pub concepts: Vec<String>,
    /// Relation used for each hop; `relations.len() == concepts.len() - 1`.
    pub relations: Vec<String>,
    pub score: f32,
    /// Shorter paths score higher.
    pub plausibility: f32,
    /// Lower average degree of intermediate concepts scores higher.
    pub selectivity: f32,
    /// Average degree of intermediate concepts, normalized by the max degree.
    pub hub_penalty: f32,
}

impl WaypointCandidate {
    /// Intermediate concepts only.
    pub fn waypoints(&self) -> &[String] {
        if self.concepts.len() <= 2 {
            return &[];
        }
        &self.concepts[1..self.concepts.len() - 1]
    }
}

fn signature_edges(ontology: &EngineeringOntology) -> BTreeMap<String, Vec<(String, String)>> {
    let mut adjacency: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for definition in ontology.relations.values() {
        for domain in &definition.domain {
            if domain == CLASS_WILDCARD || !ontology.classes.contains_key(domain) {
                continue;
            }
            for range in &definition.range {
                if range == CLASS_WILDCARD || !ontology.classes.contains_key(range) {
                    continue;
                }
                adjacency
                    .entry(domain.clone())
                    .or_default()
                    .push((definition.id.clone(), range.clone()));
            }
        }
    }
    for edges in adjacency.values_mut() {
        edges.sort();
        edges.dedup();
    }
    adjacency
}

fn degrees(adjacency: &BTreeMap<String, Vec<(String, String)>>) -> BTreeMap<String, usize> {
    let mut degrees: BTreeMap<String, usize> = BTreeMap::new();
    for (from, edges) in adjacency {
        *degrees.entry(from.clone()).or_default() += edges.len();
        for (_, to) in edges {
            *degrees.entry(to.clone()).or_default() += 1;
        }
    }
    degrees
}

fn score_path(
    concepts: &[String],
    relations: &[String],
    degrees: &BTreeMap<String, usize>,
    max_degree: usize,
) -> WaypointCandidate {
    let steps = relations.len().max(1);
    let plausibility = 1.0 / steps as f32;
    let intermediates = &concepts[1..concepts.len().saturating_sub(1)];
    let (selectivity, hub_penalty) = if intermediates.is_empty() || max_degree == 0 {
        (1.0, 0.0)
    } else {
        let total: usize = intermediates
            .iter()
            .map(|concept| degrees.get(concept).copied().unwrap_or(0))
            .sum();
        let average = total as f32 / intermediates.len() as f32 / max_degree as f32;
        ((1.0 - average).clamp(0.0, 1.0), average.clamp(0.0, 1.0))
    };
    let score = 0.5 * plausibility + 0.3 * selectivity - 0.2 * hub_penalty;
    WaypointCandidate {
        concepts: concepts.to_vec(),
        relations: relations.to_vec(),
        score,
        plausibility,
        selectivity,
        hub_penalty,
    }
}

/// Enumerate and score concept paths from `start` to `goal`.
///
/// Returns at most [`BEAM_WIDTH`] candidates sorted by score desc and concept
/// chain ascending. Unknown concepts or no signature path yields an empty list.
pub fn generate_waypoint_candidates(
    ontology: &EngineeringOntology,
    start: &str,
    goal: &str,
) -> Vec<WaypointCandidate> {
    if start == goal
        || !ontology.classes.contains_key(start)
        || !ontology.classes.contains_key(goal)
    {
        return Vec::new();
    }
    let adjacency = signature_edges(ontology);
    let degrees = degrees(&adjacency);
    let max_degree = degrees.values().copied().max().unwrap_or(0);
    let max_edges = MAX_WAYPOINTS + 1;

    let mut candidates = Vec::new();
    let mut explored = 0usize;
    let mut stack: Vec<(Vec<String>, Vec<String>, BTreeSet<String>)> = vec![(
        vec![start.to_string()],
        Vec::new(),
        BTreeSet::from([start.to_string()]),
    )];
    while let Some((concepts, relations, seen)) = stack.pop() {
        if explored >= MAX_EXPLORED_PATHS {
            break;
        }
        let current = concepts.last().cloned().unwrap_or_default();
        if current == goal {
            explored += 1;
            candidates.push(score_path(&concepts, &relations, &degrees, max_degree));
            continue;
        }
        if relations.len() >= max_edges {
            explored += 1;
            continue;
        }
        if let Some(edges) = adjacency.get(&current) {
            for (relation, next) in edges.iter().rev() {
                if seen.contains(next) {
                    continue;
                }
                let mut next_concepts = concepts.clone();
                next_concepts.push(next.clone());
                let mut next_relations = relations.clone();
                next_relations.push(relation.clone());
                let mut next_seen = seen.clone();
                next_seen.insert(next.clone());
                stack.push((next_concepts, next_relations, next_seen));
            }
        }
    }
    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.concepts.cmp(&right.concepts))
            .then_with(|| left.relations.cmp(&right.relations))
    });
    candidates.truncate(BEAM_WIDTH);
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engineering_ontology::{
        BfoCategory, EngineeringClassDef, EngineeringRelationDef, default_engineering_ontology,
    };

    fn ontology_with(edges: &[(&str, &str, &str)], classes: &[&str]) -> EngineeringOntology {
        let mut ontology = EngineeringOntology::new("test", 1, "test ontology");
        for class in classes {
            ontology = ontology.with_class(EngineeringClassDef::new(
                *class,
                class.to_uppercase(),
                BfoCategory::IndependentContinuant,
            ));
        }
        for (relation, from, to) in edges {
            ontology = ontology.with_relation(EngineeringRelationDef::new(
                *relation,
                *relation,
                &[from],
                &[to],
            ));
        }
        ontology
    }

    #[test]
    fn prefers_direct_signature_paths() {
        let ontology = ontology_with(
            &[("hop1", "a", "b"), ("hop2", "b", "c"), ("direct", "a", "c")],
            &["a", "b", "c"],
        );
        let candidates = generate_waypoint_candidates(&ontology, "a", "c");
        assert_eq!(candidates.len(), 2, "{candidates:?}");
        assert_eq!(candidates[0].relations, vec!["direct".to_string()]);
        assert!(candidates[0].waypoints().is_empty());
        assert_eq!(
            candidates[1].concepts,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(candidates[1].waypoints(), &["b".to_string()]);
        assert!(candidates[0].score > candidates[1].score);
    }

    #[test]
    fn penalizes_hub_waypoints_via_selectivity() {
        // a - hop -> b - hop2 -> c beats a -> hub -> c because hub has a high degree.
        let mut edges = vec![
            ("hop", "a", "b"),
            ("hop2", "b", "c"),
            ("h1", "a", "hub"),
            ("h2", "hub", "c"),
        ];
        for index in 0..6 {
            let leaf: &'static str = Box::leak(format!("leaf{index}").into_boxed_str());
            edges.push(("s", "hub", leaf));
        }
        let ontology = ontology_with(
            &edges,
            &[
                "a", "b", "c", "hub", "leaf0", "leaf1", "leaf2", "leaf3", "leaf4", "leaf5",
            ],
        );
        let candidates = generate_waypoint_candidates(&ontology, "a", "c");
        let direct_via_b = candidates
            .iter()
            .find(|candidate| candidate.concepts == vec!["a", "b", "c"])
            .expect("b route");
        let via_hub = candidates
            .iter()
            .find(|candidate| candidate.concepts == vec!["a", "hub", "c"])
            .expect("hub route");
        assert!(direct_via_b.selectivity > via_hub.selectivity);
        assert!(direct_via_b.hub_penalty < via_hub.hub_penalty);
        assert_eq!(candidates[0].concepts, vec!["a", "b", "c"]);
    }

    #[test]
    fn returns_empty_for_unknown_or_disconnected_concepts() {
        let ontology = ontology_with(&[("hop1", "a", "b")], &["a", "b", "c"]);
        assert!(generate_waypoint_candidates(&ontology, "a", "missing").is_empty());
        assert!(generate_waypoint_candidates(&ontology, "b", "a").is_empty());
        assert!(generate_waypoint_candidates(&ontology, "a", "a").is_empty());
    }

    #[test]
    fn caps_waypoints_on_a_single_long_chain() {
        let ontology = ontology_with(
            &[
                ("r1", "a", "x1"),
                ("r2", "x1", "x2"),
                ("r3", "x2", "x3"),
                ("r4", "x3", "c"),
            ],
            &["a", "c", "x1", "x2", "x3"],
        );
        let candidates = generate_waypoint_candidates(&ontology, "a", "c");
        assert_eq!(candidates.len(), 1, "{candidates:?}");
        assert_eq!(candidates[0].waypoints().len(), MAX_WAYPOINTS);
        assert_eq!(
            candidates[0].waypoints(),
            &["x1".to_string(), "x2".to_string(), "x3".to_string()]
        );
    }

    #[test]
    fn caps_beam_width() {
        let chains = [
            ("s1", "a", "y1"),
            ("s2", "y1", "c"),
            ("t1", "a", "z1"),
            ("t2", "z1", "c"),
            ("u1", "a", "w1"),
            ("u2", "w1", "c"),
        ];
        let mut classes = vec!["a", "c"];
        for (_, from, to) in chains {
            if !classes.contains(&from) {
                classes.push(from);
            }
            if !classes.contains(&to) {
                classes.push(to);
            }
        }
        let ontology = ontology_with(&chains, &classes);
        let candidates = generate_waypoint_candidates(&ontology, "a", "c");
        assert_eq!(candidates.len(), BEAM_WIDTH, "{candidates:?}");
    }

    #[test]
    fn works_on_default_ontology_signatures() {
        let ontology = default_engineering_ontology();
        let candidates = generate_waypoint_candidates(&ontology, "file", "api");
        assert!(!candidates.is_empty(), "{candidates:?}");
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.relations.contains(&"defines".to_string())),
            "{candidates:?}"
        );
    }
}
