use super::*;
use crate::memory_types::EdgeKind;

fn test_skill(name: &str, dir: &Path, content: &str) -> Skill {
    Skill {
        name: name.to_string(),
        description: format!("Test skill {name}"),
        allowed_tools: None,
        evidence: Vec::new(),
        content: content.to_string(),
        path: dir.join("SKILL.md"),
        search_text: std::sync::OnceLock::new(),
    }
}

const BODY: &str = r#"# Optimization

Define the target metrics before changing code.

- subject: `api:/api/example`
- weight (corrections count double): 33.0

## Workflow

- Measure the full path before changing any code.
- Attribute bottlenecks to concrete components.
- Validate improvements with repeated benchmarks.

```bash
cargo test --profile selfdev
```

## Guardrails

Never optimize without a baseline.

## Evidence

- mem_1791955097411_820374521488728637
- review-task-put-required-role

## TODO

- [ ] Turn this draft into concrete steps.
"#;

#[test]
fn builds_sections_steps_code_and_script_metadata() {
    let temp = tempfile::tempdir().expect("tempdir");
    let scripts = temp.path().join("scripts");
    std::fs::create_dir_all(&scripts).expect("mkdir scripts");
    std::fs::write(
        scripts.join("run.py"),
        "def main():\n    pass\n\nclass Runner:\n    pass\n",
    )
    .expect("write script");

    let skill = test_skill("optimization", temp.path(), BODY);
    let parts = build_skill_parts(&skill);

    assert!(
        parts
            .iter()
            .any(|part| part.id.contains("::section:optimization:"))
    );
    assert!(
        parts
            .iter()
            .any(|part| part.id.contains("::section:workflow:"))
    );
    assert!(
        parts
            .iter()
            .any(|part| part.id.contains("::section:guardrails:"))
    );
    assert!(
        !parts
            .iter()
            .any(|part| part.id.contains("::section:evidence:")),
        "evidence lists are represented by artifact metadata, not index prose"
    );
    assert!(
        !parts.iter().any(|part| part.id.contains("::section:todo:")),
        "TODO sections are not procedural index content"
    );
    assert_eq!(
        parts
            .iter()
            .filter(|part| part.tags.iter().any(|tag| tag == "skill-part:step"))
            .count(),
        3
    );
    assert!(
        !parts.iter().any(|part| {
            part.tags.iter().any(|tag| tag == "skill-section:evidence")
                && part.tags.iter().any(|tag| tag == "skill-part:step")
        }),
        "evidence lists must not become executable step nodes"
    );
    assert!(
        !parts
            .iter()
            .any(|part| part.content.contains("weight (corrections count double)")),
        "generated metadata bullets must not become steps"
    );
    assert!(parts.iter().any(|part| {
        part.id.contains("::code:")
            && part.tags.iter().any(|tag| tag == "lang:bash")
            && part.content.contains("cargo test")
    }));
    let script = parts
        .iter()
        .find(|part| part.id.contains("::script:"))
        .expect("script metadata node");
    assert!(script.content.contains("scripts/run.py"));
    assert!(script.content.contains("Symbols: main, Runner"));
    assert!(script.tags.iter().any(|tag| tag == "lang:python"));

    let mut ids = HashSet::new();
    for part in &parts {
        assert!(part.id.starts_with("skill:optimization::"));
        assert!(ids.insert(part.id.clone()), "duplicate id {}", part.id);
        assert!(
            part.subject_key
                .as_deref()
                .is_some_and(|key| key.starts_with("skill:optimization::"))
        );
    }
}

#[test]
fn ids_are_stable_until_the_source_changes() {
    let temp = tempfile::tempdir().expect("tempdir");
    let skill = test_skill("stable", temp.path(), BODY);
    let first: Vec<String> = build_skill_parts(&skill)
        .into_iter()
        .map(|part| part.id)
        .collect();
    let second: Vec<String> = build_skill_parts(&skill)
        .into_iter()
        .map(|part| part.id)
        .collect();
    assert_eq!(first, second);

    let changed = test_skill(
        "stable",
        temp.path(),
        &BODY.replace(
            "Never optimize without a baseline.",
            "Never optimize without a baseline or a rollback plan.",
        ),
    );
    let third: Vec<String> = build_skill_parts(&changed)
        .into_iter()
        .map(|part| part.id)
        .collect();
    assert_ne!(first, third);
    assert!(third.iter().any(|id| id.contains("section:guardrails")));
}

#[test]
fn sync_creates_parent_edges_and_prunes_removed_parts() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut graph = MemoryGraph::new();
    graph.add_memory(
        MemoryEntry::new(MemoryCategory::Observation, "Skill `/stable`")
            .with_id("skill:stable".to_string()),
    );
    let skill = test_skill("stable", temp.path(), BODY);

    let (mirrored, pruned) = sync_skill_parts(&mut graph, &skill);
    assert!(mirrored > 0);
    assert_eq!(pruned, 0);
    let part_ids: Vec<String> = graph
        .memories
        .keys()
        .filter(|id| is_skill_part_id(id))
        .cloned()
        .collect();
    assert!(!part_ids.is_empty());
    for id in &part_ids {
        assert!(
            graph
                .get_edges(id)
                .iter()
                .any(|edge| edge.target == "skill:stable"
                    && matches!(edge.kind, EdgeKind::DerivedFrom)),
            "missing parent edge for {id}"
        );
    }

    let (mirrored, pruned) = sync_skill_parts(&mut graph, &skill);
    assert_eq!((mirrored, pruned), (0, 0), "rerun must be idempotent");

    let changed = test_skill(
        "stable",
        temp.path(),
        &BODY.replace("- Validate improvements with repeated benchmarks.\n", ""),
    );
    let (_, pruned) = sync_skill_parts(&mut graph, &changed);
    assert!(pruned > 0, "edited body must prune old derived nodes");
    assert!(
        graph.memories.keys().all(|id| !id.contains("step:3-")),
        "removed step node must be gone"
    );
}
