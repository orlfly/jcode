use super::*;
use jcode_memory_types::default_ontology;

fn ontology_pin(ontology: &Ontology) -> OntologyPin {
    OntologyPin {
        id: ontology.id.clone(),
        version: ontology.version,
        hash: ontology_content_hash(ontology),
    }
}

fn write_source(root: &Path, relative: &str, declaration: &str) {
    let path = root.join(relative);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    std::fs::write(path, declaration).expect("write source");
}

fn valid_document(root: &Path, ontology: &Ontology) -> CodeOntology {
    write_source(root, "src/contract.rs", "pub struct Contract;\n");
    CodeOntology {
        version: 1,
        concepts: vec![CodeConcept {
            id: "contract".into(),
            kind: CodeConceptKind::Entity,
            rust_type: "Contract".into(),
            source: "src/contract.rs".into(),
            ontology_type: Some("contract".into()),
            unmapped_reason: None,
            fields: Vec::new(),
            invariants: vec!["content must not be empty".into()],
            lifecycle: Some(CodeLifecycle {
                states: vec!["active".into(), "archived".into()],
                transitions: vec![CodeTransition {
                    from: "active".into(),
                    to: "archived".into(),
                }],
            }),
            ontology: Some(ontology_pin(ontology)),
        }],
    }
}

#[test]
fn valid_code_ontology_passes() {
    let temp = tempfile::tempdir().expect("temp");
    let ontology = default_ontology();
    let document = valid_document(temp.path(), &ontology);
    let report = validate_code_ontology(&document, &ontology, temp.path());
    assert!(report.is_valid(), "{:?}", report.issues);
    assert_eq!((report.checked, report.mapped, report.unmapped), (1, 1, 0));
    assert!(report.summary().contains("1 mapped"));
}

#[test]
fn unknown_type_missing_pin_and_bad_source_are_errors() {
    let temp = tempfile::tempdir().expect("temp");
    let ontology = default_ontology();
    let document = CodeOntology {
        version: 1,
        concepts: vec![CodeConcept {
            id: "missing".into(),
            kind: CodeConceptKind::Entity,
            rust_type: "Missing".into(),
            source: "src/missing.rs".into(),
            ontology_type: Some("no_such_type".into()),
            unmapped_reason: None,
            fields: Vec::new(),
            invariants: Vec::new(),
            lifecycle: None,
            ontology: None,
        }],
    };
    let report = validate_code_ontology(&document, &ontology, temp.path());
    let codes: Vec<_> = report
        .issues
        .iter()
        .map(|issue| issue.code.as_str())
        .collect();
    assert!(codes.contains(&"C004"), "{codes:?}");
    assert!(codes.contains(&"C014"), "{codes:?}");
    assert!(!report.is_valid());
}

#[test]
fn lifecycle_and_unmapped_rules_are_checked() {
    let temp = tempfile::tempdir().expect("temp");
    let ontology = default_ontology();
    write_source(
        temp.path(),
        "src/contract.rs",
        "pub enum Contract { Active }\n",
    );
    let document = CodeOntology {
        version: 1,
        concepts: vec![
            CodeConcept {
                id: "contract".into(),
                kind: CodeConceptKind::Entity,
                rust_type: "Contract".into(),
                source: "src/contract.rs".into(),
                ontology_type: Some("contract".into()),
                unmapped_reason: None,
                fields: Vec::new(),
                invariants: Vec::new(),
                lifecycle: Some(CodeLifecycle {
                    states: vec!["archived".into(), "deleted".into()],
                    transitions: vec![CodeTransition {
                        from: "archived".into(),
                        to: "deleted".into(),
                    }],
                }),
                ontology: Some(ontology_pin(&ontology)),
            },
            CodeConcept {
                id: "infra".into(),
                kind: CodeConceptKind::Infrastructure,
                rust_type: "Contract".into(),
                source: "src/contract.rs".into(),
                ontology_type: None,
                unmapped_reason: Some("runtime container".into()),
                fields: Vec::new(),
                invariants: Vec::new(),
                lifecycle: None,
                ontology: None,
            },
        ],
    };
    let report = validate_code_ontology(&document, &ontology, temp.path());
    let codes: Vec<_> = report
        .issues
        .iter()
        .map(|issue| issue.code.as_str())
        .collect();
    assert!(codes.contains(&"C010"), "{codes:?}");
    assert!(codes.contains(&"C011"), "{codes:?}");
    assert!(codes.contains(&"C002"), "{codes:?}");
    assert_eq!(report.unmapped, 1);
}

#[test]
fn yaml_loader_round_trips() {
    let temp = tempfile::tempdir().expect("temp");
    let path = temp.path().join("code-ontology.yaml");
    std::fs::write(
        &path,
        "version: 1\nconcepts:\n  - id: contract\n    kind: entity\n    rust_type: Contract\n    source: src/contract.rs\n    ontology_type: contract\n    fields: []\n",
    )
    .expect("write map");
    let document = load_code_ontology(&path).expect("load");
    assert_eq!(document.concepts.len(), 1);
    assert_eq!(document.concepts[0].kind, CodeConceptKind::Entity);
}
