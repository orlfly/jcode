# Code design ontology bridge

Status: implemented 2026-11-09

This bridge lets a project record how runtime memory ontology types map to Rust
types. It is a validation and review aid, not a code generator. A class and an
ontology type are not one-to-one: one ontology type can span several code types,
and infrastructure types may be explicitly unmapped.

## Design map

Put the map at `.jcode/code-ontology.yaml` (YAML or JSON):

```yaml
version: 1
concepts:
  - id: skill
    kind: aggregate
    rust_type: Skill
    source: crates/jcode-base/src/skill.rs
    ontology_type: process_rule
    fields: []
    invariants:
      - "Version must increase when a skill contract changes."
      - "L3 contract changes require --confirm."
    ontology:
      id: jcode/default/v1
      version: 3
      hash: 97e4e5ba401d8927

  - id: memory-review-item
    kind: value_object
    rust_type: MemoryReviewItem
    source: crates/jcode-memory-types/src/review.rs
    unmapped_reason: "queue transport record; old/new payloads are the instances"
```

Kinds: `entity`, `value_object`, `aggregate`, `service`, `policy`,
`infrastructure`.

Every mapped concept should pin the ontology `id`, `version` and `hash`. A pin
mismatch is an error: the design decision must be revisited when the ontology
changes. Unmapped concepts need a non-empty `unmapped_reason`.

## Validation

```bash
jcode code validate .jcode/code-ontology.yaml
jcode code validate .jcode/code-ontology.yaml --json
jcode code validate .jcode/code-ontology.yaml --root . --ontology ontology.json
```

Checks:

| Code | Meaning |
|---|---|
| C001 | duplicate concept id |
| C002 | the same Rust type is claimed by two concepts (warning) |
| C003 | unmapped concept without a reason |
| C004 | ontology type does not exist |
| C005-C007 | ontology pin id/version/hash mismatch |
| C008 | mapped concept has no ontology pin (warning) |
| C009 | required ontology property has no mapped code field |
| C010 | lifecycle state is not allowed by the ontology type |
| C011 | lifecycle transition is not allowed |
| C012-C015 | missing Rust type, source path or declaration |

The default ontology is the compiled ontology. To validate against a patched
project ontology, export it first:

```bash
jcode memory ontology show --scope project --json \
  | jq '.ontology' > /tmp/project-ontology.json
jcode code validate .jcode/code-ontology.yaml --ontology /tmp/project-ontology.json
```

## How to use it for class design

1. Read the ontology type's properties, lifecycle and rules.
2. Decide whether the code type is an entity, value object, aggregate, service
   or policy; do not mirror ontology inheritance.
3. Map required properties to constructor fields or validated values.
4. Map lifecycle states to an enum and transitions to guarded methods.
5. Put rules into policies/services and tests, not into an ever-growing entity.
6. Record deliberate gaps in `unmapped_reason`.
7. Validate the map in CI so ontology changes surface stale class designs.

## Limits

- Source checks are declaration scans (`struct`/`enum`/`trait`/`type`), not a
  Rust parser; they catch missing files and renamed types, not signatures.
- `invariants` are design notes. They are not executable checks.
- The map does not inspect generics, visibility or trait implementations.
- Runtime ontology patches and code changes remain separate review flows.
