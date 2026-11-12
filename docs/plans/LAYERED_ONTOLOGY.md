# Layered memory ontology

Status: project overlay implemented (phases 1-3), global packs and promotion
pending, 2026-11-10

## 1. Problem

A project needs concepts that only make sense for that repository:
`acme.billing.invoice_flow`, `acme.release.rollback_owner`, `acme.legacy.import_job`.
The user also needs general concepts that apply everywhere:
software contracts, deployment procedures, cloud services, incidents, runbooks.

Today those two sets have no shared structure. Each memory store persists a full
`Ontology` snapshot in `GraphMetadata::ontology`. A project graph and the global
graph are independent, so:

- a project does not inherit new global concepts;
- project concepts cannot be proposed globally without copying them by hand;
- two projects can use the same flat type id for different meanings;
- the extractor prompt is hardcoded to seven core types, so project types are
  not extracted even when the project ontology declares them;
- a project ontology update has no reviewable diff in the repository.

The goal is one management model with two scopes, not one giant ontology and not
two disconnected ontologies.

## 2. Design: three tiers, one effective ontology

```text
core        compiled, immutable  general lifecycle primitives
packs       user-installed     software / systems / cloud / security
project     repository-owned   project concepts and local overrides
                              |
                              v
                 effective ontology for a store
```

### Tier 1: core

The compiled-in `jcode/default/v1` ontology. It keeps the stable primitives:
`env_fact`, `contract`, `process_rule`, `observation`, `preference`, `entity`,
`correction`. Core changes are rare and versioned with the binary.

### Tier 2: global packs

Optional partial ontologies installed under the user's home:

```text
~/.jcode/memory/ontology/layers/*.yaml
```

A pack describes a domain, not an individual project:

```yaml
id: jcode/pack/cloud
version: 1
priority: 20
description: Cloud services, deployments and operational procedures.
types:
  cloud.service:
    display_name: Cloud service
    parent: entity
    description: A managed service such as S3, RDS, Lambda, or a self-hosted equivalent.
  cloud.deployment:
    display_name: Deployment
    parent: process_rule
    description: How a service is built, released, rolled back and verified.
  systems.runbook:
    display_name: Runbook
    parent: process_rule
    description: Operational steps for an incident, migration or recovery.
type_patches:
  cloud.service:
    ttl_days: 365
```

Packs are user-controlled, can add rules/activities, and are applied to every
project store. They are not fetched from a repository automatically.

### Tier 3: project overlay

One reviewable file in the repository:

```text
<repo>/.jcode/memory/ontology.yaml
```

```yaml
id: project/acme-billing
version: 3
priority: 100
description: Concepts and overrides specific to the acme billing repository.
namespace: acme.billing
types:
  acme.billing.invoice_flow:
    display_name: Invoice flow
    parent: cloud.deployment
    description: Steps, invariants and owners for invoice generation and settlement.
  acme.billing.settlement_job:
    display_name: Settlement job
    parent: process_rule
type_patches:
  acme.billing.invoice_flow:
    ttl_days: 730
  acme.billing.settlement_job:
    present_in_prompt: false
```

The overlay is committed with the project, so changes are reviewed like code and
can be reverted with git. It never contains global concepts. Core memory
classes are code-defined and cannot be patched, disabled or redefined here.

## 3. Effective ontology and scope resolution

For a store, the effective ontology is a deterministic fold:

```text
global scope  = compose(core, global packs)
project scope = compose(core, global packs, project overlay)
```

Rules:

1. Layers are sorted by `priority`, then by file name for stable ties.
2. A layer may add a full type, patch fields of an existing type, or disable a
   type.
3. `type_patches` are field-level. A project that changes `contract.ttl_days`
   keeps future global changes to `contract.scoring` and `contract.properties`.
4. A full type entry replaces the definition. Use it only when changing fields a
   `TypePatch` cannot express (`display_name`, `parent`, `lifecycle`, `persistent`,
   `embeddable`).
5. `disabled_types` removes the type from the effective ontology. The operation
   is idempotent, so recomposition is safe.
6. Relationships, rules and activities are merged by id. Later layers replace an
   entry with the same id.
7. `category_aliases` are unioned. Later layers win on conflicting aliases.
8. Type inheritance uses the existing `OntologyType::parent`. A project concept
   such as `acme.billing.invoice_flow` should parent a general type, not copy its
   properties. Property lookup already follows `parent`.
9. Type ids are namespaced: `core` types stay flat for compatibility, packs use
   `<pack>.<concept>`, project overlays use `<project namespace>.<concept>`.
10. `compose` validates the result with `validate_ontology` and fails closed. An
    invalid overlay must not change runtime behavior.

The effective ontology keeps the parent id and version is
`max(base.version, layer.version)`. The content hash covers the composed result,
so a global or project change changes the hash.

## 4. Runtime storage

The current per-store snapshot remains the materialized effective ontology. It
is a cache, not the source of truth.

Add three `GraphMetadata` fields:

```rust
pub ontology_parent_hash: String;   // global effective hash for a project store
pub ontology_layer_hash: String;    // hash of the project overlay/pack stack
pub ontology_layer_ids: Vec<String>; // human-readable layer provenance
```

On `load_project_graph`:

1. Load the effective global ontology.
2. Read `~/.jcode/memory/ontology/layers/*.yaml`.
3. Read `<repo>/.jcode/memory/ontology.yaml`.
4. If both hashes match the stored metadata, reuse the materialized ontology.
5. Otherwise compose, validate, and replace `metadata.ontology`.
6. Persist on the next normal save. A recomposition is not a data migration.

On `apply_ontology_patch(scope = Project)`:

1. Resolve the effective global ontology and the current overlay.
2. Validate the patch against `compose(global, overlay)`.
3. Compute the new overlay as `diff(global, patched_effective)`.
4. Write the overlay YAML atomically.
5. Materialize the new effective ontology in the project graph metadata.
6. Push the previous effective ontology into `ontology_history`.

Global patches keep their current behavior: they patch the global ontology
directly. A project never installs its ontology into the process-wide registry.

`rollback_ontology(scope = Project)` restores the previous effective snapshot and
rewrites the overlay as `diff(global, snapshot)`. Data migrations are still not
reversed; restore a store backup when the data itself must be undone.

### Legacy stores

A project store that already has a `metadata.ontology` but no overlay gets a
one-time migration:

```text
overlay = diff(current_global_effective, stored_project_ontology)
```

Write that overlay to `.jcode/memory/ontology.yaml`, then continue with normal
composition. If the diff is empty, no file is needed.

## 5. Extraction and recall

The extractor must receive the effective type catalog, not a hardcoded list.

```text
Type: acme.billing.invoice_flow
Parent: cloud.deployment
Description: Steps, invariants and owners for invoice generation and settlement.
```

Rules:

- Include types with `present_in_prompt` and `persistent`.
- Put project overlay types first, then packs, then core.
- Cap the catalog at about 40 types to bound prompt size.
- The model must return the exact type id. Unknown ids are stored as
  `observation` and parked for review, never silently downgraded to `fact`.
- `MemoryCategory::Custom(<type id>)` already round-trips, so no enum change is
  required.
- Classification and TTL sweep call `type_from_category`, which already resolves
  exact ids and aliases from the effective ontology.

Recall remains store-scoped. A project store can retrieve global and project
memories. The global store never retrieves project-only types.

## 6. Governance and trust

- Core and global packs are L3 changes: schema validation, version bump, history,
  confirm required.
- Project overlays are L2: reviewable in git, confirm required when written by a
  session, and limited to types, aliases and relationships. Rules and activities
  are ignored in project overlays because a cloned repository is untrusted input
  and must not change memory side effects.
- Automatic finalize reflection writes to the narrowest writable layer:
  project overlay first. It can propose `add_type` and `update_type`, but applying
  still requires confirmation.
- Promotion from project to global is a separate proposal. Require evidence from
  at least three distinct projects, the same threshold idea as skill distillation,
  and an explicit confirm.
- Never bulk-rename stored `MemoryCategory` strings. New type ids are additive;
  migrations map old categories explicitly.

## 7. CLI

Implemented today:

```bash
jcode memory ontology show --scope project          # effective ontology + layers + parent_hash
jcode memory ontology validate proposal.json
jcode memory ontology apply proposal.json --scope project --apply
jcode memory ontology history --scope project
jcode memory ontology rollback <version> --scope project
```

Planned with the pack/promotion phases:

```bash
jcode memory ontology layers --scope project
jcode memory ontology compose --scope project --json
jcode memory ontology explain acme.billing.invoice_flow --scope project
jcode memory ontology promote acme.billing.invoice_flow --to-global
```

`explain` prints the defining layer, parent chain, inherited properties, effective
TTL/scoring, and whether the type is visible to the extractor.

## 8. Implementation phases

### Phase 1: composition primitives, no runtime change (implemented)

- Add `OntologyLayer`, `compose_ontology_layers`, and
  `diff_ontology_layer` to `jcode-memory-types`.
- Cover add/patch/disable, priority order, parent inheritance, rules/activities
  dedupe, alias conflicts, and idempotent recomposition.
- Add a JSON round-trip test for a project overlay file.

### Phase 2: project resolution and extraction catalog (implemented)

- Load packs and the project overlay in `MemoryManager` (packs remain future).
- Add parent/layer hash metadata and recomposition on load.
- Feed the effective type catalog to `Sidecar::extract_memories`.
- Test that a project type is extracted as `Custom(id)` and is invisible in the
  global store.

### Phase 3: patch and rollback redirection (implemented)

- Make project `apply_ontology_patch` write an overlay diff.
- Make project rollback rewrite the overlay.
- Adopt legacy project snapshots as overlays only when they are provably
  additive over the current global ontology. A snapshot that changes a core
  definition keeps its own snapshot until an explicit migration.
- Test that deleting an overlay type removes it after reload and that a core field
  change still flows through when only another field was overridden.

### Phase 4: packs, promotion and CLI

- Add global pack loading, `layers`, `compose`, and `explain`.
- Add cross-project promotion proposals and confirmation.
- Extend finalize reflection to target the project overlay.

## 9. Acceptance tests

1. A global pack type is usable in a project store without copying it.
2. A project overlay type never appears in the global store.
3. A project field patch on a core type keeps unrelated global updates flowing.
4. Removing a project overlay entry takes effect on reload, with no stale
   materialized type.
5. A project overlay containing rules or activities is rejected or stripped.
6. Extraction emits a project type id and the stored category is `Custom(id)`.
7. Applying a project patch writes a diff, not a full ontology copy.
8. Effective hash changes when either global or project inputs change, and data
   migrations run once.
9. Rollback restores the previous effective ontology and overlay.
10. A promotion proposal requires cross-project evidence and confirm.

## 10. Alternatives rejected

- **One global ontology with project tags.** Simple, but leaks every project
  concept into every project, creates id collisions, and makes the global prompt
  grow without bound.
- **Full ontology copy per project, current behavior.** No inheritance, no
  reviewable diff, and global changes never reach existing projects.
- **Namespaced ids without composition.** Avoids collisions but does not solve
  inheritance, field-level overrides, extraction prompting, or staged rollout.
