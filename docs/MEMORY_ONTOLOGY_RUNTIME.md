# Runtime ontology storage, reflection and migration

Updated: 2026-10-29

This document describes the implemented runtime path that lets the memory
ontology live inside the memory store, be adjusted during a reflection pass,
and carry existing memory instances along when it changes.

Detailed design (Chinese): [MEMORY_ONTOLOGY_DESIGN.zh.md](./MEMORY_ONTOLOGY_DESIGN.zh.md).

## 1. Storage

The ontology is stored with each memory store in `GraphMetadata`:

```rust
pub struct GraphMetadata {
    pub ontology_id: String,
    pub ontology_version: u32,
    pub ontology: Option<Ontology>,          // None = compiled default; Some = non-default override
    pub ontology_hash: String,               // effective ontology hash (default hash when None)
    pub ontology_history: Vec<OntologySnapshot>,
    pub applied_migrations: Vec<String>,
    pub migration_state: Option<MigrationState>,
    // ... counters ...
}
```

Both backends persist it without extra schema work:

- JSON: the whole `MemoryGraph` (including metadata) is one document.
- SQLite/gvec: `jcode_graph_meta(store, json)` stores the serialized
  metadata per named graph.

Default ontologies are stored by reference: `ontology = None` plus
`ontology_id`, `ontology_version` and `ontology_hash`. A full snapshot is
written only when the effective ontology differs from the compiled default.
Loading a redundant default snapshot clears it, and `jcode memory migrate
--apply` compacts existing stores.

`MemoryManager::bind_and_check_ontology` records the effective id, version and
hash on load and drops a redundant default snapshot. A store that predates
runtime ontology storage therefore reads correctly without migration; the
compiled `default_ontology()` is the fallback whenever `ontology` is `None`.

## 2. Runtime resolution

`MemoryManager::ontology_for_graph(graph)` resolves the active ontology:

1. `graph.metadata.ontology` when present.
2. Otherwise the process registry default.

Every consumer that makes an ontology decision uses this resolver:

- write-time reconcile thresholds (`activity.write_reconcile`)
- TTL GC (`type.ttl_days`)
- residual reclassification
- skill evidence selection
- the global scope gate
- category/type resolution

`OntologyRegistry` is now mutable behind an `RwLock` and clones share the same
map. `OntologyRegistry::install(ontology)` replaces the active default without
restarting the process, so subsequent writes and recalls use the new
definition. In-flight operations that already hold an `Arc<Ontology>` keep
using the version they pinned.

## 3. Patch format

A reflection pass produces an `OntologyPatch`, never a mutated ontology:

```json
{
  "base_id": "jcode/default/v1",
  "base_version": 2,
  "base_hash": "…",
  "rationale": "env_fact rows are re-created after TTL expiry",
  "evidence": ["session_…", "mem_…"],
  "ops": [
    {
      "op": "update_type",
      "id": "env_fact",
      "patch": { "ttl_days": 45.0 }
    }
  ],
  "migrations": [
    { "step": "recompute_subject_key", "overwrite": false }
  ]
}
```

Supported operations:

- `add_type`, `update_type`, `remove_type`
- `add_relationship`, `remove_relationship`
- `add_rule`, `remove_rule`
- `add_activity`, `remove_activity`
- `set_category_alias`, `remove_category_alias`

`update_type` only exposes fields that are safe to change at runtime:
`ttl_days`, `clear_ttl`, `scoring`, `decay`, `present_in_prompt`,
`extraction_methods`, `legacy_category`. Structural fields (`id`, `parent`,
lifecycle state machine) still require a code change.

Validation (`migration::validate_ontology`) rejects:

- empty ids, non-positive TTLs, non-positive half-lives
- aliases pointing at unknown types
- rules applying to unknown types
- rules linking unknown relationships
- removal of a type without a matching data migration
- removal of a relationship without a `rewire_edge` migration
- stale `base_version` / `base_hash`

Applying a patch bumps `ontology.version` by one and leaves the base untouched,
which makes dry runs and rollback trivial.

## 4. Data migrations

Structural changes ship with deterministic `MigrationStep`s:

| Step | Effect |
|---|---|
| `remap_category` | rewrite one category into another |
| `split_category` | first-match content markers into new categories |
| `retire_category` | move rows to a fallback type, or archive them |
| `recompute_subject_key` | derive anchors, or recompute all |
| `resanitize_tags` | re-run the tag vocabulary and refresh search text |
| `normalize_lifecycle` | repair archived-but-active rows |
| `demote_identityless_entities` | entity without identity becomes a fact |
| `drop_dangling_edges` | remove edges whose endpoints vanished |
| `rewire_edge` | delete an edge kind or replace it |
| `reclassify_residual` | deterministic procedure/contract/env split |

The executor is `migration::apply_migrations`. It is local, deterministic and
idempotent. A dry run executes the same steps on a cloned graph and reports
counts without writing.

## 5. Reflection integration

Review/autoreview/judge sessions may inspect the stored ontology and write a
proposal, but may not apply it:

```json
{ "action": "ontology", "ontology_action": "show", "scope": "project" }
{ "action": "ontology", "ontology_action": "propose", "scope": "project",
  "patch": { ... } }
```

The tool writes proposals to:

```
<jcode_dir>/memory/ontology/.proposals/<timestamp>-<ontology-id>.json
```

Applying from a session requires `confirm: true` and is rejected for review
sessions. The deliberate path is the CLI:

```bash
jcode memory ontology show --scope project
jcode memory ontology validate proposal.json
jcode memory ontology apply proposal.json --scope project          # dry run
jcode memory ontology apply proposal.json --scope project --apply  # write
jcode memory ontology history --scope project
jcode memory ontology rollback 2 --scope project
```

Applying persists the new ontology, hash, version, history snapshot and
applied migration ids, then installs it into the process registry.

### Automatic finalize reflection

Session finalize also runs a proposal-only reflection pass after memory
extraction succeeds (`run_final_extraction` -> `propose_ontology_from_session`).
It samples the memories extracted from that session and asks the sidecar for a
small patch. It never applies the result.

The automatic envelope is narrower than the tool/CLI path:

- only `update_type` ops are kept; add/remove type, rule, activity,
  relationship, alias and all migrations are dropped;
- evidence ids are restricted to the session's extracted memory ids;
- base id/version/hash are overwritten from the current ontology and the patch
  must pass `apply_patch` plus `validate_ontology`, or nothing is written;
- at least 5 extracted memories are required;
- at most one automatic proposal per project per 24 hours;
- output is written as `<timestamp>-auto-<ontology-id>-<nonce>.json` in the
  proposals directory above.

Disable it with `JCODE_ONTOLOGY_REFLECTION=0`. A no-change (`NONE`) response is
recorded in `memory/ontology/.reflection-<project-store-key>.json` and still
consumes that project's daily window. Applying an automatic proposal uses the
same confirmed `ontology` tool call or CLI path as any other proposal.

### Project overlay

A project can carry a reviewable, repository-owned overlay:

```
<repo>/.jcode/memory/ontology.yaml
```

The effective project ontology is `compose(global ontology, overlay)`. Global
types are inherited, so a project sees `contract`, `process_rule`, and any
global domain additions without copying them. Project types stay local and never
appear in the global store.

A project with no overlay still inherits the global ontology when its stored
snapshot is exactly the compiled default or only a same-definition subset of
the global ontology. A snapshot that changes a core definition is treated as a
possible legacy customization and kept until explicitly migrated.

Overlay shape (see `jcode_memory_types::layered::OntologyLayer`):

```yaml
id: project/acme-billing
version: 4
priority: 100
namespace: acme.billing
types:
  acme.billing.invoice_flow:
    display_name: Invoice flow
    parent: contract
    description: Invoice generation and settlement procedure.
type_patches:
  acme.billing.invoice_flow:
    ttl_days: 45.0
```

Rules:

- `types` adds or fully replaces a type. Prefer `parent` inheritance over
  copying an existing definition.
- `type_patches` is field-level: a project TTL override does not pin scoring,
  decay, properties, or future global changes to those fields.
- `disabled_types` removes a business type from the project's effective
  ontology. Core memory classes (`env_fact`, `contract`, ...) are code-defined
  and cannot be patched, disabled or redefined by an overlay.
- Overlay `rules` and `activities` are stripped on load. A cloned repository is
  untrusted input and must not change memory side effects.

Applying `jcode memory ontology apply ... --scope project --apply` writes the
diff back to this file instead of copying a full snapshot. Rollback rewrites the
overlay and removes it when the restored state equals the global ontology.
Metadata records `ontology_parent_hash`, `ontology_layer_hash` and
`ontology_layer_ids`; a changed global ontology or overlay invalidates the
effective metadata on the next load.

An older project store that persisted a full ontology snapshot is adopted
automatically only when the snapshot is provably additive: every current global
type, relationship, rule and activity is unchanged and the store only adds extra
project types. A snapshot that changes a core definition may contain stale
compiled defaults, so it keeps its own snapshot until an explicit migration.

Extraction uses the effective type catalog. Project types are appended to the
extractor prompt with their description and parent, and the returned id is
stored as `MemoryCategory::Custom(id)`. Types with `present_in_prompt: false`
are excluded.

`jcode memory ontology show --scope project` prints `layers` and `parent_hash`
alongside the effective ontology, so it is clear which overlay produced the
result. JSON output carries the same fields.

## 6. Rollback

Each apply pushes the previous ontology into `ontology_history` (bounded to 10
snapshots). `rollback_ontology` restores a snapshot and installs it. Data
migrations are not reversed automatically; use a store backup when the data
change itself must be undone.

## 7. Tests

- `crates/jcode-memory-types/src/migration.rs`: patch validation, hash
  stability, category split, tag/anchor/edge migrations.
- `crates/jcode-base/src/memory_tests.rs`:
  `ontology_patch_persists_migrates_and_rolls_back` covers apply, persistence
  across manager instances, and rollback.
  `project_overlay_inherits_global_and_stays_local`,
  `project_patch_writes_overlay_diff_and_rolls_back`, and
  `project_field_patch_keeps_global_changes_flowing` cover the layered path.
- `crates/jcode-memory-types/src/layered.rs`: composition, priority, disable,
  idempotence, diff round-trips, and layer stack hashing.
- `crates/jcode-base/src/sidecar.rs`:
  `extraction_catalog_includes_project_types` covers prompt catalog filtering.
- `crates/jcode-base/src/memory/backend/sqlite_gvec.rs`:
  `ontology_metadata_round_trips_through_the_meta_table`.
