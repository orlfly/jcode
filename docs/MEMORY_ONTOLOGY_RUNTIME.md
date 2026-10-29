# Runtime ontology storage, reflection and migration

Updated: 2026-10-29

This document describes the implemented runtime path that lets the memory
ontology live inside the memory store, be adjusted during a reflection pass,
and carry existing memory instances along when it changes.

## 1. Storage

The ontology is stored with each memory store in `GraphMetadata`:

```rust
pub struct GraphMetadata {
    pub ontology_id: String,
    pub ontology_version: u32,
    pub ontology: Option<Ontology>,          // full serialized definition
    pub ontology_hash: String,               // deterministic content hash
    pub ontology_history: Vec<OntologySnapshot>,
    pub applied_migrations: Vec<String>,
    pub migration_state: Option<MigrationState>,
    // ... counters ...
}
```

Both backends persist it without extra schema work:

- JSON: the whole `MemoryGraph` (including metadata) is one document.
- SQLite/gvec: `jcode_graph_meta(store, json)` already stores the serialized
  metadata per named graph.

On the first load of a store that predates this feature,
`MemoryManager::bind_and_check_ontology` writes the compiled default ontology
into the metadata. On later loads the stored ontology is authoritative; the
compiled `default_ontology()` remains a fallback for stores that carry none.

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
- `crates/jcode-base/src/memory/backend/sqlite_gvec.rs`:
  `ontology_metadata_round_trips_through_the_meta_table`.
