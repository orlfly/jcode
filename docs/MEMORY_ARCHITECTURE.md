# Memory architecture

Updated: 2026-10-14 (v2 rewrite; describes what is implemented, not what was
planned)

## What the system is

Local, durable memory for coding sessions: typed instances (the v2 ontology),
stored as graph nodes in one SQLite database (with a JSON fallback), indexed by
FTS5 full-text search and 22%-coverage embeddings, recalled each turn through
two tiers with a skill-linkage layer on top.

Three storage layers cooperate:

| Layer | Purpose | Where |
|---|---|---|
| Ontology (v2 types) | What a memory *is*, which drives decay, TTL, extraction methods and scoping | `crates/jcode-memory-types/src/ontology.rs` |
| Graph (nodes + edges) | Identity, links between instances, provenance, tags | `crates/jcode-memory-types/src/graph.rs` |
| Indexes (FTS5 + embeddings) | lexical and semantic lookup for the retrieval stages | `jcode_fts_*` tables, optional `embedding` field |

## The v2 ontology

Types: `env_fact` (30-day TTL), `contract`, `process_rule`, `observation`
(90-day TTL, the residual bucket), `preference`, `entity`, `correction`,
plus `goal:...` / `skill:...` **link nodes**, which are indexes of external
artifacts, not memories: they skip write reconciliation, TTL GC and clustering,
and only inject their bounded pointer line.

`MemoryCategory` carries the v2 types as real enum variants (`EnvFact`,
`Contract`, `ProcessRule`, `Observation`). Older string-shaped rows
(`Custom("env_fact")`) are canonicalized into variants by
`prepare_for_write` on load/write, so stores converge without a migration.

## Recall: two tiers

**Tier 1 (L1, local, default-on)**: `score_l1_relevance` scores every active
instance from the query: anchor match (`subject_key` contained in the
normalized query) → 1.0, tag hit → 0.6, two-plus term overlap → ≤0.5.
Threshold `L1_MIN_SCORE = 0.34`, top `L1_MAX_RESULTS = 3`.
Used when the judged path is unavailable (`agents.memory_l1_recall`, default
true). It needs no credentials, no LLM and no embedder.

**Tier 2 (Jev-judged)**: scoped candidates (BM25 prefilter to
`MAX_JEV_RECALL_CANDIDATES = 72` when the store exceeds it) are sent to Jev's
typed Decisions API as `noul` questions; answers above
`memory_jev_threshold` (default 0.8) are injected, max 5, no padding. A failed
batch invalidates the whole selection; there is no fallback mid-request - the
agent falls back to Tier 1 on the *next* turn.

Skill pointers: when L1 (or the query alone) matches a `skill:<name>` mirror
that was not already selected, a one-line `# Available Skill` hint is appended
so the model learns "this workspace already has a procedure for this".

Read feedback: every actually-injected memory gets `access_count += 1`
(`record_retrievals`, detached task). This feeds TTL refresh
(`touch()`), confidence decay with access boost, and the ranking prior.

## Write path (how a conversation becomes memory)

Triggered per fresh user turn into the async coordinator and at **session
finalize** (the only lifecycle event measured to fire in practice). The
finalize maintenance runs even without any LLM:

1. `noise_reason` gate rejects extraction noise (too-short, bare identifiers,
   tool payloads, commit hashes) - errors for LLM-extracted rows only.
2. `prepare_for_write`: derive `subject_key` (api/file/host/mr/task), sanitize
   tags, canonicalize categories, demote identity-less entities.
3. `reconcile` decides the relationship with the existing store by lexical
   similarity (deterministic, no embedder):
   `Contradict` (conflict) → `Reinforce` (≥0.85 restatement, +strength, no new
   row) → `Supersede` (≥0.5, same anchor or correction, authority ≥ existing:
   new row + supersedes edge, old archived) → `Relate` (≥0.3, relates_to edge)
   → `Insert`.
   Authority order: `UserStated 5 > StructuredMapping 4 > RuleExtraction /
   SkillInvocation 3 > LlmExtraction 2 > RuleInference 1`. An LLM paraphrase
   can never replace a user statement.
4. On save, FTS5 indexes are refreshed; links and tags become graph edges.

### Maintenance at finalize

- **TTL GC** (`gc_expired_memories`): archive instances whose type TTL elapsed
  since `updated_at`. `touch()` on injection restarts the clock.
- **Local reclassify** (`reclassify_project_store(apply=true)`): deterministic
  classifier for residual rows (procedures → contracts → env facts).
- **Skill distillation** (`distill_skill_evidence`, once per day per machine):
  see the next section.

## Skill linkage (memory ↔ skills)

Skills are files (`SKILL.md`); memory only mirrors and evidence-boards them.
Both halves are implemented:

- **Skills → memory**: `sync_skill_memory_mirrors` (finalize) writes a
  `skill:<name>` link node per loaded skill - global skills into the global
  store, project-local skills into that project's store. The mirror content is
  a ≤200-char pointer; the full body lives in `search_text` (match-only).
- **Memory → skills**: `collect_skill_evidence` walks every store, groups
  `process_rule` (weight 1.0) and `correction` (weight 2.0) instances by
  canonical `subject_key`, requires `evidence ≥ 3` across `≥ 2 stores`, and
  writes a draft `SKILL.md` to `~/.jcode/skills/.proposals/`. Nothing is
  installed silently.
- **Acceptance** (`jcode memory skill-accept <draft>`): installs the draft
  verbatim into the skills dir, deletes the draft, writes the mirror node with
  `skill:<name> -derived_from→ mem_*` edges for evidence ids that exist, and
  records one `SkillInvocation` row. Refuses drafts without evidence and never
  overwrites an installed skill.
- **Invocation loopback** (`record_skill_invocation`): every skill run records
  `id = skill-invocation:<name>` with `ExtractionMethod::SkillInvocation`
  (authority rank 3), idempotent per skill.

## Storage layout

- SQLite backend (default, `sqlite-gvec`): `~/.jcode/memory/backend-sqlite/gvec.sqlite`,
  one graph per project hash plus a global graph, `jcode_fts_*` index tables,
  `jcode_graph_meta` for last-saved provenance.
- Legacy JSON layout (`memory/global.json`,
  `memory/projects/<hash>.json`) is still readable; a one-time migration
  imports a non-empty JSON store into an empty SQLite one and renames the
  source.
- Embeddings: `embedding` (384-dim) written historically and by opted-in
  builds; recall does not require them (the Jev path predates and ignores
  them; the L1 path never used them).

## Provider access (Tier 2 only)

`auto` picks the first configured route: Jcode subscription (requires live
`capabilities.memory_jev` from `GET /v1/me`), OpenRouter, TypeSafe,
AI/ML API. No fallback across routes on failure. Batches ≤ 24 memories /
64 KiB; 60-second whole-selection deadline; UTF-8-bounded context (8 KiB).
Any failed batch invalidates the whole selection.

## Learning vs recall

Extraction (periodic / finalize; optional, `memory_sidecar_enabled`) writes
through the reconcile pipeline above. **Recall never depends on extraction:
a machine with no credentials recalls nothing from Jev, still gets L1 hits.**

## Privacy

Storage is local. Tier 2 sends the focused query plus candidate memories in
the selected scope to the selected provider; disabling memory or setting
`memory_l1_recall` without a Jev route keeps recall fully local. Do not store
secrets in memory.

## Runtime ontology (added 2026-10-29)

The ontology is now stored beside each memory graph (`GraphMetadata.ontology`),
not only compiled into the binary. A reflection pass can propose an
`OntologyPatch` (`memory` tool action `ontology`, or
`jcode memory ontology ...`), and applying it runs deterministic data
migrations before installing the new definition. See
[MEMORY_ONTOLOGY_RUNTIME.md](./MEMORY_ONTOLOGY_RUNTIME.md).

## Implementation map

- `crates/jcode-memory-types/src/{lib,graph,ontology,instance,reconcile}.rs`:
  types, graph, ontology, provenance, write reconcile.
- `crates/jcode-base/src/memory.rs`: manager, recall tiers, GC, distillation.
- `crates/jcode-base/src/memory_jev.rs`: Jev Decisions client and batching.
- `crates/jcode-base/src/memory_agent.rs`: async coordinator, finalize.
- `crates/jcode-base/src/memory/pending.rs`: scope-bound injection and
  retrieval writeback.
- `crates/jcode-base/src/skill.rs`: mirrors, invocation records, acceptance.
- `crates/jcode-app-core/src/tool/memory.rs`: the `memory` tool surface.

## Known gaps (tracked in the landing record)

- `supersedes`/`contradicts` edges: the reconcile branches exist and are
  wired, but the real store still carries almost exclusively v1-era edges;
  verify production counts before relying on conflict surfacing.
- `memory_judge_metrics` auto-disable: driven by the judged path; on a machine
  whose judge never runs, the metric stays silent by design.
- macOS/Windows behaviors of the runtime are untested on this machine.

Historical references: `docs/plans/MEMORY_GRAPH_PLAN.md` (graph-rerank
proposal, partially implemented), `docs/plans/MCP_SKILLS_PLAN.md`.