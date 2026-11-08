# Memory and skill update plan

Status: implemented (P0/P1/P2 core), 2026-11-08

Implemented surfaces:

- `jcode memory review list|accept|reject|defer`
- `jcode memory health`
- `jcode memory source status|resume`
- `jcode skill list|show|validate|update|rollback|history|health`
- `memory` tool actions `review`, `health`
- `skill_manage` tool actions `validate`, `health`, `update`, `rollback`

Remaining from the design: field-level G4 batch-diff refinement (extraction
batches now force review when more than five rows or more than 20% of an
active category change, but the diff is category-level rather than per
property field), body-similarity proposal merging (dedup is subject-key only),
skill-health signals for unused/failing skills and disappeared evidence,
enforced skill effects at the tool layer (manifests are currently advisory
metadata), and a hard query router (routing is currently a lexical
skill-prior, not a filter).

Reference: `/home/cc/Downloads/enterprise-ontology` skill design, read in full
(SKILL.md plus modeling-rules, lifecycle-ops, data-pipeline and query-routing).

This document analyzes jcode's current memory/skill update strategy against
that design and proposes a staged, selective adoption plan. It deliberately
does **not** copy the enterprise business-ontology model wholesale: jcode is a
coding assistant, not a CRM. The useful part is the update control plane:
schema/instance separation, provenance, diff gates, review queues, versioning,
quality inspection and source health.

## 1. What to borrow, and what not to borrow

### Adopt directly

| Enterprise design | jcode adaptation |
|---|---|
| Schema + instances are separate | Memory ontology + typed instances; skill contract + Markdown body |
| Mandatory provenance with locator | Add `locator` to `ProvenanceRecord` (message id, tool call id, file:line) |
| Confidence admission gates | `>=0.9` auto, `0.7-0.9` auto + review badge, `<0.7` pending only |
| L1/L2/L3 change levels | Memory instance / memory schema+skill metadata / ontology+skill contract |
| Diff before merge | Deterministic memory diff and skill-body diff |
| Human confirmation gates | Critical memories, authoritative overwrites, status loss, bulk anomalies, skill contract changes |
| Conflict priority | Method authority > source authority > recency > confidence; otherwise review |
| Soft delete and archive | Already present in lifecycle; make every destructive path use it |
| Periodic quality inspection | `jcode memory health` and `jcode skill health` |
| Source health and pause | Extraction/skill source health with automatic pause and alert |
| Sedimentation loop | Review queue + knowledge-gap list + action/rule candidates |
| Action contract | Skill manifest declares inputs, allowed tools, preconditions, effects and audit |
| Query routing | Route between memory facts, skill procedures and skill execution |

### Adapt, do not copy

- **Business entity skeleton** (`Organization`, `Person`, `Offering`,
  `Process`, `Artifact`): jcode's memory types are already task-shaped
  (`process_rule`, `contract`, `env_fact`, `correction`, `preference`,
  `entity`, `observation`). Keep them; do not introduce a second business
  taxonomy.
- **Action role permissions**: coding sessions do not have business roles.
  Use capability/approval gates (destructive command, file edits outside
  workspace, network/credential access) instead.
- **Full JSON schema per memory**: too heavy for single-sentence memories.
  Use typed properties only where structure pays off (contract endpoints,
  env facts, skill parts, review records).
- **Confirm everything critical**: review fatigue would kill adoption. Limit
  hard gates to user-stated/corrected knowledge, contracts, destructive skill
  upgrades and bulk anomalies.

### Do not adopt

- Manual business alignment questions before every ontology change.
- Business action roles and IM/email alert channels.
- Versioned schema files as the only source of truth; jcode already stores the
  ontology inside each graph and has runtime patches/migrations.

## 2. Current strategy: strengths and gaps

### Strengths already in place

- Runtime ontology v3 stored per graph, with hashes, history, patches,
  migrations and rollback (`MEMORY_ONTOLOGY_RUNTIME.md`).
- Deterministic write-time reconciliation: reinforce / supersede / contradict /
  relate, with authority ordering and `subject_key` anchors.
- Mandatory provenance method/confidence, trust level, lifecycle and TTL.
- L1 + Jev recall, pending-injection TTL, duplicate/dedup suppression and
  retrieval/injection/outcome counters.
- Skill mirrors, typed skill parts (`contains`, `next_step`), evidence edges,
  invocation recording and outcome-driven validation/invalidation.
- Deterministic skill distillation with cross-store thresholds.

### Gaps, in update-control terms (before this change)

| Area | Current behavior | Risk |
|---|---|---|
| Extraction | One pass, flat `TYPE|CONTENT|TRUST`, no locator, no structured properties | Cannot audit a claim back to the transcript; no way to validate schema-level facts |
| Confidence gates | Trust maps to a confidence, but no `needs_review` state; all accepted rows look equally healthy | Low-confidence claims silently enter recall |
| Memory diff | Reconcile compares one incoming row to the store; no batch diff or change report | Bulk regressions and authoritative overwrites are invisible |
| Review queue | Only skill artifacts have a `needs_review` health flag | No place to park disputed memories or await user confirmation |
| Change levels | Ontology patches have version/migration; memory rows and skills do not | No way to classify a change as safe vs structural |
| Conflicts | `contradicts` edges exist but are not surfaced in recall or review | Users see one side without knowing a conflict exists |
| Skill contract | Markdown skeleton with `TODO`; only name/description/evidence are machine-read | Skills cannot declare inputs, tools, preconditions or effects |
| Skill versioning | Installed skills are never overwritten; no diff, changelog, snapshot or rollback | Improvements cannot be shipped; stale skills persist |
| Proposal quality | 44 pending proposals, 1 installed skill at baseline; drafts are TODO skeletons and near-duplicate API subjects are common | Proposal backlog is not actionable and hides real candidates |
| Skill feedback | Invocation/outcome tags exist, but proposal ranking does not use usage, failure or recency | Failed or unused procedures can still rank highly |
| Health inspection | TTL GC and residual reclassification exist | No schema/instance/freshness/coverage/review-backlog report |
| Source health | Extraction failures are logged | No failure-rate, confidence-drift or duplicate-rate pause/alert |

### Live evidence (2026-11-08)

- Baseline current-project + global store: 522 memories (487 + 35), 0
  identityless entities and 0 deprecated `in_cluster` edges after the ontology
  migration. The migration-era zero `fact` count has since drifted: the active
  project store now has 2 active `fact` rows and the all-store scan has 4, so
  residual reclassification is still a live check. That scan reports 10,316
  memories across 68 stores, 271 detected contradictions and 9,368 missing
  locators; a manual sample includes lexical false positives, so conflict
  quality still needs review. This is why provenance/health work matters
  beyond one project.
- 44 files in `~/.jcode/skills/.proposals/`; 2 installed skills.
- Skill evidence graph: 41 `contains`, 26 `next_step`, 42 `derived_from` in the
  active project store.
- At baseline no memory carried `critical=true` and no memory-level review
  queue existed. After implementation the queue is empty on existing data
  (no new gated writes yet) and `skill health` reports 44 proposals, 0
  actionable and 44 draft skeletons.

This is the core conclusion: jcode has a good **write and recall** pipeline but
a weak **change-governance** pipeline. This stage optimizes update quality
instead of adding more memory types.

## 3. Target update model

### 3.1 Change levels

Use one vocabulary for memory and skills:

| Level | Scope | Examples | Required flow |
|---|---|---|---|
| L1 | Instance update | New memory, reinforce, relate, tag change, skill body wording | Diff -> auto if safe -> review if gated |
| L2 | Backward-compatible schema | New property/tag/anchor rule, skill input/allowed-tools addition, enum expansion | Version bump (minor) + validation + changelog |
| L3 | Structural/breaking | Type removal, contract change, skill precondition/effect change, bulk archive | Impact report + migration + human confirmation + version bump (major) |

### 3.2 Confidence and review gates

Admission policy:

| Confidence | Behavior |
|---|---|
| `>= 0.9` | Auto-admit |
| `0.7 - 0.9` | Admit with `review.needs_review=true`; include in report |
| `< 0.7` | Do not admit; write to review queue |

Hard confirmation gates (any one => `REQUIRE_CONFIRM`):

- **G1 critical**: memory is marked `critical`, or the change touches a
  user-stated preference/correction or a contract. Skill contract fields
  (triggers, allowed tools, preconditions, effects) are always critical.
- **G2 authoritative overwrite**: an existing value from `user_stated` or
  `structured_mapping` is replaced.
- **G3 status loss**: active -> archived/expired, or an active instance
  disappears from a batch, unless it is TTL GC or an explicit user forget.
- **G4 bulk anomaly**: more than 20% of a type, or more than 5 records in one
  batch, change in the same field.
- **G5 skill upgrade**: installed skill body/contract changes, even if the new
  draft has higher evidence; patch-level typo fixes may auto-admit.
- **G6 contradiction**: a `contradicts` edge would be created between two
  authoritative memories.

Everything else is `AUTO_OK`.

### 3.3 Review queue contract

A review item must contain:

```json
{
  "id": "review_...",
  "created_at": "...",
  "level": "L1|L2|L3",
  "gate": ["G2", "G6"],
  "target": {"kind": "memory|skill", "id": "..."},
  "old": {"content": "...", "provenance": {...}},
  "new": {"content": "...", "provenance": {...}},
  "diff": ["property:customer_level", "status:active->archived"],
  "recommendation": "accept|keep_old|needs_human",
  "status": "pending|accepted|rejected|deferred",
  "confirmed_by": null,
  "confirmed_at": null
}
```

Storage: one durable JSONL file per scope under
`<jcode_dir>/memory/reviews/` initially. Revisit SQLite tables only if the
queue exceeds a few hundred items. The review queue must be visible in
`memory health` and must alert when the oldest pending item exceeds 7 days.

## 4. Skill contract v2

Keep `SKILL.md` and Markdown instructions. Extend frontmatter with a
machine-checkable contract. All new fields are optional for existing skills.

```yaml
---
name: enterprise-ontology
description: ...
version: 1.2.0
status: active            # active | deprecated | archived
triggers:
  - "build or update an ontology"
  - "schema modeling"
inputs:
  - id: source
    type: text
    required: true
allowed-tools: [read, write, apply_patch, bash]
preconditions:
  - schema_validation_available
effects:
  - kind: write_file
    scope: workspace
  - kind: run_command
    scope: workspace
validation:
  - validate_skill_frontmatter
  - reference_integrity
evidence:
  - mem_...
changelog:
  - version: 1.2.0
    level: L2
    summary: add action/rule modeling reference
provenance:
  source: user_dialogue
  method: user_stated
  confidence: 0.95
---
```

Rules:

- Keep existing `name`, `description`, `evidence`, `allowed-tools` compatible.
- `version` starts at `0.1.0` for distilled drafts; acceptance promotes to
  `1.0.0`.
- Contract fields are validated before install: unknown `allowed-tools`,
  empty triggers, missing evidence, malformed changelog => reject.
- Skill body changes are diffed and classified L1/L2/L3.
- Installed skill updates write a snapshot to
  `~/.jcode/skills/.history/<name>/<version>.md` before replacing.
- `jcode skill rollback <name> <version>` restores a snapshot.
- Never silently overwrite an installed skill; the current "refuse overwrite"
  behavior remains the default, with an explicit `--update` path.

## 5. Memory update flow

```
session turn / finalize
  -> structured extraction candidate
     {content, type, properties, provenance{locator,method,confidence},
      effective_from, tags, subject_key}
  -> normalize (noise gate, subject_key, tags, type alias)
  -> diff against current store + batch diff
  -> classify AUTO_OK / REQUIRE_CONFIRM
     -> AUTO_OK: reconcile (reinforce/supersede/relate/contradict) + write
     -> REQUIRE_CONFIRM: write review item, keep old value active
  -> post-write validation (schema, dangling edges, duplicate check)
  -> metrics + extraction source health
  -> next health inspection surfaces backlog, conflicts and stale rows
```

Specific changes:

1. **Structured extraction output**: replace `TYPE|CONTENT|TRUST` with one
   JSON object per candidate (or JSONL). Include `locator` (message id / tool
   call id / file:line), `method`, `confidence`, `properties`, `subject_hint`.
2. **Batch diff**: compare the candidate set with the target store before any
   write. Produce `auto`, `confirm`, `review` buckets.
3. **Critical policy**: mark user-stated preferences, corrections and
   contracts as critical by default. Do not mark every observation critical.
4. **Conflict surfacing**: when injecting a memory that has a `contradicts`
   edge to another active memory, append one line: "conflicts with <id>
   (<source>, <date>)" and mark the pair in the review queue.
5. **Reviewer actions**: `jcode memory review list|accept|reject|defer`,
   plus the `memory` tool's `review` action. Accepted items append
   `confirmed_by/confirmed_at` to provenance and raise confidence to 0.95.
6. **TTL/GC exemption**: never route TTL archive through the confirmation gate;
   it is deterministic and reversible. Manual forget is a separate explicit
   user action.

## 6. Skill update flow

```
evidence scan (process_rule, correction, invocation outcomes)
  -> candidate scoring (evidence weight + store/session diversity
     + success rate + recency - failure rate - duplicate penalty)
  -> group by normalized subject; merge near-duplicate proposals
  -> draft skill v0.1.0 with contract + TODO body
  -> validate manifest + references
  -> diff against installed skill (if any)
  -> classify L1/L2/L3
     -> L1 safe body change: auto-update if evidence threshold met
     -> L2 contract addition: minor bump + review
     -> L3 contract/breaking change: major bump + impact report + review
  -> human accept -> snapshot old version -> install -> update evidence edges
  -> invoke -> record outcome -> reinforce/invalidate evidence
```

Specific changes:

1. **Proposal ranking**: do not emit every subject that reaches the threshold.
   Score and cap proposals (for example top 10 per pass, 1 per subject, merge
   near-duplicates by normalized subject and body similarity).
2. **Negative evidence**: incorporate `outcome_failures` and
   `skill-invalidated` tags. A procedure with repeated failures should be
   demoted or archived, not promoted.
3. **Staleness**: expire proposals not reviewed within 30 days; keep a
   knowledge-gap list instead of resurrecting the same draft forever.
4. **Draft body**: the current `TODO` skeleton is not acceptable as the only
   output. The sidecar should either generate concrete steps from evidence or
   mark the proposal as `needs_human_draft` and not count it as actionable.
5. **Skill health**: report unused skills (no invocation in 90 days), failing
   skills, skills whose evidence disappeared, and skills with deprecated
   contracts.

## 7. Health inspection and source health

### `jcode memory health`

Report:

- schema validation errors and ontology hash drift
- type distribution, TTL near-expiry, active/archived
- duplicate clusters and contradiction pairs
- missing `subject_key`, missing provenance locator, low-confidence rows
- review backlog by gate and age
- injection -> outcome conversion, stale-hit rate, never-injected rows
- per-store freshness and skill-evidence coverage

### `jcode skill health`

Report:

- installed skills, versions, last invocation, last successful outcome
- proposals: count, age, duplicates, evidence strength, blocked drafts
- skills whose evidence is archived/missing
- skills with failing outcomes or `needs_review` artifacts

### Extraction source health

Track per extraction batch/session:

- candidate count, admission rate, validation failure rate
- average confidence and drift from rolling mean
- duplicate/paraphrase ratio
- extraction failure count

If failure rate > 5%, confidence drops > 0.1, or duplicate ratio spikes, pause
automatic extraction for that source and surface an alert in the next session.
Do not silently resume; require `jcode memory source resume <id>`.

## 8. Retrieval and execution routing

Add a lightweight route decision before retrieval:

| Query shape | Route |
|---|---|
| Fact/state question | Memory recall (L1/Jev) with `active` filter |
| Procedure/how-to question | Skill first; memory evidence second |
| Multi-hop "why/which" | Graph traversal with explicit path |
| Operation request | Skill execution contract; no ad-hoc direct mutation |
| Conflicting memories | Present both + provenance, do not choose silently |

The route result should include a one-line trace: route, selected IDs, and
whether any selected item is `needs_review` or conflicted.

## 9. Code-level work breakdown

### Phase P0: governance foundations (1 week, implemented; field-level G4 refinement remaining)

- `jcode-memory-types`: add `ProvenanceRecord.locator`, `needs_review` flag,
  review-item type, change-level enum.
- `jcode-base`: confidence gate in `remember_in_graph`; review queue writer;
  `memory health` report; batch diff for extracted candidates.
- `jcode-app-core`: `memory review list|accept|reject|defer`; `memory health`.
- Tests: low-confidence routing, critical overwrite gate, bulk gate, review
  accept/reject, health report counts.

Acceptance: a low-confidence extraction is not admitted; a user-stated
preference overwrite creates a review item; health shows the backlog.

### Phase P1: skill contract and lifecycle (2-3 weeks, implemented; body-similarity merge and extended health remaining)

- `jcode-base/src/skill.rs`: manifest v2 parser/validator, version snapshots,
  diff/classify, update/rollback APIs.
- Distillation: proposal scoring, dedup, cap, negative evidence, staleness.
- CLI/tool: `skill list|show|validate|update|rollback`, `skill health`.
- Tests: v1 compatibility, L2/L3 diff gates, snapshot restore, proposal dedup.

Acceptance: a distillation pass caps emission at 20 and dedupes by subject
key; files not refreshed expire after 30 days, so the existing 44-file queue
drains over time rather than all at once. An installed skill can be updated
with a changelog and rolled back.

### Phase P2: source health and routing (3-5 weeks, implemented core; optional action audit remaining)

- Extraction source health metrics and pause/resume.
- Memory health source scan and coverage report.
- Retrieval route decision + conflict/review badges in prompt.
- Optional action audit for skill execution (tool calls, files touched,
  outcome, duration).

Acceptance: a synthetic extraction regression pauses the source; a query
selects the skill route for how-to questions; conflicting memories are shown
with provenance.

## 10. Metrics to prove the plan works

- Review backlog: count, oldest age, accept/reject ratio.
- Admission quality: duplicates prevented, authoritative overwrites blocked,
  low-confidence rows kept out.
- Skill quality: proposal -> accepted ratio, skill success rate, unused skills,
  rollback count.
- Recall quality: injection -> outcome conversion, stale-hit rate, conflict
  exposure rate.
- Source health: extraction failure rate, confidence drift, duplicate ratio.

Targets after P1 (not yet measured on a fresh review cycle):

- Review backlog < 50 items, oldest < 7 days.
- At least one in five actionable proposals is accepted, instead of leaving
  the current 44-file unreviewed backlog.
- Zero silent overwrites of `user_stated`/`structured_mapping` values.
- Every installed skill has version, evidence, and last outcome.

## 11. Risks and open decisions

- **Review fatigue**: auto-admit threshold must be tuned on real sessions.
  Start conservative for `critical` and authoritative rows only.
- **Lexical reconcile false positives**: keep the current conservative
  thresholds; review gates are the safety net, not a reason to loosen them.
- **Skill format compatibility**: existing 2 installed skills and 44 proposals
  must continue to parse; v2 fields optional.
- **Storage for reviews**: JSONL is enough initially; moving to SQLite is a
  later optimization if the queue grows.
- **Action layer scope**: decide whether skill effects are advisory metadata
  or enforced at the tool layer. Enforcing is safer but requires tool context
  plumbing.
- **Which memories are critical**: proposed default is user-stated preference,
  correction and contract; revisit after 2 weeks of review data.
