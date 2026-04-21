<!--
WAVE BACKLOG DRAFT — COPY-TO-USE TEMPLATE

Purpose:
  Draft a whole wave of GitHub issues as one reviewable markdown file
  BEFORE opening the issues. Keeps shared context in one place, uses
  stable draft IDs so children can cross-reference each other, and
  lets the whole backlog be reviewed in a single scroll.

How to use:
  1. Copy this file into your project's `.plans/` directory
     (gitignored). Rename to describe the wave, e.g.
     `.plans/wave-1-diagonal-backlog.md`.
  2. Pick a short project prefix (e.g. PYX for pyrox, OBX for
     optax_bayes) and number drafts sequentially: <PREFIX>-01,
     <PREFIX>-02, … Draft IDs survive the draft→publish conversion
     so references between issues remain meaningful.
  3. Fill in the wave-level shared context once at the top, then
     draft each issue body. Review the whole file as a unit.
  4. When ready, open each draft as a real GitHub issue using the
     matching .github/ISSUE_TEMPLATE — the epic-wave, epic-theme,
     feature, design, bug, and research templates all take bodies
     compatible with what this file produces.
  5. After publishing, either delete this file or replace draft IDs
     with the assigned GitHub issue numbers to keep the record.

Conventions:
  - For algorithmic / numerical issues, math is part of the spec, not
    decoration. Include the equations another implementer will need:
    update rules, factorization identities, sign conventions,
    approximations, and invariants to test.
  - Use normal GitHub math syntax for issue-ready drafts:
    inline `$...$`, display `$$...$$`.
  - Unicode math in prose (σ², ∑, Λ⁻¹, O(d³)) is also encouraged when
    it improves readability in plain text.
  - Delete sections that don't apply. Every header below is optional
    except Wave Goal, one Theme Epic, and one child issue.
  - Rename headings to fit the content (see .github/ISSUE_TEMPLATE/
    guidance for examples — "Design Snapshot" → "Demo To Implement"
    etc.).
-->

# [Wave N] <title>

---

## Shared Context

<!--
One or two paragraphs that are TRUE for every issue in this wave. Source
material the drafter used (design docs, papers, existing code). The goal
is that another contributor (human or AI agent) can implement this wave
without opening the private design docs or asking follow-up questions.
-->

## Design Snapshot

<!--
Patterns, conventions, or excerpted API signatures that cross-cut every
issue in this wave. Copy the signatures here so children don't have to
duplicate them. Examples of what belongs here:

- Cross-cutting interface or protocol: `class Foo(eqx.Module): ...`
- A convention every child must respect: natural-parameter state
  (η, s) not moment (m, v); log-likelihood convention; `_for_loss`
  wrapper pattern.
- Shared types: `class WaveState(NamedTuple): ...`

Keep prose short — lead with code.
-->

```python
# Lead with the most important cross-cutting signature or pattern.
```

## Intended Package Layout

<!--
If this wave introduces new modules, show the intended layout. Later
waves get to assume this shape. Keep it to the modules touched by
THIS wave — don't sketch the whole package.
-->

```text
src/<package>/
  __init__.py
  _core/
    __init__.py
    <new_module>.py
  ...
```

## Runtime Boundary

<!-- Optional. Dependency boundaries this wave introduces or enforces. -->

- **Required runtime**: `<deps>`
- **Optional runtime**: `<deps>`
- **Dev / test only**: `<deps>`
- **Examples / docs only**: `<deps>`

---

# [Wave N] <wave title>
Draft ID: `<PREFIX>-01`

## Goal
<!-- One sentence: what outcome does this wave deliver? Maps to a milestone / release. -->

## Why This Wave Exists
<!-- Narrative motivation. Explains what the wave unlocks and what's blocked without it. -->

## Canonical Epics
- [ ] `<PREFIX>-02` [Epic] N.A <theme title>
- [ ] `<PREFIX>-03` [Epic] N.B <theme title>
- [ ] `<PREFIX>-04` [Epic] N.C <theme title>

## Sequential Dependencies
<!-- e.g. A must land before B; C can run in parallel with both. -->

## Definition of Done
- [ ] All theme epics closed
- [ ] Milestone released (tag + changelog)
- [ ] Tests / lint / format / typecheck all pass on `main`

## Relationships
- Blocks `<PREFIX-of-next-wave>-01`
- Related: <design-doc references>

---

# [Epic] N.A <theme title>
Draft ID: `<PREFIX>-02`

## Theme
<!-- One-sentence theme / outcome for this group. -->

## Parent Wave
`<PREFIX>-01`

## Motivation
<!-- Why this group exists; what it ships together. -->

## Canonical Child Issues
- [ ] `<PREFIX>-05` <feature title>
- [ ] `<PREFIX>-06` <feature title>

## Execution Notes
<!--
Describes ordering / sequencing between child issues. e.g. "PYX-05
should land first because most path rewrites depend on the final
package name. PYX-06 can start once the target module layout is
agreed."
Optional — delete if children are fully parallel.
-->

## Definition of Done
- [ ] All child issues closed
- [ ] Tests for the theme's surface land and pass

## Relationships
- Parent wave: `<PREFIX>-01`

---

# [Epic] N.B <theme title>
Draft ID: `<PREFIX>-03`

## Theme
## Parent Wave
`<PREFIX>-01`

## Motivation

## Canonical Child Issues
- [ ] `<PREFIX>-07` <title>

## Execution Notes

## Definition of Done
- [ ]

## Relationships
- Parent wave: `<PREFIX>-01`

---

# <scope>: <feature title>
Draft ID: `<PREFIX>-05`

## Problem / Request
<!-- Two sentences max. -->

## User Story
> As a <role>, I want <capability>, so that <outcome>.

## Proposed API
```python
# Signatures, types, docstring stubs. Lead with code, prose second.
```

## Design Snapshot
<!--
Cross-cutting context lives at the top of the wave file — don't
duplicate it here. Use this section for content specific to THIS
issue. Rename if warranted ("Demo To Implement", "Config Snippet",
etc.). Delete if not needed.
-->

## Mathematical Notes
<!--
Issue-specific equations. For algorithmic issues this section should be
treated as REQUIRED and should carry enough math that another agent can
implement the issue without reopening the design docs.

Recommended contents:
- defining equations
- parameterization / sign conventions
- approximation / factorization used
- identities or invariants tests should pin down

Use GitHub math:
- inline: `$...$`
- display: `$$...$$`

Delete only if the issue is truly non-algorithmic.
-->

## Implementation Notes
<!--
Prose bullets describing HOW to implement. Use this for issues where
the checklist-style "Implementation Steps" below is too granular
(e.g. when implementation shape depends on a design decision that
resolves during the work).
-->

- <decision / approach / consideration>
- <decision / approach / consideration>

## Implementation Steps
<!-- Concrete, file-level checklist. Each item should be checkable. -->
- [ ] Add `<symbol>` at `src/<package>/<module>.py`
- [ ] Export via `src/<package>/__init__.py` (if public)

## Definition of Done
- [ ] Code lands at the intended path
- [ ] Tests pass: `make test`
- [ ] Lint + typecheck pass: `make lint && make typecheck`

## Testing
- [ ] Unit test: `<what it asserts>`
- [ ] Property / regression test: `<what it asserts>` (if applicable)

## Documentation
- [ ] API reference page / section
- [ ] Notebook or recipe (if user-facing)

## Relationships
- Parent epic: `<PREFIX>-02`
- Blocked by: `<PREFIX>-NN>`
- Blocks: `<PREFIX>-NN>`
- Related: `<PREFIX>-NN>`

---

# <scope>: <feature title>
Draft ID: `<PREFIX>-06`

## Problem / Request

## User Story

## Proposed API

## Implementation Notes

## Definition of Done
- [ ]

## Testing

## Documentation

## Relationships
- Parent epic: `<PREFIX>-02`

---

# [Design] <design question>
Draft ID: `<PREFIX>-07`

## Problem / Question

## Proposed Options
```python
# Option A
```
```python
# Option B
```

## Alternatives Considered
- **Option A** — pros / cons
- **Option B** — pros / cons

## Decision
<!-- Filled in when resolved. -->

## Consequences

## Relationships
- Parent epic: `<PREFIX>-03`
- Blocks: <list of child issues that can't start until this resolves>

---

## Publishing Checklist

When the draft is ready to become real GitHub issues:

- [ ] For each `Draft ID`, open a GH issue using the matching
      `.github/ISSUE_TEMPLATE/*.md` template. Copy the body verbatim
      into the issue.
- [ ] After opening, record the GH issue number next to the draft ID
      in this file (`<PREFIX>-05` → `#42`) OR replace the draft IDs
      with GH numbers throughout.
- [ ] Update cross-references (`Parent epic`, `Blocked by`, `Blocks`,
      `Related`) from draft IDs to GH numbers so relationships survive
      in the GH UI.
- [ ] **Apply native relationships.** The prose `Parent:` / `Blocked
      by:` / `Blocks:` lines are the human-readable record; they also
      need the native GitHub sub-issue and dependency links so the UI
      and automation pick up the hierarchy.
      - Sub-issues (wave → themes → features):
        `make gh-sub PARENT=<parent#> CHILDREN="<child1#> <child2#> ..."`
      - Blocked-by:
        `make gh-block ISSUE=<this#> BLOCKED_BY=<other#>`
      - For bulk application, see `.github/scripts/link-issues.sh`.
- [ ] Either delete this file or keep it as historical record in
      `.plans/archive/`.
