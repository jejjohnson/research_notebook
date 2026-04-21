---
name: Feature / Enhancement
about: A single deliverable that rolls up to a theme epic.
title: "<scope>: <short description>"
labels: ["type:feature"]
---

## Problem / Request
<!-- What's needed? One or two sentences. -->

## User Story
> As a <role>, I want <capability>, so that <outcome>.

## Motivation
<!-- Why now; what it enables; what breaks if we don't have it. -->

## Proposed API
```python
# Signatures, types, docstring stubs.
```

<!--
OPTIONAL — Inline context ("Design Snapshot")
Copy here any API signatures, code snippets, configuration examples, or
excerpts from external / private design docs that the implementer needs to
work on this issue in isolation. Goal: another contributor (human or AI
agent) can implement this issue without opening other repos or chats.

RENAME the heading to fit the issue type. Examples:
  ## Design Snapshot            (typical library feature)
  ## Demo To Implement          (walkthrough / example / notebook issue)
  ## Demo Snippet To Include    (docs / API-reference issue)
  ## Config Snippet             (CI / infra issue)
  ## Reference Trace            (bug reproduction)

Lead with the exact snippet the implementer will reproduce — prose last.
Delete this section if not relevant.
-->

## Design Snapshot
<!-- Delete or rename. -->
```python
# Lead with the exact code/config the implementer will reproduce.
```

<!--
REQUIRED FOR ALGORITHMIC / NUMERICAL ISSUES — Inline math / numerical
context ("Mathematical Notes")

If the issue implements an algorithm, numerical method, probabilistic
update, optimizer, filter, solver, linearization, approximation, or any
other mathematically-defined behavior, DO NOT delete this section.

Treat this section as part of the spec, not optional commentary. Include
as much math as another implementer needs to succeed without opening the
private design docs or re-deriving the method from scratch.

Minimum content for algorithmic issues:
  - defining equations
  - parameterization / sign conventions
  - approximation being used
  - identities or invariants the tests should pin down
  - numerical-stability notes or edge cases

Examples of the level of detail we want:
  - "K_post⁻¹ = K_prior⁻¹ + Λ_sites"
  - "η₂ = -½Λ"
  - "f_post = f_prior + K_fy (K_yy + R)⁻¹ (y - y_prior)"
  - "A ≈ Q_k Λ_k Q_kᵀ,  R = Q_k Λ_k^{1/2}"

STYLE — use normal GitHub math syntax:
  - inline math: `$...$`
  - display math: `$$...$$`

Prefer real math notation over vague prose whenever the algorithm is
defined mathematically. If plain-text readability matters, unicode math
in prose (σ², E₁, ∑, ⊗, ≈, Λ⁻¹, O(d³)) is also encouraged.

Rename the heading if the content type warrants (e.g. "Numerical Notes",
"Stability Notes", "Equations To Test"). Delete only when the issue is
truly non-algorithmic.
-->

## Mathematical Notes
<!-- Delete or rename. -->
Use this section as the implementation spec for algorithms.

Suggested prompts:
- Defining equations:
  $$ ... $$
- Parameterization / sign conventions:
  $$ ... $$
- Approximation / factorization used:
  $$ ... $$
- Identities or invariants tests should assert:
  $$ ... $$

## References & Existing Code
- Design doc / spec: `<path or URL>`
- Reference impl: `<path:line>`
- Related prior art: `<repo / paper / issue>`

## Implementation Steps
<!-- Concrete, file-level steps. Each should be checkable. -->
- [ ] Add `<symbol>` in `src/<package>/<module>.py`
- [ ] Wire <symbol> into `src/<package>/__init__.py` (if public)
- [ ] ...

## Definition of Done
- [ ] Code lands at the intended path
- [ ] Public API exported via `src/<package>/__init__.py` (if user-facing)
- [ ] Tests pass: `make test`
- [ ] Lint + typecheck pass: `make lint && make typecheck`
- [ ] Docstrings (Google-style) on all public symbols

## Testing
<!-- One checkbox per test, so progress is trackable. -->
- [ ] Unit test: `<what it asserts>`
- [ ] Property / round-trip test: `<what it asserts>` (if applicable)
- [ ] Regression test: `<what it asserts>` (if applicable)

## Documentation
- [ ] API reference page / section
- [ ] Notebook or recipe (if user-facing flow)
- [ ] Docstrings (covered by Definition of Done)

## Relationships
<!--
Each prose line below has a native GitHub feature you should also apply
after the issue is opened. Keep the prose as a human-readable record AND
apply the native link so GitHub's UI (sub-issue panel, dependency graph,
"unblocks on close") works.

  Parent:      → Sub-issue link. UI: Issue side-panel → Sub-issues →
                 "Create sub-issue" on the parent, or "Convert to
                 sub-issue" on the child.
                 CLI: make gh-sub PARENT=<parent#> CHILDREN="<this#>"

  Blocked by:  → Typed dependency. UI: side-panel → Development →
                 "Mark as blocked by".
                 CLI: make gh-block ISSUE=<this#> BLOCKED_BY=<other#>

  Blocks:      → Inverse of Blocked by; apply on the OTHER issue.
                 CLI: make gh-block ISSUE=<other#> BLOCKED_BY=<this#>

  Related:     → No native feature; mention only.

For bulk / scripted linking see `.github/scripts/link-issues.sh`.
-->
- Parent (theme epic): #
- Blocked by: #
- Blocks: #
- Related: #
