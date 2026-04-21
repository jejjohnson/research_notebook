---
name: Design / ADR
about: Resolve an open design question for a new API or architectural decision.
title: "[Design] <question>"
labels: ["type:design"]
---

## Problem / Question
<!-- The design question being resolved. -->

## User Story
> As a <role>, I want <capability>, so that <outcome>.

## Motivation
<!-- Why this needs a decision now. Upstream / downstream consumers. -->

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
- **Option C (rejected)** — pros / cons

<!--
OPTIONAL — Inline context ("Design Snapshot")
Relevant excerpts from external / private design docs, prior art, or
existing implementations that inform the decision. Lead with the exact
signature / snippet; prose last.

Rename the heading to fit the content type: "Prior Art Snippets",
"Reference Implementations", "Existing Code Excerpt". Delete if N/A.
-->

## Design Snapshot
<!-- Delete or rename. -->
```python
# Lead with the most relevant prior-art excerpt or API sketch.
```

<!--
OPTIONAL — Inline math / numerical context ("Mathematical Notes")
Equations, sign conventions, numerical considerations that scope the
decision.

STYLE — prefer unicode math in prose (σ², Λ⁻¹, ∑, O(d³)). Use `text`
code fences for multi-line equation blocks so pseudo-math isn't mangled
by syntax highlighting:

    ```text
    ELBO(λ) = 𝔼_q[ℓ(θ)] − KL(q ‖ p)
    ∇_λ ELBO = ∇_μ 𝔼[ℓ] − F(λ − λ₀)
    ```

Rename if the content warrants ("Numerical Considerations",
"Stability Notes"). Delete if N/A.
-->

## Mathematical Notes
<!-- Delete or rename. -->
```text
<equations, conventions, edge cases>
```

## Decision
<!-- Filled in when the issue is resolved. Short + explicit. -->

## Consequences
<!-- Ripple effects, downstream changes, back-compat implications. -->

## References & Existing Code
- Design doc / ADR log: `<path or URL>`
- Related prior art / issue / PR: #

## Implementation Steps (post-decision)
- [ ] Land reference implementation at `<path>`
- [ ] Update ADR log / decisions doc
- [ ] Open follow-up feature issues if the surface changes

## Definition of Done
- [ ] Decision + Consequences written into the issue body
- [ ] ADR entry added to `docs/decisions.md` (or equivalent)
- [ ] Follow-up feature issues opened and linked

## Testing
- [ ] Test that encodes the decision (if reference impl lands in the same PR)

## Documentation
- [ ] ADR page
- [ ] Docstring notes referencing the decision

## Relationships
<!--
Each prose line has a native GitHub feature — apply it after the issue
is opened so GitHub's UI and automation pick up the hierarchy.

  Parent:      → Sub-issue link.  make gh-sub PARENT=<parent#> CHILDREN="<this#>"
  Blocked by:  → Typed dependency.  make gh-block ISSUE=<this#> BLOCKED_BY=<other#>
  Blocks:      → Inverse — apply on the OTHER issue.
                 make gh-block ISSUE=<other#> BLOCKED_BY=<this#>
  Related:     → Prose only; no native feature.

Helper: `.github/scripts/link-issues.sh`.
-->
- Parent (theme epic): #
- Blocked by: #
- Blocks: # (issues that can't start until this decision resolves)
- Related: #
