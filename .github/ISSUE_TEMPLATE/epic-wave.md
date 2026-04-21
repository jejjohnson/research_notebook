---
name: Epic — Wave (L1)
about: A release-scoped mega-epic grouping theme epics under one milestone. See CONTRIBUTING.md for the two-layer epic model.
title: "[EPIC] Wave N: <title>"
labels: ["type:epic-wave"]
---

<!--
A Wave epic is the L1 container for a release / milestone. It groups several
Theme epics (L2) that can run in parallel. Keep this body short — the
details live in the Theme epics and their child issues.
-->

## Goal
<!-- One sentence: what outcome does this wave deliver? Maps to a milestone / release. -->

## Wave / Milestone
- Wave: `wave:N` (matches the bootstrap label set; swap to `wave:N-<slug>` if you've added descriptive labels)
- Milestone: `vX.Y-<slug>`

## Motivation
<!--
Why this wave now; what it unlocks; what it blocks.

Rename to "Why This Wave Exists" if a narrative framing fits better
than pure rationale — useful for waves that exist to undo a historical
state (e.g. "the repo is still a template, every later wave depends
on a real package identity").
-->

## Theme Epics (parallel-safe)
<!--
One section per theme. Sections can run in parallel unless noted.

Rename to "Canonical Epics" if you prefer that wording — matches the
wave-backlog drafting convention in docs/templates/wave-backlog.md.

After the wave epic is opened, each theme epic listed below should also
be linked as a SUB-ISSUE of this wave. One-shot:
    make gh-sub PARENT=<this#> CHILDREN="<theme1#> <theme2#> <theme3#>"
-->


### Section A — <theme>
- [ ] #<theme-epic-issue>

### Section B — <theme>
- [ ] #<theme-epic-issue>

## Sequential Dependencies
<!-- e.g. Section A → Section B; Section C can start anytime. -->

## Definition of Done (Wave)
- [ ] All theme epics closed
- [ ] Milestone released (tag + changelog)
- [ ] Tests, lint, format, typecheck all pass on `main`
- [ ] Docs published

## Relationships
<!--
After opening, apply native links:
  Sub-issues:  each Theme epic above becomes a sub-issue of this wave
               (make gh-sub PARENT=<this#> CHILDREN="<theme#> <theme#>")
  Blocked by:  make gh-block ISSUE=<this#> BLOCKED_BY=<prior-wave#>
  Blocks:      make gh-block ISSUE=<next-wave#> BLOCKED_BY=<this#>
Helper: `.github/scripts/link-issues.sh`.
-->
- Blocked by: #
- Blocks: #
- Related: #
