---
name: Epic — Theme (L2)
about: A parallel-safe group of issues under a Wave epic. See CONTRIBUTING.md for the two-layer epic model.
title: "[EPIC] <theme title>"
labels: ["type:epic-theme"]
---

<!--
A Theme epic is the L2 container under a Wave. It groups several concrete
issues (features / designs / chores) that ship together as a coherent slice.
-->

## Theme
<!-- One-sentence theme / outcome for this group. -->

## Parent Wave
- Wave epic: #
- Wave label: `wave:N` (matches the bootstrap label set; swap to `wave:N-<slug>` if you've added descriptive labels)
- Milestone: `vX.Y-<slug>`

## Motivation
<!-- Why this group exists; what it ships together. -->

## Issues
<!--
Rename to "Canonical Child Issues" if that reads more naturally.

After opening this epic and its children, link each child issue below
as a SUB-ISSUE of this epic. One-shot:
    make gh-sub PARENT=<this#> CHILDREN="<child1#> <child2#> <child3#>"
-->
- [ ] #<issue> — <short description>
- [ ] #<issue> — <short description>

<!--
OPTIONAL — Execution Notes
Describes ordering / sequencing between child issues inside this theme.
Useful when some children block others. Examples:

  "#42 lands first because most path rewrites depend on the final
   module layout. #43 can start once #42's public API is stable."

  "#50 and #51 are fully parallel; #52 depends on both."

Delete this section if the children are fully parallel and no ordering
decisions need to be recorded.
-->

## Execution Notes
<!-- Delete if children are fully parallel. -->

<!--
If this theme is algorithmic / numerical, use this section to record the
shared mathematical conventions that every child issue must preserve.
Examples: parameterization choice, sign conventions, factorization form,
or which equations the child issues are expected to implement.
-->

## Parallelism
- Can run in parallel with: # (other theme epics in the same wave)
- Blocked by (inside this wave): #
- Must complete before: #

## Definition of Done
- [ ] All child issues closed
- [ ] Tests for the theme's surface land and pass
- [ ] Docs / API pages for the theme's surface land

## Relationships
<!--
After opening, apply native links:
  Parent (wave):  make gh-sub PARENT=<wave#> CHILDREN="<this#>"
  Sub-issues:     make gh-sub PARENT=<this#> CHILDREN="<child1#> <child2#>"
  Blocked by:     make gh-block ISSUE=<this#> BLOCKED_BY=<other-theme#>
Helper: `.github/scripts/link-issues.sh`.
-->
- Parent: #<wave-epic>
- Related: #
