---
name: Bug report
about: Something isn't working correctly.
title: "bug: <short description>"
labels: ["type:bug"]
---

## Problem
<!-- What's broken? One or two sentences. -->

## Reproduction
```python
# Minimal reproducing example.
```

## Expected Behavior
<!-- What should happen. -->

## Actual Behavior
<!-- What happens instead. Include traceback if relevant. -->

## Environment
- Package version:
- Python:
- Platform:
- Key dependency versions:

## References & Existing Code
- Related code: `<path>`
- Related issue / PR: #

## Implementation Steps (fix)
- [ ] Reproduce locally
- [ ] Root-cause analysis
- [ ] Fix at `<path>`
- [ ] Add regression test

## Definition of Done
- [ ] Regression test captures the bug
- [ ] Fix lands; regression test green
- [ ] `make test && make lint && make typecheck` green

## Testing
- [ ] Regression test at `tests/<path>::<name>` — asserts <what>

## Documentation
<!-- If user-facing behaviour changed, update relevant pages. -->
- [ ] N/A, or: update `<path>`

## Relationships
<!--
Apply the native GitHub links after the issue is opened:
  Parent:      make gh-sub PARENT=<parent#> CHILDREN="<this#>"
  Blocked by:  make gh-block ISSUE=<this#> BLOCKED_BY=<other#>
  Blocks:      make gh-block ISSUE=<other#> BLOCKED_BY=<this#>
Helper: `.github/scripts/link-issues.sh`.
-->
- Parent (theme epic, if any): #
- Blocked by: #
- Blocks: #
