---
applyTo: "**"
---

# Code Review Instructions

When performing code review, use `/CODE_REVIEW.md` as the source of truth for:

- Review checklist (style, idioms, packaging, docs, error handling, testing, performance, security)
- Python-specific checks (type hints, modern syntax, dataclasses, path handling, exceptions)
- Output format and priority levels
- Suggestion type emojis and review tone

Key principles:
- Sacrifice *cleverness* for *clarity*. Sacrifice *brevity* for *explicitness*.
- Don't worry about formatting — CI (ruff format, pre-commit) handles that automatically.
- Be **constructive** and **specific**. Acknowledge good patterns with 👍.
- Every suggestion must include a concrete alternative, not just criticism.
