# Contributing

This template uses a lightweight GitHub project scaffold so issues, PRs, and release waves stay consistent across research, implementation, and docs work.

## Development Setup

```bash
make install
```

Alternative full environment with pixi:

```bash
pixi install
```

## Quality Gates

Before opening a PR, run the full local checks from the repo root:

```bash
uv run pytest -v
uv run --group lint ruff check .
uv run --group lint ruff format --check .
uv run --group typecheck ty check src/research_notebook
```

Equivalent Make targets:

```bash
make test-cov
make lint
make typecheck
```

## Label Taxonomy

This template uses a small label taxonomy for issue planning and PR routing.

- `type:*` identifies the work item kind: `feature`, `bug`, `design`, `research`, `docs`, `chore`, `epic-wave`, `epic-theme`
- `area:*` identifies the repo surface: `engineering`, `testing`, `docs`, `data`, `models`, `experiments`, `code`
- `wave:*` identifies the release wave: `wave:0`, `wave:1`, ...
- `priority:*` identifies urgency: `p0`, `p1`, `p2`

Bootstrap the standard label set with:

```bash
make gh-labels
```

## Epic Model

The template supports a two-layer epic model:

1. Wave epic: a release-scoped container
2. Theme epic: a parallel-safe slice within that wave
3. Concrete issues: feature, bug, design, research, docs, or chores

Use consistent issue bodies to keep the structure predictable.

## Relationships

Record relationships in the issue body and also apply the matching native GitHub relationship:

- `Parent:` use a sub-issue link
- `Blocked by:` use a typed dependency
- `Blocks:` apply the inverse typed dependency on the other issue
- `Related:` prose only

Helper commands:

```bash
make gh-sub PARENT=<parent#> CHILDREN="<child1#> <child2#>"
make gh-block ISSUE=<issue#> BLOCKED_BY=<other#>
make gh-show ISSUE=<issue#>
```

The underlying helper is [.github/scripts/link-issues.sh](.github/scripts/link-issues.sh).

## Docs And Examples

- Documentation source lives in [docs](docs/)
- Notebooks live in [notebooks](notebooks/) (Jupytext `.py`) and [marimo_notebooks](marimo_notebooks/)
- Docs build locally with `make docs-serve` (or `pixi run -e docs docs-serve`)
- If a notebook produces figures for docs, save committed artifacts under `docs/images/<notebook_name>/`

## Commit And PR Conventions

- Use Conventional Commits for commit messages and PR titles
- Keep diffs surgical; avoid adjacent refactors
- Add or update tests for behavior changes
- Prefer explicit paths and checklists in issue bodies so work is easy to hand off
