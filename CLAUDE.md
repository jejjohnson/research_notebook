# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A scientific research project template combining Hydra configs, DVC pipelines, MyST documentation, and notebooks. The template supports two environment managers: [uv](https://github.com/astral-sh/uv) (via `Makefile`) and [pixi](https://pixi.sh) (via `pixi.toml`).

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Run tests: uv run pytest -v -o addopts=
make test-cov             # Tests with coverage
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check src/research_notebook
make precommit            # Run pre-commit on all files
make docs-serve           # Local MyST docs server
```

### Running a single test

```bash
uv run pytest tests/test_example.py::test_case -v
```

### Alternative pixi tasks

```bash
pixi run test
pixi run lint
pixi run typecheck
pixi run -e docs docs-serve
```

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -v
uv run --group lint ruff check .
uv run --group lint ruff format --check .
uv run --group typecheck ty check src/research_notebook
```

**Critical**: Always lint the entire repo with `.` from the root. The template includes tests, configs, scripts, and docs glue outside the package directory.

## Architecture

### Package structure

The installable package lives in [src/research_notebook](src/research_notebook/).

### Key directories

| Path | Purpose |
|------|---------|
| `src/research_notebook/` | Installable library and public exports |
| `src/research_notebook/data/` | Data loading utilities |
| `src/research_notebook/models/` | Model implementations |
| `src/research_notebook/trainers/` | Training loops |
| `src/research_notebook/utils/` | Utility functions |
| `configs/` | Hydra configuration hierarchy |
| `data/` | DVC-managed data directories |
| `results/` | DVC-managed experiment results |
| `scripts/` | Entry-point scripts (train, evaluate, preprocess) |
| `docs/` | MyST documentation source |
| `notebooks/` | Jupytext percent-format `.py` notebooks |
| `marimo_notebooks/` | Marimo reactive notebooks |
| `tests/` | Test suite |

## Documentation Examples

Docs pages live in [docs](docs/). Notebooks may live in [notebooks](notebooks/) as jupytext percent-format `.py` files. When notebooks produce figures for docs pages:

1. Run them locally
2. Save figures under `docs/images/{notebook_name}/`
3. Reference those assets from the relevant MyST page in `docs/`
4. Commit the notebook source and generated assets together

## Coding Conventions

- `from __future__ import annotations` at the top of Python modules
- Type hints on public functions and methods
- Use `pathlib.Path` for filesystem work
- Keep scientific computations pure; isolate IO and CLI side effects
- Match existing numerical style and avoid refactoring unrelated code

## Plans

Plans and scratch implementation docs go in `.plans/` and should not be committed.

## PR Review Comments

When addressing PR review comments, resolve each review thread after fixing it via the GitHub GraphQL API. Use the workflow documented in [AGENTS.md](AGENTS.md).

## Code Review

Follow the guidance in [CODE_REVIEW.md](CODE_REVIEW.md) for all code review tasks.
