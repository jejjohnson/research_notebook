# Agent Guidelines

This document provides instructions for AI coding agents working on this
research project.

## Environment Setup

This project uses [Pixi](https://pixi.sh) for environment management.
All commands should be run using `pixi run <task>`.

```bash
# Install all environments
pixi install

# Install specific environment
pixi install -e jupyterlab
```

## Project Structure

```
research_template/
├── configs/          # Hydra configuration hierarchy
├── data/             # Data directories (managed by DVC)
├── docs/             # MyST documentation
├── marimo_notebooks/ # Marimo reactive notebooks
├── notebooks/        # Jupytext percent-format .py scripts (no .ipynb in repo)
├── results/          # Experiment results (managed by DVC)
├── scripts/          # Entry point scripts
├── src/research_notebook/    # Source package
└── tests/            # Test suite
```

## Development Commands

```bash
# Run tests
pixi run test

# Lint code
pixi run lint

# Format code
pixi run format

# Type check
pixi run typecheck

# Run all pre-commit hooks
pixi run precommit
```

## Running Experiments

```bash
# Preprocess data
pixi run preprocess

# Train with default config
pixi run train

# Train with hydra-zen
pixi run train-zen

# Train with overrides
pixi run train training.lr=0.001 model=transformer

# Evaluate
pixi run evaluate
```

## DVC Conventions

- **Never commit raw data** to git. Use `dvc add` and `dvc push`.
- Run `dvc repro` to reproduce the full pipeline.
- Check pipeline status with `dvc status`.
- View pipeline DAG with `dvc dag`.
- Commit `.dvc` pointer files and `dvc.lock` to git.

```bash
# Add a data file to DVC tracking
dvc add data/raw/dataset.csv

# Push data to remote storage
dvc push

# Pull data from remote storage
dvc pull

# Reproduce the pipeline
dvc repro

# Check what has changed
dvc status
```

## Hydra / hydra-zen Conventions

- Config files live in `configs/`.
- Use `configs/train.yaml` as the main entry point config.
- Override values on the command line: `python scripts/train.py training.lr=0.01`.
- Use multirun for sweeps: `python scripts/train.py -m training.lr=0.001,0.01,0.1`.
- hydra-zen configs are defined in `scripts/train_zen.py` using `builds()` and `make_config()`.

## Documentation

```bash
# Serve docs locally
pixi run -e docs docs-serve

# Build static HTML
pixi run -e docs docs-build
```

## Notebook Environments

```bash
# JupyterLab
pixi run -e jupyterlab lab

# Marimo
pixi run -e marimo marimo-edit
```

## Code Style

- Python >= 3.12
- Ruff for linting and formatting (line-length 88)
- Type annotations required for all public functions
- Use `from __future__ import annotations` for forward references

---

## Karpathy Coding Principles

Four behavioral principles to reduce the most common LLM coding mistakes. These bias toward caution over speed — for trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State assumptions explicitly. If uncertain, ask before writing code.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

Test: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

Test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform imperative tasks into declarative goals with verification:

- "Add validation" → Write tests for invalid inputs, then make them pass
- "Fix the bug" → Write a test that reproduces it, then make it pass
- "Optimize X" → Write the naive correct version first, then optimize while preserving correctness

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Before Every Commit

**All agents must** verify that every one of the following passes before creating a commit or reporting progress. No exceptions.

1. **Tests** – `pixi run test` must have 0 failures.
2. **Lint** – `pixi run lint` must report no issues.
3. **Format** – `pixi run format` must leave the working tree clean (no diff) after formatting.
4. **Type checks** – `pixi run typecheck` must report no errors in changed files.

## Pull Request Descriptions

**Never replace or remove an existing PR title or description.** When reporting progress on a PR that already has a title and description, only append new checklist items or update the status of existing ones. The original content must be preserved in full.

This is a common failure mode: an agent called to make a small follow-up change will supply a fresh description scoped only to its own work, silently discarding all prior context. Always read the existing description first and treat it as the base.

## GIT Safety Rules

- **NEVER** push to `main`/`master` or merge into `main`/`master` unless the user explicitly says "push to main" or "merge to main".
- **NEVER** push to any remote branch or run `git push` unless the user explicitly asks you to push. Only commit locally.
- Always work on feature branches.
- When the user says "merge the changes" or "merge the branch", they mean push the local branch to the remote — NOT merge into main.
- Always confirm before any action that affects shared branches (main, master, production, etc.).

## Commit Messages

All commit messages **must** follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Valid types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

Examples:
- `feat: add new data loading utility`
- `fix: correct off-by-one error in slice computation`
- `docs: update installation instructions`
- `chore: bump ruff to 0.9.7`
