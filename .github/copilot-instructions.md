# Copilot Instructions

This is a scientific research project template. Follow these conventions:

## Environment

- Use `pixi run <task>` for all commands (never `pip install` or `conda install` directly)
- Python >= 3.12, type annotations required
- Use `from __future__ import annotations` for forward references

## Code Style

- Ruff linting + formatting (line-length 88)
- Select: E, F, I, UP, B, SIM, RUF

## Project Conventions

- Data files managed by DVC (never commit raw data to git)
- Configs managed by Hydra in `configs/` directory
- Results saved to `results/` (DVC-managed)
- Notebooks stored as Jupytext percent-format `.py` files in `notebooks/` (no `.ipynb` committed)

## Testing

```bash
pixi run test        # Run tests
pixi run lint        # Lint
pixi run typecheck   # Type check
```

## Key Directories

| Path | Purpose |
|------|---------|
| `src/research_notebook/` | Main package source code |
| `tests/` | Test suite |
| `docs/` | Documentation (MyST) |
| `notebooks/` | Jupytext percent-format .py notebooks |
| `marimo_notebooks/` | Marimo reactive notebooks |
| `configs/` | Hydra configuration hierarchy |
| `scripts/` | Entry point scripts |
| `data/` | Data directories (DVC-managed) |
| `results/` | Experiment results (DVC-managed) |

## Behavioral Guidelines

### Do Not Nitpick
- Ignore style issues that linters/formatters catch (formatting, import order, quote style)
- Don't suggest changes to code you weren't asked to modify
- Match existing patterns even if you'd do it differently

### Always Propose Tests
When implementing features or fixing bugs:
1. Write a test that verifies the expected behavior
2. Implement the change
3. Verify the test passes

### Never Suggest Without a Proposal
Bad: "You should add validation here"
Good: "Add validation here. Proposed implementation:"
```python
if value < 0:
    raise ValueError('Value must be non-negative')
```

### Simplicity First
- No abstractions for single-use code
- No speculative features beyond what was asked
- If 200 lines could be 50, propose the simpler version

### Surgical Changes
- Only modify lines directly related to the request
- Don't refactor adjacent code
- Don't add docstrings/comments to code you didn't change
- Remove only imports/functions that YOUR changes made unused

## Code Review

For all code review tasks, follow the guidance in `/CODE_REVIEW.md`.
