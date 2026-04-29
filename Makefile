# =============================================================================
# Research project Makefile
# =============================================================================
#
# PREREQUISITES:
#   - uv installed  (https://github.com/astral-sh/uv) — primary task runner
#   - git available in PATH
#   - Copy .env.example to .env for local overrides (optional)
#
#   Pixi remains the alternative environment manager. See pixi.toml for the
#   equivalent tasks (test, lint, format, typecheck, precommit, docs-serve).
#
# QUICK START:
#   make help          # Show all available commands
#   make install       # Install all dependency groups
#   make test          # Run tests
#   make lint          # Lint with ruff
#   make format        # Format with ruff
#
# =============================================================================

# ---------------------------------------------------------------------------
# .env support
# ---------------------------------------------------------------------------
-include .env
ifneq (,$(wildcard .env))
ENV_VARS := $(shell grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env | cut -d= -f1 | xargs)
ifneq ($(strip $(ENV_VARS)),)
export $(ENV_VARS)
endif
endif

# ---------------------------------------------------------------------------
# Calculated variables
# ---------------------------------------------------------------------------
GIT_HASH := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
PKG_VERSION := $(shell grep -E '^version\s*=' pyproject.toml 2>/dev/null \
	| sed -E 's/.*"([^"]+)".*/\1/' || echo "unknown")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PKGROOT ?= src/myproject

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
BLUE   := \033[36m
YELLOW := \033[33m
GREEN  := \033[32m
RED    := \033[31m
RESET  := \033[0m

# ---------------------------------------------------------------------------
# Phony declarations
# ---------------------------------------------------------------------------
.PHONY: help install lint format typecheck test test-cov \
	precommit build clean version docs docs-serve docs-deploy \
	init gh-labels gh-sub gh-block gh-show

.DEFAULT_GOAL := help

# ===========================================================================
##@ Meta
# ===========================================================================

help: ## Show this help menu
	@printf "$(YELLOW)research-project$(RESET)\n"
	@printf "%s\n" "-----------------------------------------------------------"
	@awk 'BEGIN {FS = ":.*##"; printf ""} \
	     /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-18s$(RESET) %s\n", $$1, $$2 } \
	     /^##@/ { printf "\n$(YELLOW)%s$(RESET)\n", substr($$0, 5) } ' \
	     $(MAKEFILE_LIST)

version: ## Display package version and git hash
	@printf "$(YELLOW)Version Info$(RESET)\n"
	@printf "%s\n" "-----------------------------------------------------------"
	@printf "$(GREEN)Package : $(PKG_VERSION)$(RESET)\n"
	@printf "$(BLUE)Git hash: $(GIT_HASH)$(RESET)\n"

# ===========================================================================
##@ Setup
# ===========================================================================

install: ## Install all dependency groups via uv + pre-commit hooks
	@printf "$(YELLOW)>>> Installing all dependencies...$(RESET)\n"
	uv sync --all-groups
	uv run pre-commit install
	@printf "$(GREEN)>>> Installation complete!$(RESET)\n"

init: ## Bootstrap .env from .env.example if needed
	@if [ -f .env ]; then \
		printf "$(YELLOW)>>> .env already exists — skipping.$(RESET)\n"; \
	else \
		cp .env.example .env; \
		printf "$(GREEN)>>> .env created from .env.example$(RESET)\n"; \
	fi

# ===========================================================================
##@ Quality
# ===========================================================================

lint: ## Lint code with ruff (no auto-fix)
	@printf "$(YELLOW)>>> Running ruff check...$(RESET)\n"
	uv run --group lint ruff check .
	@printf "$(GREEN)>>> Lint passed!$(RESET)\n"

format: ## Format code with ruff (format + auto-fix)
	@printf "$(YELLOW)>>> Running ruff format + fix...$(RESET)\n"
	uv run --group lint ruff format .
	uv run --group lint ruff check --fix .
	@printf "$(GREEN)>>> Format complete!$(RESET)\n"

typecheck: ## Type-check with ty
	@printf "$(YELLOW)>>> Running type checks...$(RESET)\n"
	uv run --group typecheck ty check $(PKGROOT)
	@printf "$(GREEN)>>> Type check passed!$(RESET)\n"

# ===========================================================================
##@ Testing
# ===========================================================================

test: ## Run tests with pytest (no coverage)
	@printf "$(YELLOW)>>> Running tests (no coverage)...$(RESET)\n"
	uv run pytest -v -o addopts=
	@printf "$(GREEN)>>> Tests passed!$(RESET)\n"

test-cov: ## Run tests with coverage report
	@printf "$(YELLOW)>>> Running tests with coverage...$(RESET)\n"
	uv run pytest -v
	@printf "$(GREEN)>>> Coverage report generated!$(RESET)\n"

# ===========================================================================
##@ Pre-commit
# ===========================================================================

precommit: ## Run pre-commit hooks on all files
	@printf "$(YELLOW)>>> Running pre-commit...$(RESET)\n"
	uv run pre-commit run --all-files
	@printf "$(GREEN)>>> Pre-commit passed!$(RESET)\n"

# ===========================================================================
##@ Build
# ===========================================================================

build: ## Build Python wheel and sdist
	@printf "$(YELLOW)>>> Building package...$(RESET)\n"
	uv build
	@printf "$(GREEN)>>> Build complete — see dist/$(RESET)\n"

clean: ## Remove build artefacts and cache directories
	@printf "$(YELLOW)>>> Cleaning up...$(RESET)\n"
	rm -rf dist/ build/ .eggs/ *.egg-info
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/
	rm -f .coverage coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@printf "$(GREEN)>>> Clean complete!$(RESET)\n"

# ===========================================================================
##@ Docs
# ===========================================================================

docs: ## Build documentation with MyST
	uv run --group docs myst build --html

docs-serve: ## Serve documentation locally (app on 3000, assets on 3100)
	uv run --group docs myst start --port 3000 --server-port 3100

docs-deploy: ## Deploy documentation to GitHub Pages
	uv run --group docs myst build --html
	uv run --group docs ghp-import -n -p _build/html

# ===========================================================================
##@ GitHub helpers
# ===========================================================================

gh-labels: ## Bootstrap the GitHub label taxonomy
	bash .github/scripts/create-labels.sh

gh-sub: ## Link CHILDREN as sub-issues of PARENT
	@test -n "$(PARENT)"   || { echo "error: PARENT=<issue-number> required" >&2; exit 1; }
	@test -n "$(CHILDREN)" || { echo "error: CHILDREN=\"<a> <b> ...\" required" >&2; exit 1; }
	bash .github/scripts/link-issues.sh sub '$(PARENT)' $(foreach c,$(CHILDREN),'$(c)')

gh-block: ## Mark ISSUE as blocked by BLOCKED_BY
	@test -n "$(ISSUE)"      || { echo "error: ISSUE=<issue-number> required" >&2; exit 1; }
	@test -n "$(BLOCKED_BY)" || { echo "error: BLOCKED_BY=<issue-number> required" >&2; exit 1; }
	bash .github/scripts/link-issues.sh block '$(ISSUE)' '$(BLOCKED_BY)'

gh-show: ## Show parent / sub-issues / blocking state for ISSUE
	@test -n "$(ISSUE)" || { echo "error: ISSUE=<issue-number> required" >&2; exit 1; }
	bash .github/scripts/link-issues.sh show '$(ISSUE)'
