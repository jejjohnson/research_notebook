#!/usr/bin/env bash

set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found on PATH. Install from https://cli.github.com/" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "error: gh is not authenticated. Run 'gh auth login' first." >&2
  exit 1
fi

create() {
  local name="$1"
  local color="$2"
  local desc="$3"
  gh label create "$name" --color "$color" --description "$desc" --force >/dev/null
  printf '  ✓ %s\n' "$name"
}

echo "Creating type:* labels..."
create "type:epic-wave"  "5319e7" "Release-scoped mega-epic (L1)"
create "type:epic-theme" "8b5cf6" "Parallel-safe theme under a wave (L2)"
create "type:feature"    "a2eeef" "New feature or enhancement"
create "type:bug"        "d73a4a" "Something is not working"
create "type:design"     "fbca04" "Design / ADR issue"
create "type:chore"      "c5def5" "Engineering maintenance"
create "type:docs"       "0075ca" "Documentation work"
create "type:research"   "bfe5bf" "Comparative analysis / prior-art survey"

echo "Creating area:* labels..."
create "area:engineering" "0e8a16" "Build, packaging, CI, tooling, DVC, repo automation"
create "area:testing"     "bfdadc" "Test suite and verification"
create "area:docs"        "1d76db" "MyST docs, notebooks, and examples"
create "area:data"        "f9d0c4" "Data loading, preprocessing, and pipelines"
create "area:models"      "d4c5f9" "Model implementations and training"
create "area:experiments" "fbca04" "Experiment configs, runs, and reporting"
create "area:code"        "ededed" "General source-code changes"

echo "Creating misc labels..."
create "dependencies" "ededed" "Pull requests that update dependency files"

echo "Creating wave:* labels..."
create "wave:0" "c2e0c6" "Wave 0"
create "wave:1" "bfd4f2" "Wave 1"
create "wave:2" "d4c5f9" "Wave 2"
create "wave:3" "f9d0c4" "Wave 3"
create "wave:4" "fef2c0" "Wave 4"

echo "Creating priority:* labels..."
create "priority:p0" "b60205" "Blocker for current wave"
create "priority:p1" "d93f0b" "High priority"
create "priority:p2" "fbca04" "Normal priority"

echo
echo "Done. View labels with: gh label list"
