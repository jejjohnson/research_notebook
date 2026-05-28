#!/usr/bin/env bash
# Source the repo-root .env so shell-level tools (earthengine, gcloud,
# aws, etc.) and pixi tasks see the geospatial-service credentials
# documented in .env.example. Sourced by pixi on activation of the
# satellite-viewer environment.
#
# Safe to source from any cwd: pixi sets PIXI_PROJECT_ROOT to the
# directory containing pixi.toml. If .env doesn't exist, this is a
# no-op — the Python side (satellite_viewer.credentials) still works
# via env vars set elsewhere or the per-service native files.

set -a
if [ -n "${PIXI_PROJECT_ROOT:-}" ] && [ -f "${PIXI_PROJECT_ROOT}/.env" ]; then
    # shellcheck disable=SC1090,SC1091
    . "${PIXI_PROJECT_ROOT}/.env"
fi
set +a
