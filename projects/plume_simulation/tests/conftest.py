"""Top-level pytest configuration for ``plume_simulation``.

Enables JAX 64-bit math at the earliest possible point: ``JAX_ENABLE_X64=1``
is set in the env *before* JAX has a chance to import (pytest plugins like
``pytest-jaxtyping`` can pull JAX in during plugin discovery, which happens
before this module is even loaded — but env-var reads still pick up the
value because JAX checks ``JAX_ENABLE_X64`` on first config access). We
also call ``jax.config.update`` as a belt-and-braces fallback.

The adjoint / round-trip invariants in the radtran and assimilation suites
assert agreements at ~1e-12 — well below float32 precision (~1e-7) — so a
silent regression to float32 turns multiple tests into mysteriously-failing
phantoms.
"""

from __future__ import annotations

import os


os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402 — must come *after* the env var is set.


jax.config.update("jax_enable_x64", True)
