"""Shared fixtures for the matched-filter test suite.

Provides:
- a seeded RNG so tests are reproducible across pytest-xdist workers,
- a compact synthetic hyperspectral scene with a known plume injected, and
- re-exports of the assimilation conftest's ``obs_model_*`` fixtures so
  target-from-obs tests share the same LUT / SRF plumbing without
  duplication. We import them rather than use ``pytest_plugins`` because
  pytest disallows the latter outside the rootdir conftest.
"""

from __future__ import annotations

import numpy as np
import pytest

# Re-export the assimilation conftest fixtures so they are available to
# matched_filter tests. pytest looks them up by name in any conftest on the
# lookup chain; re-binding them here makes the ``obs_model_no_optics``,
# ``synthetic_lut``, ``hyperspectral_srf`` fixtures resolvable from
# tests/matched_filter/*.
from tests.assimilation.conftest import (  # noqa: F401
    hyperspectral_srf,
    obs_model_no_optics,
    obs_model_with_gsd,
    obs_model_with_psf,
    synthetic_lut,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def hyperspectral_scene(rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic ``(cube, amplitude_map, target)``.

    - ``cube``: shape ``(H, W, n_bands)``, Gaussian-noise background plus a
      rectangular plume enhancement aligned with a known target signature.
    - ``amplitude_map``: ground-truth per-pixel plume amplitude (zero
      background, non-zero within the plume rectangle).
    - ``target``: the spectral signature used to generate the plume —
      matched-filter estimators should recover it exactly in the noiseless
      limit.
    """
    H, W, B = 20, 24, 16
    target = rng.standard_normal(B) * 0.05
    amp_map = np.zeros((H, W), dtype=float)
    amp_map[8:13, 9:15] = 0.25  # the "plume"
    noise = rng.standard_normal((H, W, B)) * 0.01
    cube = 1.0 + amp_map[..., None] * target + noise
    return cube, amp_map, target
