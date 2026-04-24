"""Tests for ``plume_simulation.matched_filter.cluster``."""

from __future__ import annotations

import numpy as np
import pytest
from plume_simulation.matched_filter.cluster import (
    adaptive_window_background,
    gmm_cluster_background,
)


def test_gmm_separates_two_populations(rng):
    """Land + water mock scene: GMM recovers a mask correlated with truth."""
    H, W, B = 20, 20, 6
    truth = np.zeros((H, W), dtype=int)
    truth[:, W // 2 :] = 1  # right half = cluster 1
    cube = np.empty((H, W, B))
    cube[truth == 0] = 0.3 + rng.standard_normal(((truth == 0).sum(), B)) * 0.01
    cube[truth == 1] = 0.8 + rng.standard_normal(((truth == 1).sum(), B)) * 0.01
    result = gmm_cluster_background(cube, n_clusters=2, random_state=0)
    assert result.labels.shape == (H, W)
    assert len(result.means) == len(result.cov_operators) == 2
    # Match cluster IDs to truth by the higher-mean cluster being "water".
    means_mag = [m.mean() for m in result.means]
    high, low = int(np.argmax(means_mag)), int(np.argmin(means_mag))
    predicted = np.where(result.labels == high, 1, 0)
    accuracy = (predicted == truth).mean()
    assert accuracy > 0.9, f"GMM cluster recovery only {accuracy:.2f}"


def test_gmm_shape_check():
    with pytest.raises(ValueError, match="must be"):
        gmm_cluster_background(np.zeros((3, 4)), n_clusters=2)


def test_adaptive_window_mean_equals_uniform_filter(rng):
    """Sliding-window mean == scipy.ndimage.uniform_filter per band."""
    from scipy.ndimage import uniform_filter

    cube = rng.standard_normal((12, 14, 3))
    mean, var = adaptive_window_background(cube, window_size=3)
    for b in range(3):
        ref = uniform_filter(cube[..., b], size=3, mode="reflect")
        np.testing.assert_allclose(mean[..., b], ref, atol=1e-12)
    assert np.all(var >= 0.0)


def test_adaptive_window_rejects_even_window(rng):
    cube = rng.standard_normal((5, 5, 2))
    with pytest.raises(ValueError, match="odd"):
        adaptive_window_background(cube, window_size=4)


def test_adaptive_window_variance_is_nonneg_and_correct(rng):
    """Variance is close to E[x²] − E[x]² inside the interior."""
    cube = rng.standard_normal((9, 9, 1)) * 0.5 + 1.0
    mean, var = adaptive_window_background(cube, window_size=3)
    # Spot-check an interior pixel against the 3x3 neighborhood by hand.
    patch = cube[3:6, 3:6, 0]
    np.testing.assert_allclose(mean[4, 4, 0], patch.mean(), atol=1e-12)
    np.testing.assert_allclose(var[4, 4, 0], patch.var(), atol=1e-12)
