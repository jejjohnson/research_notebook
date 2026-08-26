"""Cached inference results, for warm-starting one notebook from another.

The curriculum fits the same stations, and often the same parameters, several
times over: the Laplace approximation of notebook 04b re-derives what notebook
04 sampled; notebook 12 refits notebook 11's ODE verbatim; every spatial model
estimates a location for *every* station, Albacete included, before the
non-stationary notebooks ever look at it.

This module lets a notebook **publish** a fit and a later notebook **start from
it** — the Bayesian analogue of a pretrained checkpoint. Warm starts are
initialisation only: they move where a sampler or optimiser *begins*, never
what it targets, so a warm-started posterior is the same posterior.

Network-free, and deliberately optional. :func:`load_fit` returns ``None`` when
the artifact is missing or was built from a different kind of data, and every
consumer falls back to fitting cold (``init_to_median``, as before). So a fresh
clone still runs every notebook standalone and offline; running the curriculum
in order simply populates the cache and makes the later notebooks faster and
better-anchored. Mirrors the cache conventions of
:mod:`spatial_extremes.features`.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .data import default_cache_root


def results_path(name: str, root: Path | None = None) -> Path:
    """Location of the cached fit ``name`` (its metadata sits alongside)."""
    base = Path(root) if root is not None else default_cache_root()
    return base / "results" / f"{name}.npz"


def _git_sha() -> str | None:
    """Short HEAD sha, for provenance — ``None`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - provenance is best-effort
        return None


def save_fit(
    name: str,
    *,
    source: str,
    is_real: bool,
    root: Path | None = None,
    **arrays,
) -> Path:
    """Cache posterior summaries (or draws) under ``name``.

    ``source`` is the publishing notebook's path and ``is_real`` flags whether
    the fit saw real CDS data — both are recorded so a consumer can print where
    its warm start came from, and refuse one built from the other data regime.
    """
    path = results_path(name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez_compressed(path, **payload)
    meta = {
        "name": name,
        "source": source,
        "is_real": bool(is_real),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "arrays": {k: list(v.shape) for k, v in payload.items()},
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def load_fit(
    name: str,
    *,
    expect_real: bool | None = None,
    root: Path | None = None,
) -> dict | None:
    """Return the cached arrays plus a ``"_meta"`` entry, or ``None``.

    ``None`` means "no usable warm start" — the artifact is absent, unreadable,
    or (when ``expect_real`` is given) was built from the other data regime.
    Callers treat that as a cue to fit cold rather than as an error.
    """
    path = results_path(name, root)
    if not path.exists():
        return None
    meta_path = path.with_suffix(".json")
    try:
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if expect_real is not None and meta.get("is_real") not in (None, expect_real):
            return None
        out = {k: v for k, v in np.load(path).items()}
    except Exception:  # pragma: no cover - a broken cache must not crash a notebook
        return None
    out["_meta"] = meta
    return out


def describe_fit(fit: dict | None) -> str:
    """One-line provenance string for a loaded fit, for notebooks to print."""
    if fit is None:
        return "no cached fit — starting cold"
    meta = fit.get("_meta", {})
    return (
        f"warm start <- {meta.get('source', '?')} "
        f"@ {meta.get('git_sha', '?')} ({meta.get('created', '?')})"
    )


def nearest_station(stations: np.ndarray, lon: float, lat: float) -> tuple[int, float]:
    """Index of the station nearest ``(lon, lat)``, and its distance in km.

    ``stations`` is the ``(S, 2)`` lon/lat array the loaders return; it carries
    no station ids, so a spatial fit is joined to a single-station record by
    position. Equirectangular distance is ample over a region this size.
    """
    s = np.asarray(stations, dtype=float)
    mean_lat = np.radians((s[:, 1] + lat) / 2.0)
    dlat = np.radians(s[:, 1] - lat)
    dlon = np.radians(s[:, 0] - lon) * np.cos(mean_lat)
    d_km = 6371.0 * np.hypot(dlat, dlon)
    i = int(np.argmin(d_km))
    return i, float(d_km[i])
