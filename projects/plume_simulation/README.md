---
title: Plume simulation (moved)
---

# Plume simulation — moved to plumax

The `plume_simulation` package, its roadmap, derivations and notebooks now live in [plumax](https://github.com/jejjohnson/plumax) ([docs](https://jejjohnson.github.io/plumax/)).

| Was (`plume_simulation`) | Now |
|---|---|
| `src/plume_simulation/<subpkg>` | `plumax.<subpkg>` (`gauss_plume`, `gauss_puff`, `les_fvm`, `hapi_lut`, `radtran`, `matched_filter`, `assimilation`) |
| `notes/roadmap/` | [plumax design roadmap](https://jejjohnson.github.io/plumax/roadmap/) |
| `notes/EQUATIONS.md`, `notes/satellites.md` | [Equations](https://jejjohnson.github.io/plumax/equations/), [Satellites](https://jejjohnson.github.io/plumax/satellites/) |

Derivations:

- [Gaussian plume](https://jejjohnson.github.io/plumax/gaussian-plume-derivation/)
- [Gaussian puff](https://jejjohnson.github.io/plumax/gaussian-puff-derivation/)
- [Eulerian dispersion](https://jejjohnson.github.io/plumax/eulerian-dispersion-derivation/)
- [HAPI LUT](https://jejjohnson.github.io/plumax/hapi-lut-derivation/)
- [Matched filter](https://jejjohnson.github.io/plumax/mf-derivation/)
- [3D-Var](https://jejjohnson.github.io/plumax/variational-derivation/)

The executed notebooks are under `docs/notebooks/` in plumax. The full history of the original sub-project is in this repository's git log.
