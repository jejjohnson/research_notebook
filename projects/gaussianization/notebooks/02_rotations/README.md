---
title: "Part 2 — Rotations & Orthogonal Mixers"
short_title: "Part 2 · Rotations & mixers"
subject: gaussianization tutorial
---

# Part 2 — Rotations & Orthogonal Mixers

The **between-coordinate** half of Gaussianization. Part 1's marginal transforms
fix each coordinate's distribution but can never remove *dependence between*
coordinates — a product of 1D maps is separable. An orthogonal **rotation**
mixes information across dimensions so the next marginal pass has something to
do; iterating the two is RBIG (Part 3). This part builds every mixer that
matters — fitting one, learning one, freezing one, and the cheap structured
linear layers that make deep flows trainable — grounded in
[`rbig`](https://github.com/jejjohnson/rbig) and
[`gauss_flows`](https://github.com/jejjohnson/gauss_flows).

Each notebook keeps the Part 0/1 pattern: derive the idea, then confirm it
against the packages.

## Notebooks

| # | notebook | master list | what you take away |
|---|----------|:-----------:|--------------------|
| 00 | [Rotation zoo & why it matters](00_rotation_zoo.ipynb) | 2.1–2.2 | PCA/ICA/random/Picard; the demo that marginal-only stalls and a rotation unsticks it |
| 01 | [Householder & trainable orthogonals](01_householder_orthogonal.ipynb) | 2.3–2.4 | reflections → products spanning $O(d)$; Cayley/matrix-exp for $SO(d)$; log-det 0 under training |
| 02 | [Fixed orthogonal & PCA warm starts](02_fixed_pca_warmstart.ipynb) | 2.5–2.6 | `FixedRotation.from_data`; why freezing matters; Householder decomposition to warm-start a trainable stack |
| 03 | [Structured linear layers: 1×1 conv & ActNorm](03_conv1x1_actnorm.ipynb) | 2.7–2.8 | LU-parameterised $1\times1$ conv ($\log|\det|=\sum\log|s|$); data-dependent ActNorm |

## The through-line: log-determinants of linear layers

Part 2 is organised by *what a linear layer costs the log-likelihood*:

- **Rotation** ($Qx$, orthogonal) — volume-preserving, $\log|\det| = 0$. Free.
- **$1\times1$ conv** ($Wx$, general linear) — rescales, $\log|\det| = \sum\log|s|$,
  read off the LU factor in $O(d)$.
- **ActNorm** ($(x-b)/s$, diagonal affine) — $\log|\det| = -\sum\log s$, with a
  data-dependent initialisation.

Each is kept a valid bijector *by construction* — orthogonal parameterisation,
LU factorisation, positive scale — so none can drift singular under training, the
lesson of notebook 02 §2.

## Threads from earlier parts

- The **"rotations are free"** fact ([Part 0, 01](../00_foundations/01_composition_logdet.ipynb))
  is realised here: every orthogonal mixer contributes log-det 0, and we watch it
  stay exactly 0 throughout training (notebook 01).
- **Marginal transforms** (Part 1) are the other half: notebook 00 shows that
  marginal-only Gaussianization plateaus at non-zero total correlation, and a
  rotation between passes drives it to 0.
- The **warm-start** recipe (notebook 02) — decompose → initialise → fine-tune —
  is the bridge to Part 3's RBIG-initialised parametric flows.

## A note on the packages

`gauss_flows` covers the whole toolkit: `HouseholderRotation`,
`OrthogonalRotation` (Cayley), `FixedRotation.from_data`, `Invertible1x1Conv`
(LU), `ActNorm`/`ActNorm1D`; `rbig` supplies the rotation zoo (`PCARotation`,
`ICARotation`, `RandomRotation`, `PicardRotation`). One small asymmetry surfaced
while writing notebook 03 — `FixedRotation` has a `from_data` factory but
`ActNorm` does not, despite data-dependent init being ActNorm's defining feature
([gauss_flows#112](https://github.com/jejjohnson/gauss_flows/issues/112)); the
notebook shows the three-line manual init in the meantime.

## Running

Same `uv` environment as [Part 0](../00_foundations/README.md#running) /
[Part 1](../01_marginal_transforms/README.md#running) (`rbig` + `gauss_flows` +
a Jupyter stack):

```bash
cd projects/gaussianization
.venv-tutorials/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/02_rotations/0*.ipynb --ExecutePreprocessor.timeout=600
```

Notebooks are paired (`jupytext`, `py:percent`) and set `jax_enable_x64`.
