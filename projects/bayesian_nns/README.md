# Bayesian Models & Neural Networks

Tutorials and applied case studies for **Bayesian linear regression, parametric Bayes, Bayesian classification, and Bayesian neural networks** — using `pyrox`, `gaussx`, and NumPyro.

The full curriculum (parts, subparts, references, gaps, cross-listing with the GP and neural-fields lists) lives in [`TUTORIAL_MASTER_LIST.md`](TUTORIAL_MASTER_LIST.md).

## Layout

- `notebooks/A_regression/` — Bayesian linear regression, basis features, RFF/HSGP, likelihood + loss zoo, 9-model regression masterclass pick-apart
- `notebooks/B_classification/` — Bayesian logistic / multinomial / Laplace / VI / MCMC / last-layer & SNGP classifiers
- `notebooks/C_nn_gp_bridges/` — NNGP, NTK, deep kernels, deep RFF, functional priors, pathwise BNN
- `notebooks/D_inference/` — point estimates, Laplace family, VI, sampling, last-layer, stochastic/implicit, tempering, distance-aware
- `notebooks/E_ensembles/` — deep ensembles, ensemble runners, diversity strategies
- `notebooks/F_calibration/` — ECE, temperature scaling, OOD, active learning
- `notebooks/G_bayesian_neural_fields/` — Bayesian INR / NeRF / BNF (companion to `../neural_fields/`)
- `notebooks/H_applied/` — UCI benchmarks, BNN emulators, Bayesian PINNs, image/signal regression, capstone progressions

## Companion lists

- Pure GPs → [`../gaussian_processes/TUTORIAL_MASTER_LIST.md`](../gaussian_processes/TUTORIAL_MASTER_LIST.md)
- Neural fields / NeRF (deterministic) → [`../neural_fields/TUTORIAL_MASTER_LIST.md`](../neural_fields/TUTORIAL_MASTER_LIST.md)
- Normalizing flows → [`../gaussianization/TUTORIAL_MASTER_LIST.md`](../gaussianization/TUTORIAL_MASTER_LIST.md)
- Filtering / variational data assimilation → [`../assimilation/docs/`](../assimilation/docs/)
