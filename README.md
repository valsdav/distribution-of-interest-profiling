# Profiling Systematics with Distributions of Interest

Reference implementation accompanying the paper *Profiling systematic uncertainties in
Simulation-Based Inference with Factorizable Normalizing Flows* (citation below). This repository
implements the **full unbinned profiled-likelihood fit** on a controllable 2-class toy: the
measurement target is a learnable, invertible **Distribution of Interest (DoI)** transformation
`T_φ`, and systematic uncertainties are **profiled** by an amortized residual flow `T_ψ(ν)` trained
over the nuisance space.

> Scope: this release covers the **downstream profiling chain** — the mixture model, the two-step
> (global + amortized) fit, the Poisson-bootstrap ensemble, the Bayesian-model-averaged (BMA)
> likelihood scans, the post-fit bands, and the Hessian/eigenvector decomposition. The **upstream
> density models** (the frozen base flows `p(x|c)`, `p(y|x,c)` and their residual *input*
> systematics) are the subject of a companion release,
> [factorizable-normalizing-flow](https://github.com/valsdav/factorizable-normalizing-flow); their
> pretrained checkpoints are shipped here as frozen inputs, and the scripts to (re)train them are
> included for completeness.

---

## The idea

A classical fit summarises the data into a few scalar parameters. Here the measurement is instead a
**transformation** `T_φ` of the feature space that maps a reference (simulation) model onto the
observed data — the *Distribution of Interest*. Systematic uncertainties are handled in two steps:

- **Step 1 — global fit.** Jointly optimize the DoI `T_φ` and the nuisances `ν` to the data,
  giving the best fit `ν̂` and the central transformation `T_φ^ν̂`. Repeated on
  **Poisson(1)-bootstrap** replicas of the data, this yields an ensemble `{T_φ^(b)}` that carries the
  **statistical** uncertainty of the DoI.

- **Step 2 — amortized systematic-aware training.** Freeze `T_φ^ν̂` and train a single shared
  **residual** `T_ψ(ν)` (a low-order polynomial in ν) over the nuisance space by maximizing the
  ν-averaged extended likelihood. One training run learns the full ν-response, so the profiling scan
  is replaced by an up-front cost. The composition `T_φ^ν̂ ∘ T_ψ(ν)` is the ν-dependent DoI.

The ensemble of base transformations is combined by **Bayesian model averaging** (an equal-weight
likelihood average, with per-member rebasing to avoid collapse onto a single replica), so a single
likelihood scan carries **both** the statistical (ensemble spread) and systematic (residual
response) uncertainty without a quadrature assumption. A Hessian at `ν̂` then exposes the
**principal modes** of the uncertainty as orthogonal eigenvector morphings of the DoI.

The toy (`generator.py`) injects two known nuisances — an anti-correlated centroid **shift**
(`ν_shift`) and a volume-preserving **squeeze** (`ν_squeeze`) — plus a fixed class-dependent
data↔simulation **distortion** that the fit must recover while profiling.

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: PyTorch, [Zuko](https://github.com/probabilists/zuko) (normalizing flows), NumPy,
SciPy, Matplotlib, PyYAML. A GPU is recommended for the scans/training but everything falls back to
CPU automatically if CUDA is unavailable.

---

## Repository contents

```
distribution-of-interest-profiling/
├── lib.py                          # FullMixtureModel, TransferModel (the DoI T_φ)
├── residual_flow.py                # SystematicCorrectedModel, IndependentPolynomialResidualTransform
├── generator.py                    # ParametricLikelihoodDataset — the 2-class toy + distortion
├── utils.py                        # LR scheduler + checkpoint helper
├── plotting.py                     # shared ATLAS-style heatmap / vector-field helpers
│
├── generate_dataset.py             # materialize the (distorted) toy dataset to disk
├── train_base_flows.py             # (upstream) train frozen base flows  p(x|c), p(y|x,c)
├── train_systematics.py            # (upstream) train the residual input-systematic models
├── train_mixture.py                # STEP 1: global fit + Poisson-bootstrap ensemble (--member N)
├── train_profiling.py              # STEP 2: amortized residual T_ψ(ν) over the ensemble
│
├── likelihood_scan.py              # 1D/2D BMA likelihood scans (ensemble)
├── plot_postfit_ensemble.py        # post-fit distributions with stat / syst / total bands
├── plot_results.py                 # DoI transfer field + score histograms
├── plot_postfit.py                 # post-fit panel helpers (imported)
├── plot_residual_field.py          # residual displacement helpers (imported)
├── hessian_bma.py                  # BMA post-fit Hessian + covariance ellipse overlay
├── visualize_hessian_distortion.py # principal systematic morphings along Hessian eigenvectors
│
├── configs/
│   ├── dataset.yaml                # toy dataset generation (nominal + distorted "data")
│   ├── base_flows.yaml             # (upstream) base-flow architecture + training
│   ├── systematics.yaml            # (upstream) residual input-systematic models
│   ├── mixture.yaml                # step-1 global fit (single)
│   ├── mixture_ensemble.yaml       # step-1 Poisson-bootstrap ensemble
│   ├── profiling.yaml              # step-2 amortized residual (single)
│   └── profiling_ensemble.yaml     # step-2 amortized residual over the ensemble
├── models/                         # pretrained checkpoints (committed)
│   ├── score_density.pt            # frozen base  p(y|x,c)        ┐ upstream density models
│   ├── kin_density.pt              # frozen base  p(x|c)          │ (companion release)
│   ├── score_density_residuals.pt  # frozen score input-systematic │
│   ├── kin_density_residuals.pt    # frozen kin   input-systematic ┘
│   ├── mixture_step1.pt            # step-1 global fit (DoI T_φ^ν̂)
│   ├── mixture_ensemble_profiled.pt# step-2 shared residual T_ψ trained over the ensemble
│   └── ensemble/
│       └── mixture_boot0.pt … mixture_boot7.pt   # 8-member Poisson-bootstrap ensemble {T_φ^(b)}
└── datasets/
    └── dataset.pt                  # the toy dataset (nominal templates + distorted "data")
```

---

## Quickstart A — reproduce the paper figures from the pretrained checkpoints

The shipped checkpoints reproduce the profiling-chain figures with **no training**. Run from the
repository root:

```bash
mkdir -p figs/scans figs/postfit

# (1) STEP 1 — global-fit (mixture) BMA likelihood scan
python likelihood_scan.py -c configs/mixture_ensemble.yaml \
    --ensemble "models/ensemble/mixture_boot*.pt" --ensemble-mode rebased-bma \
    --scan-2d --pairs 0,1 --scan2d-name scan2d_ensemble_mixture.npz --out-dir figs/scans

# (2) STEP 1 — post-fit distribution with the data-statistical (ensemble) band
python plot_postfit_ensemble.py -c configs/mixture_ensemble.yaml \
    --ensemble "models/ensemble/mixture_boot*.pt" --out-dir figs/postfit

# (3) STEP 1 — the learned DoI transfer field
python plot_results.py -c configs/mixture.yaml --ckpt models/mixture_step1.pt --out-dir figs

# (4) STEP 2 — profiled BMA likelihood scan (also writes the scan npz used by 5 & 6)
python likelihood_scan.py -c configs/profiling_ensemble.yaml \
    --ensemble "models/ensemble/mixture_boot*.pt" --ensemble-mode rebased-bma \
    --scan-2d --pairs 0,1 --scan2d-name scan2d_ensemble_profiled.npz --out-dir figs/scans

# (5) STEP 2 — post-fit with statistical + systematic + total bands
python plot_postfit_ensemble.py -c configs/profiling_ensemble.yaml \
    --ensemble "models/ensemble/mixture_boot*.pt" \
    --syst-scan2d figs/scans/scan2d_ensemble_profiled.npz --pair 0,1 --out-dir figs/postfit

# (6) Orthogonal decomposition — BMA Hessian overlay (writes hessian_bma_ensemble_profiled.npz)
python hessian_bma.py -c configs/profiling_ensemble.yaml \
    --ensemble "models/ensemble/mixture_boot*.pt" --ensemble-mode rebased-bma \
    --scan2d figs/scans/scan2d_ensemble_profiled.npz --pair 0,1 \
    --overlay-scan2d figs/scans/scan2d_ensemble_profiled.npz --out-dir figs/scans

# (7) Principal systematic morphings along the Hessian eigenvectors
python visualize_hessian_distortion.py -c configs/profiling_ensemble.yaml \
    --hessian-npz figs/scans/hessian_bma_ensemble_profiled.npz \
    --flavour 0 --out-dir figs/scans
```

Each command writes both `.png` and `.pdf`. The mapping to the paper figures:

| Command | Output file | Paper figure |
|---|---|---|
| 1 | `figs/scans/scan2d_members_rebased-bma_ensemble_mixture_nuis01` | Step-1 scan (per-member) |
| 1 | `figs/scans/scan2d_rebased-bma_ensemble_mixture_nuis01` | Step-1 scan (combined BMA) |
| 2 | `figs/postfit/postfit_distortion_xbinned_ensemble_mixture` | Step-1 post-fit |
| 3 | `figs/mixture_step1_transfer_field` | Step-1 DoI transfer field¹ |
| 4 | `figs/scans/scan2d_members_rebased-bma_ensemble_profiled_nuis01` | Step-2 scan (per-member) |
| 4 | `figs/scans/scan2d_rebased-bma_ensemble_profiled_nuis01` | Step-2 scan (combined BMA) |
| 5 | `figs/postfit/postfit_total_xbinned_ensemble_profiled` | Step-2 post-fit (stat+syst) |
| 6 | `figs/scans/hessian_bma_overlay_ensemble_profiled` | Hessian / eigenvector overlay |
| 7 | `figs/scans/hessian_distortion_profiled` | Principal systematic morphings |

¹ `plot_results.py` derives this filename from the checkpoint basename, so it is
`mixture_step1_transfer_field` here (the paper's `full_mixture_model_v15_transfer_field`).

The `--ensemble-mode rebased-bma` flag is the combination used in the paper (rebase each member to
its own minimum, then BMA-average). The data hand-offs are explicit: the **step-2 scan** npz feeds
both the step-2 post-fit (`--syst-scan2d`) and the Hessian (`--scan2d`/`--overlay-scan2d`), and the
Hessian npz feeds the morphing plot (`--hessian-npz`). Increase `--steps-2d` for a finer scan
surface.

---

## Quickstart B — train the full chain from scratch

```bash
# 0. materialize the toy dataset (nominal templates + the distorted "data")
python generate_dataset.py -c configs/dataset.yaml

# 1. (upstream) frozen base flows  p(x|c), p(y|x,c)   — data generated on the fly
python train_base_flows.py -c configs/base_flows.yaml -s train_score,train_kin

# 2. (upstream) residual input-systematic models on top of the frozen bases
python train_systematics.py -c configs/systematics.yaml -s train_kin,train_score

# 3. STEP 1 — global fit (single) and the Poisson-bootstrap ensemble
python train_mixture.py -c configs/mixture.yaml -s train                  # -> models/mixture_step1.pt
for i in 0 1 2 3 4 5 6 7; do
    python train_mixture.py -c configs/mixture_ensemble.yaml -s train --member $i --seed 0
done                                                                       # -> models/ensemble/mixture_boot<i>.pt

# 4. STEP 2 — amortized residual T_ψ(ν) trained over the ensemble
python train_profiling.py -c configs/profiling_ensemble.yaml -s train      # -> models/mixture_ensemble_profiled.pt

# 5. reproduce the figures as in Quickstart A
```

Before step 4, the residual's expansion anchor `ν₀` (`mixture_model.m_vector_override` in
`configs/profiling_ensemble.yaml`) should be set to the best fit from the step-1 BMA mixture scan
(command 1 above writes a `bestfit` next to the scan). The single-model `configs/profiling.yaml`
trains the non-ensemble step-2 residual (`models/mixture_profiled.pt`) the same way.

---

## Configuration

All architecture, training, generator, and path settings live in the YAML files under `configs/`.
Paths are resolved relative to the config file (`../models`, `../datasets`). Key sections:

- `paths` — frozen input checkpoints, the step-1 mixture / ensemble, and the output checkpoint.
- `score_flow` / `kin_flow`, `residual_score_model` / `residual_kin_model` — the (frozen) upstream
  density architectures, sized to match the shipped checkpoints.
- `mixture_model` — class fractions, `lnN` normalisation nuisances, the profiled-nuisance mask, and
  `m_vector_override` (the step-1 best fit `ν̂` / residual anchor `ν₀`).
- `transfer_model` — the DoI `T_φ` architecture (note `mixture.yaml` uses `nbins: 30` while the
  ensemble configs use `nbins: 16`, matching the respective checkpoints — keep them consistent).
- `residual_transfer_model` — the step-2 residual `T_ψ`: `num_residual_layers`, `nuisance_scales`,
  `quadratic_damping`.
- `training.m_sampling: quadrature` with `m_oversample` — the Gauss–Legendre grid over `range_m`
  used to integrate the ν-averaged objective (the residual is polynomial in ν, so the grid
  *identifies* its coefficients).

---

## Citation

```bibtex
@article{doi_profiling,
  title  = {Profiling systematic uncertainties in Simulation-Based Inference
            with Factorizable Normalizing Flows},
  author = {Valsecchi, Davide and others},
  year   = {2026},
  note   = {TODO: fill in once published}
}
```

## License

See [LICENSE](LICENSE).
