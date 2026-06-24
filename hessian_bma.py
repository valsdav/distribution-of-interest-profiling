"""
BMA post-fit Hessian / covariance of the profiling fit.

After step-2 profiling, the best-fit shape-nuisance vector ν̂ is found from a
Bayesian-Model-Averaged (BMA) likelihood scan over the step-1 transfer-T bootstrap
ensemble (`likelihood_scan.py --ensemble --ensemble-mode bma`). This script builds the
local-Gaussian post-fit covariance at ν̂ *including* the BMA over ensemble members, i.e.
an uncertainty that propagates the step-1 statistical spread of the transfer model into
the combined fit.

Math
----
BMA log-posterior to maximise (per nuisance ν):

    S_BMA(ν) = logsumexp_b[ ℓ_b(ν) ] − ln K + c(ν)

with per-member data log-likelihood  ℓ_b(ν) = Σ_events log p(event | ν, T_b, R_ν)  and
member-independent shape constraint  c(ν) = −0.5·Σ_{profiled} ν²  (the lnN term is
constant in an m-scan ⇒ zero gradient/Hessian, ignored). NLL_BMA = −S_BMA.

Writing nll̃_b = NLL_data_b + 0.5·Σν² (the per-member NLL *including* the constraint),
with  ℓ̃_b = −nll̃_b(ν̂),  g̃_b = ∇nll̃_b(ν̂),  H̃_b = ∇²nll̃_b(ν̂),  and BMA softmax weights
w_b = softmax_b(ℓ̃_b) (the constraint cancels in the softmax):

    H_within  = Σ_b w_b H̃_b                       (weighted-avg per-member curvature, PD)
    ḡ         = Σ_b w_b g̃_b                        (≈ 0 at the BMA min — reported)
    Cov_w[g]  = Σ_b w_b g̃_b g̃_bᵀ − ḡ ḡᵀ           (between-member gradient spread, PSD)
    H_BMA     = H_within − Cov_w[g]                 (combined BMA NLL Hessian)

Identity: ∇²S_BMA = −Σ w_b H̃_b + Cov_w[∇ℓ̃_b], and H_BMA = −∇²S_BMA. The −Cov_w[g] term
REDUCES curvature ⇒ INFLATES the covariance: this is the propagated step-1 BMA
uncertainty. C_BMA = inv(H_BMA); the within-only covariance C_within = inv(H_within) and
the inflation C_BMA − C_within are reported alongside.

Usage
-----
    # 1. BMA 2D scan (gives ν̂ and the contour to validate against)
    python likelihood_scan.py -c configs/profiling_v15_ensemble.yaml \
        --ensemble 'models/.../member_*.pt' --ensemble-mode bma --scan-2d \
        --out-dir scans/bma_v15

    # 2. BMA Hessian with verification + ellipse overlay
    python hessian_bma.py -c configs/profiling_v15_ensemble.yaml \
        --ensemble 'models/.../member_*.pt' \
        --scan2d scans/bma_v15/scan2d.npz --pair 0,1 \
        --out-dir hessian_bma/v15 --verify --overlay-scan2d scans/bma_v15/scan2d.npz

Without --ensemble the script loads a single model (K=1); the decomposition collapses
(w=[1], Cov_w=0, H_BMA=H_within) to the ordinary single-model Hessian — a useful
cross-check against paper_plots.py:fig_hessian.
"""
import argparse
import json
import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as mtransforms
import numpy as np
import torch
import torch.autograd.functional as AF
import yaml

from likelihood_scan import load_ensemble, load_models, _resolve, _save
from plot_postfit import _bestfit_from_scan2d


# ---------------------------------------------------------------------------
# Per-member gradient + Hessian of the data NLL at ν̂ (additive over events)
# ---------------------------------------------------------------------------

def _member_grad_hess(model, T_model, y, x, nu_hat, mask_idx, batch_size, n_batches, device):
    """DATA NLL value, gradient (d) and Hessian (d×d) at ν̂ for ONE member, summed over
    events. The shape constraint is NOT included here (added analytically by the caller).

    The data NLL is a sum over events, so its gradient and Hessian are the sums of the
    per-batch gradients/Hessians. We accumulate them batch by batch and FREE each batch's
    autograd graph before the next, so peak memory is bounded by ONE batch (`--batch-size`),
    not the whole dataset. Per batch we do a single forward, then d+1 backward passes
    (one for the gradient with create_graph=True, then one per Hessian row) reusing that
    batch's graph. The free variable spans only the profiled indices `mask_idx`;
    `model.log_prob` zeroes and grad-blocks the rest (lib.py:258-279), so frozen nuisances
    never enter H.
    """
    N = y.shape[0]
    nb = min(n_batches, (N + batch_size - 1) // batch_size)
    d = mask_idx.numel()
    theta0 = nu_hat.index_select(0, mask_idx).detach().clone()
    ell_data = 0.0                                          # ℓ_data = -NLL_data(ν̂)
    g_data = torch.zeros(d, dtype=torch.float64, device=device)
    H_data = torch.zeros(d, d, dtype=torch.float64, device=device)
    for b in range(nb):
        y_b = y[b * batch_size:(b + 1) * batch_size]
        x_b = x[b * batch_size:(b + 1) * batch_size]
        theta = theta0.clone().requires_grad_(True)        # fresh leaf per batch
        # Inject the d profiled params into a full m-vector; index_copy keeps the graph
        # flowing only through `theta` (general-d analogue of plot_postfit.py:88-90).
        m_full = nu_hat.clone().index_copy(0, mask_idx, theta)
        m_b = m_full.unsqueeze(0).expand(y_b.shape[0], -1)
        lp, _ = model.log_prob(y_b, x_b, T_model, m_b)
        nll_b = (-lp.sum()).double()                       # float64 (per-batch sum)
        g_b = torch.autograd.grad(nll_b, theta, create_graph=True)[0]   # [d]
        for i in range(d):
            H_row = torch.autograd.grad(g_b[i], theta, retain_graph=(i < d - 1))[0]
            H_data[i] += H_row.detach().double()
        g_data += g_b.detach().double()
        ell_data += float(-nll_b.detach().item())
        del lp, nll_b, g_b, m_b, m_full, theta             # release this batch's graph
    return ell_data, g_data, H_data


def _rebased_logweights(G, Hs, floor=1e-6):
    """Per-member rebased log-weights r_b = −½ g̃_bᵀ H̃_b⁻¹ g̃_b (one per member).

    This is the local (Laplace) equivalent of likelihood_scan.py's `rebased-*` rebasing:
    subtracting each member's OWN minimum NLL removes the ν-independent absolute-depth
    offsets (overfitting/calibration) that otherwise collapse the plain-bma softmax onto a
    single member (K_eff→1). ½ g̃_bᵀ H̃_b⁻¹ g̃_b is exactly member b's quadratic NLL deficit
    at ν̂ relative to its own optimum, so r_b is its rebased log-likelihood at ν̂. H̃_b is
    inverted through a floored eigen-decomposition so a non-PD member degrades gracefully."""
    K = G.shape[0]
    r = torch.empty(K, dtype=torch.float64, device=G.device)
    for b in range(K):
        evals, evecs = torch.linalg.eigh(0.5 * (Hs[b] + Hs[b].T))
        gb = evecs.T @ G[b]
        deficit = 0.5 * (gb * gb / evals.clamp(min=floor)).sum()   # ½ g̃ᵀ H̃⁻¹ g̃ ≥ 0
        r[b] = -deficit
    return r


def compute_bma_hessian(model, residual_T, member_Ts, y, x, nu_hat, mask_idx,
                        batch_size, n_batches, device, include_shape=True, mode="bma"):
    """The decomposition. Loops members (swapping each base transfer into the shared
    residual, as likelihood_scan.py:_member_logliks does), accumulates per-member
    (ℓ̃_b, g̃_b, H̃_b), then combines with the BMA softmax weights.

    `mode` selects the weighting (only the WEIGHTS change; the per-member g̃_b/H̃_b and the
    decomposition H_BMA = Σ w_b H̃_b − Cov_w[g] are identical):
      'bma'          w_b = softmax_b(ℓ̃_b(ν̂))            — absolute per-member likelihood;
                     collapses to one member when depth offsets dominate (K_eff→1).
      'rebased-bma'  w_b = softmax_b(−½ g̃_bᵀ H̃_b⁻¹ g̃_b) — strips the per-member depth
                     offset first (matches likelihood_scan.py --ensemble-mode rebased-bma)."""
    K = len(member_Ts)
    d = mask_idx.numel()
    nu_d = nu_hat.index_select(0, mask_idx).double()
    eye = torch.eye(d, dtype=torch.float64, device=device)
    shape_val = 0.5 * (nu_d ** 2).sum() if include_shape else torch.zeros((), dtype=torch.float64, device=device)

    ell = torch.empty(K, dtype=torch.float64, device=device)
    G = torch.empty(K, d, dtype=torch.float64, device=device)
    Hs = torch.empty(K, d, d, dtype=torch.float64, device=device)
    for b, T_b in enumerate(member_Ts):
        if residual_T is not None:
            residual_T.base_model = T_b              # swap member b's base transfer
            T_model = residual_T
        else:
            T_model = T_b
        e_data, g_data, h_data = _member_grad_hess(
            model, T_model, y, x, nu_hat, mask_idx, batch_size, n_batches, device)
        # Add the shape constraint analytically (once per member, not per batch):
        #   ℓ̃_b = ℓ_data − 0.5Σν²,  g̃_b = g_data + ν,  H̃_b = H_data + I
        ell[b] = e_data - shape_val
        G[b] = g_data + (nu_d if include_shape else 0.0)
        Hs[b] = h_data + (eye if include_shape else 0.0)

    if mode == "rebased-bma":
        logw = _rebased_logweights(G, Hs)
    elif mode == "bma":
        logw = ell
    else:
        raise ValueError(f"unknown mode '{mode}' (use 'bma' or 'rebased-bma')")
    w = torch.softmax(logw, dim=0)
    H_within = torch.einsum('b,bij->ij', w, Hs)
    g_bar = torch.einsum('b,bi->i', w, G)
    Cov_w = torch.einsum('b,bi,bj->ij', w, G, G) - torch.outer(g_bar, g_bar)
    H_BMA = H_within - Cov_w
    return {
        "H_BMA": 0.5 * (H_BMA + H_BMA.T),
        "H_within": 0.5 * (H_within + H_within.T),
        "Cov_w": 0.5 * (Cov_w + Cov_w.T),
        "w": w, "g_bar": g_bar, "ell": ell, "G": G, "Hs": Hs, "mode": mode,
    }


def _safe_inv(H, floor=1e-6, name="H"):
    """Invert a (NLL) Hessian via a floored eigen-decomposition; warn if not PD.

    Mirrors plot_postfit.py:104-109. A non-PD H_BMA means Cov_w[g] over-subtracted the
    curvature (members disagree strongly) — the local-Gaussian approximation is then
    unreliable and the full BMA scan contour should be preferred."""
    H = 0.5 * (H + H.T)
    evals, evecs = torch.linalg.eigh(H)
    npd = int((evals <= floor).sum().item())
    if npd > 0:
        print(f"  ⚠ {name} not positive-definite "
              f"(eigvals={['%.3e' % v for v in evals.tolist()]}); clamping to {floor:g}. "
              "Prefer the full BMA scan contour (likelihood_scan.py --scan-2d).")
    cov = (evecs * (1.0 / evals.clamp(min=floor))) @ evecs.T
    return 0.5 * (cov + cov.T), evals, evecs, npd


# ---------------------------------------------------------------------------
# Optional exact cross-check: direct autograd Hessian of S_BMA on a subsample
# ---------------------------------------------------------------------------

def _verify_full(model, residual_T, member_Ts, y, x, nu_hat, mask_idx, batch_size, device,
                 include_shape=True):
    """Hessian of −S_BMA(θ) built directly through the logsumexp over members (exact, but
    keeps the full members×events graph ⇒ subsample only). Should match the decomposition."""
    K = len(member_Ts)
    logK = torch.log(torch.tensor(float(K), dtype=torch.float64, device=device))

    def neg_S_BMA(theta):
        m_full = nu_hat.clone().index_copy(0, mask_idx, theta)
        per = []
        for T_b in member_Ts:
            if residual_T is not None:
                residual_T.base_model = T_b
                T_model = residual_T
            else:
                T_model = T_b
            tot = torch.zeros((), device=device, dtype=torch.float64)
            for b in range(0, y.shape[0], batch_size):
                y_b = y[b:b + batch_size]
                x_b = x[b:b + batch_size]
                m_b = m_full.unsqueeze(0).expand(y_b.shape[0], -1)
                lp, _ = model.log_prob(y_b, x_b, T_model, m_b)
                tot = tot + lp.sum().double()
            per.append(tot)
        ell_data = torch.stack(per)                         # [K]
        S = torch.logsumexp(ell_data, 0) - logK
        if include_shape:
            S = S - 0.5 * (theta.double() ** 2).sum()
        return -S

    theta0 = nu_hat.index_select(0, mask_idx).detach().clone().requires_grad_(True)
    H_full = AF.hessian(neg_S_BMA, theta0).detach().double()
    return 0.5 * (H_full + H_full.T)


# ---------------------------------------------------------------------------
# ν̂ resolution
# ---------------------------------------------------------------------------

def _resolve_nu_hat(args, model, num_nuis, device):
    """Precedence: --best > --scan2d argmin > --bestfit-json > model.m_vector anchor."""
    base = model.m_vector.detach().to(device).float().clone()
    if args.best is not None:
        vals = [float(v) for v in args.best.split(",")]
        if len(vals) != num_nuis:
            raise ValueError(f"--best needs {num_nuis} values, got {len(vals)}")
        return torch.tensor(vals, device=device, dtype=torch.float32), "--best"
    if args.scan2d is not None:
        pair = tuple(int(v) for v in args.pair.split(","))
        bi, bj = _bestfit_from_scan2d(args.scan2d, pair)
        base[pair[0]] = bi
        base[pair[1]] = bj
        return base, f"scan2d argmin {os.path.basename(args.scan2d)}"
    if args.bestfit_json is not None:
        with open(args.bestfit_json) as f:
            bf = json.load(f)
        for k, v in bf["nuisances"].items():
            base[int(k)] = float(v["best_fit"])
        return base, f"bestfit.json {os.path.basename(args.bestfit_json)}"
    return base, "model.m_vector anchor"


# ---------------------------------------------------------------------------
# Ellipse overlay on the 2D BMA scan
# ---------------------------------------------------------------------------

def _confidence_ellipse(cov_m, pos, n_std, ax, **kw):
    """Draw the {δ : δᵀ C⁻¹ δ = n_std²} ellipse of covariance `cov_m` at `pos`
    (replicates the helper nested in paper_plots.py:fig_hessian)."""
    ev, evec = np.linalg.eigh(cov_m)
    order = ev.argsort()[::-1]
    ev, evec = ev[order], evec[:, order]
    angle = np.degrees(np.arctan2(evec[1, 0], evec[0, 0]))
    ell = Ellipse((0, 0), width=2 * n_std * np.sqrt(ev[0]), height=2 * n_std * np.sqrt(ev[1]),
                  angle=angle, **kw)
    ell.set_transform(mtransforms.Affine2D().translate(*pos) + ax.transData)
    return ax.add_patch(ell)


def _draw_eigenvectors(ax, cov_m, pos):
    """Overlay the principal axes of `cov_m` as ±1σ line segments through `pos`.

    The variance along eigenvector k is its eigenvalue, so the 1σ semi-axis is √λ_k; we
    scale by √2.30 so the segment ends ON the 1σ (2-dof) ellipse. Matches the eigenvector
    quiver of paper_plots.py:fig_hessian (which uses the covariance eigen-decomposition)."""
    evals, evecs = np.linalg.eigh(cov_m)
    order = evals.argsort()[::-1]                      # major axis first
    evals, evecs = evals[order], evecs[:, order]       # major axis first
    n1 = np.sqrt(2.30)
    names = ["major axis", "minor axis"]
    for k, col in zip(range(len(evals)), ["magenta", "deepskyblue"]):
        sig_k = np.sqrt(max(float(evals[k]), 0.0))
        v = evecs[:, k] * sig_k * n1                   # reaches the 1σ ellipse
        name = names[k] if k < len(names) else f"axis {k}"
        ax.plot([pos[0] - v[0], pos[0] + v[0]], [pos[1] - v[1], pos[1] + v[1]],
                color=col, lw=2.0, solid_capstyle="round",
                label=f"{name} (σ={sig_k:.4f})")


def _plot_overlay(scan2d_path, pair, C_BMA, C_within, nu_pair, labels, out_dir, label,
                  draw_eigenvectors=False):
    """C_BMA 1σ/2σ ellipse (+ within-member 1σ, + optional principal axes) over the 2D
    −2Δln L scan, styled like likelihood_scan.py:plot_scan_2d (RdYlBu_r heatmap, ROOT-style
    triangular colorbar, white/yellow/orange 1σ/2σ/3σ CL contours, white best-fit star)."""
    d_npz = np.load(scan2d_path, allow_pickle=True)
    key = f"{pair[0]}{pair[1]}"
    try:
        axis_i, axis_j, m2dnll = d_npz[f"axis_i_{key}"], d_npz[f"axis_j_{key}"], d_npz[f"m2dnll_{key}"]
    except KeyError:
        print(f"  ⚠ {scan2d_path} has no 2D scan for pair {pair}; skipping overlay.")
        return
    Xg, Yg = np.meshgrid(axis_i, axis_j)
    Z = m2dnll.T                                          # axis_i -> x, axis_j -> y

    fig, ax = plt.subplots(figsize=(9, 8))
    vmax = float(np.nanmax(Z))
    cf = ax.contourf(Xg, Yg, Z, levels=np.linspace(0.0, vmax, 41), cmap="RdYlBu_r", extend="both")
    cbar = fig.colorbar(cf, ax=ax, extend="both", pad=0.02)
    cbar.set_label(r"$-2\Delta\log L$", fontsize=13)
    cs = ax.contour(Xg, Yg, Z, levels=[2.30, 6.18, 11.83],
                    colors=["white", "yellow", "orange"], linewidths=1.8)
    ax.clabel(cs, inline=True, fontsize=10, fmt={2.30: "1σ", 6.18: "2σ", 11.83: "3σ"})

    pos = (float(nu_pair[0]), float(nu_pair[1]))
    _confidence_ellipse(C_BMA, pos, np.sqrt(2.30), ax, edgecolor="black", facecolor="none",
                        lw=2.2, label="1σ Hessian (T-averaged)")
    _confidence_ellipse(C_BMA, pos, np.sqrt(6.18), ax, edgecolor="black", facecolor="none",
                        lw=1.6, ls=":", label="2σ Hessian (T-averaged)")
    _confidence_ellipse(C_within, pos, np.sqrt(2.30), ax, edgecolor="lime", facecolor="none",
                        lw=1.6, label="1σ Hessian (T fixed)")
    if draw_eigenvectors:
        _draw_eigenvectors(ax, C_BMA, pos)
    ax.plot(*pos, marker="*", color="white", ms=18, markeredgecolor="black",
            markeredgewidth=0.8, linestyle="none", label=r"$\hat\nu$ (best fit)")

    lbl_i = labels[0] if len(labels) > 0 else f"nuis {pair[0]}"
    lbl_j = labels[1] if len(labels) > 1 else f"nuis {pair[1]}"
    ax.set_xlabel(lbl_i, fontsize=14)
    ax.set_ylabel(lbl_j, fontsize=14)
    ax.set_title("Hessian covariance vs. profile-likelihood scan", fontsize=15)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    _save(fig, out_dir, f"hessian_bma_overlay_{label}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="BMA post-fit Hessian / covariance at ν̂.")
    p.add_argument("-c", "--cfg", required=True, help="YAML config (ensemble or single-model).")
    p.add_argument("--ensemble", default=None,
                   help="Glob of bootstrap member checkpoints. Omit for a single-model (K=1) Hessian.")
    p.add_argument("--ensemble-mode", choices=["bma", "rebased-bma"], default="bma",
                   help="Member weighting. 'bma' = softmax of the absolute per-member likelihood "
                        "(collapses to one member when depth offsets dominate, K_eff→1). "
                        "'rebased-bma' = strip each member's own depth offset first "
                        "(matches likelihood_scan.py --ensemble-mode rebased-bma).")
    p.add_argument("--ckpt", default=None,
                   help="Shared residual/profiled checkpoint (default: config output_checkpoint).")
    p.add_argument("--dataset", default=None, help="Override dataset .pt (default: config paths.dataset).")
    p.add_argument("--out-dir", default="hessian_bma", help="Output directory.")
    p.add_argument("--label", default=None, help="Name for outputs (default: stage).")
    # ν̂ source (precedence: --best > --scan2d > --bestfit-json > model.m_vector)
    p.add_argument("--best", default=None, help="Explicit ν̂: comma list of length num_nuisances.")
    p.add_argument("--scan2d", default=None, help="Take ν̂ from the argmin of a saved 2D scan npz.")
    p.add_argument("--bestfit-json", default=None, help="Take ν̂ from a bestfit.json (nuisances[*].best_fit).")
    p.add_argument("--pair", default="0,1", help="Nuisance pair 'i,j' for --scan2d argmin and the overlay.")
    p.add_argument("--indices", default=None,
                   help="Comma list of nuisance indices for the Hessian (default: the profiled mask).")
    p.add_argument("--batch-size", type=int, default=10_000)
    p.add_argument("--n-batches", type=int, default=None, help="Cap batches per member (default: all).")
    p.add_argument("--n-events", type=int, default=None, help="Subsample dataset (default: all).")
    p.add_argument("--no-shape-constraint", dest="shape_constraint", action="store_false",
                   help="Drop the -0.5 m² shape prior (data-only Hessian).")
    p.add_argument("--verify", action="store_true",
                   help="Cross-check the decomposition against a direct autograd Hessian of S_BMA.")
    p.add_argument("--verify-events", type=int, default=20_000, help="Subsample size for --verify.")
    p.add_argument("--verify-tol", type=float, default=1e-4, help="Pass threshold on ‖ΔH‖_F/‖H‖_F.")
    p.add_argument("--overlay-scan2d", default=None,
                   help="Draw the C_BMA 1σ/2σ ellipse over this 2D scan's contour (needs d=2).")
    p.add_argument("--overlay-eigenvectors", action="store_true",
                   help="Also draw the C_BMA principal axes (±1σ segments) on the overlay.")
    args = p.parse_args()

    # --- config / device (mirrors likelihood_scan.py:1055-1066) ---
    cfg_path = os.path.abspath(args.cfg)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(cfg_path)
    for key, val in list(cfg["paths"].items()):
        if isinstance(val, str):
            cfg["paths"][key] = _resolve(val, cfg_dir)
    device = cfg.get("runtime", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        device = "cpu"

    # --- models ---
    if args.ensemble:
        res_ckpt = args.ckpt or cfg["paths"].get("output_checkpoint")
        print(f"Ensemble residual checkpoint: {res_ckpt}")
        model, residual_T, member_Ts, stage = load_ensemble(cfg, res_ckpt, args.ensemble, device)
    else:
        ckpt_path = args.ckpt or cfg["paths"]["output_checkpoint"]
        print(f"Checkpoint: {ckpt_path}")
        model, T, residual_T, stage = load_models(cfg, ckpt_path, device)
        member_Ts = [T]                       # K=1: residual_T (if any) swaps this single base
    K = len(member_Ts)

    # --- dataset (mirrors likelihood_scan.py:1111-1117) ---
    dataset_path = args.dataset or cfg["paths"]["dataset"]
    dataset = torch.load(dataset_path, map_location=device)
    y_all = dataset["y_data_distorted"].to(device)
    X_all = dataset["X_data_distorted"].to(device)
    if args.n_events is not None and args.n_events < y_all.shape[0]:
        perm = torch.randperm(y_all.shape[0])[:args.n_events]
        y_all, X_all = y_all[perm], X_all[perm]

    # --- profiled-nuisance subset ---
    num_nuis = model.m_vector.shape[0]
    if args.indices is not None:
        mask_list = [int(v) for v in args.indices.split(",")]
    else:
        mask = model.m_vector_profile_mask.detach().cpu().numpy()
        mask_list = [i for i in range(num_nuis) if bool(mask[i])]
    if not mask_list:
        raise SystemExit("No profiled nuisances to differentiate (empty mask). Use --indices.")
    mask_idx = torch.tensor(mask_list, device=device, dtype=torch.long)
    d = len(mask_list)
    labels_full = list(cfg.get("plotting", {}).get("shape_nuisance_labels",
                       [f"nuis {i}" for i in range(num_nuis)]))
    labels = [labels_full[i] for i in mask_list]
    label = args.label or stage

    nu_hat, nu_source = _resolve_nu_hat(args, model, num_nuis, device)
    n_batches = args.n_batches if args.n_batches is not None else 10 ** 9

    mode = args.ensemble_mode if K > 1 else "bma"   # weighting is moot for a single member
    print(f"\nStage: {stage} | members K={K} | mode={mode} | params d={d} {labels}")
    print(f"ν̂ ({nu_source}): {np.array2string(nu_hat.detach().cpu().numpy(), precision=5)}")
    print(f"Events: {y_all.shape[0]}  shape-constraint: {args.shape_constraint}")

    # --- the decomposition ---
    res = compute_bma_hessian(model, residual_T, member_Ts, y_all, X_all, nu_hat, mask_idx,
                              args.batch_size, n_batches, device,
                              include_shape=args.shape_constraint, mode=mode)
    H_BMA, H_within, Cov_w = res["H_BMA"], res["H_within"], res["Cov_w"]
    w, g_bar = res["w"], res["g_bar"]

    C_BMA, evH, evecH, npd = _safe_inv(H_BMA, name="H_BMA")
    C_within, _, _, _ = _safe_inv(H_within, name="H_within")
    sigmas = torch.sqrt(torch.diag(C_BMA).clamp(min=0))
    sig_within = torch.sqrt(torch.diag(C_within).clamp(min=0))
    corr = C_BMA / torch.outer(sigmas, sigmas).clamp(min=1e-30)
    inflation = C_BMA - C_within
    cov_evals, cov_evecs = torch.linalg.eigh(C_BMA)

    # Stationarity check: the implied Newton shift from ν̂ to the BMA optimum, in σ units.
    newton = C_BMA @ g_bar
    newton_rel = (newton.abs() / sigmas.clamp(min=1e-30))
    g_bar_norm = float(g_bar.norm().item())

    # --- printed summary ---
    np.set_printoptions(precision=4, suppress=True)
    print("\n=== BMA post-fit covariance ===")
    for i, lab in enumerate(labels):
        infl = (sigmas[i] / sig_within[i]).item() if sig_within[i] > 0 else float("nan")
        print(f"  {lab:>16s}:  σ_BMA = {sigmas[i].item():.5f}   "
              f"σ_within = {sig_within[i].item():.5f}   inflation ×{infl:.3f}")
    if d == 2:
        print(f"  correlation ρ = {corr[0, 1].item():+.3f}")
    print(f"  H_BMA eigenvalues: {evH.cpu().numpy()}  (PD: {npd == 0})")
    print(f"  ‖ḡ‖ = {g_bar_norm:.3e}   implied Newton shift = "
          f"{newton_rel.max().item():.3f}σ (max over nuisances)")
    if newton_rel.max().item() > 0.1:
        print("  ⚠ ν̂ may be off the BMA minimum (Newton shift > 0.1σ): use a finer "
              f"scan2d (matching --ensemble-mode {mode}) or pass --best.")
    keff = (1.0 / (w ** 2).sum()).item()
    print(f"  {mode} weights w_b (K={K}): {np.array2string(w.cpu().numpy(), precision=4)}  "
          f"(K_eff = {keff:.2f})")
    if mode == "bma" and K > 1 and keff < 0.5 * K:
        print(f"  ⚠ bma weighting collapsed (K_eff={keff:.2f}≪{K}): the absolute per-member depth "
              "offsets dominate, so this is ~a single member's Hessian (no T-inflation). "
              "Use --ensemble-mode rebased-bma to match a rebased-bma scan.")

    # --- optional verification ---
    verify_info = None
    if args.verify:
        ne = min(args.verify_events, y_all.shape[0])
        ys, xs = y_all[:ne], X_all[:ne]
        # The direct autograd Hessian goes through the plain-bma logsumexp, so verify the
        # decomposition in 'bma' mode (this validates the per-member g̃_b/H̃_b + combination;
        # 'rebased-bma' only reweights those same validated pieces analytically).
        H_full = _verify_full(model, residual_T, member_Ts, ys, xs, nu_hat, mask_idx,
                              args.batch_size, device, include_shape=args.shape_constraint)
        res_sub = compute_bma_hessian(model, residual_T, member_Ts, ys, xs, nu_hat, mask_idx,
                                      args.batch_size, 10 ** 9, device,
                                      include_shape=args.shape_constraint, mode="bma")
        H_dec = res_sub["H_BMA"]
        dH = H_full - H_dec
        max_abs = float(dH.abs().max().item())
        rel_fro = float(dH.norm().item() / max(H_dec.norm().item(), 1e-30))
        passed = rel_fro < args.verify_tol
        print(f"\n=== Verify (n={ne}, bma-mode machinery check): decomposition vs direct autograd(−S_BMA) ===")
        print(f"  max|ΔH| = {max_abs:.3e}   ‖ΔH‖_F/‖H‖_F = {rel_fro:.3e}   "
              f"{'PASS' if passed else 'FAIL'} (tol {args.verify_tol:g})")
        verify_info = {"n_events": ne, "max_abs": max_abs, "rel_fro": rel_fro, "passed": passed}

    # --- save ---
    os.makedirs(args.out_dir, exist_ok=True)
    npz = {
        "H_BMA": H_BMA.cpu().numpy(), "H_within": H_within.cpu().numpy(),
        "Cov_w_g": Cov_w.cpu().numpy(), "C_BMA": C_BMA.cpu().numpy(),
        "C_within": C_within.cpu().numpy(), "inflation": inflation.cpu().numpy(),
        "sigmas": sigmas.cpu().numpy(), "sigmas_within": sig_within.cpu().numpy(),
        "correlation": corr.cpu().numpy(),
        "H_eigvals": evH.cpu().numpy(), "H_eigvecs": evecH.cpu().numpy(),
        "cov_eigvals": cov_evals.cpu().numpy(), "cov_eigvecs": cov_evecs.cpu().numpy(),
        "weights": w.cpu().numpy(), "g_bar": g_bar.cpu().numpy(),
        "g_bar_norm": np.array(g_bar_norm), "newton_shift_sigma": newton_rel.cpu().numpy(),
        "nu_hat": nu_hat.cpu().numpy(), "indices": np.array(mask_list),
        "labels": np.array(labels, dtype=object), "stage": np.array(stage),
        "nu_source": np.array(nu_source), "h_bma_pd": np.array(npd == 0),
        "mode": np.array(mode),
    }
    if verify_info is not None:
        npz["verify_rel_fro"] = np.array(verify_info["rel_fro"])
        npz["verify_max_abs"] = np.array(verify_info["max_abs"])
        npz["verify_passed"] = np.array(verify_info["passed"])
    np.savez(os.path.join(args.out_dir, f"hessian_bma_{label}.npz"), **npz)

    summary = {
        "stage": stage, "mode": mode, "nu_source": nu_source, "K": K,
        "nu_hat": [float(v) for v in nu_hat.cpu().numpy()],
        "indices": mask_list, "labels": labels,
        "sigma_bma": {labels[i]: float(sigmas[i]) for i in range(d)},
        "sigma_within": {labels[i]: float(sig_within[i]) for i in range(d)},
        "inflation_factor": {labels[i]: (float(sigmas[i] / sig_within[i]) if sig_within[i] > 0 else None)
                             for i in range(d)},
        "correlation": corr.cpu().numpy().tolist(),
        "H_BMA_eigvals": evH.cpu().numpy().tolist(),
        "h_bma_positive_definite": bool(npd == 0),
        "g_bar_norm": g_bar_norm,
        "newton_shift_sigma_max": float(newton_rel.max().item()),
        "weights": w.cpu().numpy().tolist(),
        "k_eff": float((1.0 / (w ** 2).sum()).item()),
        "verify": verify_info,
    }
    with open(os.path.join(args.out_dir, f"hessian_bma_{label}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved hessian_bma_{label}.npz / .json to {args.out_dir}/")

    # --- optional ellipse overlay ---
    if args.overlay_scan2d is not None:
        if d != 2:
            print("  ⚠ overlay needs exactly 2 profiled nuisances; skipping.")
        else:
            pair = tuple(int(v) for v in args.pair.split(","))
            if tuple(mask_list) != pair:
                print(f"  ⚠ overlay --pair {pair} differs from the Hessian indices "
                      f"{tuple(mask_list)}; ellipse assumes index order {tuple(mask_list)}.")
                pair = tuple(mask_list)
            nu_pair = nu_hat.index_select(0, mask_idx).cpu().numpy()
            _plot_overlay(args.overlay_scan2d, pair, C_BMA.cpu().numpy(), C_within.cpu().numpy(),
                          nu_pair, labels, args.out_dir, label,
                          draw_eigenvectors=args.overlay_eigenvectors)


if __name__ == "__main__":
    main()
