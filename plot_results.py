"""
Post-training analysis plots for a mixture or profiling model.

Produces:
  1. Data vs model score histograms (per flavour, per score component, binned in X)
  2. 1D NLL scans for each nuisance parameter
  3. Residual transfer map visualization (profiling model only)

Usage:
    # After train_mixture.py:
    python plot_results.py -c configs/mixture_v9.yaml --ckpt models/full_mixture_model_shift_dataset9_systematics_v9a.pt

    # After train_profiling.py (profiling model):
    python plot_results.py -c configs/profiling_v9.yaml --ckpt models/full_mixture_model_shift_dataset9_systematics_v9a_profiled_condor.pt --profiling

    # Save to a specific directory:
    python plot_results.py -c configs/mixture_v9.yaml --out-dir plots/
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import yaml
import zuko

from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

from lib import FullMixtureModel, TransferModel
from residual_flow import SystematicCorrectedModel
from plotting import visualize_residual_map, field_quiver_panel, truncate_cmap
from utils import load_state_dict_checked


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_path(path: str, cfg_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(cfg_dir, path))


def _to_tensor(value, device, dtype=torch.float32):
    return torch.as_tensor(value, device=device, dtype=dtype)


def _auto_range(*arrs, pct=(0.5, 99.5), pad_frac=0.05):
    """Percentile range over the concatenation of all arrays, with a small pad."""
    v = np.concatenate([np.asarray(a).ravel() for a in arrs if a is not None and len(a)])
    lo, hi = np.percentile(v, pct)
    pad = pad_frac * (hi - lo + 1e-6)
    return float(lo - pad), float(hi + pad)


def _save_both(out_path, dpi=120):
    """Save the current figure as both .png and .pdf (`out_path` is the .png path)."""
    base = os.path.splitext(out_path)[0]
    for ext in (".png", ".pdf"):
        plt.savefig(base + ext, dpi=dpi, bbox_inches="tight")
    print(f"Saved {base}.png / .pdf")


# ---------------------------------------------------------------------------
# Batched evaluation helpers (cap peak memory for large n_mc / n_scan)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _rsample_batched(model, n, T_model, device, batch, m_value=None):
    """model.rsample in chunks of `batch`, moved to numpy each chunk.

    Returns (scores [n, F], kin [n, Z], fl [n]) as numpy arrays. Statistically
    equivalent to a single rsample(n) but caps peak GPU memory (sampling expands
    per flavour internally). `batch <= 0` falls back to a single call.
    `m_value` (nuisance vector [num_nuis]) overrides the sampling point; defaults
    to the best-fit `model.m_vector`."""
    nuis = model.m_vector.shape[0]
    mv = model.m_vector if m_value is None else m_value
    if batch is None or batch <= 0:
        batch = n
    s_l, k_l, f_l = [], [], []
    done = 0
    while done < n:
        b = min(batch, n - done)
        m = torch.ones((b, nuis), device=device) * mv
        s, k, f = model.rsample(b, T_model, m)
        s_l.append(s.detach().cpu().numpy())
        k_l.append(k.detach().cpu().numpy())
        f_l.append(f.detach().cpu().numpy())
        done += b
    return np.concatenate(s_l), np.concatenate(k_l), np.concatenate(f_l)


@torch.no_grad()
def _logp_sum(model, y, X, T_model, m, batch):
    """Σ_i log p(y_i, X_i | m_i) over the dataset in chunks of `batch`.

    Exact (not stochastic) — only the summation is chunked — so NLL scans are
    unchanged while peak memory (log_prob expands ×n_flavours) stays bounded."""
    if batch is None or batch <= 0:
        batch = y.shape[0]
    total = 0.0
    for i in range(0, y.shape[0], batch):
        logp, _ = model.log_prob(y[i:i + batch], X[i:i + batch], T_model, m[i:i + batch])
        total += float(logp.sum().item())
    return total


# ---------------------------------------------------------------------------
# Model loading (mirrors train_profiling.py / train_mixture.py)
# ---------------------------------------------------------------------------

def _load_mixture_model(cfg, device):
    score_cfg = cfg["score_flow"]
    score_base = zuko.flows.NSF(
        features=score_cfg["features"],
        context=score_cfg["context"],
        bins=score_cfg["bins"],
        transforms=score_cfg["transforms"],
        hidden_features=tuple(score_cfg["hidden_features"]),
    ).to(device)

    # v12 has no score-space systematic. If `residual_score_model` is absent, wrap the
    # score flow in an identity SystematicCorrectedModel with dims from score_flow.
    res_score_cfg = cfg.get("residual_score_model") or {
        "features_dim": score_cfg["features"],
        "context_dim": score_cfg["context"],
        "num_nuisances": 0,
        "num_residual_layers": 1,
        "hidden_features": [64, 64],
        "type": "flow",
    }
    residual_score = SystematicCorrectedModel(
        score_base,
        features_dim=res_score_cfg["features_dim"],
        context_dim=res_score_cfg["context_dim"],
        num_nuisances=res_score_cfg["num_nuisances"],
        num_residual_layers=res_score_cfg["num_residual_layers"],
        hidden_features=res_score_cfg["hidden_features"],
        type=res_score_cfg["type"],
    ).to(device)

    kin_cfg = cfg["kin_flow"]
    kin_base = zuko.flows.NSF(
        features=kin_cfg["features"],
        context=kin_cfg["context"],
        bins=kin_cfg["bins"],
        transforms=kin_cfg["transforms"],
        hidden_features=tuple(kin_cfg["hidden_features"]),
    ).to(device)

    res_kin_cfg = cfg["residual_kin_model"]
    residual_kin = SystematicCorrectedModel(
        kin_base,
        features_dim=res_kin_cfg["features_dim"],
        context_dim=res_kin_cfg["context_dim"],
        num_nuisances=res_kin_cfg["num_nuisances"],
        num_residual_layers=res_kin_cfg["num_residual_layers"],
        hidden_features=res_kin_cfg["hidden_features"],
        type=res_kin_cfg["type"],
    ).to(device)

    mix_cfg = cfg["mixture_model"]
    _lnN_mix = mix_cfg.get("lnN_mix_matrix", None)
    lnN_mix_matrix = _to_tensor(_lnN_mix, device) if _lnN_mix is not None else None
    _profile_mask = mix_cfg.get("norm_nuisance_profile_mask", None)
    norm_nuisance_profile_mask = (
        torch.as_tensor(_profile_mask, dtype=torch.bool, device=device)
        if _profile_mask is not None else None
    )
    _m_mask = mix_cfg.get("m_vector_profile_mask", None)
    m_vector_profile_mask = (
        torch.as_tensor(_m_mask, dtype=torch.bool, device=device)
        if _m_mask is not None else None
    )
    model = FullMixtureModel(
        features_dim=mix_cfg["features_dim"],
        n_flavours=mix_cfg["n_flavours"],
        num_nuisances=mix_cfg["num_nuisances"],
        norm_factors=_to_tensor(mix_cfg["norm_factors"], device),
        scores_model=residual_score,
        kin_model=residual_kin,
        lnN_constraints=_to_tensor(mix_cfg["lnN_constraints"], device),
        fit_conditional_pdf=bool(mix_cfg["fit_conditional_pdf"]),
        lnN_mix_matrix=lnN_mix_matrix,
        norm_nuisance_profile_mask=norm_nuisance_profile_mask,
        m_vector_profile_mask=m_vector_profile_mask,
    ).to(device)

    transfer_cfg = cfg["transfer_model"]
    T = TransferModel(
        features_dim=transfer_cfg["features_dim"],
        context_dim=transfer_cfg["context_dim"],
        n_transforms=transfer_cfg["n_transforms"],
        nbins=transfer_cfg["nbins"],
        hidden_net=transfer_cfg["hidden_net"],
        add_rotation=bool(transfer_cfg["add_rotation"]),
    ).to(device)

    return model, T


def _load_profiling_model(cfg, model, T, device):
    res_T_cfg = cfg["residual_transfer_model"]
    residual_T = SystematicCorrectedModel(
        T,
        features_dim=res_T_cfg["features_dim"],
        context_dim=res_T_cfg["context_dim"],
        num_nuisances=res_T_cfg["num_nuisances"],
        num_residual_layers=res_T_cfg["num_residual_layers"],
        hidden_features=res_T_cfg["hidden_features"],
        type=res_T_cfg["type"],
        central_nuisance_values=model.m_vector,
        nuisance_scales=res_T_cfg["nuisance_scales"],
        quadratic_damping=float(res_T_cfg["quadratic_damping"]),
    ).to(device)
    return residual_T


# ---------------------------------------------------------------------------
# Plot 1: Data vs model histograms
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_histograms(model, T_model, dataset, device, out_path, n_scan=50_000, batch_size=50_000):
    """Marginal distributions: stacked MC per flavour vs data (integrated over X)."""
    y_data = dataset["y_data_distorted"].to(device)
    X_data = dataset["X_data_distorted"].to(device)
    N_data = y_data.shape[0]

    y_np = y_data.cpu().numpy()
    X_np = X_data.cpu().numpy()

    n_mc = n_scan
    scores_mc, kin_mc, fl_mc = _rsample_batched(model, n_mc, T_model, device, batch_size)
    kin_x1    = kin_mc[:, 0]                   # bin on x₁

    w         = N_data / n_mc
    colors_fl = ["tab:red", "tab:blue", "tab:green", "tab:orange"]

    # Auto-ranged bins: y is sigmoid (0,1)² in v12, x is centred near ±0.5.
    panels = [
        (y_np[:, 0], scores_mc[:, 0], np.linspace(*_auto_range(y_np[:, 0], scores_mc[:, 0]), 61), "y₁", "score dim 0"),
        (y_np[:, 1], scores_mc[:, 1], np.linspace(*_auto_range(y_np[:, 1], scores_mc[:, 1]), 61), "y₂", "score dim 1"),
        (X_np[:, 0], kin_x1,          np.linspace(*_auto_range(X_np[:, 0], kin_x1),          61), "x₁", "kinematic x₁"),
    ]

    fig = plt.figure(figsize=(15, 7))
    gs  = GridSpec(2, 3, figure=fig, height_ratios=[3, 1], hspace=0.05)

    for col, (data_arr, mc_arr, bins, xlabel, title) in enumerate(panels):
        ax       = fig.add_subplot(gs[0, col])
        ax_ratio = fig.add_subplot(gs[1, col], sharex=ax)
        bc = 0.5 * (bins[:-1] + bins[1:])
        bottoms    = np.zeros(len(bins) - 1)
        h_mc_total = np.zeros(len(bins) - 1)

        for fl in range(model.n_flavours):
            mask = fl_mc == fl
            h, _ = np.histogram(mc_arr[mask], bins=bins,
                                 weights=np.full(mask.sum(), w))
            ax.bar(bins[:-1], h, width=np.diff(bins), bottom=bottoms,
                   align='edge', color=colors_fl[fl % len(colors_fl)],
                   alpha=0.7, label=f"MC flav {fl}")
            bottoms    += h
            h_mc_total += h

        h_d, _ = np.histogram(data_arr, bins=bins)
        err_d   = np.sqrt(np.maximum(h_d, 1))
        ax.errorbar(bc, h_d, yerr=err_d,
                    fmt="ko", ms=3, lw=1, zorder=5, label="Data")
        ax.set_ylabel("Events")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.tick_params(labelbottom=False)

        safe  = np.where(h_mc_total > 0, h_mc_total, np.nan)
        ratio = h_d / safe
        ax_ratio.errorbar(bc, ratio, yerr=err_d / safe,
                          fmt="ko", ms=2, lw=1)
        ax_ratio.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax_ratio.fill_between(bc, 0.9, 1.1, alpha=0.1, color="gray")
        ax_ratio.set_ylim(0., 2.)
        ax_ratio.set_xlabel(xlabel)
        if col == 0:
            ax_ratio.set_ylabel("Data/MC", fontsize=9)
        ax_ratio.tick_params(labelsize=8)

    plt.suptitle(
        f"Marginal distributions  [ν = {model.m_vector.detach().cpu().numpy().round(4)}]",
        fontsize=12,
    )
    plt.tight_layout()
    _save_both(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: 1D NLL scans
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_nll_scans(model, T_model, dataset, device, out_path, n_scan=50_000, n_steps=60,
                   shape_labels=None, batch_size=50_000):
    y_data = dataset["y_data_distorted"].to(device)
    X_data = dataset["X_data_distorted"].to(device)

    # n_scan must match training.n_events: the trainer's per-epoch NLL sums logp over
    # exactly n_events with the prior at full strength (train_mixture.py:297,301),
    # so any other n shifts the MAP relative to the trained best fit.
    n = min(n_scan, y_data.shape[0])
    idx = torch.randperm(y_data.shape[0])[:n]
    y_obs = y_data[idx]
    X_obs = X_data[idx]
    N_data_64     = torch.tensor(n, dtype=torch.float64, device=device)
    log_N_data_64 = torch.log(N_data_64)

    num_shape_nuis = model.m_vector.shape[0]
    num_lnN_nuis   = model.norm_nuisance.shape[0]

    # Skip frozen nuisances — their NLL is flat by construction (mask zeros them out
    # before they reach the loss), so the scan would just show the prior.
    m_mask = model.m_vector_profile_mask.detach().cpu().numpy().astype(bool)
    lnN_mask = model.norm_nuisance_profile_mask.detach().cpu().numpy().astype(bool)
    active_shape = [i for i in range(num_shape_nuis) if m_mask[i]]
    active_lnN   = [i for i in range(num_lnN_nuis)   if lnN_mask[i]]
    n_panels = len(active_shape) + len(active_lnN)

    if n_panels == 0:
        print("No active (unfrozen) nuisances — skipping NLL scan plot.")
        return

    best_fit_m   = model.m_vector.detach().clone()
    best_fit_lnN = model.norm_nuisance.detach().clone()

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    m_fixed = best_fit_m.unsqueeze(0).expand(n, -1).clone()

    def _apply_nll_panel(ax, x_vals, nlls_delta2, best_x, title, xlabel):
        """Plot a -2Δln L profile and auto-zoom x to the ±3σ window."""
        ax.plot(x_vals, nlls_delta2)
        ax.axvline(best_x, color="C1", ls="--", label=f"best fit = {best_x:.4f}")
        ax.axhline(1, color="gray", lw=0.5, ls=":")
        ax.axhline(4, color="gray", lw=0.5, ls=":")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("−2 Δln L")
        in_window = x_vals[nlls_delta2 < 9]
        if len(in_window) >= 2:
            margin = (in_window[-1] - in_window[0]) * 0.10
            ax.set_xlim(in_window[0] - margin, in_window[-1] + margin)
        ax.set_ylim(0, 15)
        ax.legend()

    def _estimate_sigma(scan_fn, center, sigma_hint):
        """Probe -ln L locally with a 3-point Hessian, return σ at the minimum.

        Uses ε = 5% of the prior σ; small enough to stay in the quadratic regime
        for typical data-tightened posteriors, large enough to be above numerical
        noise. Falls back to sigma_hint when the local curvature is degenerate.
        """
        eps = max(0.05 * float(sigma_hint), 1e-3)
        nll_0 = scan_fn(center)
        nll_p = scan_fn(center + eps)
        nll_m = scan_fn(center - eps)
        # curvature = d²(-ln L)/dx² ≈ (nll(+eps) + nll(-eps) - 2 nll(0)) / eps²
        # σ from the −2Δln L = 1 convention: σ = 1/sqrt(curvature).
        curvature = (nll_p + nll_m - 2.0 * nll_0) / (eps * eps)
        if curvature > 0:
            return float(1.0 / np.sqrt(curvature))
        return float(sigma_hint)

    def _scan_around(scan_fn, center, half_range, n_steps_local):
        deltas = torch.linspace(-half_range, half_range, n_steps_local, device=device)
        nlls = np.array([scan_fn(center + float(d.item())) for d in deltas])
        nlls = nlls - nlls.min()
        return center + deltas.cpu().numpy(), 2.0 * nlls

    plot_idx = 0

    # Shape-nuisance labels: caller passes cfg["plotting"]["shape_nuisance_labels"] when
    # available. v12 default is [ν_shift, ν_rot]; per-index ν_<i> fallback below.
    if shape_labels is None:
        shape_labels = {0: "ν_shift", 1: "ν_rot"}
    elif isinstance(shape_labels, list):
        shape_labels = {i: lbl for i, lbl in enumerate(shape_labels)}

    # --- shape nuisance scans (m_vector). Prior: 0.5 * m² → σ_prior = 1 ---
    for ni in active_shape:
        def _scan_shape(m_val, _ni=ni):
            m_scan = m_fixed.clone()
            m_scan[:, _ni] = m_val
            constraint = 0.5 * float((m_scan[0] ** 2).sum().item())
            return -_logp_sum(model, y_obs, X_obs, T_model, m_scan, batch_size) + constraint

        center = best_fit_m[ni].item()
        sigma  = _estimate_sigma(_scan_shape, center, sigma_hint=1.0)
        x_vals, nll2 = _scan_around(_scan_shape, center, 5.0 * sigma, n_steps)
        slabel = shape_labels.get(ni, f"ν_{ni}")
        _apply_nll_panel(axes[plot_idx], x_vals, nll2, center,
                         f"1D NLL scan — {slabel} (shape)", slabel)
        plot_idx += 1

    # --- lnN nuisance scans (norm_nuisance). Prior σ = lnN_constraints[ni] ---
    lnN_labels = {0: "θ_glob", 1: "θ_ratio"}
    for ni in active_lnN:
        sigma_prior = float(model.lnN_constraints[ni].item()) or 0.1

        def _scan_lnN(theta_val, _ni=ni):
            model.norm_nuisance.data[_ni] = theta_val
            logp_sum = _logp_sum(model, y_obs, X_obs, T_model, m_fixed, batch_size)
            log_nu  = torch.logsumexp(model.modified_log_normalization, dim=0).double()
            poisson = (torch.exp(log_nu) - N_data_64 - N_data_64 * (log_nu - log_N_data_64)).item()
            lnN_pen = (-model.get_lnN_likelihood_term()).double().item()
            return -logp_sum + poisson + lnN_pen

        center = best_fit_lnN[ni].item()
        sigma  = _estimate_sigma(_scan_lnN, center, sigma_hint=sigma_prior)
        x_vals, nll2 = _scan_around(_scan_lnN, center, 5.0 * sigma, n_steps)
        model.norm_nuisance.data[ni] = best_fit_lnN[ni]  # restore

        label = lnN_labels.get(ni, f"θ_{ni}")
        _apply_nll_panel(axes[plot_idx], x_vals, nll2, center,
                         f"1D NLL scan — {label} (lnN)", label)
        plot_idx += 1

    plt.tight_layout()
    _save_both(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Residual transfer map (profiling model only)
# ---------------------------------------------------------------------------

def plot_residual_maps(residual_T, out_path):
    num_nuis = residual_T.num_nuisances
    fig, axes = plt.subplots(1, num_nuis * 2, figsize=(9 * num_nuis, 7))
    if num_nuis * 2 == 1:
        axes = [axes]

    col = 0
    for ni in range(num_nuis):
        for delta, sign in [(+0.1, "+"), (-0.1, "−")]:
            visualize_residual_map(
                residual_T,
                delta=delta,
                nuisance_idx=ni,
                ax=axes[col],
                title=f"Residual T  ν{ni} {sign}0.1",
            )
            col += 1

    plt.tight_layout()
    _save_both(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Model vs data — stacked MC per X-bin (sampled from mixture)
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_model_vs_data(model, T_model, dataset, device, out_path, n_mc=200_000, batch_size=50_000):
    y_data  = dataset["y_data_distorted"].to(device)
    X_data  = dataset["X_data_distorted"].to(device)
    N_data  = y_data.shape[0]

    scores_mc, _kin, fl_mc = _rsample_batched(model, n_mc, T_model, device, batch_size)
    kin_mc = _kin[:, 0]    # bin on x₁ (kin is [n_mc, 2] in v12)

    x_data_np = X_data[:, 0].cpu().numpy()
    y_data_np = y_data.cpu().numpy()

    x_edges      = np.array([-3., -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, 3.])
    nbins_y      = 30
    y_range      = _auto_range(y_data_np, scores_mc)   # sigmoid (0,1)² in v12
    weight_scale = N_data / n_mc
    colors_fl    = ["tab:red", "tab:blue", "tab:green", "tab:orange"]

    n_xbins = len(x_edges) - 1
    ncols   = 4
    n_figs  = int(np.ceil(n_xbins / ncols))

    for fig_idx in range(n_figs):
        bin_range = range(fig_idx * ncols, min((fig_idx + 1) * ncols, n_xbins))
        fig, axes = plt.subplots(2, len(bin_range), figsize=(6 * len(bin_range), 10))
        if len(bin_range) == 1:
            axes = axes[:, None]

        for col_idx, xb in enumerate(bin_range):
            xl, xr = x_edges[xb], x_edges[xb + 1]
            mask_d = (x_data_np >= xl) & (x_data_np < xr)
            mask_m = (kin_mc    >= xl) & (kin_mc    < xr)
            n_d    = mask_d.sum()
            n_m    = mask_m.sum()
            w      = n_d / n_m if n_m > 0 else weight_scale

            bins = np.linspace(*y_range, nbins_y + 1)
            bc   = 0.5 * (bins[:-1] + bins[1:])

            for row, dim in enumerate([0, 1]):
                ax = axes[row, col_idx]
                bottoms = np.zeros(nbins_y)
                for fl in range(model.n_flavours):
                    y_fl = scores_mc[mask_m & (fl_mc == fl), dim]
                    h, _ = np.histogram(y_fl, bins=bins,
                                        weights=np.full(y_fl.shape, w))
                    ax.bar(bins[:-1], h, width=np.diff(bins), bottom=bottoms,
                           align='edge',
                           color=colors_fl[fl % len(colors_fl)], alpha=0.7,
                           label=f"Flav {fl}")
                    bottoms += h

                h_d, _ = np.histogram(y_data_np[mask_d, dim], bins=bins)
                ax.errorbar(bc, h_d, yerr=np.sqrt(np.maximum(h_d, 1)),
                            fmt="ko", ms=3, lw=1, zorder=5, label="Data")
                ax.set_yscale("log")
                ax.set_xlim(y_range)
                ax.set_xlabel(f"y{'₁' if dim == 0 else '₂'}")
                ax.set_title(f"x∈[{xl:.1f},{xr:.1f})  y{'₁' if dim == 0 else '₂'}")
                if col_idx == 0:
                    ax.set_ylabel("Events")
                if col_idx == 0 and row == 0:
                    ax.legend(fontsize=8)

        plt.suptitle(f"Model vs data — stacked MC  (part {fig_idx + 1})", fontsize=12)
        plt.tight_layout()
        stem = os.path.splitext(os.path.basename(out_path))[0]
        part_path = os.path.join(os.path.dirname(out_path),
                                 f"{stem}_part{fig_idx + 1}.png")
        _save_both(part_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4b: Transfer map T(y; c, x) — quiver of y → z faceted by class and x-bin
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_transfer_quiver(model, T_model, dataset, device, out_path,
                          n_ybins=16, x1_centers=None, grid_pct=(0.5, 99.5),
                          cmap_name="viridis", cmap_floor=0.18):
    """Quiver of the transfer map y → T(y; c, x), faceted by class (rows) and
    x₁-bin centre (cols). x₂ is fixed at the data median. Evaluated at the
    best-fit m_vector so the residual correction (if present) is included.

    The score grid spans the real (pre-sigmoid) data score range per dimension (from the
    `grid_pct` percentiles), NOT the sigmoid box [0,1]. Styled like
    visualize_hessian_distortion.py: a shared |z−y| heatmap background per panel
    (brightened colormap) with white black-outlined unit-length direction arrows on top,
    and a single shared colorbar — so the flow direction reads in every cell and the
    magnitude is comparable across panels."""
    y_data = dataset["y_data_distorted"]
    X_data = dataset["X_data_distorted"]
    if x1_centers is None:
        x1_centers = [-1.5, -0.5, 0.5, 1.5]
    x2_med = float(X_data[:, 1].median().item())
    y_np = y_data.cpu().numpy()

    n_cols = len(x1_centers)
    n_rows = model.n_flavours
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.6 * n_rows),
                              squeeze=False)

    # Grid over the real score range per dimension (pre-sigmoid; not the [0,1] box).
    rng = [_auto_range(y_np[:, d], pct=grid_pct, pad_frac=0.08) for d in range(2)]
    g0 = np.linspace(rng[0][0], rng[0][1], n_ybins)
    g1 = np.linspace(rng[1][0], rng[1][1], n_ybins)
    YY1, YY2 = np.meshgrid(g0, g1)
    y_grid = torch.tensor(np.stack([YY1.ravel(), YY2.ravel()], 1),
                          dtype=torch.float32, device=device)
    N = y_grid.shape[0]
    m_vec = model.m_vector.detach().unsqueeze(0).expand(N, -1)

    # First pass: evaluate every panel and collect (u, v) + the global |z−y| max so all
    # panels share one normalisation / colorbar.
    uv = {}
    for cl in range(n_rows):
        c_onehot = torch.zeros(N, model.n_flavours, device=device)
        c_onehot[:, cl] = 1.0
        for col, xc1 in enumerate(x1_centers):
            x_ctx = torch.tensor([xc1, x2_med], device=device,
                                  dtype=torch.float32).unsqueeze(0).expand(N, -1)
            context = torch.cat([c_onehot, x_ctx], dim=1)
            z, _ = T_model(y_grid, context, m_vec)
            u = (z[:, 0] - y_grid[:, 0]).reshape(n_ybins, n_ybins).cpu().numpy()
            v = (z[:, 1] - y_grid[:, 1]).reshape(n_ybins, n_ybins).cpu().numpy()
            uv[(cl, col)] = (u, v)

    speed_max = max(float(np.sqrt(u * u + v * v).max()) for u, v in uv.values())
    norm = Normalize(0.0, speed_max if speed_max > 0 else 1.0)
    cmap = truncate_cmap(cmap_name, cmap_floor)
    cell = min(rng[0][1] - rng[0][0], rng[1][1] - rng[1][0]) / max(n_ybins - 1, 1)
    arrow_len = 1.6 * cell   # ~1.6 grid cells

    for cl in range(n_rows):
        for col, xc1 in enumerate(x1_centers):
            u, v = uv[(cl, col)]
            ax = axes[cl, col]
            field_quiver_panel(ax, g0, g1, YY1, YY2, u, v, norm, cmap,
                               background=True, arrows="unit", arrow_len=arrow_len)
            ax.set_aspect("equal")
            ax.set_title(f"flav {cl} | x=({xc1:+.2f}, {x2_med:+.2f})", fontsize=9)
            if col == 0:
                ax.set_ylabel("y₂")
            if cl == n_rows - 1:
                ax.set_xlabel("y₁")

    fig.suptitle(
        f"Transfer map T(y; c, x):  arrows = direction of z − y, colour = |z − y|   "
        f"[ν = {model.m_vector.detach().cpu().numpy().round(3)}]",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    cax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="|z−y|")
    _save_both(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4c: Transfer field T(y; c, x) — per-class arrows over the score space,
#          on the data density, faceted by X interval
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_transfer_field_classes(model, T_model, dataset, device, out_path,
                                 x_intervals=None, n_grid=22, n_density=4000,
                                 grid_pct=(0.5, 99.5), pad_frac=0.15, arrow_scale=1.0,
                                 dens_thresh=0.02):
    """Per-class transfer vector field T(y; c, x) over the (pre-sigmoid) score space,
    one panel per X interval. Class-0 (A) red and class-1 (B) blue arrows show z − y on a
    grid; the data score density in that X bin is a faint gray scatter behind. Evaluated at
    the best-fit m_vector (residual correction included). Unlike plot_transfer_quiver (which
    grids the sigmoid box [0,1]), the grid spans the real data score range per dimension.

    The frame is zoomed out (`pad_frac`) and arrows are drawn ONLY where the in-bin data
    density exceeds `dens_thresh × peak` (a KDE mask), so the field traces the populated
    region and the empty margin stays clean."""
    y_data = dataset["y_data_distorted"].to(device)
    X_data = dataset["X_data_distorted"].to(device)
    y_np = y_data.cpu().numpy()
    x1_np = X_data[:, 0].cpu().numpy()
    x2_med = float(X_data[:, 1].median().item())

    if x_intervals is None:
        x_intervals = [(-2.0, -1.0), (-0.5, 0.5), (1.0, 2.0)]
    colors_fl = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    cls_name = ["A", "B", "C", "D"]

    # Score-space grid range from the data percentiles (per dim), widened by `pad_frac`
    # to zoom out so the populated region sits with margin around it.
    rng = [_auto_range(y_np[:, d], pct=grid_pct, pad_frac=pad_frac) for d in range(2)]
    g0 = np.linspace(rng[0][0], rng[0][1], n_grid)
    g1 = np.linspace(rng[1][0], rng[1][1], n_grid)
    YY1, YY2 = np.meshgrid(g0, g1)
    y_grid = torch.tensor(np.stack([YY1.ravel(), YY2.ravel()], 1),
                          dtype=torch.float32, device=device)
    N = y_grid.shape[0]
    m_vec = model.m_vector.detach().unsqueeze(0).expand(N, -1)

    ncols = len(x_intervals)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 5.0), squeeze=False)
    rng_g = np.random.default_rng(0)

    for col, (xl, xr) in enumerate(x_intervals):
        ax = axes[0][col]
        in_bin = (x1_np >= xl) & (x1_np < xr)
        yb_all = y_np[in_bin]
        # Representative x for the arrows: data mean of x₁ within the bin (fallback: mid).
        x1c = float(x1_np[in_bin].mean()) if in_bin.any() else 0.5 * (xl + xr)

        # Data density (gray scatter), subsampled.
        yb = yb_all if len(yb_all) <= n_density else yb_all[rng_g.choice(len(yb_all), n_density, replace=False)]
        ax.scatter(yb[:, 0], yb[:, 1], s=5, c="0.6", alpha=0.18, linewidths=0,
                   zorder=0, label="Data Density")

        # Density mask: KDE of the in-bin data at the grid points → keep only cells above
        # dens_thresh × peak, so no arrows are drawn over the empty (low-data) margin.
        keep = np.ones(N, dtype=bool)
        sub = yb_all if len(yb_all) <= 4000 else yb_all[rng_g.choice(len(yb_all), 4000, replace=False)]
        if len(sub) >= 10:
            try:
                dens = gaussian_kde(sub.T)(np.vstack([YY1.ravel(), YY2.ravel()]))
                keep = dens >= dens_thresh * float(dens.max())
            except np.linalg.LinAlgError:
                pass
        gx, gy = YY1.ravel()[keep], YY2.ravel()[keep]

        # Per-class transfer arrows (z − y), drawn only where there is data.
        for cl in range(model.n_flavours):
            c_onehot = torch.zeros(N, model.n_flavours, device=device)
            c_onehot[:, cl] = 1.0
            x_ctx = torch.tensor([x1c, x2_med], device=device,
                                 dtype=torch.float32).unsqueeze(0).expand(N, -1)
            context = torch.cat([c_onehot, x_ctx], dim=1)
            z, _ = T_model(y_grid, context, m_vec)
            u = (z[:, 0] - y_grid[:, 0]).cpu().numpy()
            v = (z[:, 1] - y_grid[:, 1]).cpu().numpy()
            ax.quiver(gx, gy, u[keep], v[keep],
                      color=colors_fl[cl % len(colors_fl)], angles="xy",
                      scale_units="xy", scale=1.0 / arrow_scale, width=0.004,
                      alpha=0.9, zorder=2 + cl, label=f"Class {cls_name[cl % len(cls_name)]}")

        ax.set_xlim(*rng[0]); ax.set_ylim(*rng[1])
        ax.set_xlabel("y₁")
        if col == 0:
            ax.set_ylabel("y₂")
        ax.set_title(rf"$T_{{\hat\nu}}(y\,|\,x,f)$:  X $\in$ [{xl:g}, {xr:g})", fontsize=11)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.2)

    nu_str = ", ".join(f"{v:.3f}" for v in model.m_vector.detach().cpu().numpy())
    fig.suptitle(f"Transfer map per class over score space   [ν = ({nu_str})]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_both(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: Fit quality with Data/MC ratio panels
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_fit_quality(model, T_model, dataset, device, out_path, n_mc=200_000, batch_size=50_000):
    y_data  = dataset["y_data_distorted"].to(device)
    X_data  = dataset["X_data_distorted"].to(device)
    N_data  = y_data.shape[0]

    scores_mc, _kin, fl_mc = _rsample_batched(model, n_mc, T_model, device, batch_size)
    kin_mc = _kin[:, 0]    # bin on x₁ (kin is [n_mc, 2] in v12)

    x_data_np    = X_data[:, 0].cpu().numpy()
    y_data_np    = y_data.cpu().numpy()
    weight_scale = N_data / n_mc
    colors_fl    = ["tab:red", "tab:blue"]

    x_edges   = np.array([-3., -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, 3.])
    # Show 5 central X bins
    xbins_show = [1, 2, 3, 4, 5]
    ncols      = len(xbins_show)
    nbins_y    = 25
    y_range    = _auto_range(y_data_np, scores_mc)   # sigmoid (0,1)² in v12

    fig = plt.figure(figsize=(5 * ncols, 14))
    gs  = GridSpec(4, ncols, figure=fig, height_ratios=[3, 1, 3, 1], hspace=0.05)
    bins_y = np.linspace(*y_range, nbins_y + 1)
    bc     = 0.5 * (bins_y[:-1] + bins_y[1:])

    for col_idx, xb in enumerate(xbins_show):
        xl, xr = x_edges[xb], x_edges[xb + 1]
        mask_d = (x_data_np >= xl) & (x_data_np < xr)
        mask_m = (kin_mc    >= xl) & (kin_mc    < xr)
        n_d    = mask_d.sum()
        n_m    = mask_m.sum()
        w      = n_d / n_m if n_m > 0 else weight_scale

        for row_pair, dim in enumerate([0, 1]):
            ax_main  = fig.add_subplot(gs[row_pair * 2,     col_idx])
            ax_ratio = fig.add_subplot(gs[row_pair * 2 + 1, col_idx])

            bottoms    = np.zeros(nbins_y)
            h_mc_total = np.zeros(nbins_y)
            for fl in range(model.n_flavours):
                y_fl = scores_mc[mask_m & (fl_mc == fl), dim]
                h, _ = np.histogram(y_fl, bins=bins_y, weights=np.full(y_fl.shape, w))
                ax_main.bar(bins_y[:-1], h, width=np.diff(bins_y), bottom=bottoms,
                            align='edge',
                            color=colors_fl[fl % len(colors_fl)], alpha=0.7,
                            label=f"Flav {fl}")
                bottoms    += h
                h_mc_total += h

            h_d, _ = np.histogram(y_data_np[mask_d, dim], bins=bins_y)
            err_d   = np.sqrt(np.maximum(h_d, 1))
            ax_main.errorbar(bc, h_d, yerr=err_d, fmt="ko", ms=3, lw=1,
                             zorder=5, label="Data")
            ax_main.set_xlim(y_range)
            ax_main.set_title(f"x∈[{xl:.1f},{xr:.1f})  y{'₁' if dim == 0 else '₂'}",
                              fontsize=9)
            if col_idx == 0:
                ax_main.set_ylabel("Events", fontsize=9)
            ax_main.tick_params(labelbottom=False)
            if col_idx == 0 and row_pair == 0:
                ax_main.legend(fontsize=7)

            safe  = np.where(h_mc_total > 0, h_mc_total, np.nan)
            ratio = h_d / safe
            ax_ratio.errorbar(bc, ratio, yerr=err_d / safe, fmt="ko", ms=2, lw=1)
            ax_ratio.axhline(1.0, color="gray", lw=0.8, ls="--")
            ax_ratio.set_ylim(0., 2.)
            ax_ratio.set_xlim(y_range)
            ax_ratio.set_xlabel(f"y{'₁' if dim == 0 else '₂'}", fontsize=9)
            if col_idx == 0:
                ax_ratio.set_ylabel("Data/MC", fontsize=9)

    plt.suptitle(
        f"Fit quality — stacked MC vs data  "
        f"[ν = {model.m_vector.detach().cpu().numpy().round(4)}]",
        fontsize=11,
    )
    _save_both(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6: Per-X-bin validation — truth (by class) vs model sample (by flavour)
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_validation(model, T_model, dataset, device, out_path, n_mc=200_000, batch_size=50_000,
                    y_ranges=None):
    """For each X bin and each score component (y₁, y₂):
      - stacked model sample per flavour (filled bars)  ← rsample at best-fit ν
      - truth per true class (step lines)               ← data split by label c
      - data (points)                                   ← total distorted data
    plus a ratio panel: Data/Sample_total (points) and True_fl/Sample_fl (lines).
    """
    if "c" not in dataset:
        print("Dataset has no truth class label 'c' — skipping validation plot.")
        return
    y_data = dataset["y_data_distorted"].to(device)
    X_data = dataset["X_data_distorted"].to(device)
    c_raw  = dataset["c"]
    c_t    = c_raw.reshape(c_raw.shape[0], -1)
    c_true = (c_t.argmax(1) if c_t.shape[1] > 1 else c_t.reshape(-1)).long().cpu().numpy()
    N_data = y_data.shape[0]

    scores_mc, _kin, fl_mc = _rsample_batched(model, n_mc, T_model, device, batch_size)
    kin_x1    = _kin[:, 0]   # bin on x₁ (kin is [n_mc, 2])

    y_np   = y_data.cpu().numpy()
    x1_np  = X_data[:, 0].cpu().numpy()
    weight = N_data / n_mc                    # global scale: tests rate + shape

    # Nominal expected overlay: the SAME model sampled at ν=0, in data space (full
    # forward chain — kin/score residuals at identity, base transfer T still applied),
    # binned by its OWN sampled x₁ and on the SAME global weight. This is consistent
    # with the best-fit "Sample" curve (only ν differs), unlike the previous raw
    # pre-distortion dataset['y'] binned by nominal X (different space + different X
    # population per bin under the X-shifting systematic).
    nuis = model.m_vector.shape[0]
    nom_scores, _nom_kin, _ = _rsample_batched(
        model, n_mc, T_model, device, batch_size,
        m_value=torch.zeros(nuis, device=device))
    nom_x1 = _nom_kin[:, 0]

    x_edges   = np.array([-5., -1.5, -0.5, 0.5, 1.5, 5.])
    ncols     = len(x_edges) - 1
    nbins_y   = 30
    colors_fl = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    cls_name  = ["A", "B", "C", "D"]

    # X-axis range for the score components y₁, y₂. v15 scores are PRE-SIGMOID
    # (unbounded). Derive a robust range per dimension from the DATA + best-fit model
    # (the two distributions actually compared), at the 99% interval. The nominal (ν=0)
    # overlay is EXCLUDED from the range and the percentile tightened from (0.2,99.8) to
    # (0.5,99.5): the nominal's heavier pre-sigmoid tails (reaching ±6–7) otherwise
    # over-widen the axis and squash the bulk. The nominal line is still drawn (clipped at
    # the edges). Every X-bin column shares this common axis — pin it via `--val-yrange`.
    if y_ranges is None:
        y_ranges = [
            _auto_range(y_np[:, d], scores_mc[:, d], pct=(0.5, 99.5), pad_frac=0.05)
            for d in range(2)
        ]

    fig = plt.figure(figsize=(4.2 * ncols, 11.5))
    # Two independent blocks (y₁ on top, y₂ below) with a clear gap between them, so the
    # lower block's y-tick numbers don't collide with the upper block's ratio panel.
    outer = fig.add_gridspec(2, 1, hspace=0.28)

    for d in range(2):
        block = outer[d].subgridspec(2, ncols, height_ratios=[3, 1], hspace=0.05, wspace=0.22)
        bins = np.linspace(*y_ranges[d], nbins_y + 1)
        bc   = 0.5 * (bins[:-1] + bins[1:])
        for col in range(ncols):
            xl, xr = x_edges[col], x_edges[col + 1]
            mask_d = (x1_np  >= xl) & (x1_np  < xr)
            mask_m = (kin_x1 >= xl) & (kin_x1 < xr)

            ax  = fig.add_subplot(block[0, col])
            axr = fig.add_subplot(block[1, col], sharex=ax)

            # truth per class (data split by true label) — STACKED step lines, so the
            # outermost line traces the truth total (= the data points).
            h_true = [np.histogram(y_np[mask_d & (c_true == fl), d], bins=bins)[0]
                      for fl in range(model.n_flavours)]
            cum_true = np.zeros(nbins_y)
            for fl in range(model.n_flavours):
                cum_true = cum_true + h_true[fl]
                ax.stairs(cum_true, bins, color=colors_fl[fl], lw=1.5, zorder=3,
                          label=f"True {cls_name[fl]}")

            # model sample per flavour — stacked filled bars
            h_samp  = []
            bottoms = np.zeros(nbins_y)
            for fl in range(model.n_flavours):
                sel = mask_m & (fl_mc == fl)
                h_s, _ = np.histogram(scores_mc[sel, d], bins=bins,
                                      weights=np.full(int(sel.sum()), weight))
                h_samp.append(h_s)
                ax.bar(bins[:-1], h_s, width=np.diff(bins), bottom=bottoms,
                       align="edge", color=colors_fl[fl], alpha=0.45, zorder=1,
                       label=f"Sample {cls_name[fl]}")
                bottoms += h_s

            # nominal expected (model at ν=0) — gray dashed line, same binning/weight
            # as the best-fit "Sample" curve so the systematic pull is read off directly
            mask_o = (nom_x1 >= xl) & (nom_x1 < xr)
            h_o, _ = np.histogram(nom_scores[mask_o, d], bins=bins,
                                  weights=np.full(int(mask_o.sum()), weight))
            ax.stairs(h_o, bins, color="0.4", lw=1.3, ls="--", zorder=2,
                      label="Model ν=0 (nominal)")

            # data (total distorted)
            h_d, _ = np.histogram(y_np[mask_d, d], bins=bins)
            err     = np.sqrt(np.maximum(h_d, 1))
            ax.errorbar(bc, h_d, yerr=err, fmt="ko", ms=2.5, lw=0.8, zorder=5, label="Data")

            ax.set_xlim(*y_ranges[d])
            ax.tick_params(labelbottom=False)
            if d == 0:
                ax.set_title(f"X = [{xl:g}, {xr:g}]", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Events (y{'₁' if d == 0 else '₂'})")
                ax.legend(fontsize=7, loc="upper right")

            # ratio panel: Data/Sample_total (points), True_fl/Sample_fl (lines)
            tot  = np.sum(h_samp, axis=0)
            safe = np.where(tot > 0, tot, np.nan)
            axr.errorbar(bc, h_d / safe, yerr=err / safe, fmt="ko", ms=2, lw=0.7, zorder=5)
            for fl in range(model.n_flavours):
                s = np.where(h_samp[fl] > 0, h_samp[fl], np.nan)
                axr.step(bc, h_true[fl] / s, where="mid", color=colors_fl[fl], lw=1.0)
            axr.axhline(1.0, color="gray", lw=0.7, ls="--")
            axr.set_ylim(0.3, 1.8)
            axr.set_yticks([0.5, 1.0, 1.5])
            axr.set_xlim(*y_ranges[d])
            axr.set_xlabel(f"y{'₁' if d == 0 else '₂'}")
            if col == 0:
                axr.set_ylabel("Ratio", fontsize=9)

    nu_str = ", ".join(f"{v:.3f}" for v in model.m_vector.detach().cpu().numpy())
    fig.suptitle(
        "Validation by X bin — truth (by class) vs model sample (by flavour)   "
        f"[ν = ({nu_str})]",
        fontsize=13, y=0.995,
    )
    _save_both(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Post-training analysis plots.")
    parser.add_argument("-c", "--cfg", required=True, help="YAML config used for training")
    parser.add_argument("--ckpt", default=None, help="Checkpoint .pt path (overrides config paths.output_checkpoint)")
    parser.add_argument("--profiling", action="store_true", help="Checkpoint is a profiling model (has residual_transfer_model)")
    parser.add_argument("--out-dir", default=None, help="Directory for output PNGs (default: same dir as checkpoint)")
    parser.add_argument("--n-scan", type=int, default=None,
                        help="Events for histogram / NLL scan. Defaults to cfg.training.n_events "
                             "so the scan matches the training-time NLL convention.")
    parser.add_argument("--n-mc",   type=int, default=200_000, help="MC samples for model-vs-data plots")
    parser.add_argument("--batch-size", type=int, default=50_000,
                        help="Chunk size for rsample / log_prob evaluation to cap peak memory "
                             "(<=0 evaluates all at once). Results are unchanged for the NLL scans "
                             "and statistically equivalent for the sampled histograms.")
    parser.add_argument("--val-yrange", default=None,
                        help="Fix the validation-plot score x-axis range as "
                             "'y1lo,y1hi,y2lo,y2hi'. Default: auto from data+MC percentiles.")
    args = parser.parse_args()

    val_yranges = None
    if args.val_yrange:
        vals = [float(v) for v in args.val_yrange.split(",")]
        if len(vals) != 4:
            parser.error("--val-yrange needs exactly 4 comma-separated floats: y1lo,y1hi,y2lo,y2hi")
        val_yranges = [(vals[0], vals[1]), (vals[2], vals[3])]

    cfg_path = os.path.abspath(args.cfg)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg_dir = os.path.dirname(cfg_path)
    for key, value in list(cfg["paths"].items()):
        if isinstance(value, str):
            cfg["paths"][key] = _resolve_path(value, cfg_dir)

    device = cfg.get("runtime", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.n_scan is None:
        args.n_scan = int(cfg.get("training", {}).get("n_events", 50000))
        print(f"n_scan defaulted to training.n_events = {args.n_scan}")

    ckpt_path = args.ckpt or cfg["paths"]["output_checkpoint"]
    out_dir = args.out_dir or os.path.dirname(ckpt_path)
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(ckpt_path))[0]

    model, T = _load_mixture_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device)

    # error=False: warn (don't abort) on key mismatches so older checkpoints still plot,
    # but a silent residual-architecture mismatch (e.g. residual_score_model not matching
    # the trained config) is surfaced loudly rather than producing wrong plots.
    if args.profiling:
        load_state_dict_checked(model, ckpt["mixture_model"], "mixture_model", error=False)
        load_state_dict_checked(T, ckpt["transfer_model"], "transfer_model", error=False)
        residual_T = _load_profiling_model(cfg, model, T, device)
        load_state_dict_checked(residual_T, ckpt["residual_transfer_model"],
                                "residual_transfer_model", error=False)
        residual_T.eval()
        T_model = residual_T
    else:
        load_state_dict_checked(model, ckpt["mixture_model"], "mixture_model", error=False)
        load_state_dict_checked(T, ckpt["transfer_model"], "transfer_model", error=False)
        T_model = T

    model.eval()
    T.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Best-fit nuisances: {model.m_vector.detach().cpu().numpy()}")
    print(f"lnN norm nuisance: {model.norm_nuisance.detach().cpu().numpy()}")

    dataset = torch.load(cfg["paths"]["dataset"], map_location=device)

    plot_histograms(
        model, T_model, dataset, device,
        out_path=os.path.join(out_dir, f"{stem}_histograms.png"),
        n_scan=args.n_scan, batch_size=args.batch_size,
    )

    plot_nll_scans(
        model, T_model, dataset, device,
        out_path=os.path.join(out_dir, f"{stem}_nll_scans.png"),
        n_scan=args.n_scan,
        shape_labels=cfg.get("plotting", {}).get("shape_nuisance_labels"),
        batch_size=args.batch_size,
    )

    plot_model_vs_data(
        model, T_model, dataset, device,
        out_path=os.path.join(out_dir, f"{stem}_model_vs_data.png"),
        n_mc=args.n_mc, batch_size=args.batch_size,
    )

    plot_fit_quality(
        model, T_model, dataset, device,
        out_path=os.path.join(out_dir, f"{stem}_fit_quality.png"),
        n_mc=args.n_mc, batch_size=args.batch_size,
    )

    plot_validation(
        model, T_model, dataset, device,
        out_path=os.path.join(out_dir, f"{stem}_validation.png"),
        n_mc=args.n_mc, batch_size=args.batch_size, y_ranges=val_yranges,
    )

    plot_transfer_quiver(
        model, T_model, dataset, device,
        out_path=os.path.join(out_dir, f"{stem}_transfer_quiver.png"),
    )

    plot_transfer_field_classes(
        model, T_model, dataset, device,
        out_path=os.path.join(out_dir, f"{stem}_transfer_field.png"),
    )

    if args.profiling:
        plot_residual_maps(
            residual_T,
            out_path=os.path.join(out_dir, f"{stem}_residual_maps.png"),
        )


if __name__ == "__main__":
    main()
