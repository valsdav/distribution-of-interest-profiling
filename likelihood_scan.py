"""
Standalone 1D / 2D likelihood scans over the shape nuisances (`m_vector`).

Works for both stages of the analysis:
  * mixture model  — checkpoint has {mixture_model, transfer_model}; the scan
                     varies `m` only through the (frozen) kinematic residual flow.
  * profiled model — checkpoint additionally has `residual_transfer_model`; the
                     scan also feels the systematic-aware residual on the scores.

The script auto-detects which case it is from the checkpoint keys, so the same
command works for either — just point `-c/--ckpt` at the matching config/.pt.

Usage
-----
    # mixture model
    python likelihood_scan.py \
        -c configs/mixture_v13.yaml \
        --ckpt models/full_mixture_model_v13.pt \
        --out-dir scans/mixture_v13 --label mixture --scan-2d

    # profiled model
    python likelihood_scan.py \
        -c configs/profiling_v13.yaml \
        --ckpt models/full_mixture_model_v13_profiled.pt \
        --out-dir scans/profiled_v13 --label profiled --scan-2d

    # overlay previously-saved 1D scans for comparison
    python likelihood_scan.py --overlay \
        scans/mixture_v13/scan1d.npz scans/profiled_v13/scan1d.npz \
        --out-dir scans/compare

    # replot only: regenerate every plot from the saved scan{1,2}d*.npz in --out-dir
    # (no model / dataset / scan needed — fast restyle of existing results)
    python likelihood_scan.py --replot --out-dir scans/profiled_v13

Notes
-----
The scanned quantity is -2 Δ ln L where, at each point, the nuisance vector `m`
is held fixed (no re-profiling). By default the Gaussian constraint on the shape
nuisances (-0.5 m², the `shape_nuis` term of the training loss) is included, so
the scan minimum lines up with the best fit found during training. Pass
`--no-shape-constraint` for the data-only likelihood. The lnN normalisation term
is constant in an `m`-only scan (and is frozen to 0 in v13), so it only shifts
the curve, not its shape.
"""
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import yaml
import zuko

from lib import FullMixtureModel, TransferModel
from residual_flow import SystematicCorrectedModel


# Teal used for the "expected ν" marker/line across all scan plots.
EXPECTED_COLOR = "#17becf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(path: str, cfg_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(cfg_dir, path))


def _to_tensor(v, device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(v, device=device, dtype=dtype)


def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved {name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loading (v13-robust: tolerates a missing residual_score_model = identity)
# ---------------------------------------------------------------------------

def _build_transfer(cfg: Dict, device: str) -> TransferModel:
    tc = cfg["transfer_model"]
    return TransferModel(
        features_dim=tc["features_dim"], context_dim=tc["context_dim"],
        n_transforms=tc["n_transforms"], nbins=tc["nbins"],
        hidden_net=tc["hidden_net"], add_rotation=bool(tc["add_rotation"]),
    ).to(device)


def _build_structures(cfg: Dict, device: str):
    """Build FullMixtureModel (+ frozen kin/score template submodels) and a base
    TransferModel, weights UNLOADED. Shared by load_models and load_ensemble."""
    score_cfg = cfg["score_flow"]
    score_base = zuko.flows.NSF(
        features=score_cfg["features"], context=score_cfg["context"],
        bins=score_cfg["bins"], transforms=score_cfg["transforms"],
        hidden_features=tuple(score_cfg["hidden_features"]),
    ).to(device)

    # When `residual_score_model` is absent, wrap the score flow in an identity
    # SystematicCorrectedModel (num_nuisances=0) sized from score_flow.
    res_sc = cfg.get("residual_score_model") or {
        "features_dim": score_cfg["features"],
        "context_dim": score_cfg["context"],
        "num_nuisances": 0,
        "num_residual_layers": 1,
        "hidden_features": [64, 64],
        "type": "flow",
    }
    residual_score = SystematicCorrectedModel(
        score_base, features_dim=res_sc["features_dim"], context_dim=res_sc["context_dim"],
        num_nuisances=res_sc["num_nuisances"], num_residual_layers=res_sc["num_residual_layers"],
        hidden_features=res_sc["hidden_features"], type=res_sc["type"],
    ).to(device)

    kin_cfg = cfg["kin_flow"]
    kin_base = zuko.flows.NSF(
        features=kin_cfg["features"], context=kin_cfg["context"],
        bins=kin_cfg["bins"], transforms=kin_cfg["transforms"],
        hidden_features=tuple(kin_cfg["hidden_features"]),
    ).to(device)

    res_kn = cfg["residual_kin_model"]
    residual_kin = SystematicCorrectedModel(
        kin_base, features_dim=res_kn["features_dim"], context_dim=res_kn["context_dim"],
        num_nuisances=res_kn["num_nuisances"], num_residual_layers=res_kn["num_residual_layers"],
        hidden_features=res_kn["hidden_features"], type=res_kn["type"],
    ).to(device)

    mix_cfg = cfg["mixture_model"]
    _profile_mask = mix_cfg.get("norm_nuisance_profile_mask")
    norm_nuisance_profile_mask = (
        torch.as_tensor(_profile_mask, dtype=torch.bool, device=device)
        if _profile_mask is not None else None
    )
    _m_mask = mix_cfg.get("m_vector_profile_mask")
    m_vector_profile_mask = (
        torch.as_tensor(_m_mask, dtype=torch.bool, device=device)
        if _m_mask is not None else None
    )
    _lnN_mix = mix_cfg.get("lnN_mix_matrix")
    lnN_mix_matrix = _to_tensor(_lnN_mix, device) if _lnN_mix is not None else None
    model = FullMixtureModel(
        features_dim=mix_cfg["features_dim"], n_flavours=mix_cfg["n_flavours"],
        num_nuisances=mix_cfg["num_nuisances"],
        norm_factors=_to_tensor(mix_cfg["norm_factors"], device),
        scores_model=residual_score, kin_model=residual_kin,
        lnN_constraints=_to_tensor(mix_cfg["lnN_constraints"], device),
        fit_conditional_pdf=bool(mix_cfg["fit_conditional_pdf"]),
        lnN_mix_matrix=lnN_mix_matrix,
        norm_nuisance_profile_mask=norm_nuisance_profile_mask,
        m_vector_profile_mask=m_vector_profile_mask,
    ).to(device)

    T = _build_transfer(cfg, device)
    return model, T


def _build_residual_T(cfg: Dict, base_T: TransferModel, model, device: str):
    """Residual transfer SystematicCorrectedModel anchored at model.m_vector (= ν₀)."""
    rc = cfg["residual_transfer_model"]
    return SystematicCorrectedModel(
        base_T, features_dim=rc["features_dim"], context_dim=rc["context_dim"],
        num_nuisances=rc["num_nuisances"], num_residual_layers=rc["num_residual_layers"],
        hidden_features=rc["hidden_features"], type=rc["type"],
        central_nuisance_values=model.m_vector,
        nuisance_scales=rc["nuisance_scales"],
        quadratic_damping=float(rc["quadratic_damping"]),
        # Must match train_profiling.py: a crossterm checkpoint registers cross_term_pairs
        # of shape [n_pairs, 2]; omitting these builds an empty [0, 2] buffer and
        # load_state_dict fails on the size mismatch.
        cross_term_pairs=rc.get("cross_term_pairs"),
        cross_term_damping=float(rc.get("cross_term_damping", 0.01)),
    ).to(device)


def load_models(cfg: Dict, ckpt_path: str, device: str):
    """Build FullMixtureModel + TransferModel (+ residual_T if present in ckpt)."""
    model, T = _build_structures(cfg, device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["mixture_model"], strict=False)
    T.load_state_dict(ckpt["transfer_model"], strict=False)

    residual_T = None
    if "residual_transfer_model" in ckpt:
        residual_T = _build_residual_T(cfg, T, model, device)
        residual_T.load_state_dict(ckpt["residual_transfer_model"], strict=False)
        residual_T.eval()

    model.eval(); T.eval()
    stage = "profiled" if residual_T is not None else "mixture"
    return model, T, residual_T, stage


def load_ensemble(cfg: Dict, residual_ckpt_path: Optional[str], member_glob: str, device: str):
    """BMA loading: one shared FullMixtureModel (frozen templates) + K member transfer
    models {T_b} + an optional shared residual transfer R_ν.

    The data term depends only on the transfer (per member) and the frozen templates;
    `model.m_vector` is the residual anchor ν₀. With a profiled-ensemble residual
    checkpoint the anchor/residual come from it; otherwise (mixture-stage anchor BMA)
    the frozen templates and a representative anchor come from member 0.
    """
    import glob as _glob
    member_paths = sorted(_glob.glob(member_glob))
    if not member_paths:
        raise FileNotFoundError(f"No ensemble members matched: {member_glob}")

    model, _T_template = _build_structures(cfg, device)

    residual_T = None
    if residual_ckpt_path is not None and os.path.isfile(residual_ckpt_path):
        rckpt = torch.load(residual_ckpt_path, map_location=device)
        model.load_state_dict(rckpt["mixture_model"], strict=False)  # anchor m_vector = ν₀
        if "residual_transfer_model" in rckpt:
            residual_T = _build_residual_T(cfg, _T_template, model, device)
            residual_T.load_state_dict(rckpt["residual_transfer_model"], strict=False)
            residual_T.eval()
    else:
        # mixture-stage BMA: frozen templates + a representative anchor from member 0.
        m0 = torch.load(member_paths[0], map_location=device)
        model.load_state_dict(m0["mixture_model"], strict=False)

    member_Ts: List[TransferModel] = []
    for p in member_paths:
        ck = torch.load(p, map_location=device)
        Tb = _build_transfer(cfg, device)
        Tb.load_state_dict(ck["transfer_model"], strict=False)
        for q in Tb.parameters():
            q.requires_grad = False
        Tb.eval()
        member_Ts.append(Tb)

    model.eval()
    stage = "ensemble_profiled" if residual_T is not None else "ensemble_mixture"
    print(f"Ensemble: {len(member_Ts)} members matched {member_glob}")
    return model, residual_T, member_Ts, stage


# ---------------------------------------------------------------------------
# Likelihood evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _data_loglik_sum(model, T_model, y_all, X_all, m_vec, batch_size) -> float:
    """Sum_i log p(data_i | m_vec) over the whole dataset for a fixed `m_vec`."""
    N = y_all.shape[0]
    total = 0.0
    for b in range(0, N, batch_size):
        y_b = y_all[b:b + batch_size]
        X_b = X_all[b:b + batch_size]
        m_b = m_vec.unsqueeze(0).expand(y_b.shape[0], -1)
        lp, _ = model.log_prob(y_b, X_b, T_model, m_b)
        total += lp.sum().item()
    return total


@torch.no_grad()
def _member_logliks(model, residual_T, member_Ts, y_all, X_all, m_vec, batch_size) -> np.ndarray:
    """Per-member summed-over-events DATA log-likelihood at fixed `m_vec` -> ndarray[K].

    Each member swaps its (frozen) base transfer into the shared residual; the residual
    parameters are untouched. The per-member curves are reduced afterwards by
    `_combine_grid` (bma / envelope / rebased-*) — collecting the raw grid first is what
    lets the rebased modes shift each member to its OWN minimum before combining, and
    lets a single scan feed all four modes.
    """
    out = np.empty(len(member_Ts), dtype=np.float64)
    for b, T_b in enumerate(member_Ts):
        if residual_T is not None:
            residual_T.base_model = T_b
            T_model = residual_T
        else:
            T_model = T_b
        out[b] = _data_loglik_sum(model, T_model, y_all, X_all, m_vec, batch_size)
    return out


# The four ensemble combine modes (+ "all" runs every one from a single scan).
ENSEMBLE_MODES = ["bma", "envelope", "rebased-bma", "rebased-envelope"]


# Figure-facing labels for the combine modes (internal names above stay for CLI/filenames).
# 'rebased-*' = each member aligned to its own minimum before combining.
_MODE_BASE = {"bma": "ensemble average", "envelope": "ensemble envelope",
              "rebased-bma": "ensemble average", "rebased-envelope": "ensemble envelope"}


def _mode_label(mode: str, inline: bool = False) -> str:
    """Display label for a combine mode. `inline=True` returns the form used inside
    'combined (…)' (', aligned') to avoid nested parentheses; otherwise '(aligned)'."""
    base = _MODE_BASE.get(mode, mode)
    if not mode.startswith("rebased"):
        return base
    return f"{base}, aligned" if inline else f"{base} (aligned)"


def _combine_member_logL(total: np.ndarray, mode: str) -> np.ndarray:
    """Combine per-member TOTAL log-likelihoods [*grid, K] -> one logL curve [*grid].

    Base combination (suffix):
      'bma'       Bayesian model average — logsumexp_b[lnL_b] - ln K (soft-OR; the
                  broadening average of the likelihoods, NOT a mean of NLLs).
      'envelope'  HEP profiled scan — max_b lnL_b (hard-OR; profile the member index).

    'rebased-' prefix: first subtract each member's OWN grid-max (rebase every member to
    its own minimum) so the ν-INDEPENDENT per-member depth offsets — overfitting /
    calibration differences that otherwise let one member dominate (K_eff→1) — are
    removed before combining. 'rebased-bma' is then the equal-height Gaussian mixture
    (width √(σ²+s²)); 'rebased-envelope' is the union of the per-member intervals.
    """
    if mode not in ENSEMBLE_MODES:
        raise ValueError(f"unknown combine mode '{mode}' (use one of {ENSEMBLE_MODES} or 'all')")
    a = total
    if mode.startswith("rebased"):
        grid_axes = tuple(range(a.ndim - 1))               # all axes except the member axis
        a = a - a.max(axis=grid_axes, keepdims=True)        # rebase each member to its own min
    K = a.shape[-1]
    if mode.endswith("envelope"):
        return a.max(axis=-1)
    mx = a.max(axis=-1, keepdims=True)
    return mx[..., 0] + np.log(np.exp(a - mx).sum(axis=-1)) - np.log(K)


def _combine_grid(member_logL: np.ndarray, constraint: np.ndarray, mode: str):
    """Reduce a per-member grid of data log-likelihoods to (m2dnll, nll_min).

      member_logL : [*grid, K]  per-member summed-over-events DATA log-likelihood
      constraint  : [*grid]     member-independent prior term (shape + lnN)

    Returns the rebased-to-global-min `-2Δln L` grid and the absolute `-2lnL` at that
    minimum (for absolute-depth reporting), both for the given combine `mode`."""
    total = member_logL + constraint[..., None]            # [*grid, K] per-member total logL
    combined = _combine_member_logL(total, mode)           # [*grid]
    cmax = float(combined.max())
    return -2.0 * (combined - cmax), -2.0 * cmax


def _member_curves(member_logL: np.ndarray, constraint: np.ndarray):
    """Per-member curves for the overlay plot (compare_scans '--absolute' style).

      member_logL : [*grid, K]  per-member summed-over-events DATA log-likelihood
      constraint  : [*grid]     member-independent prior term

    Returns (m2dnll_members [*grid, K], nll_min_members [K]) where each member is
    rebased to ITS OWN minimum (m2dnll_b ≥ 0) and nll_min_b = -2·max_grid lnL_b is the
    absolute -2lnL at that member's best fit. Plotting m2dnll_b + nll_min_b therefore
    shows the absolute -2lnL curve, so a deeper-fitting member bottoms out lower.
    """
    total = member_logL + constraint[..., None]            # [*grid, K]
    grid_axes = tuple(range(total.ndim - 1))
    mx = total.max(axis=grid_axes, keepdims=True)          # [1..1, K]
    m2dnll_members = -2.0 * (total - mx)
    nll_min_members = -2.0 * np.squeeze(mx, axis=grid_axes)  # [K]
    return m2dnll_members, np.atleast_1d(nll_min_members)


@torch.no_grad()
def _constraint_loglik(model, m_vec, include_shape: bool, include_lnN: bool) -> float:
    """Additive log-prior terms held alongside the data likelihood."""
    val = 0.0
    if include_lnN:
        # Uses the (frozen) norm_nuisance; constant in an m-only scan, 0 in v13.
        val += model.get_lnN_likelihood_term().item()
    if include_shape:
        mask = model.m_vector_profile_mask.detach()
        val += float((-0.5 * (m_vec[mask] ** 2).sum()).item())
    return val


# ---------------------------------------------------------------------------
# 1D scan
# ---------------------------------------------------------------------------

@torch.no_grad()
def scan_1d(model, member_loglik_fn, center, nuis_indices, half,
            n_steps, include_shape, include_lnN):
    """Returns dict[ni] -> (axis_abs[np], member_logL[n_steps,K], constraint[n_steps]).

    `member_loglik_fn(m_vec) -> ndarray[K]` returns the per-member summed-over-events
    data log-likelihood (K=1 for a single model). The raw per-member grid is returned
    uncombined so the caller can apply any/all combine modes (`_combine_grid`) and the
    per-member overlay (`_member_curves`) from a SINGLE scan."""
    results = {}
    for ni in nuis_indices:
        deltas = torch.linspace(-half[ni], half[ni], n_steps, device=center.device)
        member_logL = None
        constraint = np.zeros(n_steps)
        for ki, d in enumerate(deltas):
            m_vec = center.clone()
            m_vec[ni] = center[ni] + d
            ml = member_loglik_fn(m_vec)
            if member_logL is None:
                member_logL = np.zeros((n_steps, ml.shape[0]))
            member_logL[ki] = ml
            constraint[ki] = _constraint_loglik(model, m_vec, include_shape, include_lnN)
        axis = (center[ni].item() + deltas).cpu().numpy()
        results[ni] = (axis, member_logL, constraint)
    return results


def _sigma_crossings(axis, m2dnll, level):
    """1D linear-interpolated crossings of `m2dnll` through `level`."""
    roots = []
    diff = m2dnll - level
    for idx in np.where(np.diff(np.sign(diff)))[0]:
        x1, x2 = axis[idx], axis[idx + 1]
        y1, y2 = m2dnll[idx], m2dnll[idx + 1]
        roots.append(x1 + (level - y1) * (x2 - x1) / (y2 - y1))
    return roots


def estimate_1d(axis, m2dnll):
    """Best-fit value and asymmetric 1σ uncertainties from a 1D -2Δln L scan.

    The minimum is refined by a parabolic interpolation of the three grid points
    around the discrete argmin (uniform grid); the ±1σ bounds come from the
    -2Δln L = 1 crossings bracketing it. Bounds that fall outside the scanned
    window are returned as NaN (widen --half).
    """
    k = int(np.argmin(m2dnll))
    best = float(axis[k])
    if 0 < k < len(axis) - 1:
        y0, y1, y2 = m2dnll[k - 1], m2dnll[k], m2dnll[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom > 0:
            best = float(axis[k] + 0.5 * (y0 - y2) / denom * (axis[1] - axis[0]))
    roots = _sigma_crossings(axis, m2dnll, 1.0)
    lows = [r for r in roots if r < best]
    highs = [r for r in roots if r > best]
    sigma_lo = best - max(lows) if lows else float("nan")
    sigma_hi = min(highs) - best if highs else float("nan")
    return best, sigma_lo, sigma_hi


def _fmt_estimate(name, best, sigma_lo, sigma_hi):
    return f"{name} = {best:+.4f}  +{sigma_hi:.4f} / -{sigma_lo:.4f}"


def _write_anchor(out_dir, name, anchor, labels, stage, mode, source, sigmas=None):
    """Write a copy-paste-ready best-fit anchor txt next to the plots.

    `anchor` is the FULL nuisance vector (length num_nuisances): scanned indices hold
    the scan minimum, the rest stay at the scan centre. The file ends with the exact
    YAML line to paste into a profiling config's `mixture_model.m_vector_override`.
    `sigmas` (optional dict ni -> (lo, hi)) adds the asymmetric 1σ per scanned nuisance."""
    nvec = [float(v) for v in anchor]
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        f.write("# likelihood_scan best-fit anchor\n")
        f.write(f"# stage={stage}  mode={mode}  source={source}\n")
        for i, v in enumerate(nvec):
            lab = labels[i] if i < len(labels) else f"nuis {i}"
            if sigmas and i in sigmas and all(np.isfinite(s) for s in sigmas[i]):
                slo, shi = sigmas[i]
                f.write(f"{lab} = {v:+.6f}  +{shi:.6f} / -{slo:.6f}\n")
            else:
                f.write(f"{lab} = {v:+.6f}\n")
        vec_str = "[" + ", ".join(f"{v:+.6f}" for v in nvec) + "]"
        f.write("\n# paste under mixture_model: in the profiling config\n")
        f.write(f"m_vector_override: {vec_str}\n")
    print(f"  anchor     -> {name}   m_vector_override: {vec_str}")
    return path


def _parabolic_vertex(axis, y0, y1, y2, idx):
    """Sub-grid minimum of a parabola through 3 uniform points; idx must be interior."""
    denom = y0 - 2.0 * y1 + y2
    if denom > 0:
        return float(axis[idx] + 0.5 * (y0 - y2) / denom * (axis[1] - axis[0]))
    return float(axis[idx])


def refine_min_2d(axis_i, axis_j, m2dnll):
    """Best-fit point = minimum of the 2D -2Δln L grid, parabolically refined.

    Returns (xi, xj, on_boundary). `on_boundary` flags that the discrete minimum
    sits on the edge of the scanned window (so the true minimum may lie outside it).
    """
    a, b = np.unravel_index(int(np.argmin(m2dnll)), m2dnll.shape)  # a->axis_i, b->axis_j
    xi, xj = float(axis_i[a]), float(axis_j[b])
    if 0 < a < len(axis_i) - 1:
        xi = _parabolic_vertex(axis_i, m2dnll[a - 1, b], m2dnll[a, b], m2dnll[a + 1, b], a)
    if 0 < b < len(axis_j) - 1:
        xj = _parabolic_vertex(axis_j, m2dnll[a, b - 1], m2dnll[a, b], m2dnll[a, b + 1], b)
    on_boundary = a in (0, len(axis_i) - 1) or b in (0, len(axis_j) - 1)
    return xi, xj, on_boundary


def plot_scan_1d(results, labels, out_dir, tag, expected=None):
    for ni, (axis, m2dnll, *_rest) in results.items():
        best, slo, shi = estimate_1d(axis, m2dnll)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(axis, m2dnll, lw=2, color="#2166ac", label="Likelihood scan")
        ax.axvline(best, color="red", lw=1.5,
                   label=rf"$\hat\nu = {best:.4f}^{{+{shi:.4f}}}_{{-{slo:.4f}}}$")
        if expected is not None:
            ax.axvline(expected[ni], color=EXPECTED_COLOR, lw=1.8, ls="--",
                       label=rf"$\nu_\mathrm{{exp}}$ = {expected[ni]:.4f}")
        for sigma, ls in [(1, "--"), (2, ":")]:
            level = float(sigma ** 2)
            ax.axhline(level, color="gray", ls=ls, lw=1.3, alpha=0.8, label=f"{sigma}σ")
            for x_root in _sigma_crossings(axis, m2dnll, level):
                ax.vlines(x_root, 0, level, colors="gray", linestyles=":", lw=1.0, alpha=0.8)
        lbl = labels[ni] if ni < len(labels) else f"nuis {ni}"
        ax.set_xlabel(lbl, fontsize=13)
        ax.set_ylabel(r"$-2\,\Delta\log L$", fontsize=13)
        ax.set_ylim(0, max(9.0, float(m2dnll.max()) * 1.05))
        ax.legend(fontsize=9, loc="upper center")
        ax.set_title(f"1D scan — {lbl}")
        _save(fig, out_dir, f"scan1d_{tag}_nuis{ni}")


# ---------------------------------------------------------------------------
# Per-member ensemble overlay (compare_scans '--absolute' style)
# ---------------------------------------------------------------------------

COMBINED_COLOR = "#b2182b"      # the combined (bma / envelope) curve
MEMBER_CMAP = "viridis"         # ensemble members coloured by fit depth (−2lnL at min)


def _member_depth_colors(nll_min_members):
    """Per-member colour by fit depth RELATIVE to the best member: Δ(−2lnL) from the
    deepest member (0 = best). Plotting the relative depth keeps the colorbar ticks small
    and meaningful (0…spread, typically O(100)) instead of the unwieldy absolute −2lnL
    (~10⁶). A reversed map makes the best (deepest, most up-weighted) members bright AND
    keeps the colorbar consistent with the member colours without an axis flip. Returns
    (colors[K], norm, cmap) — `norm`/`cmap` drive a matching Δ(−2lnL) colorbar."""
    from matplotlib.colors import Normalize
    rel = np.asarray(nll_min_members, dtype=np.float64)
    rel = rel - rel.min()                          # 0 = best (deepest) member
    vmax = float(rel.max()) or 1e-9
    norm = Normalize(0.0, vmax)
    cmap = plt.get_cmap(MEMBER_CMAP + "_r")         # reversed: best (Δ=0) → bright end
    colors = [cmap(norm(v)) for v in rel]
    return colors, norm, cmap


MEMBER_SHOW = 12.0   # vertical window (−2Δln L units, ~3.5σ) shown in the per-member overlays


def _member_window(nll_min_members, combined_nll_min, rebased):
    """Vertical window for the overlay, zoomed to ~MEMBER_SHOW units of curve structure
    (NOT the member-depth spread, which can be hundreds and squashes the curves).

    rebased modes plot every member at its own minimum (all bottoming at 0) → window
    starts at 0; non-rebased plot absolute −2lnL → anchor at the global deepest point.
    Deep-offset members fall off the top in the non-rebased view (the colorbar still
    conveys their depth); that is the intended zoom-to-the-action."""
    if rebased:
        return -0.5, MEMBER_SHOW
    floor = float(min(float(np.min(nll_min_members)), combined_nll_min))
    return floor - 0.5, floor + MEMBER_SHOW


def _keff_grid(member_logL: np.ndarray, rebased: bool = False) -> np.ndarray:
    """Effective number of ensemble members under the BMA likelihood weighting,
    per grid point.  member_logL : [*grid, K]  ->  K_eff [*grid].

    Weights w_b ∝ exp(lnL_b) (the member-independent constraint cancels in the
    softmax, so only the per-member DATA log-likelihood enters). Kish ESS:
        K_eff = (Σ_b w_b)² / Σ_b w_b²  ∈ [1, K]
    K_eff ≈ K → all members contribute (the BMA averages over the full ensemble);
    K_eff ≈ 1 → one member dominates (the BMA has collapsed onto the best fit, so it
    no longer propagates the member spread).

    `rebased=True` first subtracts each member's own grid-max (matching the 'rebased-*'
    combine modes), i.e. weights the members by how close each is to its OWN best fit —
    removing the depth offsets, so K_eff reflects genuine ν̂_b competition rather than
    overall calibration. This is why rebased-bma does not collapse."""
    a = np.asarray(member_logL, dtype=np.float64)
    if rebased:
        grid_axes = tuple(range(a.ndim - 1))
        a = a - a.max(axis=grid_axes, keepdims=True)
    e = np.exp(a - a.max(axis=-1, keepdims=True))   # exp(lnL_b - max_b); in (0, 1]
    s1 = e.sum(axis=-1)
    s2 = (e * e).sum(axis=-1)
    return (s1 * s1) / s2


def plot_keff_1d(axis, keff, K, combined_m2dnll, lbl, out_dir, tag, ni, mode, expected=None):
    """K_eff(ν) along one nuisance — how many members the likelihood weighting keeps
    in play as ν moves (drops where a single member dominates)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(axis, keff, "-", color="#762a83", lw=2.2, zorder=3)
    ax.axhline(K, color="gray", ls=":", lw=1.2, label=f"K = {K}")
    k0 = int(np.argmin(combined_m2dnll))
    ax.axvline(axis[k0], color=COMBINED_COLOR, ls="--", lw=1.4,
               label=f"combined min ($K_{{eff}}$={keff[k0]:.1f})")
    if expected is not None:
        ax.axvline(expected, color="black", ls="-.", lw=1.3, label="truth")
    ax.set_xlabel(lbl, fontsize=13)
    ax.set_ylabel(r"$K_\mathrm{eff}$ (effective # members)", fontsize=13)
    ax.set_ylim(0.8, K * 1.05)
    ax.set_title(f"Effective ensemble size — {lbl}  [K={K}, {_mode_label(mode)}]")
    ax.legend(fontsize=9)
    _save(fig, out_dir, f"scan1d_keff_{tag}_nuis{ni}")


def plot_keff_2d(axis_i, axis_j, keff, K, combined_m2dnll, pair, labels, out_dir, tag,
                 mode, expected=None):
    """Heatmap of K_eff over the 2D nuisance grid, with the combined confidence
    contour overlaid so you can read K_eff *inside* the interval (where it matters):
    low K_eff in the 1σ region ⇒ the BMA leans on a few members there."""
    i, j = pair
    Xg, Yg = np.meshgrid(axis_i, axis_j)
    fig, ax = plt.subplots(figsize=(8.6, 7))
    pcm = ax.pcolormesh(Xg, Yg, keff.T, cmap="magma", vmin=1.0, vmax=float(K), shading="auto")
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(rf"$K_\mathrm{{eff}}$ (effective # members, of {K})", fontsize=11)
    cs = ax.contour(Xg, Yg, combined_m2dnll.T, levels=[2.296, 6.180], colors="white",
                    linewidths=1.6, linestyles=["-", "--"])
    ax.clabel(cs, inline=True, fontsize=8, fmt={2.296: "1σ", 6.180: "2σ"})
    ca, cb_ = np.unravel_index(int(np.argmin(combined_m2dnll)), combined_m2dnll.shape)
    ax.plot(axis_i[ca], axis_j[cb_], "*", color="white", ms=16, mec="black", mew=0.8,
            label=rf"combined min ($K_{{eff}}$={keff[ca, cb_]:.1f})")
    if expected is not None:
        ax.plot(expected[i], expected[j], "P", color="#39ff14", ms=13, mec="black",
                mew=0.8, label="truth")
    lbl_i = labels[i] if i < len(labels) else f"nuis {i}"
    lbl_j = labels[j] if j < len(labels) else f"nuis {j}"
    ax.set_xlabel(lbl_i, fontsize=13); ax.set_ylabel(lbl_j, fontsize=13)
    ax.set_title(rf"Effective ensemble size $K_\mathrm{{eff}}$ — {lbl_i} vs {lbl_j}  "
                 f"[K={K}, {_mode_label(mode)}]", fontsize=12)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    _save(fig, out_dir, f"scan2d_keff_{tag}_nuis{i}{j}")


def _profiled_sigma_lines(ax, base, vertical=False, alpha=0.7):
    """Draw the 1σ (−2Δln L = 1) and 2σ (= 4) PROFILED 1D levels offset by `base`
    (the combined curve's own minimum), so the 1D interval can be read off where the
    combined projection crosses them. `vertical=True` for a rotated (right) panel."""
    for lev, ls in [(1.0, "--"), (4.0, ":")]:
        line = ax.axvline if vertical else ax.axhline
        line(base + lev, color="gray", lw=1.0, ls=ls, alpha=alpha)


def plot_members_1d(axis, m2dnll_members, nll_min_members, combined_m2dnll,
                    combined_nll_min, lbl, out_dir, tag, ni, mode, expected=None):
    """Overlay each member's curve so the ν̂_b spread (horizontal scatter of minima) and
    the depth spread (colour) are visible. Non-rebased modes show ABSOLUTE −2lnL (member
    b at its true depth nll_min_b, the compare_scans '--absolute' look); rebased modes
    show each member at its OWN minimum (all bottoming at 0), matching what the combine
    actually does — and keeping the combined curve on the SAME scale as the members."""
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D
    K = m2dnll_members.shape[1]
    rebased = mode.startswith("rebased")
    colors, norm, cmap = _member_depth_colors(nll_min_members)
    lo, hi = _member_window(nll_min_members, combined_nll_min, rebased)
    moff = np.zeros(K) if rebased else nll_min_members          # per-member vertical offset
    coff = 0.0 if rebased else combined_nll_min                 # combined offset
    ylab = (r"$-2\Delta\ln\mathcal{L}$ (rebased to each min)" if rebased
            else r"$-2\ln\mathcal{L}$ (absolute)")
    fig, ax = plt.subplots(figsize=(8.2, 6))
    for b in np.argsort(nll_min_members)[::-1]:            # worst first → best drawn on top
        ax.plot(axis, m2dnll_members[:, b] + moff[b], "-",
                color=colors[b], lw=1.1, alpha=0.85, zorder=2)
        kb = int(np.argmin(m2dnll_members[:, b]))
        ax.plot(axis[kb], moff[b], "o", color=colors[b], ms=4, mec="white", mew=0.4, zorder=3)
    ax.plot(axis, combined_m2dnll + coff, "-", color=COMBINED_COLOR, lw=2.6, zorder=4)
    ax.plot(axis[int(np.argmin(combined_m2dnll))], coff, "*",
            color=COMBINED_COLOR, ms=15, mec="white", mew=0.6, zorder=5)
    _profiled_sigma_lines(ax, coff)                        # 1σ/2σ levels for the combined curve
    if expected is not None:
        ax.axvline(expected, color="black", ls="-.", lw=1.4, zorder=1)
    ax.set_xlabel(lbl, fontsize=13)
    ax.set_ylabel(ylab, fontsize=13)
    ax.set_ylim(lo, hi)
    ax.set_title(f"Per-member scans — {lbl}  [{K} members, {_mode_label(mode)}]")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(r"member fit depth  $\Delta(-2\ln\mathcal{L})$ vs best", fontsize=9)
    handles = [Line2D([], [], color=cmap(0.7), lw=1.4, label=f"members ({K})"),
               Line2D([], [], color=COMBINED_COLOR, lw=2.6,
                      label=f"combined ({_mode_label(mode, inline=True)})"),
               Line2D([], [], color="gray", lw=1.0, ls="--", label="1σ level"),
               Line2D([], [], color="gray", lw=1.0, ls=":", label="2σ level")]
    if expected is not None:
        handles.append(Line2D([], [], color="black", ls="-.", lw=1.4, label="truth"))
    ax.legend(handles=handles, fontsize=9, loc="upper center")
    _save(fig, out_dir, f"scan1d_members_{tag}_nuis{ni}")


def plot_members_2d(axis_i, axis_j, m2dnll_members, nll_min_members, combined_m2dnll,
                    combined_nll_min, pair, labels, out_dir, tag, mode, expected=None):
    """Corner overlay (compare_scans '--absolute' style) for many members: each
    member's 1σ joint contour (coloured by fit depth) in the main panel + its ABSOLUTE
    profiled projections (offset by nll_min_b) in the top/right panels; the combined
    contour and its absolute projections drawn bold on top."""
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D
    i, j = pair
    K = m2dnll_members.shape[-1]
    rebased = mode.startswith("rebased")
    colors, norm, cmap = _member_depth_colors(nll_min_members)
    lo, hi = _member_window(nll_min_members, combined_nll_min, rebased)
    coff = 0.0 if rebased else combined_nll_min
    ylab = (r"$-2\Delta\ln\mathcal{L}$ (rebased)" if rebased else r"$-2\ln\mathcal{L}$")
    fig = plt.figure(figsize=(9.5, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.4], height_ratios=[1.4, 4],
                          hspace=0.06, wspace=0.06, top=0.93)
    axm = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=axm)
    axr = fig.add_subplot(gs[1, 1], sharey=axm)
    axl = fig.add_subplot(gs[0, 1]); axl.axis("off")
    for b in np.argsort(nll_min_members)[::-1]:            # worst first → best drawn on top
        mb = m2dnll_members[..., b]
        off = 0.0 if rebased else float(nll_min_members[b])
        axm.contour(axis_i, axis_j, mb.T, levels=[2.296], colors=[colors[b]],
                    linewidths=1.1, alpha=0.8)
        a0, b0 = np.unravel_index(int(np.argmin(mb)), mb.shape)
        axm.plot(axis_i[a0], axis_j[b0], "o", color=colors[b], ms=4, mec="white", mew=0.4)
        axt.plot(axis_i, mb.min(axis=1) + off, "-", color=colors[b], lw=1.0, alpha=0.8)
        axr.plot(mb.min(axis=0) + off, axis_j, "-", color=colors[b], lw=1.0, alpha=0.8)
    # combined: contour (relative Δχ²) in the main panel + profiled projections + min star
    axm.contour(axis_i, axis_j, combined_m2dnll.T, levels=[2.296, 6.180],
                colors=[COMBINED_COLOR], linestyles=["-", "--"], linewidths=2.0)
    ca, cb_ = np.unravel_index(int(np.argmin(combined_m2dnll)), combined_m2dnll.shape)
    axm.plot(axis_i[ca], axis_j[cb_], "*", color=COMBINED_COLOR, ms=15, mec="white",
             mew=0.6, zorder=6)
    axt.plot(axis_i, combined_m2dnll.min(axis=1) + coff, "-",
             color=COMBINED_COLOR, lw=2.4, zorder=6)
    axr.plot(combined_m2dnll.min(axis=0) + coff, axis_j, "-",
             color=COMBINED_COLOR, lw=2.4, zorder=6)
    # 1σ/2σ PROFILED levels in the projections — read the 1D interval off the combined
    # curve's crossings (top panel = horizontal lines; right panel = vertical lines).
    _profiled_sigma_lines(axt, coff, vertical=False)
    _profiled_sigma_lines(axr, coff, vertical=True)
    if expected is not None:
        axm.plot(expected[i], expected[j], "P", color="black", ms=12, zorder=7)
        axt.axvline(expected[i], color="black", ls="-.", lw=1.2)
        axr.axhline(expected[j], color="black", ls="-.", lw=1.2)
    lbl_i = labels[i] if i < len(labels) else f"nuis {i}"
    lbl_j = labels[j] if j < len(labels) else f"nuis {j}"
    axm.set_xlabel(lbl_i); axm.set_ylabel(lbl_j)
    axt.set_ylabel(ylab, fontsize=8); axt.set_ylim(lo, hi)
    axr.set_xlabel(ylab, fontsize=8); axr.set_xlim(lo, hi)
    plt.setp(axt.get_xticklabels(), visible=False)
    plt.setp(axr.get_yticklabels(), visible=False)
    # short labels (the combination mode is already in the suptitle) + compact spacing keep the
    # legend inside the narrow top-right corner panel, clear of the top projection panel.
    handles = [Line2D([], [], color=cmap(0.7), lw=1.5, label=f"members ({K}), 1σ"),
               Line2D([], [], color=COMBINED_COLOR, lw=2, ls="-", label="combined 1σ"),
               Line2D([], [], color=COMBINED_COLOR, lw=2, ls="--", label="combined 2σ"),
               Line2D([], [], color="gray", lw=1.0, ls="--", label="1σ proj."),
               Line2D([], [], color="gray", lw=1.0, ls=":", label="2σ proj.")]
    if expected is not None:
        handles.append(Line2D([], [], color="black", marker="P", ls="none", ms=10, label="truth"))
    axl.legend(handles=handles, fontsize=7, loc="upper center", framealpha=0.9,
               handlelength=1.5, handletextpad=0.5, borderpad=0.4, labelspacing=0.4)
    # member-depth colorbar as a horizontal inset in the MIDDLE of the empty corner panel
    # (clear of the legend above and the right projection panel below), label ABOVE the bar,
    # ticks below, ≤3 ticks — Δ(−2lnL) relative depths keep them short and aligned.
    from matplotlib.ticker import MaxNLocator
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = axl.inset_axes([0.16, 0.22, 0.70, 0.045])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cax.xaxis.set_label_position("top")
    cax.xaxis.set_ticks_position("bottom")
    cax.xaxis.set_major_locator(MaxNLocator(3))
    cax.tick_params(labelsize=6)
    cb.set_label(r"member fit depth  $\Delta(-2\ln\mathcal{L})$", fontsize=7, labelpad=3)
    fig.suptitle(f"Per-member scans — {lbl_i} vs {lbl_j}  [{K} members, {_mode_label(mode)}]",
                 fontsize=12, y=0.965)
    _save(fig, out_dir, f"scan2d_members_{tag}_nuis{i}{j}")


# ---------------------------------------------------------------------------
# 2D scan
# ---------------------------------------------------------------------------

@torch.no_grad()
def scan_2d(model, member_loglik_fn, center, pair, half,
            n_steps, include_shape, include_lnN):
    """Returns (axis_i, axis_j, member_logL[ni,nj,K], constraint[ni,nj]) — the raw
    per-member grid, uncombined, so any/all combine modes can be applied once."""
    i, j = pair
    di = torch.linspace(-half[i], half[i], n_steps, device=center.device)
    dj = torch.linspace(-half[j], half[j], n_steps, device=center.device)
    member_logL = None
    constraint = np.zeros((n_steps, n_steps))
    for a in range(n_steps):
        for b_ in range(n_steps):
            m_vec = center.clone()
            m_vec[i] = center[i] + di[a]
            m_vec[j] = center[j] + dj[b_]
            print("    pair {}, step {}/{} (ni={}, nj={})".format(
                pair, a * n_steps + b_ + 1, n_steps ** 2, i, j), end="\r")
            ml = member_loglik_fn(m_vec)
            if member_logL is None:
                member_logL = np.zeros((n_steps, n_steps, ml.shape[0]))
            member_logL[a, b_] = ml
            constraint[a, b_] = _constraint_loglik(model, m_vec, include_shape, include_lnN)
    axis_i = (center[i].item() + di).cpu().numpy()
    axis_j = (center[j].item() + dj).cpu().numpy()
    return axis_i, axis_j, member_logL, constraint


def plot_scan_2d(axis_i, axis_j, m2dnll, pair, labels, out_dir, tag,
                 best_fit_point=None, expected=None, show_anchor=False):
    i, j = pair
    Xg, Yg = np.meshgrid(axis_i, axis_j)
    Z = m2dnll.T  # so axis_i -> x, axis_j -> y

    fig, ax = plt.subplots(figsize=(9, 8))
    # Diverging RdYlBu_r heatmap: dark blue at the minimum -> cream -> dark red,
    # with ROOT-style triangular colorbar extensions.
    vmax = float(np.nanmax(Z))
    levels = np.linspace(0.0, vmax, 41)
    cf = ax.contourf(Xg, Yg, Z, levels=levels, cmap="RdYlBu_r", extend="both")
    cbar = fig.colorbar(cf, ax=ax, extend="both", pad=0.02)
    cbar.set_label(r"$-2\Delta\log L$", fontsize=13)

    # 2-parameter CL levels (Δχ² for 2 dof): 1σ white, 2σ yellow, 3σ orange.
    cl_levels = [2.30, 6.18, 11.83]
    cl_colors = ["white", "yellow", "orange"]
    cs = ax.contour(Xg, Yg, Z, levels=cl_levels, colors=cl_colors, linewidths=1.8)
    ax.clabel(cs, inline=True, fontsize=10, fmt={2.30: "1σ", 6.18: "2σ", 11.83: "3σ"})

    # Best fit = minimum of the scanned likelihood (refined), NOT the frozen m_vector.
    bx, by, on_boundary = refine_min_2d(axis_i, axis_j, m2dnll)
    if on_boundary:
        print(f"  ⚠ 2D minimum sits on the scan boundary for pair {pair}: "
              "the true best fit may lie outside the window — widen --half or re-center.")

    # Markers: expected ν "+", best fit star, and (only on request) the residual anchor ν₀
    # open circle. The anchor is opt-in via show_anchor — it is hidden by default to keep the
    # scan uncluttered; `best_fit_point` is still only meaningful for profiled models.
    if expected is not None:
        ax.plot(expected[i], expected[j], marker="+", color=EXPECTED_COLOR, ms=16,
                markeredgewidth=3, linestyle="none", label=r"$\nu_\mathrm{exp}$ (expected)")
    if show_anchor and best_fit_point is not None:
        ax.plot(best_fit_point[i], best_fit_point[j], marker="o", mfc="none",
                mec="black", ms=12, markeredgewidth=1.6, linestyle="none",
                label=r"$\nu_0$ (anchor)")
    ax.plot(bx, by, marker="*", color="white", ms=18, markeredgecolor="black",
            markeredgewidth=0.8, linestyle="none", label=r"$\hat\nu$ (best fit, scan min)")

    lbl_i = labels[i] if i < len(labels) else f"nuis {i}"
    lbl_j = labels[j] if j < len(labels) else f"nuis {j}"
    ax.set_xlabel(lbl_i, fontsize=14)
    ax.set_ylabel(lbl_j, fontsize=14)
    ax.set_title(r"Likelihood Scan $-2\Delta\log L(\nu)$", fontsize=15)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    _save(fig, out_dir, f"scan2d_{tag}_nuis{i}{j}")


# ---------------------------------------------------------------------------
# Overlay mode (compare saved scans, e.g. mixture vs profiled)
# ---------------------------------------------------------------------------

def overlay_1d(npz_paths: List[str], out_dir: str):
    print(f"Overlay of {len(npz_paths)} scans")
    scans = [(p, np.load(p, allow_pickle=True)) for p in npz_paths]
    # union of nuisance indices present in any scan
    all_ni = sorted({int(k.split("_")[1]) for _, d in scans for k in d.files if k.startswith("axis_")})
    labels0 = list(scans[0][1]["labels"]) if "labels" in scans[0][1].files else []
    # Expected ν is a property of the dataset, not the model — take it from the first scan.
    expected0 = scans[0][1]["expected"] if "expected" in scans[0][1].files else None
    for ni in all_ni:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        for path, d in scans:
            ax_key, z_key = f"axis_{ni}", f"m2dnll_{ni}"
            if ax_key not in d.files:
                continue
            name = str(d["label"]) if "label" in d.files else os.path.basename(os.path.dirname(path))
            ax.plot(d[ax_key], d[z_key], lw=2, label=name)
        if expected0 is not None:
            ax.axvline(float(expected0[ni]), color=EXPECTED_COLOR, lw=1.8, ls="--",
                       label=r"$\nu_\mathrm{exp}$ (expected)")
        for sigma, ls in [(1, "--"), (2, ":")]:
            ax.axhline(sigma ** 2, color="gray", ls=ls, lw=1.2, alpha=0.7)
        lbl = labels0[ni] if ni < len(labels0) else f"nuis {ni}"
        ax.set_xlabel(lbl, fontsize=13)
        ax.set_ylabel(r"$-2\,\Delta\log L$", fontsize=13)
        ax.set_ylim(0, 9.0)
        ax.legend(fontsize=10)
        ax.set_title(f"Likelihood scan comparison — {lbl}")
        _save(fig, out_dir, f"overlay_nuis{ni}")


# ---------------------------------------------------------------------------
# Replot mode (regenerate plots from a saved scan npz — no model/scan needed)
# ---------------------------------------------------------------------------

def _infer_mode(d) -> str:
    """Combine mode for the member/keff overlays: from the npz `mode` key if present
    (new scans), else parsed from the label `<mode>_<stage>` (old scans), else 'bma'."""
    if "mode" in d.files:
        return str(d["mode"])
    tag = str(d["label"]) if "label" in d.files else ""
    for m in sorted(ENSEMBLE_MODES, key=len, reverse=True):
        if tag == m or tag.startswith(m + "_"):
            return m
    return "bma"


def _npz_labels_expected(d):
    labels = list(d["labels"]) if "labels" in d.files else []
    expected = d["expected"].tolist() if "expected" in d.files else None
    return [str(x) for x in labels], expected


def replot_1d(d, out_dir):
    """Regenerate the 1D scan / per-member / K_eff plots from a saved scan1d npz."""
    tag = str(d["label"]); mode = _infer_mode(d)
    labels, expected = _npz_labels_expected(d)
    nis = sorted(int(k[len("axis_"):]) for k in d.files
                 if k.startswith("axis_") and not k.startswith(("axis_i_", "axis_j_")))
    plot_scan_1d({ni: (d[f"axis_{ni}"], d[f"m2dnll_{ni}"]) for ni in nis},
                 labels, out_dir, tag, expected=expected)
    for ni in nis:
        if f"member_m2dnll_{ni}" not in d.files:
            continue
        lbl = labels[ni] if ni < len(labels) else f"nuis {ni}"
        exp_ni = expected[ni] if expected is not None else None
        mc = d[f"member_m2dnll_{ni}"].T            # stored [K, n] -> plot wants [n, K]
        mnll = d[f"member_nll_min_{ni}"]
        plot_members_1d(d[f"axis_{ni}"], mc, mnll, d[f"m2dnll_{ni}"], float(d[f"nll_min_{ni}"]),
                        lbl, out_dir, tag, ni, mode, expected=exp_ni)
        if f"keff_{ni}" in d.files:
            plot_keff_1d(d[f"axis_{ni}"], d[f"keff_{ni}"], int(mnll.shape[0]),
                         d[f"m2dnll_{ni}"], lbl, out_dir, tag, ni, mode, expected=exp_ni)


def replot_2d(d, out_dir, show_anchor=False):
    """Regenerate the 2D scan / per-member / K_eff plots from a saved scan2d npz."""
    tag = str(d["label"]); mode = _infer_mode(d)
    labels, expected = _npz_labels_expected(d)
    # anchor (ν₀) marker only for profiled scans; mixture-stage scans have no residual anchor.
    # Old npz without a 'stage' key keep showing it (backward compatible).
    profiled = ("stage" not in d.files) or ("profiled" in str(d["stage"]))
    best_fit_point = d["center"] if ("center" in d.files and profiled) else None
    sfxs = [k[len("m2dnll_"):] for k in d.files
            if k.startswith("m2dnll_") and f"axis_i_{k[len('m2dnll_'):]}" in d.files]
    for sfx in sfxs:
        i, j = int(sfx[0]), int(sfx[1])
        ai, aj, m = d[f"axis_i_{sfx}"], d[f"axis_j_{sfx}"], d[f"m2dnll_{sfx}"]
        plot_scan_2d(ai, aj, m, (i, j), labels, out_dir, tag,
                     best_fit_point=best_fit_point, expected=expected, show_anchor=show_anchor)
        if f"member_m2dnll_{sfx}" not in d.files:
            continue
        mc = np.moveaxis(d[f"member_m2dnll_{sfx}"], 0, -1)   # [K, ni, nj] -> [ni, nj, K]
        mnll = d[f"member_nll_min_{sfx}"]
        plot_members_2d(ai, aj, mc, mnll, m, float(d[f"nll_min_{sfx}"]), (i, j),
                        labels, out_dir, tag, mode, expected=expected)
        if f"keff_{sfx}" in d.files:
            plot_keff_2d(ai, aj, d[f"keff_{sfx}"], int(mnll.shape[0]), m, (i, j),
                         labels, out_dir, tag, mode, expected=expected)


def replot_dir(out_dir: str, scan1d_name: str, scan2d_name: str, show_anchor: bool = False) -> None:
    """Replot every saved scan npz in `out_dir` (scan1d*.npz and scan2d*.npz, covering
    single-mode `scanNd.npz` and the per-mode `scanNd_<mode>.npz` from --ensemble-mode all)."""
    import glob as _glob
    s1 = _glob.glob(os.path.join(out_dir, scan1d_name.replace(".npz", "*.npz")))
    s2 = _glob.glob(os.path.join(out_dir, scan2d_name.replace(".npz", "*.npz")))
    npzs = sorted(set(s1) | set(s2))
    if not npzs:
        raise FileNotFoundError(
            f"--replot: no {scan1d_name.replace('.npz', '*.npz')} / "
            f"{scan2d_name.replace('.npz', '*.npz')} found in {out_dir}")
    for path in npzs:
        d = np.load(path, allow_pickle=True)
        is2d = any(k.startswith("axis_i_") for k in d.files)
        print(f"replot {os.path.basename(path)}  ({'2D' if is2d else '1D'}, mode={_infer_mode(d)})")
        if is2d:
            replot_2d(d, out_dir, show_anchor=show_anchor)
        else:
            replot_1d(d, out_dir)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def _parse_center(spec: str, model, num_nuisances, device) -> torch.Tensor:
    if spec == "best":
        return model.m_vector.detach().clone()
    if spec == "zero":
        return torch.zeros(num_nuisances, device=device)
    vals = [float(v) for v in spec.split(",")]
    if len(vals) != num_nuisances:
        raise ValueError(f"--center expects {num_nuisances} values, got {len(vals)}")
    return torch.tensor(vals, device=device, dtype=torch.float32)


def _parse_half(spec: Optional[str], num_nuisances: int) -> List[float]:
    if spec is None:
        return [0.1] * num_nuisances
    vals = [float(v) for v in spec.split(",")]
    if len(vals) == 1:
        return vals * num_nuisances
    if len(vals) != num_nuisances:
        raise ValueError(f"--half expects 1 or {num_nuisances} values, got {len(vals)}")
    return vals


def _parse_pairs(spec: Optional[str], num_nuisances: int) -> List[Tuple[int, int]]:
    if spec is None:
        return [(i, j) for i in range(num_nuisances) for j in range(i + 1, num_nuisances)]
    pairs = []
    for chunk in spec.split(";"):
        a, b = chunk.split(",")
        pairs.append((int(a), int(b)))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description="1D/2D likelihood scans for mixture / profiled models.")
    p.add_argument("--overlay", nargs="+", default=None,
                   help="Overlay mode: list of saved scan1d .npz files to compare.")
    p.add_argument("--replot", action="store_true",
                   help="Replot-only: regenerate all plots from the saved scan{1,2}d*.npz in "
                        "--out-dir (no model/dataset/scan needed). Honours --scan1d-name/--scan2d-name.")
    p.add_argument("-c", "--cfg", help="YAML config (mixture or profiling).")
    p.add_argument("--ckpt", help="Checkpoint .pt (default: config output_checkpoint).")
    p.add_argument("--ensemble", default=None,
                   help="Glob of bootstrap member checkpoints for an ensemble scan. With "
                        "--ensemble, --ckpt points at the shared profiled-ensemble (residual) "
                        "checkpoint, or is omitted for the mixture-stage anchor scan. "
                        "See --ensemble-mode for how members are combined.")
    p.add_argument("--ensemble-mode", choices=ENSEMBLE_MODES + ["all"], default="bma",
                   help="How to combine ensemble members (with --ensemble): "
                        "'bma' = logsumexp of per-member likelihoods; "
                        "'envelope' = HEP profiled scan (max over members, single global rebasing); "
                        "'rebased-bma' / 'rebased-envelope' = rebase each member to its OWN minimum "
                        "first (removes the ν-independent depth offsets that collapse bma), then "
                        "combine; 'all' = emit all four from a single scan (one set of plots each).")
    p.add_argument("--no-member-scans", dest="member_scans", action="store_false",
                   help="Skip the per-member overlay output (scan{1,2}d_members_* plots + "
                        "member_* arrays in the npz). By default, in --ensemble mode each "
                        "member's own absolute -2lnL curve is overlaid (compare_scans "
                        "'--absolute' style), which is ~free (reuses the combined-scan evals).")
    p.set_defaults(member_scans=True)
    p.add_argument("--dataset", default=None, help="Override dataset .pt (default: config paths.dataset).")
    p.add_argument("--out-dir", default="scans", help="Output directory.")
    p.add_argument("--label", default=None, help="Name for this scan (legend + plot filenames).")
    p.add_argument("--scan1d-name", default="scan1d.npz", help="Filename for the saved 1D scan npz.")
    p.add_argument("--scan2d-name", default="scan2d.npz", help="Filename for the saved 2D scan npz.")
    p.add_argument("--scan-1d", action="store_true", help="Run 1D scans (default if neither flag given).")
    p.add_argument("--scan-2d", action="store_true", help="Run 2D scan(s) (expensive).")
    p.add_argument("--nuis", default=None, help="Comma list of nuisance indices for 1D (default: profiled ones).")
    p.add_argument("--pairs", default=None, help="2D pairs 'i,j;k,l' (default: all unique pairs).")
    p.add_argument("--center", default="best", help="'best' (m_vector), 'zero', or comma list.")
    p.add_argument("--expected", default=None,
                   help="Expected ν shown as a marker: comma list (default: config plotting.expected_nuisance).")
    p.add_argument("--half", default=None, help="Scan half-width: single value or comma list per nuisance.")
    p.add_argument("--steps-1d", type=int, default=41)
    p.add_argument("--steps-2d", type=int, default=21)
    p.add_argument("--batch-size", type=int, default=10_000)
    p.add_argument("--n-events", type=int, default=None, help="Subsample dataset (default: all).")
    p.add_argument("--no-shape-constraint", dest="shape_constraint", action="store_false",
                   help="Drop the -0.5 m² shape prior (data-only likelihood).")
    p.add_argument("--no-lnN", dest="lnN", action="store_false", help="Drop the lnN term (constant anyway).")
    p.add_argument("--show-anchor", action="store_true",
                   help="Draw the residual expansion anchor ν₀ (open circle) on the 2D scan "
                        "(profiled models only). Hidden by default.")
    args = p.parse_args()

    if args.overlay:
        overlay_1d(args.overlay, args.out_dir)
        print(f"\nOverlay written to {args.out_dir}/")
        return

    if args.replot:
        replot_dir(args.out_dir, args.scan1d_name, args.scan2d_name, show_anchor=args.show_anchor)
        print(f"\nReplotted into {args.out_dir}/")
        return

    if not args.cfg:
        p.error("-c/--cfg is required unless --overlay or --replot is used.")

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

    dataset_path = args.dataset or cfg["paths"]["dataset"]
    print(f"Device: {device}\nDataset: {dataset_path}")

    member_Ts = None
    T_model = None
    if args.ensemble:
        # The shared residual R_ν + anchor ν₀ live ONLY in the profiled checkpoint (the member
        # files hold just the base T_b). Default it from the config's output_checkpoint — like
        # the single-model path below — so a profiling config picks up its residual without
        # needing --ckpt. If that file is absent / has no residual, load_ensemble falls back to
        # the mixture-stage scan (used for anchor finding, where there is no residual).
        res_ckpt = args.ckpt or cfg["paths"].get("output_checkpoint")
        print(f"Ensemble residual checkpoint: {res_ckpt}")
        model, residual_T, member_Ts, stage = load_ensemble(cfg, res_ckpt, args.ensemble, device)
        modes = list(ENSEMBLE_MODES) if args.ensemble_mode == "all" else [args.ensemble_mode]
        print(f"Ensemble combine mode(s): {modes}")
    else:
        ckpt_path = args.ckpt or cfg["paths"]["output_checkpoint"]
        print(f"Checkpoint: {ckpt_path}")
        model, T, residual_T, stage = load_models(cfg, ckpt_path, device)
        T_model = residual_T if residual_T is not None else T
        modes = ["bma"]   # K=1: every mode reduces to the single-model curve
    multi = len(modes) > 1
    # Per-member overlay: only meaningful for a genuine ensemble (>1 member).
    do_members = bool(args.member_scans) and member_Ts is not None and len(member_Ts) > 1

    def out_label(mode):
        """Plot/legend label for a mode: single-model uses --label/stage as-is; an
        ensemble prefixes the mode so the four 'all' outputs land in distinct files."""
        if member_Ts is None:
            return args.label or stage
        if not multi and args.label:
            return args.label
        return f"{mode}_{stage}"

    def mode_name(kind, mode, default):
        """Output filename per mode: keep the legacy name for a single mode (so
        coverage_test / compare_scans find scan{1,2}d.npz), suffix by mode for 'all'."""
        return f"{kind}_{mode}.npz" if multi else default

    print(f"Stage: {stage}")
    print(f"Anchor m_vector: {model.m_vector.detach().cpu().numpy()}")

    dataset = torch.load(dataset_path, map_location=device)
    y_all = dataset["y_data_distorted"].to(device)
    X_all = dataset["X_data_distorted"].to(device)
    if args.n_events is not None and args.n_events < y_all.shape[0]:
        perm = torch.randperm(y_all.shape[0])[:args.n_events]
        y_all, X_all = y_all[perm], X_all[perm]
    print(f"Events: {y_all.shape[0]}")

    # Per-point per-member summed-over-events data log-likelihood -> ndarray[K] (K=1 single
    # model). Combined per requested mode(s) by `_combine_grid` after the single scan.
    if member_Ts is not None:
        def member_loglik_fn(m_vec):
            return _member_logliks(model, residual_T, member_Ts, y_all, X_all, m_vec, args.batch_size)
    else:
        def member_loglik_fn(m_vec):
            return np.array([_data_loglik_sum(model, T_model, y_all, X_all, m_vec, args.batch_size)])

    num_nuisances = model.m_vector.shape[0]
    center = _parse_center(args.center, model, num_nuisances, device)
    half = _parse_half(args.half, num_nuisances)
    labels = list(cfg.get("plotting", {}).get("shape_nuisance_labels",
                  [f"nuis {i}" for i in range(num_nuisances)]))

    # Post-fit best fit (white star) and expected ν (teal marker, from config or CLI).
    best_fit_point = model.m_vector.detach().cpu().numpy()
    _exp_cfg = cfg.get("plotting", {}).get("expected_nuisance")
    if args.expected is not None:
        expected = [float(v) for v in args.expected.split(",")]
    elif _exp_cfg is not None:
        expected = [float(v) for v in _exp_cfg]
    else:
        expected = None
    if expected is not None and len(expected) != num_nuisances:
        raise ValueError(f"expected ν needs {num_nuisances} values, got {len(expected)}")

    # default: scan only profiled nuisances (frozen ones give a flat curve)
    if args.nuis is not None:
        nuis_indices = [int(v) for v in args.nuis.split(",")]
    else:
        mask = model.m_vector_profile_mask.detach().cpu().numpy()
        nuis_indices = [i for i in range(num_nuisances) if bool(mask[i])]

    do_1d = args.scan_1d or not args.scan_2d  # default to 1D if nothing requested
    os.makedirs(args.out_dir, exist_ok=True)

    def _new_blob(tag, mode):
        blob = {"label": np.array(tag), "labels": np.array(labels, dtype=object),
                "center": center.cpu().numpy(), "stage": np.array(stage), "mode": np.array(mode)}
        if expected is not None:
            blob["expected"] = np.array(expected, dtype=float)
        return blob

    if do_1d:
        print("1D scan…")
        # Scan ONCE: collect the raw per-member grid, then combine every requested mode.
        raw = scan_1d(model, member_loglik_fn, center, nuis_indices, half,
                      args.steps_1d, args.shape_constraint, args.lnN)
        for mode in modes:
            tag = out_label(mode)
            save_blob = _new_blob(tag, mode)
            bestfit = {"label": tag, "stage": stage, "mode": mode, "nuisances": {}}
            plot_results = {}
            print(f"\nBest-fit estimation (1D, mode={mode}):")
            for ni, (axis, member_logL, constraint) in raw.items():
                m2dnll, nll_min = _combine_grid(member_logL, constraint, mode)
                plot_results[ni] = (axis, m2dnll)
                best, slo, shi = estimate_1d(axis, m2dnll)
                lbl = labels[ni] if ni < len(labels) else f"nuis {ni}"
                print("  " + _fmt_estimate(lbl, best, slo, shi)
                      + f"   [model m_vector = {best_fit_point[ni]:+.4f}]")
                bestfit["nuisances"][str(ni)] = {
                    "label": lbl, "best_fit": best, "sigma_lo": slo, "sigma_hi": shi,
                    "model_m_vector": float(best_fit_point[ni]),
                    "expected": (float(expected[ni]) if expected is not None else None),
                    "abs_nll_min": nll_min,
                }
                save_blob[f"axis_{ni}"] = axis
                save_blob[f"m2dnll_{ni}"] = m2dnll
                save_blob[f"bestfit_{ni}"] = np.array([best, slo, shi])
                save_blob[f"nll_min_{ni}"] = np.array(nll_min)
                if do_members:
                    mc, mnll = _member_curves(member_logL, constraint)   # [n,K], [K]
                    K = member_logL.shape[-1]
                    save_blob[f"member_m2dnll_{ni}"] = mc.T               # [K, n_steps]
                    save_blob[f"member_nll_min_{ni}"] = mnll              # [K]
                    plot_members_1d(axis, mc, mnll, m2dnll, float(nll_min), lbl,
                                    args.out_dir, tag, ni, mode,
                                    expected=(expected[ni] if expected is not None else None))
                    if mode.endswith("bma"):     # K_eff is the BMA effective sample size
                        keff = _keff_grid(member_logL, rebased=mode.startswith("rebased"))
                        save_blob[f"keff_{ni}"] = keff
                        print(f"    K_eff at scan min = {keff[int(np.argmin(m2dnll))]:.2f} / {K}")
                        plot_keff_1d(axis, keff, K, m2dnll, lbl, args.out_dir, tag, ni, mode,
                                     expected=(expected[ni] if expected is not None else None))
            plot_scan_1d(plot_results, labels, args.out_dir, tag, expected=expected)
            scan1d_name = mode_name("scan1d", mode, args.scan1d_name)
            bestfit_name = f"bestfit_{mode}.json" if multi else "bestfit.json"
            np.savez(os.path.join(args.out_dir, scan1d_name), **save_blob)
            with open(os.path.join(args.out_dir, bestfit_name), "w") as _bf:
                json.dump(bestfit, _bf, indent=2)
            print(f"  [{mode}] 1D -> {scan1d_name}   best fit -> {bestfit_name}")

            # copy-paste anchor txt: full m_vector (scanned nuis at the 1D min, rest at centre)
            anchor = list(np.asarray(center.detach().cpu().numpy(), dtype=float))
            sigmas = {}
            for ni in nuis_indices:
                bf = bestfit["nuisances"][str(ni)]
                anchor[ni] = bf["best_fit"]
                sigmas[ni] = (bf["sigma_lo"], bf["sigma_hi"])
            _write_anchor(args.out_dir, f"anchor_{mode}.txt" if multi else "anchor.txt",
                          anchor, labels, stage, mode, "1D scan", sigmas)

    if args.scan_2d:
        print("2D scan…")
        pairs = _parse_pairs(args.pairs, num_nuisances)
        # Scan ONCE per pair (the expensive step); combine every requested mode after.
        raw2d = {}
        for pair in pairs:
            print(f"  pair {pair} (this may take a while)…")
            axis_i, axis_j, member_logL, constraint = scan_2d(
                model, member_loglik_fn, center, pair, half,
                args.steps_2d, args.shape_constraint, args.lnN)
            raw2d[pair] = (axis_i, axis_j, member_logL, constraint)
        for mode in modes:
            tag = out_label(mode)
            scan2d_blob = _new_blob(tag, mode)
            # joint best-fit anchor: full m_vector with each pair's two nuisances at the 2D min
            anchor2 = list(np.asarray(center.detach().cpu().numpy(), dtype=float))
            print(f"\n2D best fits (mode={mode}):")
            for pair in pairs:
                axis_i, axis_j, member_logL, constraint = raw2d[pair]
                m2dnll, nll_min = _combine_grid(member_logL, constraint, mode)
                # anchor marker only for profiled models (the residual's ν₀ expansion point);
                # the mixture stage has no residual so there is no anchor to show.
                plot_scan_2d(axis_i, axis_j, m2dnll, pair, labels, args.out_dir, tag,
                             best_fit_point=(best_fit_point if residual_T is not None else None),
                             expected=expected, show_anchor=args.show_anchor)
                bx, by, _ = refine_min_2d(axis_i, axis_j, m2dnll)
                anchor2[pair[0]], anchor2[pair[1]] = bx, by
                li = labels[pair[0]] if pair[0] < len(labels) else f"nuis {pair[0]}"
                lj = labels[pair[1]] if pair[1] < len(labels) else f"nuis {pair[1]}"
                print(f"  {li} = {bx:+.4f}, {lj} = {by:+.4f}")
                sfx = f"{pair[0]}{pair[1]}"
                scan2d_blob[f"axis_i_{sfx}"] = axis_i
                scan2d_blob[f"axis_j_{sfx}"] = axis_j
                scan2d_blob[f"m2dnll_{sfx}"] = m2dnll
                scan2d_blob[f"nll_min_{sfx}"] = np.array(nll_min)
                if do_members:
                    mc, mnll = _member_curves(member_logL, constraint)   # [ni,nj,K], [K]
                    K = member_logL.shape[-1]
                    scan2d_blob[f"member_m2dnll_{sfx}"] = np.moveaxis(mc, -1, 0)  # [K, ni, nj]
                    scan2d_blob[f"member_nll_min_{sfx}"] = mnll                   # [K]
                    plot_members_2d(axis_i, axis_j, mc, mnll, m2dnll, float(nll_min), pair,
                                    labels, args.out_dir, tag, mode, expected=expected)
                    if mode.endswith("bma"):     # K_eff is the BMA effective sample size
                        keff = _keff_grid(member_logL, rebased=mode.startswith("rebased"))
                        scan2d_blob[f"keff_{sfx}"] = keff
                        print(f"    K_eff at 2D min = "
                              f"{keff[np.unravel_index(int(np.argmin(m2dnll)), m2dnll.shape)]:.2f} / {K}")
                        plot_keff_2d(axis_i, axis_j, keff, K, m2dnll, pair, labels,
                                     args.out_dir, tag, mode, expected=expected)
            scan2d_name = mode_name("scan2d", mode, args.scan2d_name)
            np.savez(os.path.join(args.out_dir, scan2d_name), **scan2d_blob)
            print(f"  [{mode}] 2D -> {scan2d_name}")
            # joint anchor (preferred over the 1D one; overwrites anchor.txt if both ran)
            _write_anchor(args.out_dir, f"anchor_{mode}.txt" if multi else "anchor.txt",
                          anchor2, labels, stage, mode, "2D scan")

    print(f"\nScans written to {args.out_dir}/")


if __name__ == "__main__":
    main()
