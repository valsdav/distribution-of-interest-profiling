"""
Postfit histograms of the scores y with a T(X) systematic uncertainty band.

Two central predictions are drawn: the *prefit* (frozen m_vector — where the nuisance
sat before profiling) and the *postfit* (the profiling minimum: the scan argmin, or
--best). The T(X) systematic band is the per-bin envelope of the model re-evaluated at
nuisance points on the `-2Δln L = level` contour around the postfit minimum.

Nuisance points on the contour come from either:
  * a 2D scan saved by `likelihood_scan.py --scan-2d` (`--scan2d scan2d.npz`), or
  * a local Gaussian (Hessian) ellipse around the postfit point (`--hessian`, no scan).

Two figures are produced: `postfit_<label>` (X-integrated) and `postfit_xbinned_<label>`
(y in bins of x, like plot_results.py). For each nuisance point the score MC is
regenerated with a *fixed seed* so the band reflects the systematic (m) variation, not
MC noise.

Usage:
    python plot_postfit.py -c configs/profiling_cross_v13.yaml \
        --ckpt models/full_mixture_model_v13_profiled_crossterm.pt \
        --scan2d scans/mixture_v13/scan2d.npz --level 2.30 \
        --out-dir postfit --label profiled_crossterm
    # or without a scan, using the Hessian ellipse:
    python plot_postfit.py -c ... --ckpt ... --hessian --level 2.30
"""
import argparse
import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

from likelihood_scan import load_models, _resolve, _save, _to_tensor


# ---------------------------------------------------------------------------
# Nuisance points along the likelihood contour
# ---------------------------------------------------------------------------

def _contour_from_scan2d(path: str, pair: Tuple[int, int], level: float, n_points: int) -> np.ndarray:
    """Vertices (ν_i, ν_j) of the -2Δln L = level contour from a saved 2D scan."""
    d = np.load(path, allow_pickle=True)
    key = f"{pair[0]}{pair[1]}"
    try:
        axis_i, axis_j, Z = d[f"axis_i_{key}"], d[f"axis_j_{key}"], d[f"m2dnll_{key}"]
    except KeyError as e:
        raise SystemExit(f"{path} has no 2D scan for pair {pair} (missing {e}).")
    fig = plt.figure()
    cs = plt.contour(axis_i, axis_j, Z.T, levels=[level])  # Z[a,b]->axis_i,axis_j; .T for contour
    segs = list(cs.allsegs[0]) if cs.allsegs else []
    plt.close(fig)
    if not segs:
        raise SystemExit(
            f"No -2Δln L = {level} contour in {path} "
            f"(scan spans {float(Z.min()):.2f}..{float(Z.max()):.2f}); widen the scan or lower --level."
        )
    pts = np.concatenate(segs, axis=0)
    idx = np.linspace(0, len(pts) - 1, n_points).round().astype(int)
    return pts[idx]


def _bestfit_from_scan2d(path: str, pair: Tuple[int, int]) -> Tuple[float, float]:
    """Scan minimum (ν_i, ν_j) = argmin of the saved 2D -2Δln L grid."""
    d = np.load(path, allow_pickle=True)
    key = f"{pair[0]}{pair[1]}"
    axis_i, axis_j, Z = d[f"axis_i_{key}"], d[f"axis_j_{key}"], d[f"m2dnll_{key}"]
    a, b = np.unravel_index(int(np.argmin(Z)), Z.shape)
    return float(axis_i[a]), float(axis_j[b])


def _contour_from_hessian(model, residual_T, y, x, best, pair, level, n_points,
                          batch_size, n_batches, device) -> np.ndarray:
    """Ellipse {δ : δᵀ H δ = level} of the local Gaussian approx at the best fit.

    The NLL includes the profiled Gaussian constraint 0.5·Σν² (matching the scan's
    likelihood), which both makes the surface have a minimum and regularises H. The
    ellipse is built from a floored eigen-decomposition so a non-PD data Hessian
    (m_vector need not be the data-NLL minimum — see the scan) degrades gracefully.
    """
    import torch.autograd.functional as AF
    i, j = pair

    def nll(m2):
        m_full = best.clone()
        m_full[i] = m2[0]
        m_full[j] = m2[1]
        total = torch.zeros((), device=device, dtype=torch.float64)
        nb = min(n_batches, (y.shape[0] + batch_size - 1) // batch_size)
        for b in range(nb):
            yb = y[b * batch_size:(b + 1) * batch_size]
            xb = x[b * batch_size:(b + 1) * batch_size]
            mb = m_full.unsqueeze(0).expand(yb.shape[0], -1)
            lp, _ = model.log_prob(yb, xb, residual_T, mb)
            total = total + (-lp.sum()).double()
        return total + 0.5 * (m2[0] ** 2 + m2[1] ** 2).double()  # Gaussian shape constraint

    centre = torch.stack([best[i], best[j]]).detach()
    H = AF.hessian(nll, centre)
    H = 0.5 * (H + H.T)
    evals, evecs = torch.linalg.eigh(H)
    floor = 1e-6
    if int((evals <= floor).sum()) > 0:
        print(f"  ⚠ Hessian not positive-definite (eigvals={evals.tolist()}); clamping. "
              "m_vector may not be the data-NLL minimum — prefer --scan2d for a faithful contour.")
    evals = evals.clamp(min=floor)
    r = float(level) ** 0.5
    pts = []
    for th in np.linspace(0, 2 * np.pi, n_points, endpoint=False):
        u = torch.tensor([np.cos(th), np.sin(th)], device=device, dtype=evals.dtype) * r
        delta = evecs @ (u / torch.sqrt(evals))   # lies on δᵀHδ = level
        pts.append((centre + delta).detach().cpu().numpy())
    return np.asarray(pts)


# ---------------------------------------------------------------------------
# MC + plotting helpers (fixed seed -> band = systematic, not MC noise)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _mc(model, residual_T, m_full, n_mc, seed):
    """MC scores [n_mc, 2] and sampled kinematic x₁ [n_mc] at a fixed nuisance point."""
    torch.manual_seed(seed)
    m = m_full.unsqueeze(0).expand(n_mc, -1)
    scores, kin, _ = model.rsample(n_mc, residual_T, m)
    return scores.cpu().numpy(), kin.cpu().numpy()[:, 0]


def _envelope(h_list):
    stack = np.vstack(h_list)
    return stack.min(axis=0), stack.max(axis=0)


def _draw_panel(ax, axr, cen, h_pre, h_pos, band_lo, band_hi, h_d, show_prefit, level,
                xlabel, log=True, band_label=None, central_label="Postfit (profiled min)",
                prefit_label="Prefit (m_vector)", band_color="tab:orange", central_color="tab:blue"):
    """One main+ratio panel: uncertainty band, prefit (dashed) & central (solid), data;
    ratio vs the central. Shared by plot_postfit.py (ν-contour band) and
    plot_postfit_ensemble.py (ensemble band) so both have the same look — the band's
    legend text and colours are parametrised via `band_label`/`*_color`."""
    if band_label is None:
        band_label = rf"T(X) syst (−2Δln L={level:g})"
    ax.fill_between(cen, band_lo, band_hi, step="mid", alpha=0.35, color=band_color,
                    label=band_label)
    if show_prefit:
        ax.step(cen, h_pre, where="mid", color="0.4", lw=1.6, ls="--", label=prefit_label)
    ax.step(cen, h_pos, where="mid", color=central_color, lw=1.8, label=central_label)
    err = np.sqrt(np.maximum(h_d, 1.0))
    ax.errorbar(cen, h_d, yerr=err, fmt="ko", ms=3, lw=1, label="Data", zorder=5)
    if log:
        ax.set_yscale("log")
    ax.tick_params(labelbottom=False)

    nom = np.where(h_pos > 0, h_pos, np.nan)
    axr.fill_between(cen, band_lo / nom, band_hi / nom, step="mid", alpha=0.35, color=band_color)
    if show_prefit:
        axr.step(cen, h_pre / nom, where="mid", color="0.4", lw=1.4, ls="--")
    axr.axhline(1.0, color=central_color, lw=1.0)
    axr.errorbar(cen, h_d / nom, yerr=err / nom, fmt="ko", ms=2.5, lw=0.8)
    axr.set_ylim(0.5, 1.5)
    axr.set_xlabel(xlabel)


def _plot_inclusive(pf, po, alts, y_np, edges, level, show_prefit, label, out_dir):
    """Inclusive (X-integrated) prefit/postfit/band, y₁ and y₂ side by side."""
    cen = 0.5 * (edges[:-1] + edges[1:])
    w = y_np.shape[0] / pf[0].shape[0]
    H = lambda s, d: np.histogram(s[:, d], bins=edges)[0] * w
    fig = plt.figure(figsize=(13, 6))
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.06, wspace=0.22)
    for d in range(2):
        h_pre, h_pos = H(pf[0], d), H(po[0], d)
        lo, hi = _envelope([h_pos] + [H(a[0], d) for a in alts])
        h_d = np.histogram(y_np[:, d], bins=edges)[0]
        ax = fig.add_subplot(gs[0, d]); axr = fig.add_subplot(gs[1, d], sharex=ax)
        _draw_panel(ax, axr, cen, h_pre, h_pos, lo, hi, h_d, show_prefit, level,
                    f"y{'₁' if d == 0 else '₂'}", log=True)
        ax.set_ylabel("Events"); ax.set_title(f"y{'₁' if d == 0 else '₂'}")
        axr.set_ylabel("Data/Postfit")
        if d == 0:
            ax.legend(fontsize=9)
    fig.suptitle(f"Prefit vs postfit scores with T(X) band — {label}", fontsize=13)
    _save(fig, out_dir, f"postfit_{label}")


def _plot_xbinned(pf, po, alts, y_np, x1_data, edges, x_edges, level, show_prefit, label, out_dir):
    """Same prefit/postfit/band, faceted in X bins (rows: y₁, y₂; cols: X bins)."""
    cen = 0.5 * (edges[:-1] + edges[1:])
    ncols = len(x_edges) - 1
    w = y_np.shape[0] / pf[0].shape[0]
    fig = plt.figure(figsize=(4.2 * ncols, 11.5))
    outer = fig.add_gridspec(2, 1, hspace=0.28)
    for d in range(2):
        block = outer[d].subgridspec(2, ncols, height_ratios=[3, 1], hspace=0.05, wspace=0.22)
        for col in range(ncols):
            xl, xr = x_edges[col], x_edges[col + 1]
            md = (x1_data >= xl) & (x1_data < xr)

            def Hx(mc, dd=d, _xl=xl, _xr=xr):
                s, k = mc
                sel = (k >= _xl) & (k < _xr)
                return np.histogram(s[sel, dd], bins=edges)[0] * w

            h_pre, h_pos = Hx(pf), Hx(po)
            lo, hi = _envelope([h_pos] + [Hx(a) for a in alts])
            h_d = np.histogram(y_np[md, d], bins=edges)[0]
            ax = fig.add_subplot(block[0, col]); axr = fig.add_subplot(block[1, col], sharex=ax)
            _draw_panel(ax, axr, cen, h_pre, h_pos, lo, hi, h_d, show_prefit, level,
                        f"y{'₁' if d == 0 else '₂'}", log=False)
            ax.set_xlim(edges[0], edges[-1])
            if d == 0:
                ax.set_title(f"X = [{xl:g}, {xr:g}]", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Events (y{'₁' if d == 0 else '₂'})")
                axr.set_ylabel("Data/Postfit", fontsize=9)
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Prefit vs postfit scores by X bin with T(X) band — {label}", fontsize=13, y=0.995)
    _save(fig, out_dir, f"postfit_xbinned_{label}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Postfit y histograms with a T(X) systematic band.")
    p.add_argument("-c", "--cfg", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--out-dir", default="postfit")
    p.add_argument("--label", default=None)
    p.add_argument("--scan2d", default=None, help="scan2d.npz from likelihood_scan.py (contour source).")
    p.add_argument("--hessian", action="store_true", help="Use a Hessian ellipse instead of a scan.")
    p.add_argument("--pair", default="0,1", help="Nuisance pair for the 2D contour.")
    p.add_argument("--best", default=None,
                   help="Profiled best fit 'vi,vj' for the pair dims (default: scan2d minimum).")
    p.add_argument("--level", type=float, default=2.30,
                   help="-2Δln L contour level. Default 2.30 = 1σ (68.27%%) for a 2D (2-dof) "
                        "contour. Use 1.0 only for a 1D interval; 6.18 = 2σ, 5.99 = 95%% CL in 2D.")
    p.add_argument("--n-contour", type=int, default=24, help="Nuisance points sampled on the contour.")
    p.add_argument("--n-mc", type=int, default=200_000)
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--range", default=None, help="Score axis range 'lo,hi' (default: data percentiles).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hessian-batch", type=int, default=10_000)
    p.add_argument("--hessian-nbatches", type=int, default=30)
    args = p.parse_args()

    cfg_path = os.path.abspath(args.cfg)
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(cfg_path)
    for k, v in list(cfg["paths"].items()):
        if isinstance(v, str):
            cfg["paths"][k] = _resolve(v, cfg_dir)

    device = cfg.get("runtime", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    ckpt = args.ckpt or cfg["paths"]["output_checkpoint"]
    dataset_path = args.dataset or cfg["paths"]["dataset"]
    pair = tuple(int(v) for v in args.pair.split(","))

    model, T, residual_T, stage = load_models(cfg, ckpt, device)
    if residual_T is None:
        raise SystemExit("Postfit band needs a profiled checkpoint with a residual_transfer_model.")
    label = args.label or stage

    ds = torch.load(dataset_path, map_location=device)
    y_data = ds["y_data_distorted"].to(device)
    x_data = ds["X_data_distorted"].to(device)
    y_np = y_data.cpu().numpy()
    x1_data = x_data[:, 0].cpu().numpy()

    # Prefit central = frozen m_vector (where the nuisance sat before profiling).
    # Postfit central = the profiling minimum (scan argmin, or --best).
    best_prefit = model.m_vector.detach().clone()
    best_postfit = best_prefit.clone()
    if args.best is not None:
        bv = [float(v) for v in args.best.split(",")]
        best_postfit[pair[0]], best_postfit[pair[1]] = bv[0], bv[1]
        post_src = "--best"
    elif args.scan2d is not None and not args.hessian:
        vi, vj = _bestfit_from_scan2d(args.scan2d, pair)
        best_postfit[pair[0]], best_postfit[pair[1]] = vi, vj
        post_src = "scan2d minimum"
    else:
        post_src = "m_vector (no scan/--best given)"
    show_prefit = not np.allclose(best_prefit.cpu().numpy(), best_postfit.cpu().numpy())
    print(f"Prefit  ν (frozen m_vector): {best_prefit.cpu().numpy()}")
    print(f"Postfit ν ({post_src}): {best_postfit.cpu().numpy()}")

    # --- contour nuisance points (around the postfit minimum) ---
    if args.hessian or args.scan2d is None:
        if not args.hessian and args.scan2d is None:
            print("No --scan2d given -> Hessian ellipse around the postfit point.")
        contour_ij = _contour_from_hessian(
            model, residual_T, y_data, x_data, best_postfit, pair, args.level, args.n_contour,
            args.hessian_batch, args.hessian_nbatches, device,
        )
        src = "Hessian ellipse"
    else:
        contour_ij = _contour_from_scan2d(args.scan2d, pair, args.level, args.n_contour)
        src = os.path.basename(args.scan2d)
    print(f"Contour: {len(contour_ij)} points at -2Δln L = {args.level} from {src}")

    # Full nuisance vectors for each contour point (non-pair dims at the postfit point).
    m_points = []
    for (vi, vj) in contour_ij:
        m = best_postfit.clone()
        m[pair[0]] = float(vi)
        m[pair[1]] = float(vj)
        m_points.append(m)

    # Score axis.
    if args.range is not None:
        lo, hi = (float(v) for v in args.range.split(","))
        edges = np.linspace(lo, hi, args.bins + 1)
    else:
        edges = np.linspace(float(np.percentile(y_np, 0.5)),
                            float(np.percentile(y_np, 99.5)), args.bins + 1)

    # Generate MC once (scores + kinematic x₁) at prefit, postfit, and each contour point.
    print(f"Sampling MC (n_mc={args.n_mc}) at prefit, postfit, and {len(m_points)} contour points…")
    pf   = _mc(model, residual_T, best_prefit,  args.n_mc, args.seed)
    po   = _mc(model, residual_T, best_postfit, args.n_mc, args.seed)
    alts = [_mc(model, residual_T, m, args.n_mc, args.seed) for m in m_points]

    x_edges = np.array([-5., -1.5, -0.5, 0.5, 1.5, 5.])
    os.makedirs(args.out_dir, exist_ok=True)
    _plot_inclusive(pf, po, alts, y_np, edges, args.level, show_prefit, label, args.out_dir)
    _plot_xbinned(pf, po, alts, y_np, x1_data, edges, x_edges, args.level, show_prefit, label, args.out_dir)
    print(f"\nWritten to {args.out_dir}/")


if __name__ == "__main__":
    main()
