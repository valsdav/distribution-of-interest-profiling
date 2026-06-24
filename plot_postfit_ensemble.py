"""
Postfit score distributions (data vs model) with a bootstrap-ENSEMBLE uncertainty band,
drawn in the same style as `plot_postfit.py` (band fill + central step + data points +
Data/Model ratio; inclusive y₁/y₂ and faceted in X bins).

This is the **ensemble-of-T** band: every bootstrap member (its base transfer `T_b`, on
top of the shared frozen kin/score residuals + the shared residual `R_ν`) is sampled at
the SAME anchor ν₀ with COMMON RANDOM NUMBERS (same seed → identical flavour draw,
kinematics, nominal scores; only `T_b` differs), so the per-bin spread across members is
the transfer-`T` statistical uncertainty, not MC noise. Use it for **step-1** (mixture)
ensembles. For the **step-2** ν-systematic (nuisance-interval) band on a single profiled
model, use `plot_postfit.py` instead — both share the same panel style.

Usage
-----
    # step-1 mixture ensemble (no residual):
    python plot_postfit_ensemble.py -c configs/mixture_v15_ensemble.yaml \
        --ensemble "models/ensemble_v15/full_mixture_model_v15_boot*.pt" \
        --out-dir postfit_ens --band envelope
    # step-2 profiled ensemble (residual; T-stat band through the residual):
    python plot_postfit_ensemble.py -c configs/profiling_v15_ensemble.yaml \
        --ckpt models/full_mixture_model_v15_ensemble_profiled.pt \
        --ensemble "models/ensemble_v15/full_mixture_model_v15_boot*.pt" --out-dir postfit_ens
    # + combined stat (ensemble T) ⊕ syst (ν contour) ⊕ total figure:
    python plot_postfit_ensemble.py -c configs/profiling_v15_ensemble.yaml \
        --ckpt models/full_mixture_model_v15_ensemble_profiled.pt \
        --ensemble "models/ensemble_v15/full_mixture_model_v15_boot*.pt" \
        --syst-scan2d scans/profiled_v15/scan2d.npz --out-dir postfit_ens

`--ckpt` defaults to the config's output_checkpoint; omit it for a mixture-stage ensemble.

Combined view (`--syst-scan2d`)
-------------------------------
Adds a postfit figure overlaying the statistical and systematic uncertainties on the
data-vs-model panel. The **total** band is built DIRECTLY — every ensemble member `T_b` is
sampled at every point of the `-2Δln L = level` ν contour, and the band is the per-bin
envelope of that full (T_b × ν) grid. No quadrature, no orthogonality assumption: the joint
T×ν variation (including any correlation) is captured exactly. The two sub-bands shown for
interpretation are
  * **stat** — vary `T_b` at the fixed anchor ν̂ (the ensemble-of-T band above);
  * **syst** — vary ν along the contour with `T` averaged over the ensemble (T-marginalised).
The ν contour comes from a 2D scan you supply via `--syst-scan2d`; generate it by running
`likelihood_scan.py --scan-2d` on the same profiled checkpoint and pass its `scan2d.npz`.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import torch
import yaml

from likelihood_scan import load_ensemble, _resolve, _save
from plot_results import _rsample_batched
from plot_postfit import _draw_panel, _contour_from_scan2d

BAND_COLOR = "#6699cc"      # ensemble (statistical) band — distinct from plot_postfit's syst orange
CENTRAL_COLOR = "black"     # ensemble-mean central
SYST_COLOR = "tab:orange"   # ν-systematic band (matches plot_postfit)
TOTAL_COLOR = "0.6"         # total = stat ⊕ syst


def _member_T(residual_T, T_b):
    """T_model for member b: swap its base into the shared residual, else use T_b."""
    if residual_T is not None:
        residual_T.base_model = T_b
        return residual_T
    return T_b


@torch.no_grad()
def _sample_member(model, T_model, n_mc, device, batch, seed, m_value=None):
    """rsample at the anchor ν (m_value; defaults to model.m_vector) with a FIXED seed
    (common random numbers across members → only T_b differs). Returns scores[n,F], x₁[n]."""
    torch.manual_seed(seed)
    s, k, _ = _rsample_batched(model, n_mc, T_model, device, batch, m_value=m_value)
    return s, k[:, 0]


@torch.no_grad()
def _sample_member_fl(model, T_model, n_mc, device, batch, seed, m_value=None):
    """Like `_sample_member`, but also returns the per-event flavour label so the sample can
    be split by class (A/B). Returns scores[n,F], x₁[n], flavour[n]."""
    torch.manual_seed(seed)
    s, k, fl = _rsample_batched(model, n_mc, T_model, device, batch, m_value=m_value)
    return s, k[:, 0], fl


def _band(Hk, band):
    """Central + band from per-member histograms Hk [K, nbins]."""
    central = Hk.mean(axis=0)
    if band == "std":
        s = Hk.std(axis=0)
        return central, central - s, central + s
    return central, Hk.min(axis=0), Hk.max(axis=0)        # envelope (min–max)


@torch.no_grad()
def _member_hists(model, residual_T, member_Ts, dataset, device, anchor_t, n_mc, batch,
                  seed, edges, x_edges):
    """Per-member, common-RN MC → score histograms, inclusive and per X bin.

    Returns (H_incl, H_xb): H_incl[d] is [K, nbins]; H_xb[d][xb] is [K, nbins]. Each
    member is weighted to the data yield (w = N_data / n_mc)."""
    K = len(member_Ts)
    nb = len(edges) - 1
    nxb = len(x_edges) - 1
    N_data = dataset["y_data_distorted"].shape[0]
    w = N_data / n_mc
    H_incl = [np.zeros((K, nb)), np.zeros((K, nb))]
    H_xb = [[np.zeros((K, nb)) for _ in range(nxb)] for _ in range(2)]
    for b, T_b in enumerate(member_Ts):
        scores, kx = _sample_member(model, _member_T(residual_T, T_b), n_mc, device, batch,
                                    seed, m_value=anchor_t)
        for d in range(2):
            H_incl[d][b] = np.histogram(scores[:, d], bins=edges, weights=np.full(n_mc, w))[0]
            for xb in range(nxb):
                sel = (kx >= x_edges[xb]) & (kx < x_edges[xb + 1])
                H_xb[d][xb][b] = np.histogram(scores[sel, d], bins=edges,
                                              weights=np.full(int(sel.sum()), w))[0]
    return H_incl, H_xb


@torch.no_grad()
def _member_hists_byclass(model, residual_T, member_Ts, dataset, device, m_value, n_mc, batch,
                          seed, edges, x_edges):
    """Per-member, per-CLASS score histograms at nuisance `m_value`, inclusive and per X bin.

    Returns (Hc_incl, Hc_xb) as dense arrays:
      Hc_incl is [2, nfl, K, nb];  Hc_xb is [2, nxb, nfl, K, nb].
    Common-seed sampling (only T_b / ν differ, not the MC noise). Each histogram is weighted
    to the data yield (w = N_data / n_mc) so the per-class stack sums to the model expectation."""
    K = len(member_Ts)
    nfl = model.n_flavours
    nb, nxb = len(edges) - 1, len(x_edges) - 1
    w = dataset["y_data_distorted"].shape[0] / n_mc
    Hc_incl = np.zeros((2, nfl, K, nb))
    Hc_xb = np.zeros((2, nxb, nfl, K, nb))
    for b, T_b in enumerate(member_Ts):
        scores, kx, fl = _sample_member_fl(model, _member_T(residual_T, T_b), n_mc, device,
                                           batch, seed, m_value=m_value)
        for c in range(nfl):
            mc = fl == c
            for d in range(2):
                Hc_incl[d, c, b] = np.histogram(scores[mc, d], bins=edges,
                                                weights=np.full(int(mc.sum()), w))[0]
                for xb in range(nxb):
                    sel = mc & (kx >= x_edges[xb]) & (kx < x_edges[xb + 1])
                    Hc_xb[d, xb, c, b] = np.histogram(scores[sel, d], bins=edges,
                                                      weights=np.full(int(sel.sum()), w))[0]
    return Hc_incl, Hc_xb


def plot_inclusive(H_incl, y_np, edges, band, K, label, out_dir):
    """Inclusive (X-integrated) y₁/y₂: ensemble-mean central + band + data, ratio vs central."""
    cen = 0.5 * (edges[:-1] + edges[1:])
    band_label = f"ensemble {band} (K={K})"
    fig = plt.figure(figsize=(13, 6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], hspace=0.06, wspace=0.22, top=0.92)
    for d in range(2):
        central, lo, hi = _band(H_incl[d], band)
        h_d = np.histogram(y_np[:, d], bins=edges)[0]
        ax = fig.add_subplot(gs[0, d]); axr = fig.add_subplot(gs[1, d], sharex=ax)
        _draw_panel(ax, axr, cen, central, central, lo, hi, h_d, False, 0.0,
                    f"y{'₁' if d == 0 else '₂'}", log=True, band_label=band_label,
                    central_label="ensemble mean", band_color=BAND_COLOR,
                    central_color=CENTRAL_COLOR)
        ax.set_ylabel("Events"); ax.set_title(f"y{'₁' if d == 0 else '₂'}")
        axr.set_ylabel("Data/Model")
        if d == 0:
            ax.legend(fontsize=9)
    fig.suptitle(f"Postfit scores — ensemble-of-T band — {label}", fontsize=13, y=0.965)
    _save(fig, out_dir, f"postfit_ensemble_{label}")


def plot_xbinned(H_xb, y_np, x1_data, edges, x_edges, band, K, label, out_dir):
    """Same band, faceted in X bins (rows: y₁, y₂; cols: X bins)."""
    cen = 0.5 * (edges[:-1] + edges[1:])
    ncols = len(x_edges) - 1
    band_label = f"ensemble {band} (K={K})"
    fig = plt.figure(figsize=(4.2 * ncols, 11.5))
    outer = fig.add_gridspec(2, 1, hspace=0.28, top=0.94)
    for d in range(2):
        block = outer[d].subgridspec(2, ncols, height_ratios=[3, 1], hspace=0.05, wspace=0.22)
        for col in range(ncols):
            xl, xr = x_edges[col], x_edges[col + 1]
            md = (x1_data >= xl) & (x1_data < xr)
            central, lo, hi = _band(H_xb[d][col], band)
            h_d = np.histogram(y_np[md, d], bins=edges)[0]
            ax = fig.add_subplot(block[0, col]); axr = fig.add_subplot(block[1, col], sharex=ax)
            _draw_panel(ax, axr, cen, central, central, lo, hi, h_d, False, 0.0,
                        f"y{'₁' if d == 0 else '₂'}", log=False, band_label=band_label,
                        central_label="ensemble mean", band_color=BAND_COLOR,
                        central_color=CENTRAL_COLOR)
            ax.set_xlim(edges[0], edges[-1])
            if d == 0:
                ax.set_title(f"X = [{xl:g}, {xr:g}]", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Events (y{'₁' if d == 0 else '₂'})")
                axr.set_ylabel("Data/Model", fontsize=9)
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Postfit scores by X bin — ensemble-of-T band — {label}", fontsize=13, y=0.975)
    _save(fig, out_dir, f"postfit_ensemble_xbinned_{label}")


# ---------------------------------------------------------------------------
# Distorted simulation (A, B at ν̂) vs original (ν=0) shape
#
# Shows the systematic distortion the fit absorbed: stacked per-class model sample at the
# best-fit ν̂ (the "distorted simulation", ensemble-mean) with each class's ORIGINAL (ν=0)
# shape overlaid as a dashed step line. The ratio panel's dashed curve = original/distorted
# (the absorbed pull); the points are Data/distorted.
# ---------------------------------------------------------------------------

def _draw_distortion_panel(ax, axr, edges, dist_c, orig_c, dist_band, h_d, cls_name, colors,
                           xlabel, log=True, legend=False, return_handles=False):
    """One main+ratio panel: stacked per-class distorted sim (filled), per-class original (ν=0)
    dashed step lines, optional total-distorted ensemble band, data points; ratio vs distorted."""
    cen = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    nfl = len(dist_c)

    bottoms = np.zeros(len(cen))
    for c in range(nfl):
        ax.bar(edges[:-1], dist_c[c], width=widths, bottom=bottoms, align="edge",
               color=colors[c % len(colors)], alpha=0.45, lw=0,
               label=f"Sim {cls_name[c % len(cls_name)]} (ν̂)")
        bottoms += dist_c[c]
    if dist_band is not None:
        tlo, thi = dist_band
        ax.fill_between(cen, tlo, thi, step="mid", color="0.5", alpha=0.25, lw=0,
                        label="ensemble band (total)")
    # original (ν=0) per class — STACKED to mirror the filled stack, so each dashed line sits
    # on top of the ones below and the outermost traces the original total.
    cum_orig = np.zeros(len(cen))
    for c in range(nfl):
        cum_orig = cum_orig + orig_c[c]
        ax.step(cen, cum_orig, where="mid", color=colors[c % len(colors)], lw=1.7, ls="--",
                label=f"{cls_name[c % len(cls_name)]} original (ν=0)")
    if h_d is not None:
        err = np.sqrt(np.maximum(h_d, 1.0))
        ax.errorbar(cen, h_d, yerr=err, fmt="ko", ms=3, lw=1, zorder=5, label="Data")
    if log:
        ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.tick_params(labelbottom=False)
    # Legend handles: main-panel artists + a proxy for the ratio panel's gray dashed
    # "original / sim" curve (which has no twin artist in the main panel).
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="0.3", lw=1.5, ls="--"))
    labels.append("original / sim (ratio)")
    if legend:
        ax.legend(handles, labels, fontsize=8, ncol=2)

    dist_tot = np.sum(dist_c, axis=0)
    orig_tot = np.sum(orig_c, axis=0)
    nom = np.where(dist_tot > 0, dist_tot, np.nan)
    if dist_band is not None:
        tlo, thi = dist_band
        axr.fill_between(cen, tlo / nom, thi / nom, step="mid", color="0.5", alpha=0.25, lw=0)
    axr.axhline(1.0, color="black", lw=1.0)
    axr.step(cen, orig_tot / nom, where="mid", color="0.3", lw=1.5, ls="--",
             label="original / sim")
    if h_d is not None:
        err = np.sqrt(np.maximum(h_d, 1.0))
        axr.errorbar(cen, h_d / nom, yerr=err / nom, fmt="ko", ms=2.5, lw=0.8)
    axr.set_ylim(0.5, 1.5)
    axr.set_xlim(edges[0], edges[-1])
    axr.set_xlabel(xlabel)
    if return_handles:
        return handles, labels


def plot_distortion(Hc_dist, Hc_orig, y_np, edges, band, K, label, out_dir, cls_name, colors):
    """Inclusive y₁/y₂: stacked per-class distorted simulation (best-fit ν̂, ensemble-mean) with
    each class's original (ν=0) shape dashed, total ensemble band, data points and ratio."""
    nfl = Hc_dist.shape[1]
    fig = plt.figure(figsize=(13, 6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], hspace=0.06, wspace=0.22, top=0.92)
    for d in range(2):
        dist_c = [Hc_dist[d, c].mean(0) for c in range(nfl)]
        orig_c = [Hc_orig[d, c].mean(0) for c in range(nfl)]
        _, tlo, thi = _band(Hc_dist[d].sum(axis=0), band)        # total over classes per member
        h_d = np.histogram(y_np[:, d], bins=edges)[0]
        ax = fig.add_subplot(gs[0, d]); axr = fig.add_subplot(gs[1, d], sharex=ax)
        _draw_distortion_panel(ax, axr, edges, dist_c, orig_c, (tlo, thi), h_d,
                               cls_name, colors, f"y{'₁' if d == 0 else '₂'}",
                               log=True, legend=(d == 0))
        ax.set_ylabel("Events"); ax.set_title(f"y{'₁' if d == 0 else '₂'}")
        axr.set_ylabel("Ratio / sim")
    fig.suptitle(f"Distorted simulation (A, B @ ν̂) vs original (ν=0) — {label}", fontsize=13,
                 y=0.965)
    _save(fig, out_dir, f"postfit_distortion_{label}")


def plot_distortion_xbinned(Hc_dist_xb, Hc_orig_xb, y_np, x1_data, edges, x_edges, band, K,
                            label, out_dir, cls_name, colors):
    """Same distorted-vs-original view, faceted in X bins (rows: y₁, y₂; cols: X bins)."""
    nfl = Hc_dist_xb.shape[2]
    ncols = len(x_edges) - 1
    fig = plt.figure(figsize=(4.2 * ncols, 11.5))
    # leave a gap between the two y-blocks (top=0.94, bottom=0.10) to host the legend strip
    outer = fig.add_gridspec(2, 1, hspace=0.30, top=0.94, bottom=0.10)
    leg_handles = leg_labels = None
    for d in range(2):
        block = outer[d].subgridspec(2, ncols, height_ratios=[3, 1], hspace=0.05, wspace=0.22)
        for col in range(ncols):
            xl, xr = x_edges[col], x_edges[col + 1]
            md = (x1_data >= xl) & (x1_data < xr)
            dist_c = [Hc_dist_xb[d, col, c].mean(0) for c in range(nfl)]
            orig_c = [Hc_orig_xb[d, col, c].mean(0) for c in range(nfl)]
            _, tlo, thi = _band(Hc_dist_xb[d, col].sum(axis=0), band)
            h_d = np.histogram(y_np[md, d], bins=edges)[0]
            ax = fig.add_subplot(block[0, col]); axr = fig.add_subplot(block[1, col], sharex=ax)
            res = _draw_distortion_panel(ax, axr, edges, dist_c, orig_c, (tlo, thi), h_d,
                                         cls_name, colors, f"y{'₁' if d == 0 else '₂'}",
                                         log=False, legend=False,
                                         return_handles=(d == 0 and col == 0))
            if res is not None:
                leg_handles, leg_labels = res
            if d == 0:
                ax.set_title(f"X = [{xl:g}, {xr:g}]", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Events (y{'₁' if d == 0 else '₂'})")
                axr.set_ylabel("Ratio / sim", fontsize=9)
    # single horizontal legend (boxed) in the gap between the two y-blocks
    if leg_handles is not None:
        fig.legend(leg_handles, leg_labels, loc="center", bbox_to_anchor=(0.5, 0.515),
                   ncol=len(leg_labels), fontsize=11, frameon=True, fancybox=False,
                   edgecolor="0.6", framealpha=0.95)
    fig.suptitle(f"Distorted sim (A, B @ ν̂) vs original (ν=0) by X bin — {label}",
                 fontsize=13, y=0.985)
    _save(fig, out_dir, f"postfit_distortion_xbinned_{label}")


# ---------------------------------------------------------------------------
# Combined stat (T) ⊕ syst (ν) view
#
# The TOTAL band is built DIRECTLY: every ensemble member is sampled at every
# ν-contour point, so the band is the true joint envelope of the (T_b × ν) grid
# — no quadrature, no orthogonality assumption. stat (members at the anchor) and
# syst (the T-averaged ν sweep) are shown as the two nested sub-bands.
# ---------------------------------------------------------------------------

@torch.no_grad()
def _contour_member_hists(model, residual_T, member_Ts, dataset, device, m_points, n_mc,
                          batch, seed, edges, x_edges):
    """Every member `T_b` sampled at every ν-contour point `m_k` (common seed → only T_b and
    ν differ, not MC noise). Returns (Hi, Hx): Hi[d] is [K, nC, nb]; Hx[d][xb] is [K, nC, nb].
    Each histogram is weighted to the data yield (w = N_data / n_mc)."""
    K, nC = len(member_Ts), len(m_points)
    nb, nxb = len(edges) - 1, len(x_edges) - 1
    w = dataset["y_data_distorted"].shape[0] / n_mc
    Hi = [np.zeros((K, nC, nb)), np.zeros((K, nC, nb))]
    Hx = [[np.zeros((K, nC, nb)) for _ in range(nxb)] for _ in range(2)]
    for b, T_b in enumerate(member_Ts):
        T_model = _member_T(residual_T, T_b)
        for k, m in enumerate(m_points):
            scores, kx = _sample_member(model, T_model, n_mc, device, batch, seed, m_value=m)
            for d in range(2):
                Hi[d][b, k] = np.histogram(scores[:, d], bins=edges, weights=np.full(n_mc, w))[0]
                for xb in range(nxb):
                    sel = (kx >= x_edges[xb]) & (kx < x_edges[xb + 1])
                    Hx[d][xb][b, k] = np.histogram(scores[sel, d], bins=edges,
                                                   weights=np.full(int(sel.sum()), w))[0]
    return Hi, Hx


def _set_band(H2d, band, central=None):
    """(lo, hi) from a [M, nb] set of histograms: per-bin min–max envelope (default), or
    ±1 std around `central` (or the set mean if None) for band='std'."""
    if band == "std":
        c = H2d.mean(axis=0) if central is None else central
        s = H2d.std(axis=0)
        return c - s, c + s
    return H2d.min(axis=0), H2d.max(axis=0)


def _combined_bands(H_anchor, H_cont, band):
    """From the anchor set H_anchor [K, nb] (members at ν̂) and the contour set H_cont
    [K, nC, nb] (members × ν-points), build the three nested bands around the best-fit
    central (= ensemble mean at ν̂):
      stat  — vary T at ν̂                  (H_anchor)
      syst  — vary ν, T-averaged           (mean over members at each ν-point)
      total — vary BOTH, joint envelope    (every member at every ν-point ∪ the anchor set)
    Returns (central, (slo,shi), (ylo,yhi), (tlo,thi))."""
    K, nC, nb = H_cont.shape
    central = H_anchor.mean(axis=0)
    stat = _set_band(H_anchor, band, central=central)
    M = H_cont.mean(axis=0)                                    # [nC, nb] T-averaged ν sweep
    syst = _set_band(np.vstack([M, central[None, :]]), band, central=central)
    joint = np.vstack([H_anchor, H_cont.reshape(K * nC, nb)])  # [K + K·nC, nb]
    total = _set_band(joint, band, central=central)
    return central, stat, syst, total


def _draw_combined(ax, axr, cen, central, stat, syst, total, h_d, xlabel, K, nC,
                   log=True, legend=False, return_handles=False):
    """One main+ratio panel: the TOTAL band as a solid fill, with the stat (vary T at ν̂)
    and syst (vary ν, T-averaged) sub-bands overlaid as boundary lines, plus the central
    step and data; ratio vs central."""
    slo, shi = stat; ylo, yhi = syst; tlo, thi = total
    ax.fill_between(cen, tlo, thi, step="mid", color=TOTAL_COLOR, alpha=0.45, lw=0,
                    label=f"total (T × ν, {K}×{nC})")
    ax.step(cen, yhi, where="mid", color=SYST_COLOR, lw=1.3, ls="--", label="syst (ν, T-averaged)")
    ax.step(cen, ylo, where="mid", color=SYST_COLOR, lw=1.3, ls="--")
    ax.step(cen, shi, where="mid", color=BAND_COLOR, lw=1.3, label=f"stat (ensemble T @ ν̂, K={K})")
    ax.step(cen, slo, where="mid", color=BAND_COLOR, lw=1.3)
    ax.step(cen, central, where="mid", color=CENTRAL_COLOR, lw=1.6, label="central (ens-mean @ ν̂)")
    err = np.sqrt(np.maximum(h_d, 1.0))
    ax.errorbar(cen, h_d, yerr=err, fmt="ko", ms=3, lw=1, zorder=5, label="Data")
    if log:
        ax.set_yscale("log")
    ax.tick_params(labelbottom=False)

    nom = np.where(central > 0, central, np.nan)
    axr.fill_between(cen, tlo / nom, thi / nom, step="mid", color=TOTAL_COLOR, alpha=0.45, lw=0)
    axr.step(cen, yhi / nom, where="mid", color=SYST_COLOR, lw=1.1, ls="--")
    axr.step(cen, ylo / nom, where="mid", color=SYST_COLOR, lw=1.1, ls="--")
    axr.step(cen, shi / nom, where="mid", color=BAND_COLOR, lw=1.1)
    axr.step(cen, slo / nom, where="mid", color=BAND_COLOR, lw=1.1)
    axr.axhline(1.0, color=CENTRAL_COLOR, lw=1.0)
    axr.errorbar(cen, h_d / nom, yerr=err / nom, fmt="ko", ms=2.5, lw=0.8)
    axr.set_ylim(0.5, 1.5)
    axr.set_xlabel(xlabel)
    handles, labels = ax.get_legend_handles_labels()
    if legend:
        ax.legend(handles, labels, fontsize=8)
    if return_handles:
        return handles, labels


def plot_inclusive_combined(H_incl, H_cont_incl, y_np, edges, band, K, nC, level, label, out_dir):
    """Inclusive y₁/y₂ with stat ⊕ syst ⊕ total bands (data vs ensemble-mean model)."""
    cen = 0.5 * (edges[:-1] + edges[1:])
    fig = plt.figure(figsize=(13, 6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], hspace=0.06, wspace=0.22, top=0.92)
    for d in range(2):
        central, stat, syst, total = _combined_bands(H_incl[d], H_cont_incl[d], band)
        h_d = np.histogram(y_np[:, d], bins=edges)[0]
        ax = fig.add_subplot(gs[0, d]); axr = fig.add_subplot(gs[1, d], sharex=ax)
        _draw_combined(ax, axr, cen, central, stat, syst, total, h_d,
                       f"y{'₁' if d == 0 else '₂'}", K, nC, log=True, legend=(d == 0))
        ax.set_ylabel("Events"); ax.set_title(f"y{'₁' if d == 0 else '₂'}")
        axr.set_ylabel("Data/Model")
    fig.suptitle(f"Postfit scores — stat (ensemble T) ⊕ syst (ν contour, −2Δln L={level:g}) — "
                 f"{label}", fontsize=13, y=0.965)
    _save(fig, out_dir, f"postfit_total_{label}")


def plot_xbinned_combined(H_xb, H_cont_xb, y_np, x1_data, edges, x_edges, band, K, nC, level,
                          label, out_dir):
    """Same stat ⊕ syst ⊕ total bands, faceted in X bins (rows: y₁, y₂; cols: X bins)."""
    cen = 0.5 * (edges[:-1] + edges[1:])
    ncols = len(x_edges) - 1
    fig = plt.figure(figsize=(4.2 * ncols, 11.5))
    # leave a gap between the two y-blocks (top=0.94, bottom=0.10) to host the legend strip
    outer = fig.add_gridspec(2, 1, hspace=0.30, top=0.94, bottom=0.10)
    leg_handles = leg_labels = None
    for d in range(2):
        block = outer[d].subgridspec(2, ncols, height_ratios=[3, 1], hspace=0.05, wspace=0.22)
        for col in range(ncols):
            xl, xr = x_edges[col], x_edges[col + 1]
            md = (x1_data >= xl) & (x1_data < xr)
            central, stat, syst, total = _combined_bands(H_xb[d][col], H_cont_xb[d][col], band)
            h_d = np.histogram(y_np[md, d], bins=edges)[0]
            ax = fig.add_subplot(block[0, col]); axr = fig.add_subplot(block[1, col], sharex=ax)
            res = _draw_combined(ax, axr, cen, central, stat, syst, total, h_d,
                                 f"y{'₁' if d == 0 else '₂'}", K, nC, log=False,
                                 legend=False, return_handles=(d == 0 and col == 0))
            if res is not None:
                leg_handles, leg_labels = res
            ax.set_xlim(edges[0], edges[-1])
            if d == 0:
                ax.set_title(f"X = [{xl:g}, {xr:g}]", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Events (y{'₁' if d == 0 else '₂'})")
                axr.set_ylabel("Data/Model", fontsize=9)
    # single horizontal legend (boxed) in the gap between the two y-blocks
    if leg_handles is not None:
        fig.legend(leg_handles, leg_labels, loc="center", bbox_to_anchor=(0.5, 0.515),
                   ncol=len(leg_labels), fontsize=11, frameon=True, fancybox=False,
                   edgecolor="0.6", framealpha=0.95)
    fig.suptitle(f"Postfit scores by X bin — stat ⊕ syst (joint T × ν) — {label}",
                 fontsize=13, y=0.985)
    _save(fig, out_dir, f"postfit_total_xbinned_{label}")


def main():
    p = argparse.ArgumentParser(description="Postfit score distributions with a bootstrap-ensemble band.")
    p.add_argument("-c", "--cfg", required=True, help="YAML config (mixture-ensemble or profiling-ensemble).")
    p.add_argument("--ensemble", required=True, help="Glob of bootstrap member checkpoints.")
    p.add_argument("--ckpt", default=None,
                   help="Shared profiled-ensemble (residual) checkpoint + anchor ν₀. "
                        "Default: config paths.output_checkpoint; omit/absent => mixture-stage (no residual).")
    p.add_argument("--dataset", default=None, help="Override dataset .pt (default: config paths.dataset).")
    p.add_argument("--out-dir", default="postfit_ensemble")
    p.add_argument("--label", default=None, help="Name for the output files (default: stage).")
    p.add_argument("--band", choices=["envelope", "std"], default="envelope",
                   help="Band = per-bin min–max envelope (default) or ±1 std across members.")
    p.add_argument("--n-mc", type=int, default=100_000, help="MC samples per member.")
    p.add_argument("--batch-size", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=12345, help="Common-random-number seed (shared by all members).")
    p.add_argument("--bins", type=int, default=40, help="Score-axis bins.")
    p.add_argument("--range", default=None, help="Score axis range 'lo,hi' (default: data percentiles).")
    p.add_argument("--x-edges", default="-5,-1.5,-0.5,0.5,1.5,5", help="x₁ bin edges for the faceted plot.")
    p.add_argument("--no-distortion", action="store_true",
                   help="Skip the per-class distorted-simulation (A,B @ ν̂) vs original (ν=0) figure.")
    p.add_argument("--anchor", default=None,
                   help="ν₀ all members are sampled at: 'zero' or a comma list (e.g. '0.48,-0.52' "
                        "from a scan's anchor.txt). Default: the checkpoint's model.m_vector.")
    p.add_argument("--syst-scan2d", default=None,
                   help="Enable the combined stat⊕syst⊕total figure. Pass a 2D likelihood scan "
                        "'scan2d.npz' (produced by `likelihood_scan.py --scan-2d` on the profiled "
                        "checkpoint); it only supplies the −2Δln L contour geometry, separate from "
                        "the --ensemble members. The band is then built by sampling EVERY ensemble "
                        "member at EVERY point of the −2Δln L=<--level> contour (in the --pair "
                        "plane) around the anchor ν: stat = vary T at ν̂; syst = vary ν along the "
                        "contour with T averaged over members; total = joint min–max envelope over "
                        "the full member×ν grid (the exact joint variation — NOT a quadrature sum).")
    p.add_argument("--level", type=float, default=2.30,
                   help="−2Δln L level of the contour read from --syst-scan2d (default 2.30 ≈ 2D 1σ, "
                        "2 dof; use 5.99 for 95%% CL, 1.0 for a 1D interval).")
    p.add_argument("--pair", default="0,1",
                   help="Nuisance pair 'i,j' whose 2D contour is used for the syst band; must be a "
                        "pair present in --syst-scan2d. Other nuisances stay at the anchor ν.")
    p.add_argument("--n-contour", type=int, default=24,
                   help="Number of ν-points sampled along the −2Δln L contour (each is sampled for "
                        "every ensemble member, so cost scales as n_members × n_contour).")
    args = p.parse_args()

    cfg_path = os.path.abspath(args.cfg)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(cfg_path)
    for k, v in list(cfg["paths"].items()):
        if isinstance(v, str):
            cfg["paths"][k] = _resolve(v, cfg_dir)

    device = cfg.get("runtime", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        device = "cpu"

    res_ckpt = args.ckpt or cfg["paths"].get("output_checkpoint")
    model, residual_T, member_Ts, stage = load_ensemble(cfg, res_ckpt, args.ensemble, device)
    model.eval()
    label = args.label or stage

    num_nuis = model.m_vector.shape[0]
    if args.anchor is None:
        anchor_t = model.m_vector
    elif args.anchor == "zero":
        anchor_t = torch.zeros(num_nuis, device=device)
    else:
        vals = [float(v) for v in args.anchor.split(",")]
        if len(vals) != num_nuis:
            raise ValueError(f"--anchor expects {num_nuis} values, got {len(vals)}")
        anchor_t = torch.tensor(vals, dtype=torch.float32, device=device)
    print(f"Stage: {stage}   members: {len(member_Ts)}   sampling anchor ν₀ = "
          f"{anchor_t.detach().cpu().numpy()}"
          + ("  (checkpoint m_vector)" if args.anchor is None else "  (--anchor)"))

    dataset_path = args.dataset or cfg["paths"]["dataset"]
    dataset = torch.load(dataset_path, map_location=device)
    y_np = dataset["y_data_distorted"].cpu().numpy()
    x1_data = dataset["X_data_distorted"][:, 0].cpu().numpy()

    if args.range is not None:
        lo, hi = (float(v) for v in args.range.split(","))
    else:
        lo, hi = float(np.percentile(y_np, 0.5)), float(np.percentile(y_np, 99.5))
    edges = np.linspace(lo, hi, args.bins + 1)
    x_edges = np.array([float(v) for v in args.x_edges.split(",")])

    print(f"Sampling {len(member_Ts)} members (n_mc={args.n_mc}, common seed)…")
    H_incl, H_xb = _member_hists(model, residual_T, member_Ts, dataset, device, anchor_t,
                                 args.n_mc, args.batch_size, args.seed, edges, x_edges)

    os.makedirs(args.out_dir, exist_ok=True)
    plot_inclusive(H_incl, y_np, edges, args.band, len(member_Ts), label, args.out_dir)
    plot_xbinned(H_xb, y_np, x1_data, edges, x_edges, args.band, len(member_Ts), label, args.out_dir)

    # ---- per-class distorted simulation (A,B @ ν̂) vs original (ν=0) ----------
    if not args.no_distortion:
        cls_name = ["A", "B", "C", "D"][:model.n_flavours]
        colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
        nominal_t = torch.zeros(num_nuis, device=device)   # the "original" undistorted shape
        print(f"\nDistortion plot: per-class sample at ν̂ (distorted) and ν=0 (original)…")
        Hc_dist, Hc_dist_xb = _member_hists_byclass(model, residual_T, member_Ts, dataset, device,
                                                    anchor_t, args.n_mc, args.batch_size, args.seed,
                                                    edges, x_edges)
        Hc_orig, Hc_orig_xb = _member_hists_byclass(model, residual_T, member_Ts, dataset, device,
                                                    nominal_t, args.n_mc, args.batch_size, args.seed,
                                                    edges, x_edges)
        plot_distortion(Hc_dist, Hc_orig, y_np, edges, args.band, len(member_Ts), label,
                        args.out_dir, cls_name, colors)
        plot_distortion_xbinned(Hc_dist_xb, Hc_orig_xb, y_np, x1_data, edges, x_edges, args.band,
                                len(member_Ts), label, args.out_dir, cls_name, colors)

    # mean fractional band per score dim in the populated core (quick magnitude readout)
    print("Mean fractional ensemble (stat) band (populated core):")
    for d in range(2):
        central, blo, bhi = _band(H_incl[d], args.band)
        core = central > 0.05 * central.max()
        safe = np.where(central > 0, central, np.nan)
        rel = np.nanmean((bhi - blo)[core] / (2.0 * safe[core])) if core.any() else float("nan")
        print(f"  y{'₁' if d == 0 else '₂'}: {rel:.3%}" if np.isfinite(rel) else f"  y{d}: n/a")

    # ---- combined stat (ensemble T) ⊕ syst (ν contour) view -----------------
    if args.syst_scan2d is not None:
        pair = tuple(int(v) for v in args.pair.split(","))
        contour_ij = _contour_from_scan2d(args.syst_scan2d, pair, args.level, args.n_contour)
        m_points = []
        for vi, vj in contour_ij:
            m = anchor_t.detach().clone()
            m[pair[0]] = float(vi)
            m[pair[1]] = float(vj)
            m_points.append(m)
        nC = len(m_points)
        print(f"\nTotal band: every member sampled at {nC} points on the "
              f"-2Δln L={args.level:g} contour of {os.path.basename(args.syst_scan2d)} "
              f"(pair {pair}, FIXED-T single model). Joint envelope over the "
              f"{len(member_Ts)}×{nC} grid — no quadrature. ({len(member_Ts) * nC} MC passes)…")
        H_cont_incl, H_cont_xb = _contour_member_hists(model, residual_T, member_Ts, dataset,
                                                       device, m_points, args.n_mc,
                                                       args.batch_size, args.seed, edges, x_edges)
        plot_inclusive_combined(H_incl, H_cont_incl, y_np, edges, args.band, len(member_Ts),
                                nC, args.level, label, args.out_dir)
        plot_xbinned_combined(H_xb, H_cont_xb, y_np, x1_data, edges, x_edges, args.band,
                              len(member_Ts), nC, args.level, label, args.out_dir)

        # stat vs syst vs total magnitude readout (inclusive, populated core)
        print("Mean fractional band (populated core)   stat | syst | total:")
        for d in range(2):
            central, stat, syst, total = _combined_bands(H_incl[d], H_cont_incl[d], args.band)
            core = central > 0.05 * central.max()
            safe = np.where(central > 0, central, np.nan)
            f = lambda lo, hi: (np.nanmean((hi - lo)[core] / (2.0 * safe[core]))
                                if core.any() else float("nan"))
            print(f"  y{'₁' if d == 0 else '₂'}: "
                  f"{f(*stat):.3%} | {f(*syst):.3%} | {f(*total):.3%}")

    print(f"\nWritten to {args.out_dir}/")


if __name__ == "__main__":
    main()
