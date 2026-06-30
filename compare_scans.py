"""
Overlay two (or more) likelihood scans on one plot, from their saved npz files
(produced by likelihood_scan.py: scan1d.npz / scan2d.npz). Auto-detects 1D vs 2D.

  2D scans -> overlaid joint confidence CONTOURS (1σ/2σ, 2-dof Δχ²) per scan,
              with each scan's minimum marked and an optional truth marker.
  1D scans -> overlaid -2Δln L curves per nuisance, with σ level lines.

Pure numpy/matplotlib (no torch) — runs anywhere.

Usage
-----
  # ALL toys of a coverage run, auto-discovered (no manual globbing):
  python compare_scans.py --coverage-dir coverage/frozen \
      --nuis-labels "ν_shift,ν_squeeze" --expected 0.5,-0.5 --no-surface \
      --out-dir figs/coverage --tag frozen
  #   (use --scan-subdir mixscan for the step-1 mixture scan; each toy npz carries
  #    its own truth, so --expected is optional.)

  # view the coverage at a MODIFIED (e.g. calibrated) threshold — 2D via --levels,
  # 1D via --levels-1d (paste the thresholds from coverage_test.py --calibrate-coverage):
  python compare_scans.py --coverage-dir coverage/frozen --levels 8.44,17.15 \
      --expected 0.5,-0.5 --no-surface --out-dir figs/coverage_infl --tag infl

  # ALSO coverage wrt the toy-MEAN ν̂ (separates bias from width: gap = the bias effect):
  python compare_scans.py --coverage-dir coverage/frozen --vs-mean \
      --expected 0.5,-0.5 --no-surface --out-dir figs/coverage --tag frozen

  # declutter a many-toy overlay: draw ONLY the inner 1σ contour (coverage still printed):
  python compare_scans.py --coverage-dir coverage/frozen --one-sigma \
      --expected 0.5,-0.5 --no-surface --out-dir figs/coverage --tag frozen

  # PAPER figure: clean single 2D panel, mono-gray contour envelope, 1σ only:
  python compare_scans.py --coverage-dir coverage/frozen --mono --no-projections \
      --one-sigma --no-surface --expected 0.5,-0.5 \
      --nuis-labels '$\nu_{\rm shift}$,$\nu_{\rm squeeze}$' \
      --out-dir figs/coverage --tag frozen_paper

  # two 2D scans (e.g. step-1 mixture vs step-2 profiled):
  python compare_scans.py A/scan2d.npz B/scan2d.npz --labels mixture profiled \
      --expected 0.5,-0.5 --out-dir figs/scan_compare

  # two 1D scans:
  python compare_scans.py A/scan1d.npz B/scan1d.npz --labels mixture profiled \
      --nuis-labels "ν_shift,ν_squeeze" --out-dir figs/scan_compare
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# 2-dof joint Δχ² thresholds (1σ / 2σ) for 2D contours; 1-dof (1 / 4) for 1D.
LEVELS_2D = [2.296, 6.180]
LEVELS_1D = [1.0, 4.0]
PALETTE = ["#2166ac", "#b2182b", "#1b7837", "#762a83", "#d95f02", "#117733"]


def load_scan(path):
    d = np.load(path, allow_pickle=True)
    is2d = any(k.startswith("axis_i_") for k in d.files)
    return d, is2d


def _toy_idx(path):
    """Integer N from a .../toy_N/... path for natural sorting (toy_2 before toy_10)."""
    m = re.search(r"toy_(\d+)", path)
    return int(m.group(1)) if m else 0


def discover_coverage_scans(cov_dir, subdir="scan", fname=None):
    """Auto-find per-toy scan npz files under a coverage run directory.

    Layout: <cov_dir>/work/toy_N/<subdir>/<scan2d.npz|scan1d.npz>.  Accepts either
    the run dir (coverage/<run>) or its work/ dir.  `subdir` picks the scan stage
    (`scan` = step-2 profiled, `mixscan` = step-1 mixture).  If `fname` is None the
    same name is chosen for the whole run (prefer scan2d.npz, else scan1d.npz) so
    1D/2D are never mixed.  Returns (files, labels=['toy_N', ...]) toy-sorted.
    """
    base = os.path.join(cov_dir, "work") if os.path.isdir(os.path.join(cov_dir, "work")) else cov_dir
    toy_dirs = sorted(glob.glob(os.path.join(base, "toy_*")), key=_toy_idx)
    if not toy_dirs:
        return [], []
    names = [fname] if fname else ["scan2d.npz", "scan1d.npz"]
    pick = next((nm for nm in names
                 if any(os.path.exists(os.path.join(td, subdir, nm)) for td in toy_dirs)), None)
    if pick is None:
        return [], []
    files = [os.path.join(td, subdir, pick) for td in toy_dirs
             if os.path.exists(os.path.join(td, subdir, pick))]
    labels = [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in files]  # toy_N
    return files, labels


def pairs_in(d):
    """(i, j, suffix) for every 2D pair stored in the npz."""
    out = []
    for k in d.files:
        if k.startswith("m2dnll_") and f"axis_i_{k[len('m2dnll_'):]}" in d.files:
            sfx = k[len("m2dnll_"):]
            out.append((int(sfx[0]), int(sfx[1]), sfx))
    return out


def nuis_in_1d(d):
    """nuisance indices stored in a 1D npz (from axis_<ni> keys, excluding axis_i/j)."""
    nis = []
    for k in d.files:
        if k.startswith("axis_") and not k.startswith(("axis_i_", "axis_j_")):
            try:
                nis.append(int(k[len("axis_"):]))
            except ValueError:
                pass
    return sorted(nis)


def label_for(d, path, override):
    if override is not None:
        return override
    if "label" in d.files:
        return str(d["label"])
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


def nuis_name(nuis_labels, d, idx):
    if nuis_labels and idx < len(nuis_labels):
        return nuis_labels[idx]
    if "labels" in d.files:
        labs = list(d["labels"])
        if idx < len(labs):
            return str(labs[idx])
    return f"ν {idx}"


def _cols(n):
    """Cycle the palette so >len(PALETTE) scans all get a colour (zip used to truncate)."""
    return [PALETTE[k % len(PALETTE)] for k in range(n)]


def _curve_at(axis, curve, x):
    """Linear-interpolated −2Δlnℒ at x (None if x outside the scanned axis)."""
    if x is None or x < float(axis.min()) or x > float(axis.max()):
        return None
    return float(np.interp(x, axis, curve))


def _truth_for(d, expected_cli, ni):
    """Per-scan truth for nuisance ni: CLI --expected wins, else the scan's own
    stored `expected` (each toy npz carries it)."""
    if expected_cli is not None and ni < len(expected_cli):
        return float(expected_cli[ni])
    if "expected" in d.files:
        e = np.atleast_1d(d["expected"])
        if ni < len(e):
            return float(e[ni])
    return None


def _offsets(scans, labels, sfx, absolute):
    """Per-label absolute -2lnL offset (nll_min) for `absolute` mode.
    Returns (offs_dict, abs_ok). abs_ok is False if any scan lacks nll_min."""
    if not absolute:
        return {}, False
    offs, abs_ok = {}, True
    for (d, _), lab in zip(scans, labels):
        if f"m2dnll_{sfx}" not in d.files:
            continue
        if f"nll_min_{sfx}" in d.files:
            offs[lab] = float(d[f"nll_min_{sfx}"])
        else:
            abs_ok = False
    if not abs_ok:
        print("  [--absolute] nll_min not stored in npz — RERUN likelihood_scan.py to "
              "save it; showing relative −2Δlnℒ instead.")
        return {}, False
    return offs, True


def _cl_label(lev):
    """Asymptotic 2-dof confidence level of a Δχ² contour (Wilks: CL=1−e^{−Δχ²/2}),
    formatted 'NN% CL'. Exact for the standard levels (2.30→68%, 6.18→95%)."""
    return f"{100 * (1 - np.exp(-lev / 2)):.0f}% CL"


def _binom(k, n):
    """(fraction, binomial 1σ error) for k of n — the MC uncertainty on a coverage %."""
    p = k / n
    return p, (p * (1 - p) / n) ** 0.5


def plot_2d(scans, labels, pair, nuis_labels, expected, levels, out_path, absolute=False,
            mean_pt=None, one_sigma=False, mono=False, no_projections=False):
    """Corner layout: 2D joint contours + the two PROFILED 1D projections.
    `absolute`: projections show the absolute −2lnℒ (offset by each scan's nll_min)
    so the deeper-fitting scan bottoms out lower; contours stay relative (Δχ²).
    `one_sigma`: draw ONLY the inner (1σ) contour/projection line (declutters
    many-toy overlays); coverage is still computed at both levels and printed.
    `mono`: PAPER style — every toy contour in one muted gray, alpha-blended, with
    small grey best-fit dots (colour carried no information across interchangeable
    toys), so the eye reads contour density/envelope; truth + mean ν̂ stay bold.
    `no_projections`: single 2D panel only (drop the top/right profile panels)."""
    i, j, sfx = pair
    show2 = len(levels) > 1 and not one_sigma          # draw the outer (2σ) contour too?
    draw_levels = levels if show2 else levels[:1]
    offs, abs_ok = _offsets(scans, labels, sfx, absolute)
    floor = min(offs.values()) if abs_ok and offs else 0.0
    yl = r"$-2\ln\mathcal{L}$" if abs_ok else r"$-2\Delta\ln\mathcal{L}$"

    if no_projections:
        fig, axm = plt.subplots(figsize=(7, 6.6))
        axt = axr = axl = None                         # no projection / legend panels
    else:
        fig = plt.figure(figsize=(9, 9))
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.3], height_ratios=[1.3, 4],
                              hspace=0.06, wspace=0.06, top=0.93)
        axm = fig.add_subplot(gs[1, 0])
        axt = fig.add_subplot(gs[0, 0], sharex=axm)
        axr = fig.add_subplot(gs[1, 1], sharey=axm)
        axl = fig.add_subplot(gs[0, 1]); axl.axis("off")
    present_n = sum(1 for (d, _) in scans if f"m2dnll_{sfx}" in d.files)
    many = present_n > 6
    lw_c = 1.0 if (many or mono) else 1.6
    lw_p = 1.0 if (many or mono) else 1.8
    al = 0.22 if mono else (0.5 if many else 1.0)
    cols = ["0.45"] * len(scans) if mono else _cols(len(scans))
    cov1 = cov2 = ntot = 0
    truth_ref = None
    handles = []
    for (d, _), lab, col in zip(scans, labels, cols):
        if f"m2dnll_{sfx}" not in d.files:
            print(f"  ({lab}: no pair {sfx} — skipped)")
            continue
        ai, aj, m = d[f"axis_i_{sfx}"], d[f"axis_j_{sfx}"], d[f"m2dnll_{sfx}"]
        off = offs.get(lab, 0.0)
        axm.contour(ai, aj, m.T, levels=sorted(draw_levels), colors=[col], alpha=al,
                    linestyles=["-", "--", ":"][:len(draw_levels)], linewidths=lw_c)
        a, b = np.unravel_index(int(np.argmin(m)), m.shape)
        if mono:
            axm.plot(ai[a], aj[b], "o", color="0.3", ms=3.0, alpha=0.85, zorder=5)
        else:
            axm.plot(ai[a], aj[b], "*", color=col, ms=11 if many else 14, zorder=5)
        if axt is not None:
            axt.plot(ai, m.min(axis=1) + off, "-", color=col, lw=lw_p, alpha=al)  # profile out nuis j
            axr.plot(m.min(axis=0) + off, aj, "-", color=col, lw=lw_p, alpha=al)  # profile out nuis i
        # NB: under rebased-bma the per-scan nll_min is an ensemble-concordance artifact
        # (−2 ln⟨member-likelihood-vs-own-peak⟩), NOT a comparable absolute −2lnℒ, so it is
        # deliberately NOT shown in the legend to avoid implying one scan "fits better".
        if not many:                              # a 10+-entry legend is unreadable
            handles.append(Line2D([], [], color=col, lw=2, label=lab))
        # joint coverage: per-scan truth (CLI --expected wins, else the toy's own stored truth)
        ti, tj = _truth_for(d, expected, i), _truth_for(d, expected, j)
        if ti is not None and tj is not None:
            truth_ref = (ti, tj)
            v = _bilinear(ai, aj, m, ti, tj)
            tag = f"{v:.2f}" if v is not None else "out-of-window"
            if v is not None:
                ntot += 1
                cov1 += int(v <= levels[0]); cov2 += int(len(levels) > 1 and v <= levels[1])
            print(f"  {lab}: min=({ai[a]:+.4f}, {aj[b]:+.4f})  −2Δlnℒ(truth)={tag}"
                  + (f"  abs −2lnℒ min={off:.2f}" if abs_ok else ""))

    if axt is not None and not abs_ok:           # relative σ levels only make sense per-scan
        for lev, ls in ([(1.0, "--"), (4.0, ":")] if show2 else [(1.0, "--")]):
            axt.axhline(lev, color="gray", ls=ls, lw=1)
            axr.axvline(lev, color="gray", ls=ls, lw=1)
    if truth_ref is not None:
        axm.plot(truth_ref[0], truth_ref[1], "P", color="black", ms=13, zorder=6)
        if axt is not None:
            axt.axvline(truth_ref[0], color="black", ls="-.", lw=1.2)
            axr.axhline(truth_ref[1], color="black", ls="-.", lw=1.2)
        handles.append(Line2D([], [], color="black", marker="P", ls="none", ms=11, label="truth"))
    if mean_pt is not None:
        axm.plot(mean_pt[0], mean_pt[1], "X", color="#2166ac", ms=12, zorder=6)
        if axt is not None:
            axt.axvline(mean_pt[0], color="#2166ac", ls=":", lw=1.2)
            axr.axhline(mean_pt[1], color="#2166ac", ls=":", lw=1.2)
        handles.append(Line2D([], [], color="#2166ac", marker="X", ls="none", ms=10, label="mean ν̂"))
    if many:
        handles.insert(0, Line2D([], [], color="0.45" if mono else "gray", lw=2,
                                 label=f"{present_n} toys"))
        if ntot:
            p1, e1 = _binom(cov1, ntot)
            box = f"{_cl_label(levels[0])} covers truth: {cov1}/{ntot} = {p1:.0%} ± {e1:.0%}"
            msg = f"  [JOINT coverage] n={ntot}  Δχ²≤{levels[0]:.2f}={cov1}/{ntot}={p1:.0%}±{e1:.0%}"
            if show2:
                p2, e2 = _binom(cov2, ntot)
                box += f"\n{_cl_label(levels[1])} covers truth: {cov2}/{ntot} = {p2:.0%} ± {e2:.0%}"
                msg += f"  Δχ²≤{levels[1]:.2f}={cov2}/{ntot}={p2:.0%}±{e2:.0%}"
            axm.text(0.02, 0.98, box, transform=axm.transAxes, va="top", ha="left", fontsize=9,
                     bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
            print(msg)
    handles += [Line2D([], [], color="0.4", lw=1.5, ls="-",
                       label=f"inner {_cl_label(levels[0])} (Δχ²={levels[0]:.2f}, 2 dof)")]
    if show2:
        handles.append(Line2D([], [], color="0.4", lw=1.5, ls="--",
                              label=f"outer {_cl_label(levels[1])} (Δχ²={levels[1]:.2f}, 2 dof)"))

    axm.set_xlabel(nuis_name(nuis_labels, scans[0][0], i))
    axm.set_ylabel(nuis_name(nuis_labels, scans[0][0], j))
    if axt is not None:
        axt.set_ylabel(yl, fontsize=8); axt.set_ylim(floor, floor + 9)
        axr.set_xlabel(yl, fontsize=8); axr.set_xlim(floor, floor + 9)
        plt.setp(axt.get_xticklabels(), visible=False)
        plt.setp(axr.get_yticklabels(), visible=False)
        axl.legend(handles=handles, fontsize=8, loc="center")
    else:
        axm.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.9)
    proj = "absolute" if abs_ok else "profiled"
    title = "Likelihood-scan comparison — joint contours" + (
        "" if no_projections else f" + {proj} projections")
    if no_projections:                            # single panel: hug the title to the axes
        axm.set_title(title, fontsize=12, pad=8)
    else:
        fig.suptitle(title, fontsize=12, y=0.965)
    for ext in ("png", "pdf"):
        fig.savefig(out_path + f".{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    if abs_ok and offs:
        deepest = min(offs, key=offs.get)
        print(f"  deepest (lowest −2lnℒ min): {deepest} = {offs[deepest]:.2f}")
    print(f"  -> {out_path}.png")


def plot_bowl_3d(scans, labels, pair, nuis_labels, expected, zmax, out_path, absolute=False):
    """3D surface of the −2[Δ]ln L 'bowl' (window of height zmax). One scan ->
    viridis; several -> translucent per-scan surfaces. `absolute` offsets each
    surface by its nll_min so a deeper-fitting scan sits lower."""
    i, j, sfx = pair
    offs, abs_ok = _offsets(scans, labels, sfx, absolute)
    floor = min(offs.values()) if abs_ok and offs else 0.0
    present = [(d, lab, col) for (d, _), lab, col in zip(scans, labels, _cols(len(scans)))
               if f"m2dnll_{sfx}" in d.files]
    single = len(present) == 1
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    handles = []
    for d, lab, col in present:
        ai, aj, m = d[f"axis_i_{sfx}"], d[f"axis_j_{sfx}"], d[f"m2dnll_{sfx}"]
        off = offs.get(lab, 0.0)
        Ai, Aj = np.meshgrid(ai, aj, indexing="ij")
        Z = np.clip(m + off, floor, floor + zmax)
        if single:
            ax.plot_surface(Ai, Aj, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
        else:
            ax.plot_surface(Ai, Aj, Z, color=col, linewidth=0, antialiased=True, alpha=0.35)
        a, b = np.unravel_index(int(np.argmin(m)), m.shape)
        ax.scatter([ai[a]], [aj[b]], [off], color=col, s=70, marker="*", depthshade=False)
        handles.append(Line2D([], [], color=col, lw=6, label=lab))  # see plot_2d: nll_min hidden
    if expected is not None:
        ax.scatter([expected[i]], [expected[j]], [floor], color="black", s=90, marker="P",
                   depthshade=False)
        handles.append(Line2D([], [], color="black", marker="P", ls="none", ms=10, label="truth"))
    ax.set_xlabel(nuis_name(nuis_labels, scans[0][0], i))
    ax.set_ylabel(nuis_name(nuis_labels, scans[0][0], j))
    ax.set_zlabel(r"$-2\ln\mathcal{L}$" if abs_ok else r"$-2\Delta\ln\mathcal{L}$")
    ax.set_zlim(floor, floor + zmax)
    ax.set_title(("Absolute " if abs_ok else "") + f"likelihood bowl depth (height {zmax:g})")
    if handles:
        ax.legend(handles=handles, fontsize=9, loc="upper left")
    ax.view_init(elev=35, azim=-60)
    for ext in ("png", "pdf"):
        fig.savefig(out_path + f".{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}.png")


def _bilinear(ai, aj, m, xi, xj):
    if not (ai[0] <= xi <= ai[-1] and aj[0] <= xj <= aj[-1]):
        return None
    a = int(np.clip(np.searchsorted(ai, xi) - 1, 0, len(ai) - 2))
    b = int(np.clip(np.searchsorted(aj, xj) - 1, 0, len(aj) - 2))
    tx = (xi - ai[a]) / (ai[a + 1] - ai[a])
    ty = (xj - aj[b]) / (aj[b + 1] - aj[b])
    return float(m[a, b] * (1 - tx) * (1 - ty) + m[a + 1, b] * tx * (1 - ty)
                 + m[a, b + 1] * (1 - tx) * ty + m[a + 1, b + 1] * tx * ty)


def plot_bestfit_scatter(scans, labels, pair, nuis_labels, expected, levels, out_path,
                         mean_pt=None, mean_cov=None, one_sigma=False):
    """Per-toy best-fit cloud vs truth — the readable coverage view when many
    contour-corners would overlap into mush. Each toy = one best-fit point
    (argmin of its 2D scan), coloured by whether truth falls within its joint 1σ
    region (−2Δlnℒ(truth) ≤ levels[0]); a faint segment links each best-fit to
    truth (the 'pull'). Prints + annotates the 1σ/2σ joint coverage fraction
    (`one_sigma` shows the 1σ count only)."""
    i, j, sfx = pair
    show2 = len(levels) > 1 and not one_sigma
    fig, ax = plt.subplots(figsize=(7, 6.2))
    cov1 = cov2 = ntot = 0
    truth_ref = None
    pts_in, pts_out = [], []
    for (d, _), lab in zip(scans, labels):
        if f"m2dnll_{sfx}" not in d.files:
            continue
        ai, aj, m = d[f"axis_i_{sfx}"], d[f"axis_j_{sfx}"], d[f"m2dnll_{sfx}"]
        a, b = np.unravel_index(int(np.argmin(m)), m.shape)
        bf = (float(ai[a]), float(aj[b]))
        ti, tj = _truth_for(d, expected, i), _truth_for(d, expected, j)
        if ti is None or tj is None:
            continue
        truth_ref = (ti, tj)
        v = _bilinear(ai, aj, m, ti, tj)
        ntot += 1
        inside = v is not None and v <= levels[0]
        cov1 += int(inside); cov2 += int(v is not None and len(levels) > 1 and v <= levels[1])
        (pts_in if inside else pts_out).append(bf)
        ax.plot([bf[0], ti], [bf[1], tj], "-", color="gray", lw=0.6, alpha=0.5, zorder=1)
    for pts, col, lab in [(pts_in, "#1b7837", f"truth in Δχ²≤{levels[0]:.2f}"),
                          (pts_out, "#b2182b", f"truth outside Δχ²≤{levels[0]:.2f}")]:
        if pts:
            a = np.asarray(pts)
            ax.scatter(a[:, 0], a[:, 1], c=col, s=55, edgecolor="k", lw=0.5,
                       zorder=3, label=lab)
    if truth_ref is not None:
        ax.plot(*truth_ref, "P", color="black", ms=15, zorder=4, label="truth")
    if mean_pt is not None:
        if truth_ref is not None:                                   # the bias vector mean→truth
            ax.plot([mean_pt[0], truth_ref[0]], [mean_pt[1], truth_ref[1]], "-",
                    color="#2166ac", lw=1.4, alpha=0.8, zorder=2)
        ax.plot(*mean_pt, "X", color="#2166ac", ms=15, zorder=5, label="mean ν̂")
    ax.set_xlabel(nuis_name(nuis_labels, scans[0][0], i))
    ax.set_ylabel(nuis_name(nuis_labels, scans[0][0], j))
    if ntot:
        title = (f"Best-fit scatter vs truth — {ntot} toys\n"
                 f"coverage vs truth  Δχ²≤{levels[0]:.2f}: {cov1}/{ntot}={cov1/ntot:.0%}"
                 + (f"   ≤{levels[1]:.2f}: {cov2}/{ntot}={cov2/ntot:.0%}" if show2 else ""))
        if mean_cov is not None:
            mc1, mc2, nm = mean_cov
            title += (f"\ncoverage vs mean ν̂  Δχ²≤{levels[0]:.2f}: {mc1}/{nm}={mc1/nm:.0%}"
                      + (f"   ≤{levels[1]:.2f}: {mc2}/{nm}={mc2/nm:.0%}" if show2 else "")
                      + "  (bias removed)")
        ax.set_title(title, fontsize=9)
        print(f"  [scatter] n={ntot}  Δχ²≤{levels[0]:.2f}={cov1}/{ntot}={cov1/ntot:.0%}"
              + (f"  ≤{levels[1]:.2f}={cov2}/{ntot}={cov2/ntot:.0%}" if show2 else ""))
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_path + f".{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}.png")


def plot_1d(scans, labels, ni, nuis_labels, expected, out_path, recenter=False, levels=(1.0, 4.0),
            mean=None, one_sigma=False):
    """Overlay the 1D profile curves of several scans for one nuisance.

    Designed for BOTH use-cases:
      * few scans (methods on one toy)   -> thick coloured curves + per-curve legend;
      * many scans (toys of one method)  -> thin translucent curves + a bold MEDIAN curve,
        and a printed/annotated COVERAGE readout (fraction of toys whose truth sits within
        1σ/2σ of the profile) + the best-fit spread vs the reported σ (the pull).
    `recenter` shifts each curve's x-axis by its own truth so the minima stack about 0
    (truth at x=0) — the cleanest view of the best-fit scatter across toys.
    `one_sigma` draws only the inner (1σ) level line + 1σ coverage readout.
    """
    show2 = len(levels) > 1 and not one_sigma
    n = len(scans)
    cols = _cols(n)
    many = n > 8
    fig, ax = plt.subplots(figsize=(7.5, 5))

    bestfits, sigmas = [], []
    cov1 = cov2 = ntot = 0
    truth_ref = None
    for (d, _), lab, col in zip(scans, labels, cols):
        if f"axis_{ni}" not in d.files:
            continue
        axis = np.asarray(d[f"axis_{ni}"], dtype=float)
        curve = np.asarray(d[f"m2dnll_{ni}"], dtype=float)
        truth = _truth_for(d, expected, ni)
        if truth is not None:
            truth_ref = truth
        x = axis - truth if (recenter and truth is not None) else axis
        if many:
            ax.plot(x, curve, "-", color=col, lw=0.9, alpha=0.4)
        else:
            ax.plot(x, curve, "-", color=col, lw=2, label=lab)
            ax.plot(x[int(np.argmin(curve))], float(np.min(curve)), "*", color=col, ms=13)
        # coverage / pull bookkeeping
        bestfits.append(float(axis[int(np.argmin(curve))]))
        if f"bestfit_{ni}" in d.files:
            b = np.atleast_1d(np.asarray(d[f"bestfit_{ni}"], dtype=float))
            if b.size > 1 and np.isfinite(b[1]):
                sigmas.append(float(b[1]))
        if truth is not None:
            v = _curve_at(axis, curve, truth)
            if v is not None:
                ntot += 1
                cov1 += int(v <= levels[0])
                cov2 += int(len(levels) > 1 and v <= levels[1])

    # bold median curve when overlaying many toys (only if on a common grid)
    if many and f"axis_{ni}" in scans[0][0].files:
        ax0 = np.asarray(scans[0][0][f"axis_{ni}"], dtype=float)
        stack = [np.asarray(d[f"m2dnll_{ni}"], dtype=float)
                 for (d, _) in scans
                 if f"axis_{ni}" in d.files
                 and np.asarray(d[f"axis_{ni}"]).shape == ax0.shape
                 and np.allclose(np.asarray(d[f"axis_{ni}"], dtype=float), ax0)]
        if stack:
            xm = ax0 - truth_ref if (recenter and truth_ref is not None) else ax0
            ax.plot(xm, np.median(np.vstack(stack), axis=0), "-", color="black",
                    lw=2.6, label=f"median ({len(stack)} toys)")

    for lev, ls in zip((levels if show2 else levels[:1]), ("--", ":")):
        ax.axhline(lev, color="gray", ls=ls, lw=1.2, alpha=0.8)
    tx = 0.0 if recenter else truth_ref
    if tx is not None:
        ax.axvline(tx, color="black", ls="-.", lw=1.4, label="truth")
    if mean is not None:
        mx = mean - truth_ref if (recenter and truth_ref is not None) else mean
        ax.axvline(mx, color="#2166ac", ls=":", lw=1.5, label="mean ν̂")

    name = nuis_name(nuis_labels, scans[0][0], ni)
    if ntot:
        f1 = cov1 / ntot
        bf = np.asarray(bestfits)
        msig = float(np.median(sigmas)) if sigmas else float("nan")
        pull = bf.std() / msig if (sigmas and msig > 0) else float("nan")
        box = f"coverage  ≤{levels[0]:g}: {cov1}/{ntot}={f1:.0%}"
        cov = f"coverage ≤{levels[0]:g}={f1:.0%}"
        if show2:
            f2 = cov2 / ntot
            box += f"   ≤{levels[1]:g}: {cov2}/{ntot}={f2:.0%}"
            cov += f" ≤{levels[1]:g}={f2:.0%}"
        ax.text(0.02, 0.97, box, transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
        print(f"  [{name}] n={ntot}  bestfit mean={bf.mean():+.4f} std={bf.std():.4f}  "
              f"median σ={msig:.4f}  pull(std/σ)={pull:.2f}  " + cov)

    ax.set_xlabel((r"$\nu - \nu_{\rm true}$ — " if recenter else "") + name)
    ax.set_ylabel(r"$-2\,\Delta\ln\mathcal{L}$")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Likelihood-scan overlay — {name}"
                 + (f"  ({n} scans)" if many else ""))
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_path + f".{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}.png")


# Pretty names for the known per-toy scan stages (used in the coverage-by-stage view).
STAGE_NAMES = {"scan": "step-2\n(profiled)", "mixscan": "step-1\n(mixture)"}


def coverage_2d(scans, pair, expected, levels):
    """(cov1, cov2, ntot): how many toys have −2Δlnℒ(truth) ≤ joint 1σ/2σ levels."""
    i, j, sfx = pair
    cov1 = cov2 = ntot = 0
    for d, _ in scans:
        if f"m2dnll_{sfx}" not in d.files:
            continue
        ti, tj = _truth_for(d, expected, i), _truth_for(d, expected, j)
        if ti is None or tj is None:
            continue
        v = _bilinear(d[f"axis_i_{sfx}"], d[f"axis_j_{sfx}"], d[f"m2dnll_{sfx}"], ti, tj)
        if v is None:
            continue
        ntot += 1; cov1 += int(v <= levels[0]); cov2 += int(v <= levels[1])
    return cov1, cov2, ntot


def coverage_1d(scans, ni, expected, levels=(1.0, 4.0)):
    """(cov1, cov2, ntot): toys with −2Δlnℒ(truth) ≤ 1/4 (1σ/2σ, 1-dof) for nuisance ni."""
    cov1 = cov2 = ntot = 0
    for d, _ in scans:
        if f"axis_{ni}" not in d.files:
            continue
        v = _curve_at(np.asarray(d[f"axis_{ni}"], float),
                      np.asarray(d[f"m2dnll_{ni}"], float), _truth_for(d, expected, ni))
        if v is None:
            continue
        ntot += 1; cov1 += int(v <= levels[0]); cov2 += int(v <= levels[1])
    return cov1, cov2, ntot


# --- coverage wrt the toy-mean ν̂ (bias removed → pure width) ----------------------------- #

def _mean_bestfit_2d(scans, pair):
    """Mean of the per-toy 2D best-fit points (argmin of each grid). (mi, mj) or None."""
    i, j, sfx = pair
    pts = []
    for d, _ in scans:
        if f"m2dnll_{sfx}" not in d.files:
            continue
        m = d[f"m2dnll_{sfx}"]
        a, b = np.unravel_index(int(np.argmin(m)), m.shape)
        pts.append((float(d[f"axis_i_{sfx}"][a]), float(d[f"axis_j_{sfx}"][b])))
    if not pts:
        return None
    arr = np.asarray(pts)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _mean_bestfit_1d(scans, ni):
    """Mean of the per-toy 1D best-fit values (argmin of each curve). float or None."""
    vals = [float(d[f"axis_{ni}"][int(np.argmin(d[f"m2dnll_{ni}"]))])
            for d, _ in scans if f"axis_{ni}" in d.files]
    return float(np.mean(vals)) if vals else None


def coverage_2d_at(scans, pair, ci, cj, levels):
    """Joint coverage wrt the FIXED point (ci, cj) (e.g. the toy-mean ν̂): toys with
    −2Δlnℒ(ci,cj) ≤ levels. Centring on the cloud mean removes the bias → pure width."""
    i, j, sfx = pair
    cov1 = cov2 = ntot = 0
    for d, _ in scans:
        if f"m2dnll_{sfx}" not in d.files:
            continue
        v = _bilinear(d[f"axis_i_{sfx}"], d[f"axis_j_{sfx}"], d[f"m2dnll_{sfx}"], ci, cj)
        if v is None:
            continue
        ntot += 1; cov1 += int(v <= levels[0]); cov2 += int(v <= levels[1])
    return cov1, cov2, ntot


def coverage_1d_at(scans, ni, center, levels):
    """Per-nuisance coverage wrt the FIXED point `center` (the toy-mean ν̂)."""
    cov1 = cov2 = ntot = 0
    for d, _ in scans:
        if f"axis_{ni}" not in d.files:
            continue
        v = _curve_at(np.asarray(d[f"axis_{ni}"], float), np.asarray(d[f"m2dnll_{ni}"], float), center)
        if v is None:
            continue
        ntot += 1; cov1 += int(v <= levels[0]); cov2 += int(v <= levels[1])
    return cov1, cov2, ntot


def _print_vs_mean(prefix, levels, truth_cov, mean_cov, bias_str):
    """Report coverage wrt truth vs wrt the toy-mean, separating bias from width."""
    tc1, _, nt = truth_cov
    mc1, _, nm = mean_cov
    ft = tc1 / nt if nt else float("nan")
    fm = mc1 / nm if nm else float("nan")
    gap = fm - ft
    hint = ("BIAS-driven (centring on the mean recovers coverage)" if gap > 0.1 and fm >= 0.55
            else "BIAS + WIDTH both contribute" if gap > 0.1
            else "WIDTH-driven (mean-centred coverage still low)")
    print(f"  [vs-mean {prefix}] ≤{levels[0]:g}:  truth-cov={tc1}/{nt}={ft:.0%}  "
          f"mean-cov={mc1}/{nm}={fm:.0%}  (gap {gap:+.0%}){bias_str}  -> {hint}")


def plot_coverage_compare(key_name, rows, out_path):
    """Grouped bar of coverage fraction per stage (e.g. step-1 mixscan vs step-2 profiled).
    rows: list of (stage_subdir, cov1, cov2, ntot); dashed lines at the 68/95% targets."""
    disp = [STAGE_NAMES.get(s, s) for s, *_ in rows]
    f1 = [c1 / nt if nt else 0.0 for _, c1, c2, nt in rows]
    f2 = [c2 / nt if nt else 0.0 for _, c1, c2, nt in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(2.6 + 1.5 * len(rows), 5))
    b1 = ax.bar(x - w / 2, f1, w, color="#2166ac", label="1σ")
    b2 = ax.bar(x + w / 2, f2, w, color="#b2182b", label="2σ")
    ax.axhline(0.6827, color="#2166ac", ls="--", lw=1.2)
    ax.axhline(0.9545, color="#b2182b", ls="--", lw=1.2)
    ax.text(len(rows) - 0.5, 0.6827, " 68%", color="#2166ac", va="bottom", ha="right", fontsize=8)
    ax.text(len(rows) - 0.5, 0.9545, " 95%", color="#b2182b", va="bottom", ha="right", fontsize=8)
    for bars, fr in [(b1, f1), (b2, f2)]:
        for rect, val in zip(bars, fr):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 0.012, f"{val:.0%}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={nt})" for d, (_, _, _, nt) in zip(disp, rows)])
    ax.set_ylim(0, 1.08); ax.set_ylabel("coverage fraction")
    ax.set_title(f"Coverage by stage — {key_name}")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_path + f".{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}.png")


def run_overlay(inputs, labels_in, tag, args, nuis_labels, expected):
    """Build scans, emit the overlay/scatter plots for `tag`, and RETURN
    (is2d, stats) with stats keyed by pair-suffix (2D) or nuisance index (1D):
    {key: (cov1, cov2, ntot)}."""
    scans = [load_scan(f) for f in inputs]
    labels = [label_for(d, f, l) for (d, _), f, l in zip(scans, inputs, labels_in)]
    dims = {is2d for _, is2d in scans}
    if len(dims) > 1:
        raise SystemExit("mixing 1D and 2D scans — compare like with like.")
    is2d = dims.pop()
    print(f"[{tag}] {len(scans)} {'2D' if is2d else '1D'} scans: {labels}")
    stats = {}
    if is2d:
        levels = [float(v) for v in args.levels.split(",")] if args.levels else LEVELS_2D
        pairs = pairs_in(scans[0][0])
        if not pairs:
            raise SystemExit("no 2D pair found in the first npz")
        for pair in pairs:
            mean_pt = _mean_bestfit_2d(scans, pair) if args.vs_mean else None
            mean_cov = (coverage_2d_at(scans, pair, mean_pt[0], mean_pt[1], levels)
                        if mean_pt is not None else None)
            plot_2d(scans, labels, pair, nuis_labels, expected, levels,
                    os.path.join(args.out_dir, f"{tag}_2d_{pair[2]}"), args.absolute,
                    mean_pt=mean_pt, one_sigma=args.one_sigma, mono=args.mono,
                    no_projections=args.no_projections)
            plot_bestfit_scatter(scans, labels, pair, nuis_labels, expected, levels,
                                 os.path.join(args.out_dir, f"{tag}_bestfit_{pair[2]}"),
                                 mean_pt=mean_pt, mean_cov=mean_cov, one_sigma=args.one_sigma)
            if args.surface:
                plot_bowl_3d(scans, labels, pair, nuis_labels, expected, args.zmax,
                             os.path.join(args.out_dir, f"{tag}_bowl_{pair[2]}"), args.absolute)
            stats[pair[2]] = coverage_2d(scans, pair, expected, levels)
            if mean_pt is not None and mean_cov is not None:
                ti, tj = _truth_for(scans[0][0], expected, pair[0]), _truth_for(scans[0][0], expected, pair[1])
                bias_str = (f"  bias=({mean_pt[0] - ti:+.4f},{mean_pt[1] - tj:+.4f})"
                            if ti is not None and tj is not None else "")
                _print_vs_mean(f"joint {pair[2]}", levels, stats[pair[2]], mean_cov, bias_str)
    else:
        levels_1d = [float(v) for v in args.levels_1d.split(",")] if args.levels_1d else LEVELS_1D
        for ni in nuis_in_1d(scans[0][0]):
            mean_ni = _mean_bestfit_1d(scans, ni) if args.vs_mean else None
            plot_1d(scans, labels, ni, nuis_labels, expected,
                    os.path.join(args.out_dir, f"{tag}_1d_nuis{ni}"), args.recenter_truth,
                    levels=levels_1d, mean=mean_ni, one_sigma=args.one_sigma)
            stats[ni] = coverage_1d(scans, ni, expected, levels=levels_1d)
            if mean_ni is not None:
                t = _truth_for(scans[0][0], expected, ni)
                bias_str = f"  bias={mean_ni - t:+.4f}" if t is not None else ""
                _print_vs_mean(nuis_name(nuis_labels, scans[0][0], ni), levels_1d,
                               stats[ni], coverage_1d_at(scans, ni, mean_ni, levels_1d), bias_str)
    return is2d, stats


def main():
    p = argparse.ArgumentParser(description="Overlay likelihood scans from npz files.")
    p.add_argument("inputs", nargs="*", help="Scan npz files (scan1d.npz / scan2d.npz). "
                   "Optional if --coverage-dir is given.")
    p.add_argument("--coverage-dir", default=None,
                   help="Coverage run dir (e.g. coverage/frozen): auto-discovers every "
                        "work/toy_*/<scan-subdir>/scan{2d,1d}.npz and labels them toy_N.")
    p.add_argument("--scan-subdir", default="scan",
                   help="Per-toy subdir holding the scan npz (default 'scan' = step-2 profiled; "
                        "'mixscan' = step-1 mixture).")
    p.add_argument("--scan-file", default=None,
                   help="Force the scan npz filename inside each toy (default: auto, prefer "
                        "scan2d.npz else scan1d.npz).")
    p.add_argument("--stages", nargs="+", default=None,
                   help="Compare coverage across several per-toy subdirs, e.g. "
                        "--stages scan mixscan (step-2 profiled vs step-1 mixture). Requires "
                        "--coverage-dir; emits per-stage overlays + a coverage-by-stage bar.")
    p.add_argument("--labels", nargs="+", default=None, help="Legend label per input.")
    p.add_argument("--nuis-labels", default=None, help="Comma list, e.g. 'ν_shift,ν_squeeze'.")
    p.add_argument("--expected", default=None, help="Truth point, comma list e.g. '0.5,-0.5'.")
    p.add_argument("--levels", default=None,
                   help="2D contour Δχ² levels, comma list (default 1σ/2σ 2-dof: 2.30,6.18). "
                        "Pass the calibrated Δχ² thresholds from coverage_test.py "
                        "--calibrate-coverage to view the joint coverage at the inflated CI.")
    p.add_argument("--levels-1d", default=None,
                   help="1D −2Δlnℒ threshold levels, comma list (default 1σ/2σ 1-dof: 1,4). "
                        "Pass a calibrated per-nuisance threshold to view 1D coverage at the "
                        "inflated CI (the σ-lines, coverage count, and bestfit-scatter all use it).")
    p.add_argument("--out-dir", default="scan_compare")
    p.add_argument("--tag", default="compare", help="Output filename prefix.")
    p.add_argument("--no-surface", dest="surface", action="store_false",
                   help="Skip the 3D bowl surface (2D mode).")
    p.add_argument("--absolute", action="store_true",
                   help="Projections/bowl show ABSOLUTE −2lnℒ (offset by each scan's "
                        "stored nll_min) so the deeper-fitting scan bottoms out lower. "
                        "Requires npz produced by the updated likelihood_scan.py; only "
                        "meaningful when the scans used the SAME data + constraints.")
    p.add_argument("--zmax", type=float, default=12.0, help="3D bowl z-window height (−2Δln L).")
    p.add_argument("--recenter-truth", action="store_true",
                   help="(1D) shift each curve's x-axis by its own truth so the minima stack "
                        "about 0 (truth at x=0) — best view of the best-fit scatter across toys.")
    p.add_argument("--one-sigma", action="store_true",
                   help="Draw ONLY the inner (1σ) contour / level line, not the 2σ outer one — "
                        "declutters many-toy overlays. Coverage is still computed at both levels "
                        "(printed to stdout); the on-plot annotation/legend show 1σ only. Also "
                        "lets you pass a single-value --levels/--levels-1d without an IndexError.")
    p.add_argument("--mono", action="store_true",
                   help="(2D) PAPER style: draw every toy contour in one muted gray (colour "
                        "carried no information across interchangeable toys), alpha-blended, with "
                        "small grey best-fit dots — the eye reads contour density/envelope. Truth "
                        "(+) and mean ν̂ (X) stay bold; the contour legend is labelled by CL.")
    p.add_argument("--no-projections", action="store_true",
                   help="(2D) Drop the top/right profiled-projection panels for a clean single "
                        "2D panel for the paper. Recommended paper figure: "
                        "--mono --no-projections --one-sigma.")
    p.add_argument("--vs-mean", action="store_true",
                   help="ALSO compute coverage wrt the toy-MEAN ν̂ (not just the truth), to "
                        "separate bias from width: truth-coverage mixes bias+width, mean-coverage "
                        "removes the bias (centres on the cloud), so the gap = the bias effect. "
                        "Marks the mean ν̂ + the bias vector on the 2D/scatter/1D plots and prints "
                        "truth-cov vs mean-cov + the bias per pair/nuisance.")
    p.set_defaults(surface=True)
    args = p.parse_args()

    nuis_labels = args.nuis_labels.split(",") if args.nuis_labels else None
    expected = [float(v) for v in args.expected.split(",")] if args.expected else None
    os.makedirs(args.out_dir, exist_ok=True)

    # Which per-toy subdir(s) to process. --stages (multi) overrides --scan-subdir.
    if args.coverage_dir:
        if args.inputs:
            p.error("pass EITHER --coverage-dir OR explicit npz files, not both")
        stages = args.stages if args.stages else [args.scan_subdir]
    else:
        if args.stages:
            p.error("--stages requires --coverage-dir")
        stages = [None]                          # single run from positional inputs
    multi = len(stages) > 1

    stage_results = {}                           # stage -> (is2d, {key: (cov1, cov2, ntot)})
    for stage in stages:
        if args.coverage_dir:
            inputs, auto_labels = discover_coverage_scans(args.coverage_dir, stage, args.scan_file)
            if not inputs:
                msg = (f"no toy scans found under {args.coverage_dir} "
                       f"(work/toy_*/{stage}/).")
                if multi:
                    print(f"  [skip stage '{stage}'] {msg}"); continue
                p.error(msg + " Try --scan-subdir / --scan-file.")
            print(f"Discovered {len(inputs)} toy scans under {args.coverage_dir} "
                  f"({stage}/{os.path.basename(inputs[0])}).")
            labels_in = args.labels or auto_labels
        else:
            inputs = list(args.inputs)
            labels_in = args.labels or [None] * len(inputs)
        if len(inputs) < 2:
            p.error("give at least two npz files to compare (via positionals or --coverage-dir)")
        if labels_in is not None and len(labels_in) != len(inputs):
            p.error(f"--labels needs {len(inputs)} entries")
        tag = f"{args.tag}_{stage}" if multi else args.tag
        stage_results[stage] = run_overlay(inputs, labels_in, tag, args, nuis_labels, expected)

    # Cross-stage coverage comparison (e.g. step-1 mixscan vs step-2 profiled).
    if multi and stage_results:
        any_is2d = next(iter(stage_results.values()))[0]
        keys = list(next(iter(stage_results.values()))[1].keys())
        print("\n=== coverage by stage ===")
        for key in keys:
            rows = [(stage, *stats[key]) for stage, (_, stats) in stage_results.items()
                    if key in stats]
            if not rows:
                continue
            if any_is2d:
                kname, ksfx = f"joint ({key})", key
            else:
                kname = nuis_labels[key] if (nuis_labels and key < len(nuis_labels)) else f"ν{key}"
                ksfx = f"nuis{key}"
            for stage, c1, c2, nt in rows:
                tg = "1σ/2σ joint" if any_is2d else "1σ/2σ"
                print(f"  {stage:9s} {kname}: {tg} = {c1}/{nt}={c1/nt:.0%} / "
                      f"{c2}/{nt}={c2/nt:.0%}" if nt else f"  {stage} {kname}: n=0")
            plot_coverage_compare(kname, rows,
                                  os.path.join(args.out_dir, f"{args.tag}_coverage_by_stage_{ksfx}"))

    print(f"Done -> {args.out_dir}/")


if __name__ == "__main__":
    main()
