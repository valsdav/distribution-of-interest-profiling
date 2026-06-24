"""
Visualise the residual-transfer distortion of the scores along the post-fit Hessian
eigenvectors.

`hessian_bma.py` gives the post-fit covariance C_BMA at the best fit ν̂; its eigenvectors
are the principal axes of the nuisance uncertainty in ν-space. This script shows what each
axis *means* for the data: the average displacement the transfer T applies to the 2D scores
when ν is moved ±1σ along a Hessian eigenvector, relative to nominal (ν̂) — the "Δ T map
from nominal", with the eigenvectors as the displacement directions.

Only the ν-dependent residual stack of `residual_transfer_model` matters (the base transfer
is frozen / ν-independent), so no ensemble loop is needed — just the residual transfer from
the profiled checkpoint. The displacement is

    Δ_k^s(y) = ⟨ stack(y, ctx ; ν̂ + s·σ_k·e_k) − stack(y, ctx ; ν̂) ⟩_ctx ,   s ∈ {−1, +1}

with e_k, σ_k=√λ_k the k-th eigenvector / 1σ extent of C_BMA, averaged over context at a
chosen flavour (a data sample of kinematics x, or a fixed `--x`).

Usage:
    python visualize_hessian_distortion.py -c configs/profiling_v15_ensemble.yaml \
        --ckpt models/full_mixture_model_v15_ensemble_profiled.pt \
        --hessian-npz hessian_bma/v15/hessian_bma_ensemble_profiled.npz \
        --flavour 0 --out-dir hessian_distortion/v15
    # fixed kinematics instead of the data average:
    python visualize_hessian_distortion.py ... --x 0.0,0.0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import numpy as np
import torch
import yaml

from likelihood_scan import load_models, _resolve, _save
from plot_residual_field import _residual_stack
from plotting import field_quiver_panel, truncate_cmap


@torch.no_grad()
def _disp_field(residual_T, grid, contexts, m_off, m_ref, ng, chunk_rows=200_000):
    """⟨ stack(grid, ctx; m_off) − stack(grid, ctx; m_ref) ⟩ over `contexts` → [ng, ng, 2].

    Averages the residual-stack displacement over the rows of `contexts` (the chosen
    flavour's one-hot + sampled/fixed x). The grid × context outer product is processed in
    chunks bounded by `chunk_rows` so peak memory stays modest."""
    G = grid.shape[0]
    Nc = contexts.shape[0]
    nc = max(1, min(Nc, chunk_rows // max(G, 1)))
    acc = torch.zeros(G, 2, dtype=torch.float64, device=grid.device)
    total = 0
    for c0 in range(0, Nc, nc):
        cb = contexts[c0:c0 + nc]
        ncb = cb.shape[0]
        grid_t = grid.repeat_interleave(ncb, dim=0)            # [(g) repeated], row g*ncb+i
        ctx_t = cb.repeat(G, 1)                                # [(ctx) tiled] -> (grid[g], cb[i])
        m_off_b = m_off.unsqueeze(0).expand(grid_t.shape[0], -1)
        m_ref_b = m_ref.unsqueeze(0).expand(grid_t.shape[0], -1)
        d = (_residual_stack(residual_T, grid_t, ctx_t, m_off_b)
             - _residual_stack(residual_T, grid_t, ctx_t, m_ref_b))
        acc += d.double().reshape(G, ncb, 2).sum(1)
        total += ncb
    return (acc / total).reshape(ng, ng, 2).cpu().numpy()


def main():
    p = argparse.ArgumentParser(
        description="Residual-T score distortion along the post-fit Hessian eigenvectors.")
    p.add_argument("-c", "--cfg", required=True)
    p.add_argument("--ckpt", default=None, help="Profiled checkpoint (default: config output_checkpoint).")
    p.add_argument("--hessian-npz", required=True, help="hessian_bma_*.npz (eigenvectors + ν̂).")
    p.add_argument("--dataset", default=None, help="Default: config paths.dataset (grid range + x sample).")
    p.add_argument("--out-dir", default="hessian_distortion")
    p.add_argument("--label", default=None)
    p.add_argument("--flavour", type=int, default=0)
    p.add_argument("--x", default=None,
                   help="Fix kinematics to this comma list (single context) instead of averaging over data.")
    p.add_argument("--n-context", type=int, default=4000, help="x-sample size for the average (x-avg mode).")
    p.add_argument("--nsigma", type=float, default=1.0, help="Displacement amplitude along each axis, in σ.")
    p.add_argument("--ngrid", type=int, default=21)
    p.add_argument("--grid-range", default=None, help="'lo,hi' (both dims) or 'l1,h1;l2,h2'.")
    p.add_argument("--no-background", dest="background", action="store_false",
                   help="Drop the |Δ| heatmap background (colour the arrows by |Δ| instead).")
    p.add_argument("--arrows", choices=["unit", "scaled"], default="unit",
                   help="'unit' = equal-length direction arrows (default); 'scaled' = length ∝ |Δ|.")
    p.add_argument("--arrow-scale", type=float, default=1.0,
                   help="Multiplier on arrow length (default 1.0; >1 = longer arrows).")
    p.add_argument("--cmap", default="viridis", help="Background/arrow colormap.")
    p.add_argument("--cmap-floor", type=float, default=0.18,
                   help="Truncate the colormap's dark low end (0=full/dark, ~0.2 brightens; default 0.18).")
    p.add_argument("--color-log", action="store_true",
                   help="Use a logarithmic color scale for |Δy| (default: linear). Helps when the "
                        "displacement spans a wide dynamic range (large in the core, tiny in the tails).")
    p.add_argument("--color-vmin", type=float, default=None,
                   help="Lower bound of the color scale. Log mode default: 5th percentile of the "
                        "non-zero |Δy|; linear mode default: 0.")
    p.add_argument("--reference", default=None, help="Nominal ν, comma list (default: npz ν̂).")
    p.add_argument("--seed", type=int, default=0)
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

    ckpt = args.ckpt or cfg["paths"]["output_checkpoint"]
    dataset_path = args.dataset or cfg["paths"]["dataset"]
    model, T, residual_T, stage = load_models(cfg, ckpt, device)
    if residual_T is None:
        raise SystemExit("This plot needs a profiled checkpoint with a residual_transfer_model.")
    label = args.label or stage

    feat = int(cfg["residual_transfer_model"]["features_dim"])
    if feat != 2:
        raise SystemExit(f"Score features_dim={feat}; the vector-field view requires 2.")
    n_flav = int(cfg["mixture_model"]["n_flavours"])
    ctx_dim = int(cfg["residual_transfer_model"]["context_dim"])
    x_dim = ctx_dim - n_flav
    num_nuisances = model.m_vector.shape[0]

    # --- Hessian eigenvectors / ν̂ from the saved npz ---
    hz = np.load(args.hessian_npz, allow_pickle=True)
    evals = np.asarray(hz["cov_eigvals"], dtype=np.float64)        # variances (ascending)
    evecs = np.asarray(hz["cov_eigvecs"], dtype=np.float64)        # columns = eigenvectors
    indices = [int(i) for i in np.asarray(hz["indices"]).ravel()]  # profiled-index map
    labels = [str(s) for s in np.asarray(hz["labels"]).ravel()]
    nu_hat_npz = np.asarray(hz["nu_hat"], dtype=np.float32).ravel()
    d = len(indices)
    if evecs.shape != (d, d) or evals.shape[0] != d:
        raise SystemExit(f"npz eigvec dims {evecs.shape}/{evals.shape} != profiled d={d}.")
    order = np.argsort(evals)[::-1]                                # major axis first

    # Reference ("nominal") ν̂ — full vector.
    if args.reference is not None:
        ref = [float(v) for v in args.reference.split(",")]
        if len(ref) != num_nuisances:
            raise SystemExit(f"--reference needs {num_nuisances} values, got {len(ref)}.")
        nu_ref = torch.tensor(ref, dtype=torch.float32, device=device)
    else:
        if nu_hat_npz.shape[0] != num_nuisances:
            raise SystemExit(f"npz ν̂ has {nu_hat_npz.shape[0]} entries, model has {num_nuisances}.")
        nu_ref = torch.tensor(nu_hat_npz, dtype=torch.float32, device=device)

    # --- dataset: grid range (y percentiles) + context ---
    ds = torch.load(dataset_path, map_location=device)
    y_all = ds["y_data_distorted"].cpu().numpy()
    X_all = ds["X_data_distorted"].float()
    if args.grid_range is not None:
        chunks = args.grid_range.split(";")
        if len(chunks) == 1:
            lo, hi = (float(v) for v in chunks[0].split(","))
            rng = [(lo, hi), (lo, hi)]
        else:
            rng = [tuple(float(v) for v in c.split(",")) for c in chunks]
    else:
        rng = [tuple(np.percentile(y_all[:, i], [1, 99])) for i in range(2)]
    span = max(rng[0][1] - rng[0][0], rng[1][1] - rng[1][0])

    ng = args.ngrid
    g0 = np.linspace(rng[0][0], rng[0][1], ng)
    g1 = np.linspace(rng[1][0], rng[1][1], ng)
    YY1, YY2 = np.meshgrid(g0, g1)
    grid = torch.tensor(np.stack([YY1.ravel(), YY2.ravel()], 1), dtype=torch.float32, device=device)

    onehot = torch.zeros(n_flav, dtype=torch.float32, device=device)
    onehot[args.flavour] = 1.0
    if args.x is not None:
        x_fixed = [float(v) for v in args.x.split(",")]
        if len(x_fixed) != x_dim:
            raise SystemExit(f"--x needs {x_dim} values, got {len(x_fixed)}.")
        x_rows = torch.tensor([x_fixed], dtype=torch.float32, device=device)            # [1, x_dim]
        ctx_tag = f"flavour {args.flavour}, x={x_fixed}"
    else:
        gen = torch.Generator().manual_seed(args.seed)
        n = min(args.n_context, X_all.shape[0])
        sel = torch.randperm(X_all.shape[0], generator=gen)[:n]
        x_rows = X_all[sel, :x_dim].to(device)                                          # [n, x_dim]
        ctx_tag = f"flavour {args.flavour}, x-averaged (N={n})"
    contexts = torch.cat([onehot.unsqueeze(0).expand(x_rows.shape[0], -1), x_rows], dim=1)

    print(f"Stage={stage}  flavour={args.flavour}  d={d}  ν̂={nu_ref.cpu().numpy()}")
    print(f"Context: {ctx_tag}   grid {ng}×{ng}  nsigma={args.nsigma}")

    # --- per-eigenvector ±nσ displacement fields ---
    fields, comps, sigs, names = [], [], [], []
    for r, k in enumerate(order):
        sig_k = float(np.sqrt(max(evals[k], 0.0)))
        e_k = evecs[:, k]
        name = "major axis" if r == 0 else ("minor axis" if r == 1 else f"axis {r}")
        comp = " ".join(f"{e_k[j]:+.2f}·{labels[j]}" for j in range(d))
        if sig_k <= 0:
            print(f"  ⚠ {name}: σ≈0 (eigval={evals[k]:.2e}); field will be ~zero.")
        row = []
        for sign in (-1.0, +1.0):
            offset = torch.zeros(num_nuisances, dtype=torch.float32, device=device)
            idx = torch.tensor(indices, dtype=torch.long, device=device)
            offset.index_copy_(0, idx, torch.tensor(sign * args.nsigma * sig_k * e_k,
                                                    dtype=torch.float32, device=device))
            m_off = nu_ref + offset
            row.append(_disp_field(residual_T, grid, contexts, m_off, nu_ref, ng))
        fields.append(row)
        comps.append(comp)
        sigs.append(sig_k)
        names.append(name)

    # --- plot: rows = eigenvectors (major first), cols = (−nσ, +nσ); shared scale ---
    all_speed = np.concatenate([np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2).ravel()
                                for row in fields for f in row])
    speed_max = float(all_speed.max())
    if args.color_log:
        pos = all_speed[all_speed > 0]
        if args.color_vmin is not None:
            vmin = float(args.color_vmin)
        elif pos.size:
            vmin = float(np.percentile(pos, 5))
        else:
            vmin = max(speed_max * 1e-3, 1e-12)
        vmax = speed_max if speed_max > vmin else vmin * 10.0
        # clip=True: zero-displacement cells map to the floor colour instead of leaving
        # masked (white) holes in the gouraud-shaded background.
        norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
        print(f"Color scale: log  vmin={vmin:.3g}  vmax={vmax:.3g}")
    else:
        vmin = args.color_vmin if args.color_vmin is not None else 0.0
        norm = Normalize(vmin, speed_max if speed_max > vmin else vmin + 1.0)
    cmap = truncate_cmap(args.cmap, args.cmap_floor)
    # Arrow length in data units: ~1.6 grid cells (longer than the default), × --arrow-scale.
    cell = min((rng[0][1] - rng[0][0]), (rng[1][1] - rng[1][0])) / max(ng - 1, 1)
    arrow_len = 1.6 * cell * args.arrow_scale

    fig, axes = plt.subplots(d, 2, figsize=(9.0, 4.2 * d), squeeze=False)
    sgn = lambda s: "−" if s < 0 else "+"
    for r in range(d):
        for c, sign in enumerate((-1.0, +1.0)):
            ax = axes[r][c]
            fld = fields[r][c]
            field_quiver_panel(ax, g0, g1, YY1, YY2, fld[..., 0], fld[..., 1], norm, cmap,
                               background=args.background, arrows=args.arrows, arrow_len=arrow_len)
            ax.set_xlabel("y₁" if r == d - 1 else "")
            ax.set_ylabel(f"{names[r]}\n{comps[r]}\nσ={sigs[r]:.4g}\n\ny₂" if c == 0 else "", fontsize=8)
            if r == 0:
                ax.set_title(f"{sgn(sign)}{args.nsigma:g}σ", fontsize=12)
    fig.suptitle(f"Residual-T distortion along Hessian eigenvectors\n{ctx_tag}, rel. to ν̂", fontsize=12)
    fig.subplots_adjust(right=0.88, top=0.9)
    cax = fig.add_axes([0.90, 0.12, 0.015, 0.74])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="|Δy|")
    _save(fig, args.out_dir, f"hessian_distortion_{label}")
    print(f"\nFigure written to {args.out_dir}/")


if __name__ == "__main__":
    main()
