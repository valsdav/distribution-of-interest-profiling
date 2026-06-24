"""
Visualise the fitted residual transfer T over score space.

Two views (``--mode``), both for a fixed kinematics x and flavour, over a grid in
the 2D score space (y1, y2). The residual "stack" here means the m-dependent
residual layers of `residual_transfer_model` only (the base transfer T is
m-independent and excluded), evaluated as scores -> nominal-score coordinates.

* field  — the residual displacement Δ(s; m) = stack(s; m) − stack(s; m_central),
           a vector field per nuisance-space point m (relative to nominal, so it
           is zero at the central/best-fit point and shows the pure systematic
           morphing as you move in nuisance space). One panel per m.

* terms  — the linear / quadratic / cross decomposition of the response at an
           amplitude `amp` (in nuisance-σ units), via finite differences of the
           stack around m_central:
             linear_i     = [stack(+a e_i) − stack(−a e_i)] / 2        (odd part)
             quadratic_i  = [stack(+a e_i) + stack(−a e_i)] / 2 − stack(0)  (even)
             cross_ij     = [s(+a,+a)+s(−a,−a)−s(+a,−a)−s(−a,+a)] / 4   (bilinear)
           with a_i = amp · nuisance_scales_i. Robust to the stacked (>quadratic)
           composition — these are the *effective* per-order contributions at `amp`.

Usage:
    python plot_residual_field.py -c configs/profiling_cross_v13.yaml \
        --ckpt models/full_mixture_model_v13_profiled_crossterm.pt \
        --out-dir residual_fields --mode both --flavour 0
"""
import argparse
import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
import yaml

from likelihood_scan import load_models, _resolve, _save


# ---------------------------------------------------------------------------
# Residual-stack evaluation (m-dependent layers only; excludes the base T)
# ---------------------------------------------------------------------------

def _residual_stack(residual_T, s, context, m):
    x = s
    for transform in residual_T.transforms:
        x, _ = transform(x, context=context, m=m)
    return x


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _quiver(ax, YY1, YY2, U, V, norm, cmap="viridis", scale=None):
    spd = np.sqrt(U ** 2 + V ** 2)
    q = ax.quiver(YY1, YY2, U, V, spd, cmap=cmap, norm=norm, pivot="mid",
                  angles="xy", scale_units="xy", scale=scale)
    ax.set_xlabel("y₁"); ax.set_ylabel("y₂")
    return q


def _auto_scale(speed_max, span, ngrid):
    """quiver scale so the largest arrow spans ~1.5 grid cells (scale_units='xy')."""
    if speed_max <= 0:
        return None
    return speed_max / (1.5 * span / ngrid)


def plot_field(deltas, m_labels, layout, YY1, YY2, span, ngrid, out_dir, tag, flavour):
    """deltas: list of [ng,ng,2] arrays; layout=(nrows, ncols)."""
    nrows, ncols = layout
    speed_max = max(float(np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2).max()) for d in deltas)
    norm = Normalize(0.0, speed_max)
    scale = _auto_scale(speed_max, span, ngrid)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False)
    for k, (d, lbl) in enumerate(zip(deltas, m_labels)):
        ax = axes[k // ncols][k % ncols]
        q = _quiver(ax, YY1, YY2, d[..., 0], d[..., 1], norm, scale=scale)
        ax.set_title(lbl, fontsize=10)
    for k in range(len(deltas), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(rf"Residual T displacement over score space  (flavour {flavour}, rel. to nominal)",
                 fontsize=13)
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax, label="|Δ|")
    _save(fig, out_dir, f"residual_field_{tag}_fl{flavour}")


def plot_terms(terms, YY1, YY2, out_dir, tag, flavour, amp):
    """terms: ordered list of (group, label, field[ng,ng,2]); per-panel colorbars."""
    n = len(terms)
    ncols = max(1, max(sum(1 for g, _, _ in terms if g == grp) for grp in {t[0] for t in terms}))
    groups = ["linear", "quadratic", "cross"]
    present = [g for g in groups if any(t[0] == g for t in terms)]
    nrows = len(present)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows), squeeze=False)
    for r, grp in enumerate(present):
        row = [t for t in terms if t[0] == grp]
        for c in range(ncols):
            ax = axes[r][c]
            if c >= len(row):
                ax.axis("off"); continue
            _, lbl, fld = row[c]
            spd = np.sqrt(fld[..., 0] ** 2 + fld[..., 1] ** 2)
            norm = Normalize(0.0, float(spd.max()) if spd.max() > 0 else 1.0)
            q = _quiver(ax, YY1, YY2, fld[..., 0], fld[..., 1], norm)
            fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04, label="|Δ|")
            ax.set_title(f"{grp}: {lbl}", fontsize=10)
    fig.suptitle(rf"Residual T term decomposition over score space "
                 rf"(flavour {flavour}, amplitude = {amp}σ)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, f"residual_terms_{tag}_fl{flavour}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_center(spec, model, num_nuisances, device):
    if spec == "best":
        return model.m_vector.detach().clone()
    vals = [float(v) for v in spec.split(",")]
    if len(vals) != num_nuisances:
        raise ValueError(f"--center expects {num_nuisances} values, got {len(vals)}")
    return torch.tensor(vals, device=device, dtype=torch.float32)


def main():
    p = argparse.ArgumentParser(description="Visualise the fitted residual T over score space.")
    p.add_argument("-c", "--cfg", required=True)
    p.add_argument("--ckpt", default=None, help="Default: config output_checkpoint.")
    p.add_argument("--dataset", default=None, help="Default: config paths.dataset (for grid range & x).")
    p.add_argument("--out-dir", default="residual_fields")
    p.add_argument("--label", default=None)
    p.add_argument("--mode", choices=["field", "terms", "both"], default="both")
    p.add_argument("--flavour", type=int, default=0)
    p.add_argument("--x", default=None, help="Fixed kinematics, comma list (default: dataset mean).")
    p.add_argument("--grid-range", default=None, help="'lo,hi' (both dims) or 'l1,h1;l2,h2'.")
    p.add_argument("--ngrid", type=int, default=21)
    p.add_argument("--center", default="best", help="'best' (m_vector) or comma list.")
    # field mode
    p.add_argument("--steps", type=int, default=3, help="m-grid steps/dim (2-nuisance layout).")
    p.add_argument("--delta", type=float, default=None, help="m-grid half-width (default from range_m).")
    p.add_argument("--m-points", default=None, help="Explicit abs m points 'a,b;c,d' (overrides grid).")
    # terms mode
    p.add_argument("--amp", type=float, default=1.0, help="Term amplitude in nuisance-σ units.")
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
    scales = np.asarray(cfg["residual_transfer_model"]["nuisance_scales"], dtype=np.float32)
    num_nuisances = model.m_vector.shape[0]
    mask = model.m_vector_profile_mask.detach().cpu().numpy().astype(bool)
    profiled = [i for i in range(num_nuisances) if mask[i]]

    # Defaults (grid range, x) from the dataset.
    ds = torch.load(dataset_path, map_location=device)
    y_all = ds["y_data_distorted"].cpu().numpy()
    X_all = ds["X_data_distorted"]
    if args.grid_range is not None:
        chunks = args.grid_range.split(";")
        if len(chunks) == 1:
            lo, hi = (float(v) for v in chunks[0].split(","))
            rng = [(lo, hi), (lo, hi)]
        else:
            rng = [tuple(float(v) for v in c.split(",")) for c in chunks]
    else:
        rng = [tuple(np.percentile(y_all[:, d], [1, 99])) for d in range(2)]
    if args.x is not None:
        x_fixed = np.asarray([float(v) for v in args.x.split(",")], dtype=np.float32)
    else:
        x_fixed = X_all.float().mean(0).cpu().numpy()[:x_dim]

    # Score-space grid + fixed context.
    ng = args.ngrid
    g0 = np.linspace(rng[0][0], rng[0][1], ng)
    g1 = np.linspace(rng[1][0], rng[1][1], ng)
    YY1, YY2 = np.meshgrid(g0, g1)
    grid = torch.tensor(np.stack([YY1.ravel(), YY2.ravel()], 1), dtype=torch.float32, device=device)
    onehot = np.zeros(n_flav, dtype=np.float32); onehot[args.flavour] = 1.0
    ctx_row = np.concatenate([onehot, x_fixed]).astype(np.float32)
    context = torch.tensor(ctx_row, device=device).unsqueeze(0).expand(grid.shape[0], -1)
    span = max(rng[0][1] - rng[0][0], rng[1][1] - rng[1][0])

    m_central = _parse_center(args.center, model, num_nuisances, device)

    @torch.no_grad()
    def stack_at(offset: np.ndarray) -> np.ndarray:
        """residual_stack(grid; m_central + offset) -> [ng, ng, 2] numpy."""
        m = (m_central + torch.tensor(offset, dtype=torch.float32, device=device)).unsqueeze(0)
        out = _residual_stack(residual_T, grid, context, m.expand(grid.shape[0], -1))
        return out.cpu().numpy().reshape(ng, ng, 2)

    base = stack_at(np.zeros(num_nuisances, dtype=np.float32))  # stack at m_central
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Stage={stage} label={label}  flavour={args.flavour}  x={x_fixed.tolist()}")
    print(f"m_central={m_central.cpu().numpy().tolist()}  profiled={profiled}  scales={scales.tolist()}")

    # ----- field mode -----
    if args.mode in ("field", "both"):
        m_points: List[Tuple[np.ndarray, str]] = []
        layout: Tuple[int, int]
        if args.m_points is not None:
            for chunk in args.m_points.split(";"):
                mv = np.asarray([float(v) for v in chunk.split(",")], dtype=np.float32)
                m_points.append((mv, "ν=(" + ", ".join(f"{v:.3f}" for v in mv) + ")"))
            layout = (1, len(m_points))
        elif len(profiled) == 2:
            i, j = profiled
            delta = args.delta if args.delta is not None else float(np.max(np.abs(
                cfg.get("training", {}).get("range_m", [[-0.05, 0.05]]))))
            vi = m_central[i].item() + np.linspace(-delta, delta, args.steps)
            vj = m_central[j].item() + np.linspace(-delta, delta, args.steps)
            for jv in vj[::-1]:                      # top row = high ν_j
                for iv in vi:
                    mv = m_central.cpu().numpy().copy()
                    mv[i], mv[j] = iv, jv
                    m_points.append((mv, f"ν{i}={iv:.3f}, ν{j}={jv:.3f}"))
            layout = (args.steps, args.steps)
        else:
            mv = m_central.cpu().numpy()
            m_points.append((mv, "ν=central"))
            layout = (1, 1)

        deltas = [stack_at(mv - m_central.cpu().numpy()) - base for mv, _ in m_points]
        plot_field(deltas, [lbl for _, lbl in m_points], layout, YY1, YY2,
                   span, ng, args.out_dir, label, args.flavour)

    # ----- terms mode -----
    if args.mode in ("terms", "both"):
        terms: List[Tuple[str, str, np.ndarray]] = []
        amp = args.amp
        for i in profiled:
            off = np.zeros(num_nuisances, dtype=np.float32); off[i] = amp * scales[i]
            sp, sm = stack_at(off), stack_at(-off)
            terms.append(("linear", f"ν{i}", 0.5 * (sp - sm)))
            terms.append(("quadratic", f"ν{i}", 0.5 * (sp + sm) - base))
        for a in range(len(profiled)):
            for b in range(a + 1, len(profiled)):
                i, j = profiled[a], profiled[b]
                opp = np.zeros(num_nuisances, dtype=np.float32); opp[i] = opp[j] = 0.0
                pp = np.zeros(num_nuisances, dtype=np.float32); pp[i] = amp * scales[i]; pp[j] = amp * scales[j]
                pm = np.zeros(num_nuisances, dtype=np.float32); pm[i] = amp * scales[i]; pm[j] = -amp * scales[j]
                mp = np.zeros(num_nuisances, dtype=np.float32); mp[i] = -amp * scales[i]; mp[j] = amp * scales[j]
                mm = np.zeros(num_nuisances, dtype=np.float32); mm[i] = -amp * scales[i]; mm[j] = -amp * scales[j]
                cross = 0.25 * (stack_at(pp) + stack_at(mm) - stack_at(pm) - stack_at(mp))
                terms.append(("cross", f"ν{i}·ν{j}", cross))
        if terms:
            plot_terms(terms, YY1, YY2, args.out_dir, label, args.flavour, amp)
        else:
            print("No profiled nuisances -> no terms to plot.")

    print(f"\nFigures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
