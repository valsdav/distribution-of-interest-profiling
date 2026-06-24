import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
import zuko
from torch.utils.data import DataLoader, TensorDataset

from lib import FullMixtureModel, TransferModel
from residual_flow import SystematicCorrectedModel
from utils import LinearWarmupCosineDecay, load_state_dict_checked


def _resolve_path(path: str, cfg_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(cfg_dir, path))


def _to_tensor(value: Any, device: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def _parse_q_per_dim(spec: Any, num_dims: int) -> List[int]:
    """GL nodes per varied nuisance dimension: scalar (broadcast) or list of length num_dims."""
    if isinstance(spec, (list, tuple)):
        q = [int(v) for v in spec]
        if len(q) != num_dims:
            raise ValueError(f"m_oversample list has {len(q)} entries, expected {num_dims}")
    else:
        q = [int(spec)] * num_dims
    if any(v < 1 for v in q):
        raise ValueError(f"m_oversample must be >= 1 per dimension, got {q}")
    return q


def _build_gl_grid(
    range_m: torch.Tensor, num_dims: int, q_per_dim: List[int], num_nuisances: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Legendre tensor grid over the first `num_dims` (varied) nuisances.

    Returns (offsets[G, num_nuisances], weights[G]) where `offsets` are added to m_vector
    and `weights` sum to 1 — the prior-average over the uniform range `range_m` per
    dimension. The m-dependence of the residual is a (stacked) polynomial, so a fixed grid
    that resolves it identifies the function over the whole range; weights summing to 1 keep
    the data term at "one dataset's worth", preserving the data/Poisson/constraint balance.
    """
    if num_dims == 0:
        return (torch.zeros(1, num_nuisances, device=device),
                torch.ones(1, device=device))
    per_dim_offsets, per_dim_weights = [], []
    for d in range(num_dims):
        nodes, weights = np.polynomial.legendre.leggauss(q_per_dim[d])  # nodes/weights on [-1, 1]
        lo, hi = float(range_m[d, 0]), float(range_m[d, 1])
        per_dim_offsets.append(0.5 * (hi - lo) * nodes + 0.5 * (hi + lo))  # map [-1,1] -> [lo,hi]
        per_dim_weights.append(weights / 2.0)  # GL weights sum to 2 on [-1,1]; /2 -> mean
    offs_mesh = np.meshgrid(*per_dim_offsets, indexing="ij")
    w_mesh = np.meshgrid(*per_dim_weights, indexing="ij")
    offsets_flat = np.stack([o.reshape(-1) for o in offs_mesh], axis=1)              # [G, num_dims]
    weights_flat = np.prod(np.stack([w.reshape(-1) for w in w_mesh], axis=1), axis=1)  # [G]
    grid_offsets = np.zeros((offsets_flat.shape[0], num_nuisances), dtype=np.float32)
    grid_offsets[:, :num_dims] = offsets_flat
    return (torch.as_tensor(grid_offsets, device=device, dtype=torch.float32),
            torch.as_tensor(weights_flat, device=device, dtype=torch.float32))


@torch.no_grad()
def _avg_event_nll_grid(
    model, residual_T, x, y, m_vector, offsets, weights, batch_size
) -> float:
    """Mean per-event NLL using a SHARED nuisance grid (prior-weighted mean over nodes)."""
    G = offsets.shape[0]
    m_grid = m_vector.unsqueeze(0) + offsets  # [G, num_nuisances]
    total, n = 0.0, 0
    for b in range(0, x.shape[0], batch_size):
        xb, yb = x[b:b + batch_size], y[b:b + batch_size]
        B = xb.shape[0]
        logp, _ = model.log_prob(
            yb.repeat_interleave(G, dim=0), xb.repeat_interleave(G, dim=0),
            residual_T, m_grid.repeat(B, 1),
        )
        per_event = (logp.view(B, G) * weights.view(1, G)).sum(dim=1)
        total += float((-per_event).sum().item())
        n += B
    return total / max(1, n)


@torch.no_grad()
def _avg_event_nll_random(
    model, residual_T, x, y, m_vector, range_m, k, batch_size, device
) -> float:
    """Mean per-event NLL using K independent random m draws per event (off-grid)."""
    D = len(range_m)
    total, n = 0.0, 0
    for b in range(0, x.shape[0], batch_size):
        xb, yb = x[b:b + batch_size], y[b:b + batch_size]
        B = xb.shape[0]
        m_rep = m_vector.unsqueeze(0).expand(B * k, -1).clone()
        for i in range(D):
            m_rep[:, i] = m_rep[:, i] + range_m[i, 0] + (range_m[i, 1] - range_m[i, 0]) * torch.rand(B * k, device=device)
        logp, _ = model.log_prob(
            yb.repeat_interleave(k, dim=0), xb.repeat_interleave(k, dim=0), residual_T, m_rep
        )
        per_event = logp.view(B, k).mean(dim=1)
        total += float((-per_event).sum().item())
        n += B
    return total / max(1, n)


def _build_residual_flow_model(
    cfg: Dict[str, Any], device: str
) -> Tuple[FullMixtureModel, TransferModel, SystematicCorrectedModel, Optional[List[TransferModel]]]:
    score_cfg = cfg["score_flow"]
    score_model = zuko.flows.NSF(
        features=score_cfg["features"],
        context=score_cfg["context"],
        bins=score_cfg["bins"],
        transforms=score_cfg["transforms"],
        hidden_features=tuple(score_cfg["hidden_features"]),
    ).to(device)

    # v12 has no score-space systematic. If `residual_score_model` is absent from the
    # config, wrap the score flow in an identity SystematicCorrectedModel (num_nuisances=0)
    # with dims derived from score_flow — FullMixtureModel still needs the wrapper API.
    residual_score_cfg = cfg.get("residual_score_model") or {
        "features_dim": score_cfg["features"],
        "context_dim": score_cfg["context"],
        "num_nuisances": 0,
        "num_residual_layers": 1,
        "hidden_features": [64, 64],
        "type": "flow",
    }
    residual_score_model = SystematicCorrectedModel(
        score_model,
        features_dim=residual_score_cfg["features_dim"],
        context_dim=residual_score_cfg["context_dim"],
        num_nuisances=residual_score_cfg["num_nuisances"],
        num_residual_layers=residual_score_cfg["num_residual_layers"],
        hidden_features=residual_score_cfg["hidden_features"],
        type=residual_score_cfg["type"],
    ).to(device)

    # v12/v13: score residual is identity (num_nuisances=0) — no state to load. When it
    # is ENABLED (v14+), require the trained weights so it can't silently stay at identity.
    score_state_path = cfg["paths"].get("score_density_state")
    if int(residual_score_cfg["num_nuisances"]) > 0:
        if not score_state_path or not os.path.isfile(score_state_path):
            raise FileNotFoundError(
                f"residual_score_model.num_nuisances="
                f"{residual_score_cfg['num_nuisances']} but the score residual weights are "
                f"missing (paths.score_density_state={score_state_path!r}). Train them with "
                "`train_systematics.py -s train_score`; otherwise the score residual would "
                "silently stay at identity init and the score systematic be ignored."
            )
        load_state_dict_checked(
            residual_score_model, torch.load(score_state_path, map_location=device),
            label="score residual",
        )
    elif score_state_path is not None and os.path.isfile(score_state_path):
        # Identity residual but a state was provided — load the (matching) base weights.
        residual_score_model.load_state_dict(
            torch.load(score_state_path, map_location=device), strict=False
        )

    kin_cfg = cfg["kin_flow"]
    kin_model = zuko.flows.NSF(
        features=kin_cfg["features"],
        context=kin_cfg["context"],
        bins=kin_cfg["bins"],
        transforms=kin_cfg["transforms"],
        hidden_features=tuple(kin_cfg["hidden_features"]),
    ).to(device)

    residual_kin_cfg = cfg["residual_kin_model"]
    residual_kin_model = SystematicCorrectedModel(
        kin_model,
        features_dim=residual_kin_cfg["features_dim"],
        context_dim=residual_kin_cfg["context_dim"],
        num_nuisances=residual_kin_cfg["num_nuisances"],
        num_residual_layers=residual_kin_cfg["num_residual_layers"],
        hidden_features=residual_kin_cfg["hidden_features"],
        type=residual_kin_cfg["type"],
    ).to(device)

    kin_state_path = cfg["paths"]["kin_density_state"]
    if not os.path.isfile(kin_state_path):
        raise FileNotFoundError(
            f"Kin residual weights not found: {kin_state_path}. Train them with "
            "`train_systematics.py -s train_kin`."
        )
    load_state_dict_checked(
        residual_kin_model, torch.load(kin_state_path, map_location=device), label="kin residual")

    model_cfg = cfg["mixture_model"]
    _lnN_mix = model_cfg.get("lnN_mix_matrix", None)
    lnN_mix_matrix = _to_tensor(_lnN_mix, device) if _lnN_mix is not None else None
    _profile_mask = model_cfg.get("norm_nuisance_profile_mask", None)
    norm_nuisance_profile_mask = (
        torch.as_tensor(_profile_mask, dtype=torch.bool, device=device)
        if _profile_mask is not None else None
    )
    _m_mask = model_cfg.get("m_vector_profile_mask", None)
    m_vector_profile_mask = (
        torch.as_tensor(_m_mask, dtype=torch.bool, device=device)
        if _m_mask is not None else None
    )
    model = FullMixtureModel(
        features_dim=model_cfg["features_dim"],
        n_flavours=model_cfg["n_flavours"],
        num_nuisances=model_cfg["num_nuisances"],
        norm_factors=_to_tensor(model_cfg["norm_factors"], device),
        scores_model=residual_score_model,
        kin_model=residual_kin_model,
        lnN_constraints=_to_tensor(model_cfg["lnN_constraints"], device),
        fit_conditional_pdf=bool(model_cfg["fit_conditional_pdf"]),
        lnN_mix_matrix=lnN_mix_matrix,
        norm_nuisance_profile_mask=norm_nuisance_profile_mask,
        m_vector_profile_mask=m_vector_profile_mask,
    ).to(device)

    transfer_cfg = cfg["transfer_model"]

    def _new_transfer():
        return TransferModel(
            features_dim=transfer_cfg["features_dim"],
            context_dim=transfer_cfg["context_dim"],
            n_transforms=transfer_cfg["n_transforms"],
            nbins=transfer_cfg["nbins"],
            hidden_net=transfer_cfg["hidden_net"],
            add_rotation=bool(transfer_cfg["add_rotation"]),
        ).to(device)

    # Ensemble mode: `init_mixture_ensemble` is a glob of bootstrap member checkpoints. The
    # single shared residual is trained against ALL of them (random member per batch), so it
    # learns the base-T-averaged ν-response. The frozen kin/score templates are identical
    # across members; load member 0's mixture_model for them. The transfer base varies per
    # member (cycled in the training loop).
    ensemble_glob = cfg["paths"].get("init_mixture_ensemble")
    member_Ts = None
    if ensemble_glob:
        import glob as _glob
        member_paths = sorted(_glob.glob(ensemble_glob))
        if not member_paths:
            raise FileNotFoundError(f"init_mixture_ensemble matched no files: {ensemble_glob}")
        init_ckpt = torch.load(member_paths[0], map_location=device)
        model.load_state_dict(init_ckpt["mixture_model"], strict=False)
        member_Ts = []
        for p in member_paths:
            ck = torch.load(p, map_location=device)
            Tb = _new_transfer()
            Tb.load_state_dict(ck["transfer_model"], strict=False)
            for q in Tb.parameters():
                q.requires_grad = False
            Tb.eval()
            member_Ts.append(Tb)
        transfer_model = member_Ts[0]   # representative base the residual is built on
        print(f"Ensemble profiling: {len(member_Ts)} members from {ensemble_glob}")
    else:
        transfer_model = _new_transfer()
        init_ckpt = torch.load(cfg["paths"]["init_mixture_checkpoint"], map_location=device)
        model.load_state_dict(init_ckpt["mixture_model"], strict=False)
        transfer_model.load_state_dict(init_ckpt["transfer_model"], strict=False)

    # The mixture-stage checkpoint may have frozen some nuisances (e.g. ν_y).
    # Re-apply the profiling-stage masks from the current config so we can unfreeze
    # them for the profile fit without being overridden by the loaded state.
    if norm_nuisance_profile_mask is not None:
        model.norm_nuisance_profile_mask.data.copy_(norm_nuisance_profile_mask)
    if m_vector_profile_mask is not None:
        model.m_vector_profile_mask.data.copy_(m_vector_profile_mask)

    # Optionally overwrite the best-fit point loaded from the checkpoint. m_vector and
    # norm_nuisance are frozen during the profile fit, so this sets the fixed central
    # point the residual T is expanded around. Must happen before residual_transfer_model
    # is built below (it takes central_nuisance_values=model.m_vector).
    _m_override = model_cfg.get("m_vector_override")
    if _m_override is not None:
        model.m_vector.data.copy_(_to_tensor(_m_override, device))
    _norm_override = model_cfg.get("norm_nuisance_override")
    if _norm_override is not None:
        model.norm_nuisance.data.copy_(_to_tensor(_norm_override, device))

    residual_transfer_cfg = cfg["residual_transfer_model"]
    residual_transfer_model = SystematicCorrectedModel(
        transfer_model,
        features_dim=residual_transfer_cfg["features_dim"],
        context_dim=residual_transfer_cfg["context_dim"],
        num_nuisances=residual_transfer_cfg["num_nuisances"],
        num_residual_layers=residual_transfer_cfg["num_residual_layers"],
        hidden_features=residual_transfer_cfg["hidden_features"],
        type=residual_transfer_cfg["type"],
        central_nuisance_values=model.m_vector,
        nuisance_scales=residual_transfer_cfg["nuisance_scales"],
        quadratic_damping=float(residual_transfer_cfg["quadratic_damping"]),
        cross_term_pairs=residual_transfer_cfg.get("cross_term_pairs"),
        cross_term_damping=float(residual_transfer_cfg.get("cross_term_damping", 0.01)),
    ).to(device)

    return model, transfer_model, residual_transfer_model, member_Ts


def _run_training(cfg: Dict[str, Any], device: str) -> None:
    model, transfer_model, residual_transfer_model, member_Ts = _build_residual_flow_model(cfg, device)
    ensemble_glob = cfg["paths"].get("init_mixture_ensemble")
    n_members = len(member_Ts) if member_Ts is not None else 0

    train_cfg = cfg["training"]
    batch_size = int(train_cfg["batch_size"])
    nepochs = int(train_cfg["nepochs"])
    n_events = int(train_cfg["n_events"])
    steps_per_epoch = n_events // batch_size
    add_distance_loss = bool(train_cfg["add_distance_loss"])
    lambda_distance = float(train_cfg["lambda_distance"])

    range_m = _to_tensor(train_cfg["range_m"], device)

    # Nuisance-space sampling for the residual-T gradient. `random` keeps the legacy
    # one-uniform-draw-per-event behaviour; `quadrature` integrates the per-event nuisance
    # average E_m[log p] on a fixed Gauss-Legendre grid (near-zero MC noise for the smooth,
    # polynomial-in-m residual). Grid is precomputed once: m_vector is frozen here.
    m_sampling = str(train_cfg.get("m_sampling", "random")).lower()
    n_random_draws = 1
    if m_sampling == "quadrature":
        q_per_dim = _parse_q_per_dim(train_cfg.get("m_oversample", 3), len(range_m))
        grid_offsets, grid_weights = _build_gl_grid(
            range_m, len(range_m), q_per_dim, model.m_vector.shape[0], device
        )
        n_grid_nodes = int(grid_offsets.shape[0])
        print(f"Nuisance sampling: Gauss-Legendre quadrature, Q={q_per_dim} "
              f"-> {n_grid_nodes} nodes/event")
    elif m_sampling == "random":
        grid_offsets = grid_weights = None
        # m_oversample here = K random draws per event (averaged); K=1 is the legacy path.
        _os = train_cfg.get("m_oversample", 1)
        if isinstance(_os, (list, tuple)):
            raise ValueError("training.m_oversample must be a scalar for m_sampling='random' (draws/event).")
        n_random_draws = int(_os)
        if n_random_draws < 1:
            raise ValueError(f"m_oversample must be >= 1, got {n_random_draws}")
        n_grid_nodes = n_random_draws
        if n_random_draws > 1:
            print(f"Nuisance sampling: random, {n_random_draws} draws/event (variance-reduced)")
        else:
            print("Nuisance sampling: random (1 uniform draw/event)")
    else:
        raise ValueError(
            f"Unknown training.m_sampling='{m_sampling}' (use 'random' or 'quadrature')"
        )

    # Off-grid validation: every N epochs, re-estimate the per-event nuisance-averaged NLL
    # with m-points the training grid never saw, and log the gap to the on-grid value. A
    # gap ~ 0 means the GL grid resolves the (stacked, >quadratic) m-dependence; a growing
    # gap means raise m_oversample (integration) or the residual order (capacity). Requires
    # quadrature (the on-grid reference is the training grid).
    val_cfg = train_cfg.get("validate_offgrid") or {}
    offgrid_every = int(val_cfg.get("every", 0))
    offgrid_mode = str(val_cfg.get("mode", "random")).lower()
    offgrid_k = int(val_cfg.get("n_random", 64))
    offgrid_batch = int(val_cfg.get("batch_size", 2000))
    offgrid_nev = int(val_cfg.get("n_events", 20000))
    offgrid_dense = None
    if offgrid_every > 0 and grid_offsets is None:
        print("validate_offgrid ignored: requires training.m_sampling='quadrature'.")
        offgrid_every = 0
    if offgrid_every > 0:
        if offgrid_mode == "dense":
            _dense_q = _parse_q_per_dim(val_cfg.get("dense_q", 2 * max(q_per_dim) + 1), len(range_m))
            offgrid_dense = _build_gl_grid(
                range_m, len(range_m), _dense_q, model.m_vector.shape[0], device
            )
            _ref_label = f"dense GL Q={_dense_q}"
        elif offgrid_mode != "random":
            raise ValueError(f"validate_offgrid.mode='{offgrid_mode}' (use 'random' or 'dense')")
        else:
            _ref_label = f"random K={offgrid_k}"
        print(f"Off-grid validation: every {offgrid_every} epochs, ref={_ref_label}, "
              f"on {offgrid_nev} events")

    dataset = torch.load(cfg["paths"]["dataset"], map_location=device)
    y_data_true = dataset["y_data_distorted"]
    x_data_true = dataset["X_data_distorted"]

    total_available = y_data_true.shape[0]
    if n_events < total_available:
        perm = torch.randperm(total_available)[:n_events]
        y_data_true = y_data_true[perm]
        x_data_true = x_data_true[perm]
        print(f"Using {n_events} / {total_available} events (statistical emulation)")
    else:
        n_events = total_available

    # Held-out validation split. The residual is fit on ONE dataset, so a more
    # flexible response can lower the *training* NLL simply by absorbing this
    # sample's fluctuations (overfitting). A genuinely held-out NLL — same ν_data,
    # events the optimiser never sees — separates overfitting (val NLL stalls /
    # rises while train falls) from a real fit. `val_fraction: 0` => legacy (no split).
    val_fraction = float(train_cfg.get("val_fraction", 0.1))
    val_every = int(train_cfg.get("val_every", 1))
    n_val = int(round(val_fraction * n_events)) if val_fraction > 0 else 0
    if n_val > 0:
        x_val_data, y_val_data = x_data_true[:n_val], y_data_true[:n_val]
        x_train, y_train = x_data_true[n_val:], y_data_true[n_val:]
        n_train = x_train.shape[0]
        # Equal-size train probe so the train/val NLL gap is a pure generalisation
        # signal — both computed by the SAME end-of-epoch snapshot (not the
        # during-epoch optimisation loss, which carries a methodology offset).
        x_train_probe, y_train_probe = x_train[:n_val], y_train[:n_val]
        print(f"Validation split: {n_train} train / {n_val} held-out val events "
              f"(val_fraction={val_fraction}, val_every={val_every})")
    else:
        x_train, y_train = x_data_true, y_data_true
        x_val_data = y_val_data = None
        x_train_probe = y_train_probe = None
        n_train = n_events
        print("Validation split disabled (val_fraction=0) — only training loss tracked")

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    steps_per_epoch = max(1, n_train // batch_size)   # train-split size drives the LR schedule

    # Snapshot NLL probe (same estimator for train and val => an apples-to-apples gap).
    def _probe_nll(xx, yy):
        if grid_offsets is not None:
            return _avg_event_nll_grid(
                model, residual_transfer_model, xx, yy,
                model.m_vector, grid_offsets, grid_weights, offgrid_batch)
        return _avg_event_nll_random(
            model, residual_transfer_model, xx, yy,
            model.m_vector, range_m, max(n_random_draws, 32), offgrid_batch, device)

    # Off-grid m-integration check runs on the held-out val events when available
    # (so it doubles as a data-generalisation probe), else falls back in-sample.
    if offgrid_every > 0:
        if x_val_data is not None:
            _nv = min(offgrid_nev, n_val)
            x_val, y_val = x_val_data[:_nv], y_val_data[:_nv]
        else:
            _nv = min(offgrid_nev, n_train)
            x_val, y_val = x_train[:_nv], y_train[:_nv]

    model.m_vector.requires_grad = False
    model.norm_nuisance.requires_grad = False
    for p in transfer_model.parameters():
        p.requires_grad = False

    # Optimise only the residual stack (not the frozen base T). In ensemble mode the base is
    # swapped per batch, so excluding base params here keeps the optimizer robust to the swap.
    opt = torch.optim.AdamW(
        [{"params": residual_transfer_model.transforms.parameters(), "lr": float(train_cfg["lr"])}]
    )

    total_steps = steps_per_epoch * nepochs
    warmup_steps = steps_per_epoch
    scheduler = LinearWarmupCosineDecay(opt, warmup_steps, total_steps)

    # Extended-likelihood normalisation counts the events actually fit (train split).
    n_global_64 = torch.tensor(n_train, dtype=torch.float64, device=device)
    log_n_global_64 = torch.log(n_global_64)

    log_loss_tot: List[float] = []
    log_prob: List[float] = []
    log_lnN_params: List[List[float]] = []
    log_nuis: List[List[float]] = []
    log_norm: List[float] = []
    log_shape_nuis_term: List[float] = []

    losses: List[float] = []
    probs: List[float] = []
    lnN_params: List[List[float]] = []
    nuis: List[List[float]] = []
    norms: List[float] = []
    nuis_constr: List[float] = []

    output_path = cfg["paths"]["output_checkpoint"]
    metrics_path = output_path.replace(".pt", "_metrics.jsonl")

    exp = None
    if exp is not None:
        exp.log_parameters(
            {
                "device": device,
                "batch_size": batch_size,
                "nepochs": nepochs,
                "n_events": n_events,
                "lr": float(train_cfg["lr"]),
                "add_distance_loss": add_distance_loss,
                "lambda_distance": lambda_distance,
                "m_sampling": m_sampling,
                "m_grid_nodes": n_grid_nodes,
                "output_checkpoint": output_path,
            }
        )

    global_step = 0

    for e in range(nepochs):
        num_batches = len(loader)
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            # Ensemble: draw a random base member per batch so the single shared residual
            # learns the base-T-averaged ν-response (marginalises over the bootstrap spread).
            if member_Ts is not None:
                residual_transfer_model.base_model = member_Ts[int(torch.randint(n_members, (1,)).item())]

            log_nu_64 = torch.logsumexp(model.modified_log_normalization, dim=0).double()
            nu_64 = torch.exp(log_nu_64)
            poisson_deviance = nu_64 - n_global_64 - n_global_64 * (log_nu_64 - log_n_global_64)

            # m conditions the residual T. m = m_vector + offset over the profiled shape
            # nuisances; frozen nuisances are zeroed inside log_prob via the profile mask.
            B = x.shape[0]
            if grid_offsets is None:
                # random: K independent uniform m draws per event over [m_vector + range_m],
                # averaged (weights 1/K) to regularise the gradient. logp_sum still spans B
                # events -> data/Poisson/constraint balance preserved. K=1 = legacy path.
                K = n_random_draws
                x_in = x.repeat_interleave(K, dim=0) if K > 1 else x   # [B*K, ...] (event-major)
                y_in = y.repeat_interleave(K, dim=0) if K > 1 else y
                BK = B * K
                m = model.m_vector.unsqueeze(0).expand(BK, -1).clone()
                for _i in range(len(range_m)):
                    m[:, _i] = m[:, _i] + range_m[_i, 0] + (range_m[_i, 1] - range_m[_i, 0]) * torch.rand(BK, device=device)
                logp, s_t = model.log_prob(y_in, x_in, residual_transfer_model, m)
                logp_sum = logp.view(B, K).mean(dim=1).sum() if K > 1 else logp.sum()

                if add_distance_loss:
                    diff = s_t - y_in.unsqueeze(0)
                    reg_loss_sum = torch.sum(torch.sqrt(diff ** 2 + 1e-6)) / K
                else:
                    reg_loss_sum = torch.zeros(1, device=device)
            else:
                # quadrature: evaluate every event at every GL node, then take the
                # prior-weighted mean over nodes (weights sum to 1). logp_sum therefore
                # still spans B events -> the data/Poisson/constraint balance is unchanged.
                G = grid_offsets.shape[0]
                m_grid = model.m_vector.unsqueeze(0) + grid_offsets        # [G, num_nuisances]
                x_rep = x.repeat_interleave(G, dim=0)                      # [B*G, ...]  (event-major)
                y_rep = y.repeat_interleave(G, dim=0)
                m_rep = m_grid.repeat(B, 1)                                # [B*G, num_nuisances]
                logp, s_t = model.log_prob(y_rep, x_rep, residual_transfer_model, m_rep)
                logp_sum = (logp.view(B, G) * grid_weights.view(1, G)).sum(dim=1).sum()

                if add_distance_loss:
                    diff = s_t - y_rep.unsqueeze(0)
                    reg_loss_sum = torch.sum(torch.sqrt(diff ** 2 + 1e-6)) / G
                else:
                    reg_loss_sum = torch.zeros(1, device=device)

            shape_nuis_term = (0.5 * (model.m_vector) ** 2).sum().double()
            lnN_nuis_term = -model.get_lnN_likelihood_term().double()

            batch_poisson_loss = poisson_deviance / num_batches
            constraint_term_for_batch = (shape_nuis_term + lnN_nuis_term) / num_batches

            if add_distance_loss:
                loss = -logp_sum + batch_poisson_loss + constraint_term_for_batch + lambda_distance * reg_loss_sum
            else:
                loss = -logp_sum + batch_poisson_loss + constraint_term_for_batch

            loss = loss / batch_size

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            losses.append(float(loss.item()))
            probs.append(float((-logp_sum / batch_size).item()))
            lnN_params.append([float(p.item()) for p in model.norm_nuisance_factor])
            nuis.append(model.m_vector.detach().cpu().tolist())
            norms.append(float(lnN_nuis_term.item()))
            nuis_constr.append(float(shape_nuis_term.item()))

            if exp is not None:
                exp.log_metrics(
                    {
                        "train/loss_step": float(loss.item()),
                        "train/prob_step": float((-logp_sum / batch_size).item()),
                        "train/poisson_step": float(batch_poisson_loss.item()),
                        "train/shape_nuis_step": float(shape_nuis_term.item()),
                        "train/lnN_nuis_step": float(lnN_nuis_term.item()),
                        **{f"train/nuis_{_i}_step": float(model.m_vector[_i].item()) for _i in range(model.m_vector.shape[0])},
                        "train/lr_step": float(scheduler.get_last_lr()[0]),
                        "train/batch_idx": float(batch_idx),
                    },
                    step=global_step,
                    epoch=e,
                )
            global_step += 1

        # Pin the residual's base to member 0 so the per-epoch probe NLL, off-grid check and
        # saved checkpoint are deterministic (not whichever member the last batch happened to use).
        if member_Ts is not None:
            residual_transfer_model.base_model = member_Ts[0]

        current_lr = scheduler.get_last_lr()[0]
        log_loss_tot.append(float(sum(losses) / max(1, len(losses))))
        log_prob.append(float(sum(probs) / max(1, len(probs))))
        # Number of lnN nuisances is config-dependent (e.g. 2 in v10/v11, 1 in v13),
        # so average each entry generically rather than assuming a fixed count.
        _n_lnN = len(lnN_params[0]) if lnN_params else 0
        log_lnN_params.append([
            float(sum(p[k] for p in lnN_params) / max(1, len(lnN_params)))
            for k in range(_n_lnN)
        ])
        _n_nuis = len(nuis[0]) if nuis else 0
        log_nuis.append([
            float(sum(p[i] for p in nuis) / max(1, len(nuis)))
            for i in range(_n_nuis)
        ])
        log_norm.append(float(sum(norms) / max(1, len(norms))))
        log_shape_nuis_term.append(float(sum(nuis_constr) / max(1, len(nuis_constr))))

        # Held-out validation NLL vs an equal-size train probe (same ν_data, same
        # estimator, end-of-epoch snapshot). A widening (val − train) gap ==
        # overfitting; both falling together == a genuine fit. `train_nll` here is the
        # snapshot probe, NOT the optimisation loss `log_prob` (kept separately above).
        train_nll = float("nan")
        val_nll = float("nan")
        if x_val_data is not None and (e % val_every == 0 or e == nepochs - 1):
            train_nll = _probe_nll(x_train_probe, y_train_probe)
            val_nll = _probe_nll(x_val_data, y_val_data)

        _nuis_str = ", ".join(f"{v:.4f}" for v in log_nuis[-1])
        _val_str = (f"  |  train_nll={train_nll:.5f}, val_nll={val_nll:.5f}, "
                    f"gap={val_nll - train_nll:+.5f}"
                    if not np.isnan(val_nll) else "")
        print(
            f"epoch={e}, loss={log_loss_tot[-1]:.6f}, "
            f"nuis=({_nuis_str}), "
            f"norm_nuis={log_norm[-1]:.3f}, lr={current_lr:.2e}{_val_str}"
        )

        with open(metrics_path, "a") as _mf:
            _mf.write(json.dumps({
                "epoch": e,
                "loss": log_loss_tot[-1],
                "logprob": log_prob[-1],
                "train_nll": (None if np.isnan(train_nll) else train_nll),
                "val_nll": (None if np.isnan(val_nll) else val_nll),
                "val_gap": (None if np.isnan(val_nll) else val_nll - train_nll),
                "lr": current_lr,
                **{f"nuis_{i}": log_nuis[-1][i] for i in range(len(log_nuis[-1]))},
                **{f"lnN_{k}": log_lnN_params[-1][k] for k in range(len(log_lnN_params[-1]))},
                "norm_constraint": log_norm[-1],
                "shape_constraint": log_shape_nuis_term[-1],
            }) + "\n")

        if exp is not None:
            exp.log_metrics(
                {
                    "train/loss_epoch": float(log_loss_tot[-1]),
                    "train/prob_epoch": float(log_prob[-1]),
                    **({"train/nll_epoch": float(train_nll),
                        "val/nll_epoch": float(val_nll),
                        "val/gap_epoch": float(val_nll - train_nll)}
                       if not np.isnan(val_nll) else {}),
                    **{f"train/lnN_{k}_epoch": float(log_lnN_params[-1][k]) for k in range(len(log_lnN_params[-1]))},
                    **{f"train/nuis_{i}_epoch": float(log_nuis[-1][i]) for i in range(len(log_nuis[-1]))},
                    "train/norm_epoch": float(log_norm[-1]),
                    "train/shape_nuis_epoch": float(log_shape_nuis_term[-1]),
                    "train/lr_epoch": float(current_lr),
                },
                step=global_step,
                epoch=e,
            )

        losses.clear()
        probs.clear()
        lnN_params.clear()
        nuis.clear()
        norms.clear()
        nuis_constr.clear()

        torch.save(
            {
                "mixture_model": model.state_dict(),
                # In ensemble mode this is member 0 (the representative base); the full set of
                # base {T_b} lives in the member checkpoints under `ensemble_glob`.
                "transfer_model": transfer_model.state_dict(),
                "residual_transfer_model": residual_transfer_model.state_dict(),
                "epoch": e,
                **({"ensemble_glob": ensemble_glob} if ensemble_glob else {}),
            },
            output_path,
        )

        # Off-grid validation every N epochs (and on the final epoch).
        if offgrid_every > 0 and ((e + 1) % offgrid_every == 0 or e == nepochs - 1):
            nll_grid = _avg_event_nll_grid(
                model, residual_transfer_model, x_val, y_val,
                model.m_vector, grid_offsets, grid_weights, offgrid_batch,
            )
            if offgrid_mode == "dense":
                nll_ref = _avg_event_nll_grid(
                    model, residual_transfer_model, x_val, y_val,
                    model.m_vector, offgrid_dense[0], offgrid_dense[1], offgrid_batch,
                )
            else:
                nll_ref = _avg_event_nll_random(
                    model, residual_transfer_model, x_val, y_val,
                    model.m_vector, range_m, offgrid_k, offgrid_batch, device,
                )
            offgrid_gap = nll_ref - nll_grid
            offgrid_rel = offgrid_gap / (abs(nll_grid) + 1e-12)
            print(f"  [off-grid check] nll_grid={nll_grid:.5f}, nll_ref({_ref_label})="
                  f"{nll_ref:.5f}, gap={offgrid_gap:+.5f} ({offgrid_rel:+.2%})")
            with open(metrics_path, "a") as _mf:
                _mf.write(json.dumps({
                    "epoch": e,
                    "offgrid_nll_grid": nll_grid,
                    "offgrid_nll_ref": nll_ref,
                    "offgrid_gap": offgrid_gap,
                    "offgrid_rel": offgrid_rel,
                    "offgrid_mode": offgrid_mode,
                }) + "\n")
            if exp is not None:
                exp.log_metrics(
                    {
                        "val/offgrid_nll_grid": nll_grid,
                        "val/offgrid_nll_ref": nll_ref,
                        "val/offgrid_gap": offgrid_gap,
                        "val/offgrid_rel": offgrid_rel,
                    },
                    step=global_step,
                    epoch=e,
                )

    if exp is not None:
        exp.end()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train profiling residual transfer model from YAML config.")
    parser.add_argument("-c", "--cfg", type=str, required=True, help="Path to YAML config")
    parser.add_argument("-s", "--steps", type=str, required=True, help="Comma-separated steps, e.g. train")
    args = parser.parse_args()

    cfg_path = os.path.abspath(args.cfg)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg_dir = os.path.dirname(cfg_path)
    cfg.setdefault("runtime", {})
    requested_device = cfg["runtime"].get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        requested_device = "cpu"

    # Resolve config file paths relative to the config file location.
    for key, value in list(cfg["paths"].items()):
        if isinstance(value, str):
            cfg["paths"][key] = _resolve_path(value, cfg_dir)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    known_steps = {"train"}
    unknown = [s for s in steps if s not in known_steps]
    if unknown:
        print(f"Skipping unknown steps: {unknown}")

    if "train" in steps:
        _run_training(cfg, requested_device)


if __name__ == "__main__":
    main()
