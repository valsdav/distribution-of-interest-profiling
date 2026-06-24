"""
Train FullMixtureModel + TransferModel jointly.
Extracted from ToyDataset_withNuisance_bestfit_v9.ipynb.

Training phases:
  - warmup epochs: T only (nuisance params frozen)
  - afterwards: joint — T steps every batch; nuisances use gradient accumulation
    (gradients are summed over nuis_accum_steps batches, then averaged and stepped)

Loss = -logp_sum + poisson_deviance/num_batches + (shape_nuis + lnN_nuis)/num_batches

Steps:
  train  -- run training
"""
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

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


def _build_models(
    cfg: Dict[str, Any], device: str
) -> Tuple[FullMixtureModel, TransferModel]:
    score_cfg = cfg["score_flow"]
    score_model = zuko.flows.NSF(
        features=score_cfg["features"],
        context=score_cfg["context"],
        bins=score_cfg["bins"],
        transforms=score_cfg["transforms"],
        hidden_features=tuple(score_cfg["hidden_features"]),
    ).to(device)

    score_model.load_state_dict(
        torch.load(cfg["paths"]["score_model"], map_location=device)
    )
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
    # Load the trained polynomial residual (the m → score map learnt in Step 2).
    # Without this the residual stays at identity init and the score systematic has no
    # path to the loss — so when it is ENABLED (num_nuisances>0) require the weights.
    score_residual_path = cfg["paths"].get("score_residual_model")
    if int(residual_score_cfg["num_nuisances"]) > 0:
        if not score_residual_path or not os.path.isfile(score_residual_path):
            raise FileNotFoundError(
                f"residual_score_model.num_nuisances="
                f"{residual_score_cfg['num_nuisances']} but the score residual weights are "
                f"missing (paths.score_residual_model={score_residual_path!r}). Train them "
                "with `train_systematics.py -s train_score`; otherwise the score residual "
                "would silently stay at identity init and the score systematic be ignored."
            )
        load_state_dict_checked(
            residual_score_model, torch.load(score_residual_path, map_location=device),
            label="score residual",
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
    _kin_residual_path = cfg["paths"]["kin_residual_model"]
    if not os.path.isfile(_kin_residual_path):
        raise FileNotFoundError(
            f"Kin residual weights not found: {_kin_residual_path}. Train them with "
            "`train_systematics.py -s train_kin`."
        )
    load_state_dict_checked(
        residual_kin_model, torch.load(_kin_residual_path, map_location=device),
        label="kin residual",
    )

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
        scores_model=residual_score_model,
        kin_model=residual_kin_model,
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


def _average_and_step_nuis(opt_nuis: torch.optim.Optimizer, n: int) -> float:
    """Average accumulated nuisance gradients over n steps, take a step, return avg grad norm."""
    total_sq = 0.0
    for group in opt_nuis.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                p.grad.div_(n)
                total_sq += p.grad.norm().item() ** 2
    opt_nuis.step()
    opt_nuis.zero_grad()
    return total_sq ** 0.5




def _run_training(cfg: Dict[str, Any], device: str, member: Optional[int] = None,
                  seed: int = 0) -> None:
    # Shared init/shuffle seed across members so the ONLY per-member difference is the
    # Poisson-bootstrap weights -> the ensemble spread is a clean data-statistical spread.
    torch.manual_seed(seed)
    model, T = _build_models(cfg, device)

    train_cfg = cfg["training"]
    batch_size = int(train_cfg["batch_size"])
    nepochs = int(train_cfg["nepochs"])
    n_events = int(train_cfg["n_events"])
    warmup_epochs = int(train_cfg.get("warmup_epochs", 3))
    nuis_accum_steps = int(train_cfg.get("nuis_accum_steps", 10))
    add_distance_loss = bool(train_cfg.get("add_distance_loss", False))
    lambda_distance = float(train_cfg.get("lambda_distance", 0.001))

    dataset = torch.load(cfg["paths"]["dataset"], map_location=device)
    y_data_true = dataset["y_data_distorted"]
    x_data_true = dataset["X_data_distorted"]

    total_available = y_data_true.shape[0]
    if member is not None:
        # Poisson(1) multiplier bootstrap: reweight the FULL dataset in place (no resampling).
        # Each event gets an independent Poisson(1) count -> one bootstrap universe (equivalent
        # to the multinomial bootstrap in expectation). The per-member generator seed is the
        # ONLY source of per-member variation (init/shuffle are shared via torch.manual_seed).
        boot_seed = int(cfg["training"].get("bootstrap_seed0", 1000)) + int(member)
        g = torch.Generator(device=device).manual_seed(boot_seed)
        w_data = torch.poisson(torch.ones(total_available, device=device), generator=g)
        n_events = total_available
        print(f"Poisson bootstrap member {member} (seed {boot_seed}): "
              f"{total_available} events, sum(w)={float(w_data.sum()):.0f}")
    else:
        if n_events < total_available:
            perm = torch.randperm(total_available)[:n_events]
            y_data_true = y_data_true[perm]
            x_data_true = x_data_true[perm]
            print(f"Using {n_events} / {total_available} events (statistical emulation)")
        else:
            n_events = total_available
        w_data = torch.ones(n_events, device=device)

    loader = DataLoader(
        TensorDataset(x_data_true, y_data_true, w_data),
        batch_size=batch_size,
        shuffle=True,
    )

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * nepochs
    warmup_steps = steps_per_epoch * warmup_epochs

    opt = torch.optim.RAdam([{"params": T.parameters(), "lr": float(train_cfg["lr"])}])
    opt_nuis = torch.optim.AdamW([
        {"params": [model.m_vector, model.norm_nuisance], "lr": float(train_cfg.get("lr_nuis", train_cfg["lr"]))}
    ])
    scheduler = LinearWarmupCosineDecay(opt, warmup_steps, total_steps)

    opt_nuis.zero_grad()
    nuis_accum_count = 0

    # Extended-likelihood normalisation counts this universe's total (weighted) yield.
    N_global_64 = w_data.sum().to(torch.float64)
    log_N_global_64 = torch.log(N_global_64)

    output_path = cfg["paths"]["output_checkpoint"]
    if member is not None:
        root, ext = os.path.splitext(output_path)
        output_path = f"{root}_boot{member}{ext}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_path = output_path.replace(".pt", "_metrics.jsonl")

    exp = None

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
    nuis_grad_norms: List[float] = []

    global_step = 0

    for e in range(nepochs):
        in_warmup = e < warmup_epochs
        model.m_vector.requires_grad_(not in_warmup)
        model.norm_nuisance.requires_grad_(not in_warmup)
        for p in T.parameters():
            p.requires_grad = True

        if e == 0:
            print(f"Epoch {e}: warmup ({warmup_epochs} epochs) — training T only")
        elif e == warmup_epochs:
            print(f"Epoch {e}: joint training — T every step, nuisances every {nuis_accum_steps} steps")

        num_batches = len(loader)
        for batch_idx, (x, y, w) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)

            log_nu_64 = torch.logsumexp(model.modified_log_normalization, dim=0).double()
            nu_64 = torch.exp(log_nu_64)
            poisson_deviance = nu_64 - N_global_64 - N_global_64 * (log_nu_64 - log_N_global_64)

            m = model.m_vector.unsqueeze(0).expand(x.shape[0], -1)
            logp, s_T = model.log_prob(y, x, T, m)
            logp_sum = (w * logp).sum()

            shape_nuis_term = (0.5 * model.m_vector ** 2).sum().double()
            lnN_nuis_term = -model.get_lnN_likelihood_term().double()

            batch_poisson_loss = poisson_deviance / num_batches
            constraint_term_for_batch = (shape_nuis_term + lnN_nuis_term) / num_batches

            if add_distance_loss:
                diff = s_T - y.unsqueeze(0)
                reg_loss_sum = torch.sum(w.view(1, -1, 1) * torch.sqrt(diff ** 2 + 1e-6))
                loss = -logp_sum + batch_poisson_loss + constraint_term_for_batch + lambda_distance * reg_loss_sum
            else:
                loss = -logp_sum + batch_poisson_loss + constraint_term_for_batch

            loss = loss / batch_size

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            # Nuisance gradient accumulation
            if not in_warmup:
                
                nuis_accum_count += 1
                if nuis_accum_count == nuis_accum_steps:
                    nuis_grad_norms.append(_average_and_step_nuis(opt_nuis, nuis_accum_count))
                    nuis_accum_count = 0

            losses.append(float(loss.item()))
            probs.append(float((-logp_sum / batch_size).item()))
            lnN_params.append([float(p.item()) for p in model.norm_nuisance_factor])
            nuis.append(model.m_vector.detach().cpu().tolist())
            norms.append(float(lnN_nuis_term.item()))
            nuis_constr.append(float(shape_nuis_term.item()))
            global_step += 1

        # Flush remaining accumulated nuisance gradients
        if not in_warmup and nuis_accum_count > 0:
            nuis_grad_norms.append(_average_and_step_nuis(opt_nuis, nuis_accum_count))
            nuis_accum_count = 0

        log_loss_tot.append(float(sum(losses) / max(1, len(losses))))
        log_prob.append(float(sum(probs) / max(1, len(probs))))
        _n_lnN = len(lnN_params[0]) if lnN_params else 0
        log_lnN_params.append([
            float(sum(p[i] for p in lnN_params) / max(1, len(lnN_params)))
            for i in range(_n_lnN)
        ])
        _n_nuis = len(nuis[0]) if nuis else 0
        log_nuis.append([
            float(sum(p[i] for p in nuis) / max(1, len(nuis)))
            for i in range(_n_nuis)
        ])
        log_norm.append(float(sum(norms) / max(1, len(norms))))
        log_shape_nuis_term.append(float(sum(nuis_constr) / max(1, len(nuis_constr))))

        avg_nuis_grad = float(sum(nuis_grad_norms) / max(1, len(nuis_grad_norms)))
        current_lr = scheduler.get_last_lr()[0]

        _nuis_str = ", ".join(f"{v:.4f}" for v in log_nuis[-1])
        print(
            f"epoch={e}, loss={log_loss_tot[-1]:.5f}, "
            f"nuis=({_nuis_str}), "
            f"nuis_grad={avg_nuis_grad:.3e}, "
            f"norm_nuis={log_norm[-1]:.3f}, lr={current_lr:.2e}"
        )

        with open(metrics_path, "a") as _mf:
            _mf.write(json.dumps({
                "epoch": e,
                "loss": log_loss_tot[-1],
                "logprob": log_prob[-1],
                "lr": current_lr,
                **{f"nuis_{i}": log_nuis[-1][i] for i in range(len(log_nuis[-1]))},
                **{f"lnN_{i}": log_lnN_params[-1][i] for i in range(len(log_lnN_params[-1]))},
                "norm_constraint": log_norm[-1],
                "shape_constraint": log_shape_nuis_term[-1],
                "nuis_grad_norm": avg_nuis_grad,
            }) + "\n")

        if exp is not None:
            exp.log_metrics(
                {
                    "train/loss_epoch": log_loss_tot[-1],
                    "train/prob_epoch": log_prob[-1],
                    **{f"train/nuis_{i}_epoch": log_nuis[-1][i] for i in range(len(log_nuis[-1]))},
                    "train/nuis_grad_norm": avg_nuis_grad,
                    "train/norm_epoch": log_norm[-1],
                    "train/shape_nuis_epoch": log_shape_nuis_term[-1],
                    "train/lr_epoch": current_lr,
                },
                step=global_step,
                epoch=e,
            )

        losses.clear(); probs.clear(); lnN_params.clear()
        nuis.clear(); norms.clear(); nuis_constr.clear()
        nuis_grad_norms.clear()

        torch.save(
            {"mixture_model": model.state_dict(), "transfer_model": T.state_dict()},
            output_path,
        )

    if exp is not None:
        exp.end()

    print(f"Training complete. Checkpoint: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mixture + transfer model from YAML config.")
    parser.add_argument("-c", "--cfg", type=str, required=True, help="Path to YAML config")
    parser.add_argument("-s", "--steps", type=str, required=True, help="Comma-separated steps, e.g. train")
    parser.add_argument("--member", type=int, default=None,
                        help="Ensemble member index: enable Poisson(1) bootstrap of the dataset and "
                             "write <output>_boot<member>.pt.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Shared init/shuffle seed (kept equal across members so only the "
                             "Poisson bootstrap weights differ between ensemble members).")
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

    for key, value in list(cfg["paths"].items()):
        if isinstance(value, str):
            cfg["paths"][key] = _resolve_path(value, cfg_dir)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    if "train" in steps:
        _run_training(cfg, requested_device, member=args.member, seed=args.seed)


if __name__ == "__main__":
    main()
