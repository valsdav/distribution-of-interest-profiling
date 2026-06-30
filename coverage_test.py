"""
Coverage test for the profiling-flows chain.

Per TOY, regenerate data at the injected truth (fresh seed) and rerun the
DATA-DEPENDENT chain, REUSING the frozen MC templates (base flows + kin/score
residuals, trained once). Each toy is one self-contained job (`--toy N`).

Mode (`--mode`) — how each toy treats the transfer `T` (the shipped analysis uses a
bootstrap ENSEMBLE of `T` + one shared residual; the single-T step-2 fit is unstable,
so a single-T-per-toy test measures optimizer noise, not interval statistics):
  frozen   (B, default): regenerate data ONLY; ensemble-scan vs the GLOBAL frozen
                         {T_b} (`--ensemble-glob`) + shared residual (`--frozen-ckpt`).
                         No per-toy training. Conditional coverage given the fixed
                         apparatus (HEP-standard: build the response once, vary data).
  ensemble (C): per toy retrain K Poisson-bootstrap members (`--n-members`) + one
                shared cycling residual, then ensemble-scan. Fully unconditional; the
                ensemble keeps step-2 stable. The gold-standard cross-check for B.
  single   : legacy one-T + one-residual per toy, single-model scan.

Combine the per-member curves with `--ensemble-mode` (default rebased-bma).
Compare two runs with `--compare DIR_B DIR_C` (frozen vs ensemble coverage side by side).

Scan / interval (`--scan-mode`)
  2d (default): 2D scan; per-nuisance interval is the PROFILED one (the other
                nuisance is profiled out → min over its axis), plus a joint
                coverage check (is truth inside the 2-dof Δχ² region).
  1d:           1D scan; per-nuisance interval is CONDITIONAL (other held at anchor).

Scan window centre (`--scan-center`)
  Where the FINAL scan window sits: `anchor` (default, the model's ν₀), `truth`
  (the injected expected_nuisance — keeps truth + ν̂ in-window so contours close;
  MC-only), `zero`, or an explicit comma list. A well-windowed CI is unchanged by
  the centre — this only fixes window placement / edge artifacts.

Step subset (`--steps`)
  Comma list from {data, mixture, profiling, scan}; default all. Skipped steps
  reuse the existing per-toy outputs (errors if a needed output is missing) — e.g.
  `--steps scan` redoes only the scan on already-trained toys.

Outputs
  <out>/work/toy_<N>/...                 per-toy configs + checkpoints + logs + scan
  <out>/toys/toy_<N>.json                per-toy result (best, sigma, covered, pull, joint)
  <out>/coverage_summary.json + plots    (after --aggregate)

Configs default to the shipped ensemble chain (configs/{dataset,mixture_ensemble,
profiling_ensemble}.yaml); override with --dataset-config / --mixture-config /
--profiling-config. frozen (B) uses the shipped global ensemble + profiled residual
(models/ensemble/mixture_boot*.pt, models/mixture_ensemble_profiled.pt) and needs NO
training, so it runs out of the box.

Usage (run from the repository root)
  # frozen (B): cheap, scan-only vs the shipped frozen ensemble — no training
  python coverage_test.py --loop 0 100 --mode frozen --out-dir coverage/frozen
  python coverage_test.py --aggregate --calibrate-coverage --out-dir coverage/frozen
  # ensemble (C): gold-standard cross-check (retrains K members + residual per toy)
  python coverage_test.py --loop 0 10 --mode ensemble --n-members 8 --out-dir coverage/ensemble
  python coverage_test.py --aggregate --out-dir coverage/ensemble
  python coverage_test.py --compare coverage/frozen coverage/ensemble
  # legacy single-T chain
  python coverage_test.py --toy 0 --mode single --out-dir coverage/single
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys

import yaml

# Directory holding this driver + the chain scripts it shells out to, so the
# subprocess stages resolve regardless of the caller's working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _script(name):
    return os.path.join(_HERE, name)


ALL_STEPS = ["data", "mixture", "mixscan", "profiling", "scan"]
# Per-mode default step list (each value is a subset of ALL_STEPS):
#   single   — legacy: train ONE T + ONE residual per toy, single-model scan.
#   frozen   — (B) regenerate data only; ensemble-scan vs the global frozen {T_b}+residual.
#   ensemble — (C) per toy retrain K bootstrap members + one shared cycling residual, ensemble scan.
MODE_STEPS = {
    "single":   list(ALL_STEPS),
    "frozen":   ["data", "scan"],
    "ensemble": list(ALL_STEPS),
}
# Δχ² thresholds for a 2-parameter joint region (2 dof).
CHI2_2DOF = {"68": 2.296, "95": 5.991, "2sigma": 6.180}


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #

def _abs(path, cfg_dir):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(cfg_dir, path))


def load_absolutized(cfg_path):
    """Load a YAML config, rewriting every paths.* entry to an absolute path."""
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(cfg_path)
    for k, v in list(cfg.get("paths", {}).items()):
        if isinstance(v, str):
            cfg["paths"][k] = _abs(v, cfg_dir)
    return cfg


def write_cfg(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def run(cmd, log_path=None):
    print(f"  $ {' '.join(cmd)}")
    if log_path:
        with open(log_path, "w") as logf:
            r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed (rc={r.returncode}): {' '.join(cmd)}"
                           + (f"\n  see {log_path}" if log_path else ""))


def require(path, what):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"step skipped but its output is missing: {what} -> {path}\n"
            "  run that step first (drop it from --steps).")


def resolve_scan_center(spec, expected, num_nuis):
    """Map --scan-center to the value passed to likelihood_scan.py --center, or None to keep
    its default ('best' = the profiled checkpoint's m_vector = anchor ν₀).

      anchor|best : None        -> scan window centred on ν₀ (the model anchor; current default)
      truth|expected            -> the injected plotting.expected_nuisance vector (MC-only: uses
                                    the known truth so it + ν̂ stay inside the window — removes the
                                    window-edge artifacts; it does NOT change a well-windowed CI,
                                    only its placement)
      zero                      -> 'zero'
      "v0,v1,..."               -> explicit centre (num_nuis values), passed through
    """
    if spec in (None, "anchor", "best"):
        return None
    if spec == "zero":
        return "zero"
    if spec in ("truth", "expected"):
        if not expected or len(expected) != num_nuis:
            raise ValueError(f"--scan-center {spec} needs plotting.expected_nuisance with "
                             f"{num_nuis} values (got {expected}).")
        return ",".join(str(float(v)) for v in expected)
    vals = [v.strip() for v in spec.split(",")]
    if len(vals) != num_nuis:
        raise ValueError(f"--scan-center '{spec}' needs {num_nuis} comma values (got {len(vals)}).")
    return ",".join(vals)


# --------------------------------------------------------------------------- #
# interval estimators (pure numpy; copied from likelihood_scan to keep the
# driver torch-free — the heavy stages run torch in subprocesses)
# --------------------------------------------------------------------------- #

def _sigma_crossings(axis, curve, level=1.0):
    import numpy as np
    roots, diff = [], curve - level
    for idx in np.where(np.diff(np.sign(diff)))[0]:
        x1, x2, y1, y2 = axis[idx], axis[idx + 1], curve[idx], curve[idx + 1]
        roots.append(x1 + (level - y1) * (x2 - x1) / (y2 - y1))
    return roots


def estimate_1d(axis, curve):
    """Best value + asymmetric 1σ from a (baseline-0) -2Δln L curve."""
    import numpy as np
    k = int(np.argmin(curve))
    best = float(axis[k])
    if 0 < k < len(axis) - 1:
        y0, y1, y2 = curve[k - 1], curve[k], curve[k + 1]
        denom = y0 - 2 * y1 + y2
        if denom > 0:
            best = float(axis[k] + 0.5 * (y0 - y2) / denom * (axis[1] - axis[0]))
    roots = _sigma_crossings(axis, curve, 1.0)
    lows = [r for r in roots if r < best]
    highs = [r for r in roots if r > best]
    slo = best - max(lows) if lows else float("nan")
    shi = min(highs) - best if highs else float("nan")
    return best, slo, shi


def bilinear(ai, aj, m, xi, xj):
    """m[a,b] on grids ai (axis_i) × aj (axis_j), interpolated at (xi, xj).
    None if (xi,xj) is outside the scanned window."""
    import numpy as np
    if not (ai[0] <= xi <= ai[-1] and aj[0] <= xj <= aj[-1]):
        return None
    ia = int(np.clip(np.searchsorted(ai, xi) - 1, 0, len(ai) - 2))
    ja = int(np.clip(np.searchsorted(aj, xj) - 1, 0, len(aj) - 2))
    tx = (xi - ai[ia]) / (ai[ia + 1] - ai[ia])
    ty = (xj - aj[ja]) / (aj[ja + 1] - aj[ja])
    return float(m[ia, ja] * (1 - tx) * (1 - ty) + m[ia + 1, ja] * tx * (1 - ty)
                 + m[ia, ja + 1] * (1 - tx) * ty + m[ia + 1, ja + 1] * tx * ty)


def min2d(ai, aj, m):
    """2D minimum location of a -2Δln L grid m[a,b] (a→ai, b→aj), parabolic-refined
    along each axis through the grid argmin. Returns (best_i, best_j, on_boundary)."""
    import numpy as np
    a, b = np.unravel_index(int(np.argmin(m)), m.shape)

    def refine(axis, vals, k):
        if 0 < k < len(axis) - 1:
            y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
            den = y0 - 2 * y1 + y2
            if den > 0:
                return float(axis[k] + 0.5 * (y0 - y2) / den * (axis[1] - axis[0]))
        return float(axis[k])

    bi = refine(ai, m[:, b], a)     # vary i at fixed j=b
    bj = refine(aj, m[a, :], b)     # vary j at fixed i=a
    on_edge = a in (0, len(ai) - 1) or b in (0, len(aj) - 1)
    return bi, bj, on_edge


def cov_pull(best, slo, shi, exp):
    """(covered, pull). covered=None if the relevant side hit the window (NaN σ)."""
    sig = shi if exp >= best else slo
    if sig is None or not math.isfinite(sig) or sig <= 0:
        return None, float("nan")
    lo = best - (slo if math.isfinite(slo) else float("inf"))
    hi = best + (shi if math.isfinite(shi) else float("inf"))
    # cast away numpy types: `covered` is np.bool_ (lo/hi are np.float64) and
    # json.dump cannot serialise np.bool_ — it would crash mid-write, truncating
    # every toy_*.json at the "covered" key (and then break --aggregate's json.load).
    return bool(lo <= exp <= hi), float((best - exp) / sig)


# --------------------------------------------------------------------------- #
# result extraction
# --------------------------------------------------------------------------- #

def result_from_1d(scan_dir):
    """Conditional per-nuisance intervals from likelihood_scan's bestfit.json."""
    with open(os.path.join(scan_dir, "bestfit.json")) as f:
        bf = json.load(f)
    nuis = {}
    for ni, d in bf["nuisances"].items():
        cov, pull = (None, float("nan")) if d.get("expected") is None \
            else cov_pull(d["best_fit"], d["sigma_lo"], d["sigma_hi"], d["expected"])
        nuis[ni] = {"label": d["label"], "best_fit": d["best_fit"],
                    "sigma_lo": d["sigma_lo"], "sigma_hi": d["sigma_hi"],
                    "expected": d.get("expected"), "covered": cov, "pull": pull,
                    "interval_type": "conditional"}
    return {"nuisances": nuis, "joint": None}


def result_from_2d(scan_dir, scan2d_name, pair, labels, expected):
    """Profiled per-nuisance intervals + joint coverage from scan2d.npz."""
    import numpy as np
    d = np.load(os.path.join(scan_dir, scan2d_name), allow_pickle=True)
    sfx = f"{pair[0]}{pair[1]}"
    ai, aj, m = d[f"axis_i_{sfx}"], d[f"axis_j_{sfx}"], d[f"m2dnll_{sfx}"]

    nuis = {}
    # profile out the OTHER nuisance (min over its axis) -> proper marginal interval
    for k, ni in enumerate(pair):
        prof = m.min(axis=1 - k)            # k=0: min over axis_j; k=1: min over axis_i
        axis = ai if k == 0 else aj
        best, slo, shi = estimate_1d(axis, prof)
        exp = expected[ni] if expected is not None else None
        cov, pull = (None, float("nan")) if exp is None else cov_pull(best, slo, shi, exp)
        nuis[str(ni)] = {"label": labels[ni] if ni < len(labels) else f"nuis {ni}",
                         "best_fit": best, "sigma_lo": slo, "sigma_hi": shi,
                         "expected": exp, "covered": cov, "pull": pull,
                         "interval_type": "profiled"}

    joint = None
    if expected is not None:
        nll_truth = bilinear(ai, aj, m, expected[pair[0]], expected[pair[1]])
        if nll_truth is None:
            joint = {"truth_in_window": False}
        else:
            joint = {"truth_in_window": True, "nll_at_truth": nll_truth,
                     "covered_68": nll_truth <= CHI2_2DOF["68"],
                     "covered_2sigma": nll_truth <= CHI2_2DOF["2sigma"]}
    return {"nuisances": nuis, "joint": joint}


# --------------------------------------------------------------------------- #
# one toy = one job
# --------------------------------------------------------------------------- #

def _run_scan(args, cfg_toy, ckpt, scan_dir, profiled_idx, log, ensemble_glob=None, center=None):
    """Final likelihood scan (1D/2D). When `ensemble_glob` is given, run in ensemble mode
    with a single `--ensemble-mode` (keeps the legacy scan{1,2}d.npz filename). `center`
    (when not None) sets where the scan window is placed (likelihood_scan.py --center)."""
    base = [sys.executable, _script("likelihood_scan.py"), "-c", cfg_toy, "--ckpt", ckpt,
            "--out-dir", scan_dir]
    if center is not None:
        base += ["--center", center]
    if ensemble_glob is not None:
        base += ["--ensemble", ensemble_glob, "--ensemble-mode", args.ensemble_mode]
    if args.scan_mode == "2d":
        if len(profiled_idx) != 2:
            raise ValueError(f"2D scan expects exactly 2 profiled nuisances, got {profiled_idx}")
        pi, pj = profiled_idx
        cmd = base + ["--scan-2d", "--pairs", f"{pi},{pj}", "--half", str(args.scan_half),
                      "--steps-2d", str(args.scan_steps_2d), "--scan2d-name", "scan2d.npz"]
    else:
        cmd = base + ["--scan-1d", "--half", str(args.scan_half),
                      "--steps-1d", str(args.scan_steps)]
    run(cmd, log)


def _run_mixscan(args, mix_cfg_toy, ckpt, mixscan_dir, profiled_idx, num_nuis, base_override,
                 log, ensemble_glob=None):
    """Mixture 2D anchor scan -> (m_vector_override, on_boundary). When `ensemble_glob` is
    given the anchor is the BMA (or chosen mode) over members (mixture-stage, no residual)."""
    import numpy as _np
    if len(profiled_idx) != 2:
        raise ValueError(f"mixscan expects exactly 2 profiled nuisances, got {profiled_idx}")
    pi, pj = profiled_idx
    cmd = [sys.executable, _script("likelihood_scan.py"), "-c", mix_cfg_toy, "--ckpt", ckpt,
           "--out-dir", mixscan_dir, "--scan-2d", "--pairs", f"{pi},{pj}",
           "--half", str(args.mixscan_half), "--steps-2d", str(args.mixscan_steps),
           "--scan2d-name", "scan2d.npz"]
    if ensemble_glob is not None:
        cmd += ["--ensemble", ensemble_glob, "--ensemble-mode", args.mixscan_ensemble_mode]
    run(cmd, log)
    d = _np.load(os.path.join(mixscan_dir, "scan2d.npz"), allow_pickle=True)
    sfx = f"{pi}{pj}"
    bi, bj, on_edge = min2d(d[f"axis_i_{sfx}"], d[f"axis_j_{sfx}"], d[f"m2dnll_{sfx}"])
    override = list(base_override or [0.0] * num_nuis)
    override[pi], override[pj] = bi, bj
    return override, on_edge


def run_toy(args):
    toy = args.toy
    seed = args.seed0 + toy
    steps = set(args.steps)
    mode = getattr(args, "mode", "single")
    work = os.path.join(args.out_dir, "work", f"toy_{toy}")
    os.makedirs(work, exist_ok=True)
    toys_dir = os.path.join(args.out_dir, "toys")
    os.makedirs(toys_dir, exist_ok=True)

    ds_cfg = load_absolutized(args.dataset_config)
    mix_cfg = load_absolutized(args.mixture_config)
    prof_cfg = load_absolutized(args.profiling_config)

    # frozen MC templates must already exist (trained once)
    shared = [prof_cfg["paths"]["score_density_state"], prof_cfg["paths"]["kin_density_state"],
              mix_cfg["paths"]["score_model"], mix_cfg["paths"]["score_residual_model"],
              mix_cfg["paths"]["kin_residual_model"]]
    miss = [s for s in set(shared) if not os.path.isfile(s)]
    if miss and ({"mixture", "profiling", "scan"} & steps):
        raise FileNotFoundError("Frozen MC templates missing — train them ONCE first:\n  "
                                + "\n  ".join(miss))

    toy_ds = os.path.join(work, "dataset.pt")
    toy_mix = os.path.join(work, "mixture.pt")
    toy_prof = os.path.join(work, "profiled.pt")
    scan_dir = os.path.join(work, "scan")
    mixscan_dir = os.path.join(work, "mixscan")
    anchor_file = os.path.join(work, "anchor.json")
    member_glob = os.path.join(work, "mixture_boot*.pt")     # ensemble mode (per-toy members)
    member0 = os.path.join(work, "mixture_boot0.pt")
    ens = mode == "ensemble"

    # profiled nuisances / pair, labels, expected (needed by mixscan + scan + result)
    num_nuis = int(prof_cfg["mixture_model"]["num_nuisances"])
    profiled_mask = prof_cfg.get("mixture_model", {}).get("m_vector_profile_mask")
    profiled_idx = list(range(num_nuis)) if profiled_mask is None \
        else [i for i, m in enumerate(profiled_mask) if m]
    labels = prof_cfg.get("plotting", {}).get("shape_nuisance_labels", [])
    expected = prof_cfg.get("plotting", {}).get("expected_nuisance")
    scan_center = resolve_scan_center(getattr(args, "scan_center", "anchor"), expected, num_nuis)

    print(f"[toy {toy}] mode={mode}  seed={seed}  steps={sorted(steps)}  scan-mode={args.scan_mode}"
          f"  scan-center={scan_center or 'anchor (ν₀)'}")

    # 1. data (all modes regenerate the toy dataset at the injected truth)
    ds_cfg["paths"]["dataset"] = toy_ds
    ds_cfg_toy = write_cfg(ds_cfg, os.path.join(work, "dataset.yaml"))
    if "data" in steps:
        run([sys.executable, _script("generate_dataset.py"), "-c", ds_cfg_toy, "--seed", str(seed)],
            os.path.join(work, "gen.log"))
    elif {"mixture", "mixscan", "profiling", "scan"} & steps:
        require(toy_ds, "dataset")

    if mode == "frozen":
        # ---- (B) ensemble scan vs the GLOBAL frozen {T_b}+residual; no per-toy training ----
        frozen_ckpt = os.path.abspath(args.frozen_ckpt)
        ens_glob = os.path.abspath(args.ensemble_glob)
        require(frozen_ckpt, "frozen profiled-ensemble checkpoint")
        if not glob.glob(ens_glob):
            raise FileNotFoundError(f"no ensemble members match {ens_glob}")
        prof_cfg["paths"]["dataset"] = toy_ds
        prof_cfg.setdefault("comet", {})["api_key"] = None
        prof_cfg_toy = write_cfg(prof_cfg, os.path.join(work, "profiling.yaml"))
        if "scan" in steps:
            _run_scan(args, prof_cfg_toy, frozen_ckpt, scan_dir, profiled_idx,
                      os.path.join(work, "scan.log"), ensemble_glob=ens_glob, center=scan_center)
    else:
        # ---- single (legacy) / ensemble (C): train per toy, then scan ----
        mix_cfg["paths"]["dataset"] = toy_ds
        mix_cfg["paths"]["output_checkpoint"] = toy_mix
        if ens:
            mix_cfg.setdefault("training", {})["bootstrap_seed0"] = \
                args.member_seed0 + toy * args.n_members
        mix_cfg.setdefault("comet", {})["api_key"] = None
        # Scatter mode (`--member b`): each member trains in its OWN job, so write a per-member
        # config — sibling member jobs must not clobber a shared mixture.yaml mid-read.
        single_member = ens and getattr(args, "member", None) is not None
        mix_cfg_toy = write_cfg(mix_cfg, os.path.join(
            work, f"mixture_m{args.member}.yaml" if single_member else "mixture.yaml"))

        # 2. mixture — single T, or K Poisson-bootstrap members (ensemble). With `--member b`
        #    train ONLY that member (scatter over jobs); otherwise train all K serially.
        if "mixture" in steps:
            if ens:
                for b in ([args.member] if single_member else range(args.n_members)):
                    run([sys.executable, _script("train_mixture.py"), "-c", mix_cfg_toy, "-s", "train",
                         "--member", str(b)], os.path.join(work, f"mixture_boot{b}.log"))
            else:
                run([sys.executable, _script("train_mixture.py"), "-c", mix_cfg_toy, "-s", "train"],
                    os.path.join(work, "mixture.log"))
        elif {"mixscan", "profiling"} & steps:
            require(member0 if ens else toy_mix, "mixture checkpoint")

        # 3. mixscan: 2D scan of the MIXTURE -> profiling anchor (BMA over members if ensemble).
        if "mixscan" in steps:
            override, on_edge = _run_mixscan(
                args, mix_cfg_toy, member0 if ens else toy_mix, mixscan_dir, profiled_idx,
                num_nuis, prof_cfg["mixture_model"].get("m_vector_override"),
                os.path.join(work, "mixscan.log"), ensemble_glob=member_glob if ens else None)
            with open(anchor_file, "w") as f:
                json.dump({"m_vector_override": override, "on_boundary": bool(on_edge)}, f, indent=2)
            print(f"[toy {toy}] mixscan anchor = {[round(v, 4) for v in override]}"
                  + ("  ⚠ min on boundary — widen --mixscan-half" if on_edge else ""))

        # 4. profiling — only prepared/run by a job that actually reaches profiling or scan.
        #    A scatter member job (`--steps mixture`) skips this, so it never writes the shared
        #    profiling.yaml concurrently with its sibling members (the gather job writes it once).
        if {"profiling", "scan"} & steps:
            # anchor (m_vector_override) from the mixscan when available
            if os.path.isfile(anchor_file):
                with open(anchor_file) as f:
                    prof_cfg["mixture_model"]["m_vector_override"] = json.load(f)["m_vector_override"]
            else:
                prof_cfg.get("mixture_model", {}).pop("m_vector_override", None)
                if "profiling" in steps:
                    print(f"[toy {toy}] no mixscan anchor — profiling anchors on the mixture m_vector fit")

            prof_cfg["paths"]["dataset"] = toy_ds
            prof_cfg["paths"]["output_checkpoint"] = toy_prof
            if ens:
                prof_cfg["paths"]["init_mixture_ensemble"] = member_glob   # cycle members per batch
                prof_cfg["paths"].pop("init_mixture_checkpoint", None)
            else:
                prof_cfg["paths"]["init_mixture_checkpoint"] = toy_mix
            prof_cfg.setdefault("comet", {})["api_key"] = None
            prof_cfg_toy = write_cfg(prof_cfg, os.path.join(work, "profiling.yaml"))

            if "profiling" in steps:
                run([sys.executable, _script("train_profiling.py"), "-c", prof_cfg_toy, "-s", "train"],
                    os.path.join(work, "profiling.log"))

            # 5. scan — single model, or ensemble (members cycled through the shared residual)
            if "scan" in steps:
                require(toy_prof, "profiled checkpoint")
                _run_scan(args, prof_cfg_toy, toy_prof, scan_dir, profiled_idx,
                          os.path.join(work, "scan.log"),
                          ensemble_glob=member_glob if ens else None, center=scan_center)

    # 6. distil result — only when a scan ran in THIS job. Scatter data/member jobs run partial
    #    --steps (no scan output yet); the per-toy gather job runs the scan and distils.
    if "scan" not in steps:
        print(f"[toy {toy}] partial steps {sorted(steps)} complete (no scan in this job) -> {work}")
        return None
    if args.scan_mode == "2d":
        res = result_from_2d(scan_dir, "scan2d.npz", profiled_idx, labels, expected)
    else:
        res = result_from_1d(scan_dir)
    result = {"toy": toy, "seed": seed, "mode": mode, "scan_mode": args.scan_mode, **res}
    out_json = os.path.join(toys_dir, f"toy_{toy}.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[toy {toy}] done -> {out_json}")
    for ni, r in result["nuisances"].items():
        print(f"    {r['label']}: ν̂={r['best_fit']:+.4f} +{r['sigma_hi']:.4f}/-{r['sigma_lo']:.4f}"
              f"  exp={r['expected']}  covered={r['covered']}  pull={r['pull']:+.2f}"
              f"  [{r['interval_type']}]")
    if result.get("joint"):
        print(f"    joint: {result['joint']}")
    return out_json


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #

def plot_accumulated_contours(out_dir, toys, level_key="68"):
    """Overlay each toy's 2D joint confidence contour (one pair only), coloured by
    whether it covers the truth, plus the ν̂ scatter and the truth marker. Reads the
    full -2Δln L grids from each toy's work/.../scan/scan2d.npz."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    keys = sorted({int(k) for t in toys for k in t["nuisances"]})
    if len(keys) != 2:
        print("  (accumulated contours: need exactly 2 nuisances — skipped)")
        return
    i, j = keys
    sfx = f"{i}{j}"
    level = CHI2_2DOF[level_key]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    nx, ny, ncov, ntot, exp, lx, ly = [], [], 0, 0, None, None, None
    for t in toys:
        npz = os.path.join(out_dir, "work", f"toy_{t['toy']}", "scan", "scan2d.npz")
        if not os.path.isfile(npz):
            continue
        try:
            d = np.load(npz, allow_pickle=True)
            ai, aj, m = d[f"axis_i_{sfx}"], d[f"axis_j_{sfx}"], d[f"m2dnll_{sfx}"]
        except Exception:
            continue
        covered = bool((t.get("joint") or {}).get(f"covered_{level_key}"))
        color = "#1b7837" if covered else "#b2182b"
        ax.contour(ai, aj, m.T, levels=[level], colors=[color], linewidths=0.8, alpha=0.45)
        bi, bj = t["nuisances"][str(i)], t["nuisances"][str(j)]
        ax.plot(bi["best_fit"], bj["best_fit"], ".", color=color, ms=4, alpha=0.8)
        nx.append(bi["best_fit"]); ny.append(bj["best_fit"])
        ntot += 1; ncov += covered
        exp = [bi["expected"], bj["expected"]]
        lx, ly = bi["label"], bj["label"]

    if ntot == 0:
        print("  (accumulated contours: no scan2d.npz found under work/ — skipped)")
        plt.close(fig)
        return
    if exp[0] is not None:
        ax.plot(exp[0], exp[1], "*", color="black", ms=20, zorder=6, label="truth")
    ax.plot(np.mean(nx), np.mean(ny), "P", color="#2166ac", ms=13, zorder=6, label="mean ν̂")
    handles = [Line2D([], [], color="#1b7837", lw=2, label="covers truth"),
               Line2D([], [], color="#b2182b", lw=2, label="misses truth"),
               Line2D([], [], color="black", marker="*", ls="none", ms=12, label="truth"),
               Line2D([], [], color="#2166ac", marker="P", ls="none", ms=10, label="mean ν̂")]
    ax.legend(handles=handles, fontsize=9, loc="best")
    ax.set_xlabel(lx); ax.set_ylabel(ly)
    ax.set_title(f"Accumulated {level_key}% joint contours — {ntot} toys "
                 f"(cover truth: {ncov}/{ntot} = {ncov / ntot:.0%})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"coverage_contours.{ext}"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Contours -> {os.path.join(out_dir, 'coverage_contours.png')} "
          f"({ncov}/{ntot} cover truth)")


def calibrate_coverage(toys, keys, targets):
    """For each target probability q (e.g. 0.6827, 0.95), find the σ-multiplier and the
    joint Δχ² threshold that make the empirical coverage equal q.

    Exact for the interval definition: scaling a per-nuisance interval to ±k·σ covers the
    truth iff |pull| ≤ k, so coverage(k)=fraction(|pull|≤k) and the calibrated multiplier is
    k*(q) = q-quantile(|pull|). The standard q-interval uses k_nom(q)=√χ²₁(q) (1.0@68.3%,
    1.96@95%), so the σ-inflation vs the standard interval is k*/k_nom. For the joint 2D,
    −2Δln L(truth) is a real 2-dof statistic, so the calibrated Δχ² threshold is its
    q-quantile, vs the nominal χ²₂(q) (2.30@68.3%, 5.99@95%). NON-Gaussian pulls ⇒ the 68%
    and 95% inflations differ (interval shape wrong, not just width). Boundary toys
    (NaN σ → NaN pull) are excluded (flagged).
    """
    import numpy as np
    from scipy.stats import chi2
    per_nuis, curves = {}, {}
    for ni in keys:
        recs = [t["nuisances"][ni] for t in toys if ni in t["nuisances"]]
        ap = np.array([abs(r["pull"]) for r in recs
                       if r.get("covered") is not None and np.isfinite(r.get("pull", np.nan))])
        if ap.size == 0:
            continue
        tg = {}
        for q in targets:
            k_nom = float(chi2.ppf(q, 1) ** 0.5)
            k_star = float(np.quantile(ap, q))
            tg[f"{q:.4f}"] = {
                "target": q, "k_nominal": k_nom, "k_star": k_star,
                "sigma_inflation": k_star / k_nom, "threshold_star": k_star ** 2,
                "coverage_nominal": float(np.mean(ap <= k_nom)),
                "coverage_calibrated": float(np.mean(ap <= k_star))}
        per_nuis[ni] = {"label": recs[0]["label"], "n_valid": int(ap.size),
                        "n_excluded": int(len(recs) - ap.size), "targets": tg}
        curves[ni] = ap
    jn = np.array([t["joint"]["nll_at_truth"] for t in toys
                   if t.get("joint") and t["joint"].get("truth_in_window")
                   and np.isfinite(t["joint"].get("nll_at_truth", np.nan))])
    joint = None
    if jn.size:
        tg = {}
        for q in targets:
            L_nom = float(chi2.ppf(q, 2))
            L_star = float(np.quantile(jn, q))
            tg[f"{q:.4f}"] = {
                "target": q, "threshold_nominal": L_nom, "threshold_star": L_star,
                "region_inflation": float((L_star / L_nom) ** 0.5),
                "coverage_nominal": float(np.mean(jn <= L_nom)),
                "coverage_calibrated": float(np.mean(jn <= L_star))}
        joint = {"n_valid": int(jn.size), "targets": tg}
    return per_nuis, curves, joint, (jn if jn.size else None)


def plot_coverage_calibration(out_dir, per_nuis, curves, joint_cal, joint_nll, targets):
    """Empirical coverage vs threshold: per nuisance vs the σ-multiplier k (coverage=
    frac(|pull|≤k)), plus the joint vs the Δχ² threshold. Each target q gets a coloured
    h-line; the calibrated k*(q)/Δχ²*(q) where the curve crosses it is the v-line."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tcols = ["#2166ac", "#b2182b", "#762a83", "#d95f02"]
    items = list(curves.items())
    n = len(items) + (1 if joint_cal else 0)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.0), squeeze=False)
    kmax = 1.5
    for ni in curves:
        kmax = max([kmax] + [per_nuis[ni]["targets"][q]["k_star"] for q in per_nuis[ni]["targets"]])
    ks = np.linspace(0, kmax * 1.3, 240)
    col = 0
    for ni, ap in items:
        ax = axes[0][col]
        ax.plot(ks, [(ap <= k).mean() for k in ks], "-", color="#444", lw=2)
        ax.axvline(1.0, color="gray", lw=1, alpha=0.5)
        for ti, e in enumerate(per_nuis[ni]["targets"].values()):
            c = tcols[ti % len(tcols)]
            ax.axhline(e["target"], color=c, ls="--", lw=1)
            ax.axvline(e["k_star"], color=c, ls=":", lw=1.6,
                       label=f"{e['target']:.0%}: ±{e['k_star']:.2f}σ (×{e['sigma_inflation']:.2f})")
        ax.set_xlabel("σ multiplier k"); ax.set_ylim(0, 1.02)
        ax.set_title(per_nuis[ni]["label"], fontsize=10)
        if col == 0:
            ax.set_ylabel("coverage = frac(|pull| ≤ k)")
        ax.legend(fontsize=8)
        col += 1
    if joint_cal:
        ax = axes[0][col]
        Lmax = max(e["threshold_star"] for e in joint_cal["targets"].values()) * 1.25
        Ls = np.linspace(0, Lmax, 240)
        ax.plot(Ls, [(joint_nll <= L).mean() for L in Ls], "-", color="#1b7837", lw=2)
        for ti, e in enumerate(joint_cal["targets"].values()):
            c = tcols[ti % len(tcols)]
            ax.axhline(e["target"], color=c, ls="--", lw=1)
            ax.axvline(e["threshold_nominal"], color=c, lw=1, alpha=0.4)
            ax.axvline(e["threshold_star"], color=c, ls=":", lw=1.6,
                       label=f"{e['target']:.0%}: Δχ²={e['threshold_star']:.2f} (×{e['region_inflation']:.2f})")
        ax.set_xlabel("joint Δχ² threshold"); ax.set_ylim(0, 1.02)
        ax.set_title("joint 2D", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Coverage calibration — empirical coverage vs threshold "
                 f"(targets {', '.join(f'{q:.0%}' for q in targets)})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"coverage_calibration.{ext}"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Calibration plot -> {os.path.join(out_dir, 'coverage_calibration.png')}")


def run_aggregate(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(glob.glob(os.path.join(args.out_dir, "toys", "toy_*.json")))
    if not files:
        raise FileNotFoundError(f"no toy_*.json under {args.out_dir}/toys")
    toys = [json.load(open(f)) for f in files]
    print(f"Aggregating {len(toys)} toys")

    keys = sorted({ni for t in toys for ni in t["nuisances"]}, key=int)
    summary = {"n_toys": len(toys), "scan_mode": toys[0].get("scan_mode"), "nuisances": {}}
    fig, axes = plt.subplots(len(keys), 2, figsize=(11, 4.0 * len(keys)), squeeze=False)

    for row, ni in enumerate(keys):
        recs = [t["nuisances"][ni] for t in toys if ni in t["nuisances"]]
        label, exp = recs[0]["label"], recs[0]["expected"]
        best = np.array([r["best_fit"] for r in recs], float)
        pulls = np.array([r["pull"] for r in recs], float)
        cov = [r["covered"] for r in recs]
        n_valid = sum(c is not None for c in cov)
        n_cov = sum(bool(c) for c in cov if c is not None)
        n_flag = sum(c is None for c in cov)
        cover = (n_cov / n_valid) if n_valid else float("nan")
        bias = float(np.mean(best - exp)) if exp is not None else float("nan")
        pf = pulls[np.isfinite(pulls)]
        summary["nuisances"][ni] = {
            "label": label, "expected": exp, "interval_type": recs[0].get("interval_type"),
            "nuhat_mean": float(np.mean(best)), "nuhat_std": float(np.std(best)),
            "bias_mean": bias, "bias_rms": float(np.std(best - exp)) if exp is not None else float("nan"),
            "coverage": cover, "n_valid": n_valid, "n_flagged": n_flag,
            "pull_mean": float(np.mean(pf)) if pf.size else float("nan"),
            "pull_std": float(np.std(pf)) if pf.size else float("nan")}
        print(f"  [{label}] ν̂={np.mean(best):+.4f}±{np.std(best):.4f}  bias={bias:+.4f}  "
              f"coverage={n_cov}/{n_valid}={cover:.0%}"
              + (f" ({n_flag} flagged: widen --scan-half)" if n_flag else "")
              + f"  pull={summary['nuisances'][ni]['pull_mean']:+.2f}±"
                f"{summary['nuisances'][ni]['pull_std']:.2f}  [{recs[0].get('interval_type')}]")

        ax = axes[row][0]
        ax.hist(best, bins=max(8, len(best) // 3), color="#4393c3", alpha=0.85)
        ax.axvline(np.mean(best), color="#2166ac", lw=2, label=f"mean={np.mean(best):.4f}")
        if exp is not None:
            ax.axvline(exp, color="black", ls="--", lw=2, label=f"truth={exp}")
        ax.set_title(f"{label}: ν̂  (coverage {cover:.0%}, {recs[0].get('interval_type')})", fontsize=10)
        ax.set_xlabel("ν̂"); ax.legend(fontsize=8)

        ax = axes[row][1]
        if pf.size:
            ax.hist(pf, bins=max(8, pf.size // 3), density=True, color="#1b7837", alpha=0.8)
            xs = np.linspace(-4, 4, 200)
            ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.5, label="N(0,1)")
            ax.axvline(np.mean(pf), color="#762a83", lw=2,
                       label=f"mean={np.mean(pf):+.2f}, std={np.std(pf):.2f}")
        ax.set_title(f"{label}: pull (ν̂−truth)/σ", fontsize=10)
        ax.set_xlabel("pull"); ax.legend(fontsize=8)

    # joint coverage (2D mode)
    joints = [t["joint"] for t in toys if t.get("joint") and t["joint"].get("truth_in_window")]
    if joints:
        n_jv = len(joints)
        c68 = sum(bool(j["covered_68"]) for j in joints) / n_jv
        c2s = sum(bool(j["covered_2sigma"]) for j in joints) / n_jv
        n_oow = sum(1 for t in toys if t.get("joint") and not t["joint"].get("truth_in_window"))
        summary["joint"] = {"n_valid": n_jv, "coverage_68": c68, "coverage_2sigma": c2s,
                            "n_truth_out_of_window": n_oow}
        print(f"  [joint 2D] 68% region coverage = {c68:.0%}  (2σ region {c2s:.0%})"
              + (f"  ({n_oow} toys had truth outside the scan window)" if n_oow else ""))

    # coverage calibration: σ-multiplier / Δχ² threshold to reach each target coverage
    if args.calibrate_coverage:
        targets = [float(v) for v in str(args.target_coverage).split(",")]
        per_nuis, curves, joint_cal, joint_nll = calibrate_coverage(toys, keys, targets)
        summary["calibration"] = {"targets": targets, "nuisances": per_nuis, "joint": joint_cal}
        print(f"\nCoverage calibration  (use ±k*·σ for the target interval; inflation = k*/k_nominal; "
              f"k>nominal ⇒ under-covered):")
        for ni, c in per_nuis.items():
            print(f"  [{c['label']}]  (n={c['n_valid']}"
                  + (f", {c['n_excluded']} excluded" if c["n_excluded"] else "") + ")")
            for e in c["targets"].values():
                print(f"      {e['target']:.1%}: nominal cov {e['coverage_nominal']:.0%} (±{e['k_nominal']:.2f}σ)"
                      f"  →  use ±{e['k_star']:.2f}σ (−2Δlnℒ thr {e['threshold_star']:.2f}, ×{e['sigma_inflation']:.2f})")
        if joint_cal:
            print(f"  [joint 2D]  (n={joint_cal['n_valid']})")
            for e in joint_cal["targets"].values():
                print(f"      {e['target']:.1%}: nominal cov {e['coverage_nominal']:.0%} (Δχ²={e['threshold_nominal']:.2f})"
                      f"  →  Δχ²={e['threshold_star']:.2f} (region ×{e['region_inflation']:.2f})")
        if not per_nuis and not joint_cal:
            print("  (no valid toys to calibrate)")
        else:
            plot_coverage_calibration(args.out_dir, per_nuis, curves, joint_cal, joint_nll, targets)

    fig.suptitle(f"Coverage test — {args.out_dir}  ({len(toys)} toys, {summary['scan_mode']})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out_dir, f"coverage_summary.{ext}"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    if summary.get("scan_mode") == "2d":
        plot_accumulated_contours(args.out_dir, toys)

    with open(os.path.join(args.out_dir, "coverage_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {os.path.join(args.out_dir, 'coverage_summary.json')}")
    print(f"Plot    -> {os.path.join(args.out_dir, 'coverage_summary.png')}")


def run_compare(dir_a, dir_b):
    """Print the coverage_summary.json of two runs side by side (e.g. frozen B vs ensemble C)."""
    def _load(d):
        p = os.path.join(os.path.abspath(d), "coverage_summary.json")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p} not found — run --aggregate on {d} first")
        return json.load(open(p))

    A, B = _load(dir_a), _load(dir_b)
    la, lb = os.path.basename(dir_a.rstrip("/")), os.path.basename(dir_b.rstrip("/"))
    fc = lambda x: f"{x:.0%}" if isinstance(x, (int, float)) else "—"
    fp = lambda x: f"{x:+.2f}" if isinstance(x, (int, float)) else "—"
    print(f"\nCoverage comparison   A={la} ({A.get('n_toys')} toys)   "
          f"B={lb} ({B.get('n_toys')} toys)\n" + "-" * 70)
    print(f"{'nuisance':<14}{'coverage A':>13}{'coverage B':>13}{'pull σ A':>12}{'pull σ B':>12}")
    for ni in sorted(set(A["nuisances"]) | set(B["nuisances"]), key=int):
        a, b = A["nuisances"].get(ni, {}), B["nuisances"].get(ni, {})
        lab = a.get("label") or b.get("label") or ni
        print(f"{lab:<14}{fc(a.get('coverage')):>13}{fc(b.get('coverage')):>13}"
              f"{fp(a.get('pull_std')):>12}{fp(b.get('pull_std')):>12}")
    ja, jb = A.get("joint"), B.get("joint")
    if ja or jb:
        jc = lambda j, k: f"{j[k]:.0%}" if j and k in j else "—"
        print("-" * 70)
        print(f"{'joint 68%':<14}{jc(ja, 'coverage_68'):>13}{jc(jb, 'coverage_68'):>13}")
        print(f"{'joint 2σ':<14}{jc(ja, 'coverage_2sigma'):>13}{jc(jb, 'coverage_2sigma'):>13}")
    print("\nNote: frozen (B) = conditional coverage (T-ensemble spread fixed across toys);\n"
          "      ensemble (C) refluctuates the ensemble. Agreement ⇒ conditional ≈ unconditional.\n"
          "      pull σ ≈ 1 ⇒ interval width is calibrated.")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Profiling-flows coverage test (one job per toy).")
    p.add_argument("--toy", type=int, default=None, help="Run a single toy (index).")
    p.add_argument("--loop", type=int, nargs=2, metavar=("START", "STOP"), default=None,
                   help="Serial loop over toys [START, STOP).")
    p.add_argument("--aggregate", action="store_true", help="Aggregate existing toy_*.json.")
    p.add_argument("--calibrate-coverage", action="store_true",
                   help="In --aggregate: for each --target-coverage, find the σ-multiplier (≡ "
                        "−2Δlnℒ threshold) and joint Δχ² threshold giving that empirical coverage; "
                        "writes a coverage_calibration plot + a summary['calibration'] entry.")
    p.add_argument("--target-coverage", type=str, default="0.6827,0.95",
                   help="Comma list of target coverages for --calibrate-coverage "
                        "(default '0.6827,0.95' = nominal 1σ and 95%%).")
    p.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"), default=None,
                   help="Print the coverage_summary.json of two runs (e.g. frozen vs ensemble) "
                        "side by side, then exit. No toys are run.")
    p.add_argument("--mode", choices=["frozen", "ensemble", "single"], default="frozen",
                   help="frozen (B, default): regenerate data only, ensemble-scan vs the global "
                        "frozen {T_b}+residual. ensemble (C): per toy retrain K members + one shared "
                        "cycling residual. single: legacy one-T-per-toy chain.")
    p.add_argument("--steps", type=str, default=None,
                   help=f"Comma list of steps to (re)run; rest are reused. Default: the mode's "
                        f"natural steps ({ {k: v for k, v in MODE_STEPS.items()} }). Choices: {ALL_STEPS}.")
    p.add_argument("--scan-mode", choices=["1d", "2d"], default="2d",
                   help="2d (default): profiled intervals + joint coverage. 1d: conditional intervals.")
    p.add_argument("--dataset-config", default="configs/dataset.yaml",
                   help="Toy-dataset generation config (re-generated per toy at a fresh seed).")
    p.add_argument("--mixture-config", default="configs/mixture_ensemble.yaml",
                   help="Step-1 (mixture) config; supplies the frozen MC template paths and, in "
                        "ensemble mode, the per-toy bootstrap members.")
    p.add_argument("--profiling-config", default="configs/profiling_ensemble.yaml",
                   help="Step-2 (profiling) config; supplies the residual + the nuisance/plotting "
                        "metadata (num_nuisances, profile mask, expected_nuisance, labels).")
    p.add_argument("--out-dir", default="coverage/frozen")
    p.add_argument("--seed0", type=int, default=1000, help="toy N uses seed0+N.")
    p.add_argument("--scan-center", default="anchor",
                   help="Where to centre the FINAL scan window: 'anchor' (default; the profiled "
                        "checkpoint's m_vector ν₀), 'truth'/'expected' (the injected "
                        "plotting.expected_nuisance — keeps truth + ν̂ in-window, removing the "
                        "window-edge contour artifacts; MC-only, uses the known truth), 'zero', "
                        "or a comma list per nuisance. (The mixscan keeps its own centre.)")
    p.add_argument("--scan-half", type=float, default=0.15)
    p.add_argument("--scan-steps", type=int, default=41, help="1D grid points.")
    p.add_argument("--scan-steps-2d", type=int, default=21, help="2D grid points per axis.")
    p.add_argument("--mixscan-half", type=float, default=0.15,
                   help="Half-width of the mixture 2D anchor scan (around the mixture m_vector).")
    p.add_argument("--mixscan-steps", type=int, default=21, help="Mixture-scan grid points per axis.")
    # ---- ensemble (frozen / ensemble modes) ----
    p.add_argument("--ensemble-glob",
                   default="models/ensemble/mixture_boot*.pt",
                   help="Glob of bootstrap member checkpoints. frozen: the global members to scan "
                        "against; ensemble: ignored (members are retrained per toy).")
    p.add_argument("--frozen-ckpt",
                   default="models/mixture_ensemble_profiled.pt",
                   help="frozen mode: global profiled-ensemble (shared residual) checkpoint + anchor ν₀.")
    p.add_argument("--ensemble-mode", default="rebased-bma",
                   choices=["bma", "envelope", "rebased-bma", "rebased-envelope"],
                   help="Combine mode for the final ensemble scan (single mode → legacy scan2d.npz).")
    p.add_argument("--mixscan-ensemble-mode", default="bma",
                   choices=["bma", "envelope", "rebased-bma", "rebased-envelope"],
                   help="ensemble mode: combine mode for the per-toy anchor (mixture) scan.")
    p.add_argument("--n-members", type=int, default=8, help="ensemble mode: K members trained per toy.")
    p.add_argument("--member-seed0", type=int, default=2000,
                   help="ensemble mode: per-toy member bootstrap base "
                        "(bootstrap_seed0 = member_seed0 + toy*n_members).")
    p.add_argument("--member", type=int, default=None,
                   help="ensemble mode: in the 'mixture' step, train ONLY this member index "
                        "(scatter-gather submission — one Condor job per member). Default: all K.")
    args = p.parse_args()

    args.out_dir = os.path.abspath(args.out_dir)

    if args.compare is not None:
        run_compare(args.compare[0], args.compare[1])
        return

    os.makedirs(args.out_dir, exist_ok=True)

    if args.steps is None:
        args.steps = list(MODE_STEPS[args.mode])
    else:
        args.steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in args.steps if s not in ALL_STEPS]
    if bad:
        p.error(f"unknown steps {bad}; choices {ALL_STEPS}")

    if args.member is not None:
        if args.mode != "ensemble":
            p.error("--member is only valid with --mode ensemble (scatter a single member).")
        if not (0 <= args.member < args.n_members):
            p.error(f"--member {args.member} out of range [0, {args.n_members}).")

    if args.aggregate:
        run_aggregate(args)
    elif args.toy is not None:
        run_toy(args)
    elif args.loop is not None:
        for n in range(args.loop[0], args.loop[1]):
            args.toy = n
            try:
                run_toy(args)
            except Exception as e:
                print(f"[toy {n}] FAILED: {e}")
        print("Loop done. Run with --aggregate to summarise.")
    else:
        p.error("give one of --toy N, --loop START STOP, or --aggregate")


if __name__ == "__main__":
    main()
