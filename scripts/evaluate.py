"""Evaluate a trained policy against the exo-off and feedforward-only baselines.

    python scripts/evaluate.py --run runs/demo                 # uses best.pt
    python scripts/evaluate.py --run runs/demo --ckpt latest.pt --episodes 5

Writes to runs/<name>/eval_<ckpt>/:
    summary.csv                per task and mode: effort, reduction vs off, tracking, torques
                               (+ "net_vs_no_exo_pct" when a CAD exo is mounted: the ACE-style
                               comparison against the pilot NOT wearing the exo at all)
    effort_by_task.png         human effort with/without exo (the ACE-relevant plot)
    gait_<task>.png            hip/knee angle vs % cycle (sim vs reference) and human torque
                               with/without exo, for each cyclic task
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exo_rl.agents.td3 import TD3Agent  # noqa: E402
from exo_rl.envs.exo_env import ExoAssistEnv  # noqa: E402
from exo_rl.evaluation import run_episode  # noqa: E402
from exo_rl.utils import load_config  # noqa: E402

MODES = ["off", "feedforward", "policy"]


def phase_average(trace, key, idx, bins=50, skip=100):
    """Average a signal over the gait cycle using the reference phase (plots only)."""
    ph = np.array([s["phase"] for s in trace[skip:]])
    val = np.array([np.asarray(s[key])[idx] for s in trace[skip:]])
    b = np.minimum((ph * bins).astype(int), bins - 1)
    return np.array([val[b == i].mean() if np.any(b == i) else np.nan for i in range(bins)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(os.path.join(args.run, "config.yaml"))
    agent = TD3Agent.load(os.path.join(args.run, "checkpoints", args.ckpt))
    policy = lambda o: agent.act(o, 0.0)  # noqa: E731
    env = ExoAssistEnv(cfg, seed=0)
    tasks = cfg["tasks"]["names"]
    out_dir = os.path.join(args.run, f"eval_{os.path.splitext(args.ckpt)[0]}")
    os.makedirs(out_dir, exist_ok=True)

    no_exo = {}
    if (cfg.get("exo_hardware") or {}).get("enabled"):
        import copy
        cfg0 = copy.deepcopy(cfg)
        cfg0["exo_hardware"]["enabled"] = False
        env0 = ExoAssistEnv(cfg0, seed=0)
        no_exo = {t: float(np.mean([run_episode(env0, None, t, "off", seed=2000 + i)["effort"]
                                    for i in range(args.episodes)])) for t in tasks}

    rows, traces = [], {}
    for task in tasks:
        base = None
        for mode in MODES:
            eps = []
            for i in range(args.episodes):
                res, tr = run_episode(env, policy, task, mode, seed=2000 + i, record=True)
                eps.append(res)
                if i == 0:
                    traces[(task, mode)] = tr
            agg = {k: float(np.mean([e[k] for e in eps])) for k in eps[0]}
            if mode == "off":
                base = agg["effort"]
            agg["effort_reduction_pct"] = 100.0 * (1 - agg["effort"] / max(base, 1e-9))
            if no_exo:
                agg["net_vs_no_exo_pct"] = 100.0 * (1 - agg["effort"] / max(no_exo[task], 1e-9))
            rows.append({"task": task, "mode": mode, **{k: round(v, 4) for k, v in agg.items()}})

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    net_hdr = f"{'net vs no exo':>15s}" if no_exo else ""
    print(f"\n{'task':15s}{'mode':13s}{'effort':>8s}{'vs exo off':>12s}{net_hdr}{'track_rms':>11s}{'peak_tau':>10s}{'fail':>6s}")
    for r in rows:
        net = f"{r['net_vs_no_exo_pct']:+14.1f}%" if no_exo else ""
        print(f"{r['task']:15s}{r['mode']:13s}{r['effort']:8.3f}{r['effort_reduction_pct']:+11.1f}%{net}"
              f"{r['tracking_rms_deg']:10.1f}°{r['peak_exo_torque']:9.1f}{r['failed']:6.2f}")
    if no_exo:
        print("\n'vs exo off' = benefit of powering the exo you're wearing; "
              "'net vs no exo' = compared with the pilot not wearing it (what ACE measures).")

    # Effort bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(tasks))
    series = ([("not wearing exo", [no_exo[t] for t in tasks])] if no_exo else []) + [
        (f"exo {mode}", [next(r["effort"] for r in rows if r["task"] == t and r["mode"] == mode) for t in tasks])
        for mode in MODES]
    w = 0.8 / len(series)
    for j, (label, vals) in enumerate(series):
        ax.bar(x + (j - (len(series) - 1) / 2) * w, vals, w, label=label)
    ax.set_xticks(x, tasks, rotation=15)
    ax.set_ylabel("human effort  mean((τ_h/τ_max)²)")
    ax.set_title("Simulated human effort by task (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "effort_by_task.png"), dpi=130)
    plt.close(fig)

    # Gait-cycle plots
    pct = np.linspace(0, 100, 50)
    for task in tasks:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        for c, (j, jname) in enumerate([(0, "hip_l"), (1, "knee_l")]):
            tr = traces[(task, "policy")]
            axes[0, c].plot(pct, np.degrees(phase_average(tr, "q_ref", j)), "k-", label="reference")
            axes[0, c].plot(pct, np.degrees(phase_average(tr, "q", j)), "r-", label="sim (with exo)")
            axes[0, c].set_title(f"{jname} angle")
            axes[0, c].set_ylabel("deg")
            for mode, style in (("off", "k--"), ("policy", "b-")):
                axes[1, c].plot(pct, phase_average(traces[(task, mode)], "tau_human", j), style,
                                label=f"human torque, exo {mode}")
            axes[1, c].plot(pct, phase_average(traces[(task, "policy")], "tau_exo_cmd", [0, 1][c]), "g-",
                            label="exo torque")
            axes[1, c].set_ylabel("Nm")
            axes[1, c].set_xlabel("% cycle")
        axes[0, 0].legend(fontsize=8)
        axes[1, 0].legend(fontsize=8)
        fig.suptitle(f"{task}: tracking and torque over the cycle")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"gait_{task}.png"), dpi=120)
        plt.close(fig)
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
