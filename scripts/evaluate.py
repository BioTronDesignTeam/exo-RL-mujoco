"""Evaluate a trained policy against the exo-off and not-wearing-the-exo baselines.

    python scripts/evaluate.py --run runs/demo                 # uses best.pt
    python scripts/evaluate.py --run runs/demo --ckpt latest.pt --episodes 5
    python scripts/evaluate.py --run runs/demo --randomized 10     # + 10 random pilots per task
    python scripts/evaluate.py --run runs/demo --set human.model=myo --tag myo   # on the muscle model

Writes to runs/<name>/eval_<ckpt>[_<tag>]/:
    summary.csv                per task and mode: effort, reduction vs off, tracking, torques
                               (+ "net_vs_no_exo_pct" when a CAD exo is mounted: the ACE-style
                               comparison against the pilot NOT wearing the exo at all)
    effort_by_task.png         human effort with/without exo (the ACE-relevant plot)
    gait_<task>.png            hip/knee angle vs % cycle (sim vs reference) and human torque
                               with/without exo, for each cyclic task
    muscle_groups.csv          muscle model (human.model: myo): mean activation per muscle group
                               (quads, glutes, ..., shin) for no exo / exo off / policy
    summary_randomized.csv     with --randomized N: the same comparison over N fixed random
                               pilots (body size, strength, latency, noise...), one episode each
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
from exo_rl.evaluation import no_exo_effort, run_episode  # noqa: E402
from exo_rl.utils import load_config  # noqa: E402

MODES = ["off", "policy"]     # feedforward-only stays available in the env / view.py --mode


def phase_average(trace, key, idx, bins=50, skip=100):
    """Average a signal over the gait cycle using the reference phase (plots only)."""
    ph = np.array([s["phase"] for s in trace[skip:]])
    val = np.array([np.asarray(s[key])[idx] for s in trace[skip:]])
    b = np.minimum((ph * bins).astype(int), bins - 1)
    return np.array([val[b == i].mean() if np.any(b == i) else np.nan for i in range(bins)])


def randomized_table(env, cfg, policy, tasks, n, out_dir):
    """Mean over n random pilots (seeds 3000+i); each pilot is compared with ITSELF exo off / no exo."""
    no_exo = no_exo_effort(cfg, tasks, n, seed0=3000, nominal=False)
    rows = []
    for task in tasks:
        eff = {m: [run_episode(env, policy, task, m, seed=3000 + i, nominal=False) for i in range(n)]
               for m in MODES}
        off = np.array([e["effort"] for e in eff["off"]])
        for mode in MODES:
            e = np.array([x["effort"] for x in eff[mode]])
            row = {"task": task, "mode": mode, "pilots": n, "effort": round(float(e.mean()), 4),
                   "effort_reduction_pct": round(float(np.mean(100 * (1 - e / off))), 2),
                   "worst_pilot_reduction_pct": round(float(np.min(100 * (1 - e / off))), 2),
                   "failed": round(float(np.mean([x["failed"] for x in eff[mode]])), 2)}
            if no_exo:
                row["net_vs_no_exo_pct"] = round(float(100 * (1 - e.mean() / no_exo[task])), 2)
            rows.append(row)
    write_csv(rows, os.path.join(out_dir, "summary_randomized.csv"))
    net_hdr = f"{'net vs no exo':>15s}" if no_exo else ""
    print(f"\nRandomized pilots ({n} per task, each compared with itself):")
    print(f"{'task':15s}{'mode':9s}{'vs exo off':>12s}{'worst pilot':>13s}{net_hdr}{'fail':>6s}")
    for r in rows:
        if r["mode"] == "off":
            continue
        net = f"{r['net_vs_no_exo_pct']:+14.1f}%" if no_exo else ""
        print(f"{r['task']:15s}{r['mode']:9s}{r['effort_reduction_pct']:+11.1f}%{r['worst_pilot_reduction_pct']:+12.1f}%"
              f"{net}{r['failed']:6.2f}")


def nominal_table(env, policy, tasks, episodes, no_exo):
    """Nominal pilot: mean over `episodes` per task and mode. Returns (rows, first-episode traces)."""
    rows, traces = [], {}
    for task in tasks:
        base = None
        for mode in MODES:                      # "off" first: it is the baseline for "policy"
            eps = []
            for i in range(episodes):
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
    return rows, traces


def muscle_group_table(rows, no_exo_full):
    """Per task and muscle group: mean activation with no exo / exo off / policy."""
    out = []
    by = {(r["task"], r["mode"]): r for r in rows}
    for task in dict.fromkeys(r["task"] for r in rows):
        for key in [k for k in rows[0] if k.startswith("act_")]:
            row = {"task": task, "group": key[4:]}
            if no_exo_full:
                row["no_exo"] = round(no_exo_full[task][key], 4)
            for mode in MODES:
                row[mode] = by[(task, mode)][key]
            if no_exo_full:
                row["policy_vs_no_exo_pct"] = round(100.0 * (row["policy"] / max(row["no_exo"], 1e-9) - 1), 1)
            out.append(row)
    return out


def print_muscle_groups(group_rows):
    note = " (policy vs no exo: + = works harder)" if "no_exo" in group_rows[0] else ""
    print("\nmean muscle activation per group" + note)
    for r in group_rows:
        base = f"no exo {r['no_exo']:.3f}  " if "no_exo" in r else ""
        diff = f"  {r['policy_vs_no_exo_pct']:+6.1f}%" if "policy_vs_no_exo_pct" in r else ""
        print(f"  {r['task']:14s} {r['group']:12s} {base}" + "  ".join(f"{m} {r[m]:.3f}" for m in MODES) + diff)


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def print_nominal(rows, no_exo):
    net_hdr = f"{'net vs no exo':>15s}" if no_exo else ""
    print(f"\n{'task':15s}{'mode':13s}{'effort':>8s}{'vs exo off':>12s}{net_hdr}{'track_rms':>11s}{'peak_tau':>10s}{'fail':>6s}")
    for r in rows:
        net = f"{r['net_vs_no_exo_pct']:+14.1f}%" if no_exo else ""
        print(f"{r['task']:15s}{r['mode']:13s}{r['effort']:8.3f}{r['effort_reduction_pct']:+11.1f}%{net}"
              f"{r['tracking_rms_deg']:10.1f}°{r['peak_exo_torque']:9.1f}{r['failed']:6.2f}")
    if no_exo:
        print("\n'vs exo off' = benefit of powering the exo you're wearing; "
              "'net vs no exo' = compared with the pilot not wearing it (what ACE measures).")


def plot_effort(rows, no_exo, tasks, path):
    """Bar chart: not wearing exo / exo off / exo policy, per task."""
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
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_gait(traces, task, path):
    """Left hip/knee over the gait cycle: angle vs reference, human torque exo off vs policy, exo torque."""
    pct = np.linspace(0, 100, 50)
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
        axes[1, c].plot(pct, phase_average(tr, "tau_exo_cmd", c), "g-", label="exo torque")
        axes[1, c].set_ylabel("Nm")
        axes[1, c].set_xlabel("% cycle")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(f"{task}: tracking and torque over the cycle")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--randomized", type=int, default=0, metavar="N",
                    help="also evaluate on N random pilots (tests adaptation across people)")
    ap.add_argument("--set", action="append", default=[],
                    help="config override on top of the run's config, e.g. human.model=myo")
    ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. myo -> eval_best_myo")
    args = ap.parse_args()

    cfg = load_config(os.path.join(args.run, "config.yaml"), args.set)
    agent = TD3Agent.load(os.path.join(args.run, "checkpoints", args.ckpt))
    policy = lambda o: agent.act(o, 0.0)  # noqa: E731
    env = ExoAssistEnv(cfg, seed=0)
    tasks = cfg["tasks"]["names"]
    out_dir = os.path.join(args.run, f"eval_{os.path.splitext(args.ckpt)[0]}" + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(out_dir, exist_ok=True)

    no_exo_full = no_exo_effort(cfg, tasks, args.episodes, seed0=2000, details=True)
    no_exo = {t: v["effort"] for t, v in no_exo_full.items()}
    rows, traces = nominal_table(env, policy, tasks, args.episodes, no_exo)
    write_csv(rows, os.path.join(out_dir, "summary.csv"))
    print_nominal(rows, no_exo)
    if any(k.startswith("act_") for k in rows[0]):
        group_rows = muscle_group_table(rows, no_exo_full)
        write_csv(group_rows, os.path.join(out_dir, "muscle_groups.csv"))
        print_muscle_groups(group_rows)
    if args.randomized:
        randomized_table(env, cfg, policy, tasks, args.randomized, out_dir)

    plot_effort(rows, no_exo, tasks, os.path.join(out_dir, "effort_by_task.png"))
    for task in tasks:
        plot_gait(traces, task, os.path.join(out_dir, f"gait_{task}.png"))
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
