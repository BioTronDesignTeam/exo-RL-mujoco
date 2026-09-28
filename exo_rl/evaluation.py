"""Evaluation: human effort with vs. without the exo, per task.

The headline metric is effort reduction (%) relative to exo-off, mirroring how ACE scores
the Athletic Test. Tracking error, peak torque, and smoothness are reported alongside.
Evaluation uses nominal (non-randomized) parameters and fixed seeds so runs are comparable.
"""
from __future__ import annotations

import numpy as np

from .envs.exo_env import ExoAssistEnv


def run_episode(env: ExoAssistEnv, policy, task: str, mode: str, seed: int, record: bool = False):
    obs, _ = env.reset(seed=seed, options={"exo_mode": mode, "task": task, "nominal": True})
    effort, err, dtau, peak, ret = [], [], [], 0.0, 0.0
    trace = []
    while True:
        a = policy(obs) if (policy is not None and mode == "policy") else np.zeros(4)
        obs, r, term, trunc, info = env.step(a)
        ret += r
        t = info["reward_terms"]
        effort.append(t["effort"])
        err.append(np.degrees(np.sqrt(np.mean((info["q"] - info["q_ref"]) ** 2))))
        dtau.append(np.sqrt(t["smooth"]))
        peak = max(peak, float(np.max(np.abs(info["tau_exo_cmd"]))))
        if record:
            trace.append({k: info[k] for k in ("q", "q_ref", "tau_human", "tau_exo_cmd", "phase", "task")})
        if term or trunc:
            break
    out = {
        "return": ret, "steps": env.steps, "failed": bool(term),
        "effort": float(np.mean(effort)), "tracking_rms_deg": float(np.mean(err)),
        "peak_exo_torque": peak, "smoothness": float(np.mean(dtau)),
    }
    return (out, trace) if record else out


def evaluate(env: ExoAssistEnv, policy, tasks, episodes: int = 3, seed0: int = 1000, baseline=None):
    """Returns (per-task results, baseline effort dict). Baseline = exo off, cached if given."""
    if baseline is None:
        baseline = {t: np.mean([run_episode(env, None, t, "off", seed0 + i)["effort"] for i in range(episodes)])
                    for t in tasks}
    results = {}
    for t in tasks:
        eps = [run_episode(env, policy, t, "policy", seed0 + i) for i in range(episodes)]
        agg = {k: float(np.mean([e[k] for e in eps])) for k in eps[0] if k != "failed"}
        agg["fail_rate"] = float(np.mean([e["failed"] for e in eps]))
        agg["effort_off"] = float(baseline[t])
        agg["effort_reduction_pct"] = 100.0 * (1.0 - agg["effort"] / max(baseline[t], 1e-9))
        results[t] = agg
    return results, baseline
