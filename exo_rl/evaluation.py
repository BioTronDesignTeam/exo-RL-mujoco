"""Evaluation: human effort with vs. without the exo, per task.

The headline metric is effort reduction (%) relative to exo-off, mirroring how ACE scores
the Athletic Test. Tracking error, peak torque, and smoothness are reported alongside.
Evaluation uses fixed seeds so runs are comparable, and by default the nominal pilot.
With nominal=False each seed draws a randomized pilot (body, strength, latency, noise...);
the draws don't depend on whether the exo is mounted, so the same seed gives the same pilot
for exo on, exo off, and no exo.
"""
from __future__ import annotations

import copy
import csv
import os
from typing import TYPE_CHECKING, Callable, NamedTuple

import numpy as np

from .envs.exo_env import ExoAssistEnv

if TYPE_CHECKING:
    from .agents.td3 import TD3Agent


def run_episode(env: ExoAssistEnv, policy, task: str, mode: str, seed: int, record: bool = False,
                nominal: bool = True):
    obs, _ = env.reset(seed=seed, options={"exo_mode": mode, "task": task, "nominal": nominal})
    effort, err, dtau, peak, ret = [], [], [], 0.0, 0.0
    trace = []
    diagnostics: dict[str, float] = {}
    while True:
        a = policy(obs) if (policy is not None and mode == "policy") else np.zeros(4)
        obs, r, term, trunc, info = env.step(a)
        ret += r
        t = info["reward_terms"]
        effort.append(t["effort"])
        err.append(np.degrees(np.sqrt(np.mean((info["q"] - info["q_ref"]) ** 2))))
        dtau.append(np.sqrt(t["smooth"]))
        peak = max(peak, float(np.max(np.abs(info["tau_exo_cmd"]))))
        for name, value in info.get("diagnostics", {}).items():
            diagnostics[name] = diagnostics.get(name, 0.0) + value
        if record:
            trace.append({k: info[k] for k in ("q", "q_ref", "tau_human", "tau_exo_cmd", "phase", "task")})
            trace[-1]["effort"] = t["effort"]
        if term or trunc:
            break
    out = {
        "return": ret, "steps": env.steps, "failed": bool(term),
        "effort": float(np.mean(effort)), "tracking_rms_deg": float(np.mean(err)),
        "peak_exo_torque": peak, "smoothness": float(np.mean(dtau)),
    }
    out.update({f"diagnostic_{name}": value if name.endswith("_work_J") else value / env.steps
                for name, value in diagnostics.items()})
    return (out, trace) if record else out


def no_exo_effort(cfg: dict, tasks, episodes: int = 3, seed0: int = 1000, nominal: bool = True) -> dict:
    """Effort of the pilot NOT wearing the exo (the ACE comparison); {} if no CAD exo is mounted."""
    if not (cfg.get("exo_hardware") or {}).get("enabled"):
        return {}
    cfg0 = copy.deepcopy(cfg)
    cfg0["exo_hardware"]["enabled"] = False
    env0 = ExoAssistEnv(cfg0, seed=0)
    return {t: float(np.mean([run_episode(env0, None, t, "off", seed0 + i, nominal=nominal)["effort"]
                              for i in range(episodes)]))
            for t in tasks}


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


def evaluate_paired(
    env: ExoAssistEnv,
    policy: Callable[[np.ndarray], np.ndarray] | None,
    task: str,
    seeds: list[int],
    nominal: bool = True,
    mode: str = "policy",
) -> tuple[list[dict], dict]:
    """Audit equal-duration powered/unpowered episodes for explicit seeds.

    Failure rates include every seed. Effort comparisons include only pairs
    where both modes complete the configured horizon; failed episodes remain
    visible in the returned rows instead of inflating the effort baseline.
    This diagnostic does not change legacy training checkpoint selection.
    """
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a nonempty list of unique integers")
    rows = []
    for seed in seeds:
        off = run_episode(env, None, task, "off", seed, nominal=nominal)
        on = run_episode(env, policy, task, mode, seed, nominal=nominal)
        complete = (
            not off["failed"]
            and not on["failed"]
            and off["steps"] == on["steps"] == env.max_steps
        )
        reduction = (
            100.0 * (1.0 - on["effort"] / max(off["effort"], 1e-9))
            if complete
            else None
        )
        rows.append(
            {
                "seed": seed,
                "task": task,
                "complete_pair": complete,
                "off_failed": off["failed"],
                "policy_failed": on["failed"],
                "off_steps": off["steps"],
                "policy_steps": on["steps"],
                "off_effort": off["effort"],
                "policy_effort": on["effort"],
                "off_tracking_rms_deg": off["tracking_rms_deg"],
                "policy_tracking_rms_deg": on["tracking_rms_deg"],
                "policy_peak_torque": on["peak_exo_torque"],
                "policy_return": on.get("return", 0.0),
                "policy_smoothness": on.get("smoothness", 0.0),
                "effort_reduction_pct": reduction,
            }
        )
        for key in off:
            if key.startswith("diagnostic_"):
                rows[-1][f"off_{key}"] = off[key]
                rows[-1][f"policy_{key}"] = on[key]
    valid = [r for r in rows if r["complete_pair"]]
    off_mean = float(np.mean([r["off_effort"] for r in valid])) if valid else None
    on_mean = float(np.mean([r["policy_effort"] for r in valid])) if valid else None
    summary = {
        "task": task,
        "pairs": len(rows),
        "complete_pairs": len(valid),
        "off_fail_rate": float(np.mean([r["off_failed"] for r in rows])),
        "policy_fail_rate": float(np.mean([r["policy_failed"] for r in rows])),
        "effort_off_complete_pairs": off_mean,
        "effort_policy_complete_pairs": on_mean,
        "effort_reduction_pct": (
            100.0 * (1 - on_mean / max(off_mean, 1e-9))
            if off_mean is not None and on_mean is not None
            else None
        ),
        "median_pair_reduction_pct": float(
            np.median([r["effort_reduction_pct"] for r in valid])
        )
        if valid
        else None,
        "improved_pairs": sum(r["effort_reduction_pct"] > 0 for r in valid),
    }
    return rows, summary


def eval_score(results: dict, metric: str = "mean") -> float:
    """Checkpoint-selection score from evaluate() results.
    mean:  mean effort reduction - 100 * mean fail rate (the original rule)
    worst: the same per task, worst task counts (a regression on one task can't hide)"""
    for value in results.values():
        if value.get("paired_evaluation", False) and (
            value["complete_pairs"] != value["evaluation_pairs"]
            or not np.isfinite(value["effort_reduction_pct"])
        ):
            return -np.inf
    per_task = [v["effort_reduction_pct"] - 100.0 * v["fail_rate"] for v in results.values()]
    if metric == "worst":
        return float(min(per_task))
    mean_red = float(np.mean([v["effort_reduction_pct"] for v in results.values()]))
    return mean_red - 100.0 * float(np.mean([v["fail_rate"] for v in results.values()]))


def _evaluation_seeds(cfg: dict, episodes: int) -> list[int]:
    configured = cfg["train"].get("eval_seeds")
    seeds = (
        list(range(1000, 1000 + episodes)) if configured is None else list(configured)
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("train.eval_seeds must be nonempty and unique")
    return seeds


def paired_training_results(
    env: ExoAssistEnv,
    policy: Callable[[np.ndarray], np.ndarray],
    tasks: list[str],
    seeds: list[int],
) -> tuple[dict, list[dict]]:
    """Build training-compatible metrics while retaining every paired episode.

    Results carry a paired-evaluation marker so eval_score rejects checkpoints
    with incomplete comparisons. Legacy training evaluation remains available.
    """
    results, records = {}, []
    for task in tasks:
        rows, summary = evaluate_paired(env, policy, task, seeds)
        records.extend(rows)
        results[task] = {
            "return": float(np.mean([r["policy_return"] for r in rows])),
            "steps": float(np.mean([r["policy_steps"] for r in rows])),
            "effort": summary["effort_policy_complete_pairs"]
            if summary["complete_pairs"]
            else float("nan"),
            "effort_off": summary["effort_off_complete_pairs"]
            if summary["complete_pairs"]
            else float("nan"),
            "effort_reduction_pct": summary["effort_reduction_pct"]
            if summary["complete_pairs"]
            else float("nan"),
            "tracking_rms_deg": float(
                np.mean([r["policy_tracking_rms_deg"] for r in rows])
            ),
            "peak_exo_torque": float(np.mean([r["policy_peak_torque"] for r in rows])),
            "smoothness": float(np.mean([r["policy_smoothness"] for r in rows])),
            "fail_rate": summary["policy_fail_rate"],
            "baseline_fail_rate": summary["off_fail_rate"],
            "complete_pairs": float(summary["complete_pairs"]),
            "evaluation_pairs": float(summary["pairs"]),
            "paired_evaluation": 1.0,
        }
    return results, records


def _write_paired_records(ckpt_dir: str, step: int, rows: list[dict]) -> None:
    path = os.path.join(os.path.dirname(ckpt_dir), "paired_eval.csv")
    exists = os.path.exists(path)
    with open(path, "a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", *rows[0]])
        if not exists:
            writer.writeheader()
        writer.writerows({"step": step, **row} for row in rows)


def frozen_policy(agent: TD3Agent, device: str) -> Callable[[np.ndarray], np.ndarray]:
    """Copy the actor for inference without drawing RNG or mutating the learner.

    Paired evaluation can use CPU inference even when learning runs on CUDA,
    matching the default viewer and export paths.
    """
    import torch

    actor = copy.deepcopy(agent.actor).to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def policy(obs: np.ndarray) -> np.ndarray:
        return (
            actor(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            .cpu()
            .numpy()[0]
        )

    return policy


def evaluate_checkpoint(cfg: dict, path: str, tasks, episodes: int, baseline=None):
    """evaluate() for a saved checkpoint; runs in a background process (train.eval_async)."""
    import torch

    from .agents.td3 import TD3Agent

    torch.set_num_threads(1)
    if cfg["train"].get("eval_paired", False):
        agent = TD3Agent.load(path, device=cfg["train"].get("eval_device", "cpu"))
        results, rows = paired_training_results(
            ExoAssistEnv(cfg, seed=0),
            lambda o: agent.act(o, 0.0),
            tasks,
            _evaluation_seeds(cfg, episodes),
        )
        _write_paired_records(os.path.dirname(path), agent.step, rows)
        return results, None
    agent = TD3Agent.load(path)
    return evaluate(
        ExoAssistEnv(cfg, seed=0),
        lambda o: agent.act(o, 0.0),
        tasks,
        episodes,
        baseline=baseline,
    )


class FinishedEval(NamedTuple):
    step: int  # env step whose weights were evaluated
    results: dict  # evaluate() output, per task
    save_best: Callable[
        [str], None
    ]  # save_best(path): write exactly those weights to path


class PeriodicEval:
    """Training-time evaluation, inline (default) or in one background process (async_=True).
    The exo-off baseline is measured once and reused (self.baseline).
    Async: the evaluated weights are saved to a pending file that becomes best.pt if they win.
    If an evaluation is still running when the next is due, the newest one waits in a queue of
    one (an older queued one is dropped)."""

    def __init__(self, cfg, eval_env, tasks, episodes, ckpt_dir, async_=False):
        self.cfg, self.env, self.tasks, self.episodes, self.dir = (
            cfg,
            eval_env,
            tasks,
            episodes,
            ckpt_dir,
        )
        self.baseline, self.pending, self.queued, self.pool = None, None, None, None
        if cfg["train"].get("eval_paired", False):
            _evaluation_seeds(cfg, episodes)
        if async_:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            self.pool = ProcessPoolExecutor(
                1, mp_context=multiprocessing.get_context("spawn")
            )

    def run(self, agent, step) -> list[FinishedEval]:
        """Evaluate the agent's current weights. Inline: returns the result now. Async: starts
        (or queues) the evaluation and returns []; collect results with poll()."""
        if self.pool is None:
            if self.cfg["train"].get("eval_paired", False):
                policy = frozen_policy(agent, self.cfg["train"].get("eval_device", "cpu"))
                res, rows = paired_training_results(self.env, policy, self.tasks,
                                                     _evaluation_seeds(self.cfg, self.episodes))
                _write_paired_records(self.dir, step, rows)
                return [FinishedEval(step, res, lambda path: agent.save(path, step))]
            res, self.baseline = evaluate(self.env, lambda o: agent.act(o, noise=0.0), self.tasks,
                                          self.episodes, baseline=self.baseline)
            return [FinishedEval(step, res, lambda path: agent.save(path, step))]
        path = os.path.join(self.dir, f"eval_pending_{step}.pt")
        agent.save(path, step)
        if self.pending is None:
            self._submit(step, path)
        else:
            if self.queued is not None:
                print(f"[eval ] step {self.queued[0]}: dropped (evaluation falling behind; raise eval_every)")
                os.remove(self.queued[1])
            self.queued = (step, path)
        return []

    def _submit(self, step, path):
        fut = self.pool.submit(evaluate_checkpoint, self.cfg, path, self.tasks, self.episodes, self.baseline)
        self.pending = (step, path, fut)

    def poll(self, wait=False) -> list[FinishedEval]:
        """Finished async evaluations (none in inline mode); wait=True blocks until all are done."""
        out = []
        while self.pending is not None and (wait or self.pending[2].done()):
            step, path, fut = self.pending
            self.pending = None
            res, self.baseline = fut.result()
            def save_best(best_path: str, src: str = path) -> None:
                os.replace(src, best_path)
            out.append(FinishedEval(step, res, save_best))
            if self.queued is not None:
                self._submit(*self.queued)
                self.queued = None
        return out

    def discard(self, finished: list[FinishedEval]):
        """Delete pending files of async results that did not become best.pt."""
        for ev in finished:
            p = os.path.join(self.dir, f"eval_pending_{ev.step}.pt")
            if os.path.exists(p):
                os.remove(p)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
        for item in (self.pending, self.queued):
            if item is not None:
                try:
                    os.remove(item[1])
                except OSError:                    # still being read by the worker
                    pass
        self.pending = self.queued = None
