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
import os
from typing import Callable, NamedTuple

import numpy as np

from .envs.exo_env import ExoAssistEnv


def run_episode(env: ExoAssistEnv, policy, task: str, mode: str, seed: int, record: bool = False,
                nominal: bool = True):
    obs, _ = env.reset(seed=seed, options={"exo_mode": mode, "task": task, "nominal": nominal})
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


def eval_score(results: dict, metric: str = "mean") -> float:
    """Checkpoint-selection score from evaluate() results.
    mean:  mean effort reduction - 100 * mean fail rate (default)
    worst: the same per task, worst task counts (a regression on one task can't hide)"""
    per_task = [v["effort_reduction_pct"] - 100.0 * v["fail_rate"] for v in results.values()]
    if metric == "worst":
        return float(min(per_task))
    mean_red = float(np.mean([v["effort_reduction_pct"] for v in results.values()]))
    return mean_red - 100.0 * float(np.mean([v["fail_rate"] for v in results.values()]))


def evaluate_checkpoint(cfg: dict, path: str, tasks, episodes: int, baseline=None):
    """evaluate() for a saved checkpoint; runs in a background process (train.eval_async)."""
    import torch

    from .agents.td3 import TD3Agent
    torch.set_num_threads(1)
    agent = TD3Agent.load(path)
    return evaluate(ExoAssistEnv(cfg, seed=0), lambda o: agent.act(o, 0.0), tasks, episodes, baseline=baseline)


class FinishedEval(NamedTuple):
    step: int                          # env step whose weights were evaluated
    results: dict                      # evaluate() output, per task
    save_best: Callable[[str], None]   # save_best(path): write exactly those weights to path


class PeriodicEval:
    """Training-time evaluation, inline (default) or in one background process (async_=True).
    The exo-off baseline is measured once and reused (self.baseline).
    Async: the evaluated weights are saved to a pending file that becomes best.pt if they win.
    If an evaluation is still running when the next is due, the newest one waits in a queue of
    one (an older queued one is dropped)."""

    def __init__(self, cfg, eval_env, tasks, episodes, ckpt_dir, async_=False):
        self.cfg, self.env, self.tasks, self.episodes, self.dir = cfg, eval_env, tasks, episodes, ckpt_dir
        self.baseline, self.pending, self.queued, self.pool = None, None, None, None
        if async_:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            self.pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))

    def run(self, agent, step) -> list[FinishedEval]:
        """Evaluate the agent's current weights. Inline: returns the result now. Async: starts
        (or queues) the evaluation and returns []; collect results with poll()."""
        if self.pool is None:
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
            out.append(FinishedEval(step, res, lambda best_path, src=path: os.replace(src, best_path)))
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
