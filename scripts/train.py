"""Train a TD3/DDPG exo assistance policy.

    python scripts/train.py --config configs/default.yaml --name baseline
    python scripts/train.py --set agent.algo=ddpg --set train.total_steps=200000
    python scripts/train.py --name baseline --max-minutes 60      # stop cleanly, save snapshot
    python scripts/train.py --name baseline --resume               # continue from snapshot
    python scripts/train.py --set train.num_envs=8 --set train.updates_per_step=0.25  # parallel
    python scripts/train.py --name ft --init-from runs/baseline/checkpoints/best.pt   # fine-tune
    python scripts/train.py --name ws --init-from runs/bc/checkpoints/bc.pt   # after pretrain_bc.py

With --init-from, the start_steps warm-up uses the initialized policy (+ exploration noise)
instead of random actions, and agent.actor_freeze_steps / bc_weight / bc_decay_steps protect
the initial policy from the still-untrained critic (see exo_rl/agents/td3.py).

Outputs in runs/<name>/:
    config.yaml        exact config used (needed for export/deploy)
    episodes.csv       per-episode return, length, task, env_id, mean of every reward term
    eval.csv           per-task effort reduction vs exo-off, tracking, torque stats
    checkpoints/       latest.pt (every train.latest_every steps), best.pt (by mean eval effort
                       reduction), step_N.pt; all written atomically (temp file + rename)
    resume/            snapshot (agent, replay buffer, counters) for --resume
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exo_rl.agents.td3 import ReplayBuffer, TD3Agent  # noqa: E402
from exo_rl.envs.exo_env import ExoAssistEnv  # noqa: E402
from exo_rl.envs.vector import make_runner  # noqa: E402
from exo_rl.evaluation import PeriodicEval, eval_score  # noqa: E402
from exo_rl.utils import atomic_write, configure_torch_threads, load_config, pick_device  # noqa: E402


class CsvLog:
    def __init__(self, path, append=False):
        self.path, self.writer, self.f = path, None, None
        self.append = append and os.path.exists(path)

    def write(self, row: dict):
        if self.writer is None:
            fields = list(row)
            if self.append:
                with open(self.path) as f:
                    fields = next(csv.reader(f))
            self.f = open(self.path, "a" if self.append else "w", newline="")
            self.writer = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore", restval="")
            if not self.append:
                self.writer.writeheader()
        self.writer.writerow(row)
        self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. agent.algo=ddpg")
    ap.add_argument("--resume", action="store_true", help="continue runs/<name> from its snapshot")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop and snapshot after this long")
    ap.add_argument("--init-from", default=None,
                    help="start actor/critic (+targets) from this checkpoint; fresh optimizers, empty buffer")
    ap.add_argument("--init-buffer", default=None,
                    help="prefill the replay buffer (e.g. bc_buffer.npz from pretrain_bc.py); learning starts at once")
    args = ap.parse_args()
    if hasattr(signal, "SIGBREAK"):          # Windows: Ctrl+Break stops cleanly, like Ctrl+C
        signal.signal(signal.SIGBREAK, signal.default_int_handler)

    name = args.name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", name)
    if args.resume:
        cfg = load_config(os.path.join(run_dir, "config.yaml"), args.set)
    else:
        cfg = load_config(args.config, args.set)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:     # persist overrides (also on resume)
        yaml.safe_dump(cfg, f, sort_keys=False)

    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    tr = cfg["train"]
    device = pick_device(tr.get("device", "auto"))
    num_envs = int(tr.get("num_envs", 1))
    updates_per_step = float(tr.get("updates_per_step", 1.0))
    threads = configure_torch_threads(int(tr.get("torch_threads", 0)), num_envs)

    eval_env = ExoAssistEnv(cfg, seed=seed + 1)
    obs_dim, act_dim = eval_env.observation_space.shape[0], eval_env.action_space.shape[0]
    agent = TD3Agent(obs_dim, act_dim, cfg["agent"], device)
    buf = ReplayBuffer(obs_dim, act_dim, int(cfg["agent"]["buffer_size"]))
    ep_log = CsvLog(os.path.join(run_dir, "episodes.csv"), append=args.resume)
    ev_log = CsvLog(os.path.join(run_dir, "eval.csv"), append=args.resume)
    tasks = cfg["tasks"]["names"]
    print(f"[train] {agent.algo.upper()} on {device} | obs {obs_dim} act {act_dim} | {num_envs} env(s), "
          f"{updates_per_step:g} updates/step, {threads} torch threads | run dir {run_dir}")

    ep_i, best, baseline, start = 0, -np.inf, None, 1
    snap_dir = os.path.join(run_dir, "resume")
    init_from = args.init_from
    if init_from and not args.resume:
        try:
            agent.init_from(init_from)
        except ValueError as e:
            sys.exit(f"[train] error: {e}")
        if float(cfg["agent"].get("bc_weight", 0.0)) > 0:
            agent.set_bc_reference()
        print(f"[train] initialized actor/critic from {init_from}")
    prefilled = 0
    if args.init_buffer and not args.resume:
        data = np.load(args.init_buffer)
        if data["obs"].shape[1] != obs_dim:
            sys.exit(f"[train] error: --init-buffer {args.init_buffer} has obs size {data['obs'].shape[1]}, "
                     f"but this config gives {obs_dim}")
        for o, a, r, o2, nd in zip(data["obs"], data["act"], data["rew"], data["next_obs"], data["not_done"]):
            buf.add(o, a, float(r[0]), o2, nd[0] == 0.0)
        prefilled = len(buf)
        print(f"[train] prefilled the replay buffer with {prefilled} transitions from {args.init_buffer}")
    if args.resume:
        agent = TD3Agent.load(os.path.join(snap_dir, "agent.pt"), device)
        data = np.load(os.path.join(snap_dir, "buffer.npz"))
        n = int(data["n"])
        for k in ("obs", "act", "rew", "next_obs", "not_done"):
            getattr(buf, k)[:n] = data[k]
        buf.ptr, buf.full = int(data["ptr"]), bool(data["full"])
        with open(os.path.join(snap_dir, "state.yaml")) as f:
            st = yaml.safe_load(f)
        start, ep_i, best = st["step"] + 1, st["episode"], st["best"]
        baseline = st.get("baseline")
        init_from = st.get("init_from")         # keeps the warm-start schedule on resume
        prefilled = st.get("prefilled", 0)
        np.random.seed(seed + start)
        print(f"[train] resumed at step {start} (buffer {len(buf)})")
    runner = make_runner(cfg, num_envs, seed + (start if args.resume else 0))   # fresh seeds on resume

    def snapshot(step):
        os.makedirs(snap_dir, exist_ok=True)
        agent.save(os.path.join(snap_dir, "agent.pt"))
        n = len(buf)

        def write_buffer(tmp):
            with open(tmp, "wb") as f:
                np.savez(f, n=n, ptr=buf.ptr, full=buf.full,
                         **{k: getattr(buf, k)[:n] for k in ("obs", "act", "rew", "next_obs", "not_done")})

        def write_state(tmp):
            with open(tmp, "w") as f:
                yaml.safe_dump({"step": step, "episode": ep_i, "best": float(best), "init_from": init_from,
                                "prefilled": prefilled,
                                "baseline": {k: float(v) for k, v in baseline.items()} if baseline else None}, f)

        atomic_write(os.path.join(snap_dir, "buffer.npz"), write_buffer)
        atomic_write(os.path.join(snap_dir, "state.yaml"), write_state)     # last: marks a complete snapshot
        print(f"[train] snapshot saved at step {step}; continue with --name {name} --resume")

    start_steps, total_steps = int(tr["start_steps"]), int(tr["total_steps"])
    if prefilled >= int(cfg["agent"]["batch_size"]):
        start_steps = 0                          # the prefilled buffer replaces the warm-up
    batch_size = int(cfg["agent"]["batch_size"])
    expl = float(cfg["agent"]["expl_noise"])
    upd_info = {"critic_loss": float("nan"), "q_mean": float("nan"), "actor_loss": float("nan")}
    upd_acc = 0.0
    # Warm-start safeguards, in env steps after learning starts; all off without --init-from.
    ag = cfg["agent"]
    freeze = int(ag.get("actor_freeze_steps", 0)) if init_from else 0
    bc_w0 = float(ag.get("bc_weight", 0.0)) if init_from else 0.0
    bc_decay = int(ag.get("bc_decay_steps", 0)) if init_from else 0
    if init_from:
        upd_info["bc_weight"] = bc_w0
        print(f"[train] warm start: actor frozen for {freeze} steps, then BC weight {bc_w0:g} "
              f"-> 0 over {bc_decay} steps; start_steps collected with the initial policy")

    def learn(n_collected):
        """This iteration's share of gradient updates: updates_per_step per collected transition."""
        nonlocal upd_acc
        if n_collected <= start_steps:
            return
        upd_acc += updates_per_step * num_envs
        n_upd = int(upd_acc)
        upd_acc -= n_upd
        u = n_collected - start_steps - freeze          # steps since the actor was unfrozen
        bc_w = bc_w0 * min(1.0, max(0.0, 1.0 - u / bc_decay)) if bc_decay > 0 else bc_w0
        for _ in range(n_upd):
            upd_info.update(agent.update(buf, batch_size, actor_frozen=u <= 0, bc_weight=bc_w))
        if init_from:
            upd_info["bc_weight"] = bc_w

    best_metric = tr.get("best_metric", "mean")
    evaluator = PeriodicEval(cfg, eval_env, tasks, int(tr["eval_episodes"]), os.path.join(run_dir, "checkpoints"),
                             async_=bool(tr.get("eval_async", False)))
    evaluator.baseline = baseline

    def report(finished):
        """Log finished evaluations; save best.pt (by train.best_metric)."""
        nonlocal best, baseline
        for ev_step, res, save_best in finished:
            baseline = evaluator.baseline
            mean_red = float(np.mean([v["effort_reduction_pct"] for v in res.values()]))
            fail = float(np.mean([v["fail_rate"] for v in res.values()]))
            for t, v in res.items():
                ev_log.write({"step": ev_step, "task": t, **{k: round(x, 4) for k, x in v.items()}})
            print(f"[eval ] step {ev_step}: mean effort reduction {mean_red:+.1f}% | fail rate {fail:.2f} | "
                  + " ".join(f"{t}:{v['effort_reduction_pct']:+.0f}%" for t, v in res.items()))
            score = eval_score(res, best_metric)
            if score > best:
                best = score
                save_best(os.path.join(run_dir, "checkpoints", "best.pt"))
                print(f"[eval ] new best ({score:+.1f}, {best_metric})")
        evaluator.discard(finished)

    step = start - 1                          # transitions collected so far (all envs)
    obs, ep_task = runner.reset()
    ep_ret, ep_len = np.zeros(num_envs), np.zeros(num_envs, int)
    ep_terms = [defaultdict(float) for _ in range(num_envs)]
    t0 = time.time()

    try:
        while step < total_steps:
            if step + 1 <= start_steps and not init_from:
                actions = runner.sample_actions()        # random warm-up, independent per env
            else:
                actions = agent.act(obs, noise=expl)     # exploration noise drawn per env
            runner.step_async(actions)
            if runner.overlap:
                learn(step)                              # envs simulate while we learn
            obs2, trans = runner.step_wait()
            prev, step = step, step + num_envs
            for i, t in enumerate(trans):
                buf.add(obs[i], actions[i], t.reward, t.next_obs, t.terminated)
            if not runner.overlap:
                learn(step)
            obs = obs2

            any_done = False
            for i, t in enumerate(trans):
                ep_ret[i] += t.reward
                ep_len[i] += 1
                for k, v in t.info["reward_terms"].items():
                    ep_terms[i][k] += v
                if not (t.terminated or t.truncated):
                    continue
                any_done = True
                row = {"step": step, "episode": ep_i, "env_id": i, "task": ep_task[i],
                       "return": round(float(ep_ret[i]), 3), "length": int(ep_len[i]),
                       "terminated": int(t.terminated), "reason": t.info["terminated_reason"]}
                row.update({f"mean_{k}": round(v / ep_len[i], 5) for k, v in ep_terms[i].items()})
                row.update({k: round(v, 4) for k, v in upd_info.items()})
                ep_log.write(row)
                if ep_i % 10 == 0:
                    sps = (step - start + 1) / (time.time() - t0)
                    print(f"[train] step {step:>8d} ep {ep_i:>5d} ret {ep_ret[i]:7.1f} len {ep_len[i]:4d} "
                          f"effort {row['mean_effort']:.3f} ({ep_task[i]}) {sps:.0f} steps/s")
                ep_task[i] = t.reset_task
                ep_ret[i], ep_len[i], ep_terms[i] = 0.0, 0, defaultdict(float)
                ep_i += 1

            if crossed(prev, step, int(tr["eval_every"])):
                report(evaluator.run(agent, step))
                agent.save(os.path.join(run_dir, "checkpoints", "latest.pt"), step)
            elif crossed(prev, step, int(tr.get("latest_every", 5000))):
                agent.save(os.path.join(run_dir, "checkpoints", "latest.pt"), step)

            report(evaluator.poll())
            if crossed(prev, step, int(tr["save_every"])):
                agent.save(os.path.join(run_dir, "checkpoints", f"step_{step}.pt"), step)

            # One env stops at an episode boundary (as before); with several, mid-episode is fine.
            if args.max_minutes and (time.time() - t0) > 60 * args.max_minutes and (runner.overlap or any_done):
                print(f"[train] {step - start + 1} steps in {(time.time() - t0) / 60:.1f} min "
                      f"({(step - start + 1) / (time.time() - t0):.0f} steps/s incl. evaluation)")
                report(evaluator.poll(wait=True))
                snapshot(step)
                return
        report(evaluator.poll(wait=True))
    except KeyboardInterrupt:
        runner.close(terminate=True)
        runner = None
        agent.save(os.path.join(run_dir, "checkpoints", "latest.pt"), step)
        print(f"[train] interrupted at step {step}; saved latest.pt (no resume snapshot)")
        return
    finally:
        evaluator.close()
        if runner is not None:
            runner.close()

    agent.save(os.path.join(run_dir, "checkpoints", "latest.pt"), step)
    print(f"[train] done in {(time.time() - t0) / 60:.1f} min "
          f"({(step - start + 1) / (time.time() - t0):.0f} steps/s incl. evaluation)")


def crossed(prev: int, cur: int, every: int) -> bool:
    """True if a multiple of `every` lies in (prev, cur]; step counts advance by num_envs."""
    return cur // every > prev // every


if __name__ == "__main__":
    main()
