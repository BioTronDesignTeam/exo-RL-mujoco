import os

import numpy as np

from exo_rl.agents.td3 import TD3Agent
from exo_rl.envs.exo_env import ExoAssistEnv
from exo_rl.evaluation import PeriodicEval, eval_score, run_episode


def test_eval_score_worst_task_counts():
    res = {"a": {"effort_reduction_pct": 30.0, "fail_rate": 0.0},
           "b": {"effort_reduction_pct": -10.0, "fail_rate": 0.0}}
    assert eval_score(res, "mean") == 10.0
    assert eval_score(res, "worst") == -10.0


def test_async_eval_matches_inline_and_queues_newest(tmp_path, cfg):
    cfg["sim"]["episode_seconds"] = 0.3
    cfg["tasks"]["names"] = ["squat"]
    env = ExoAssistEnv(cfg, seed=0)
    agents = [TD3Agent(env.observation_space.shape[0], 4, cfg["agent"]) for _ in range(3)]
    inline = PeriodicEval(cfg, env, ["squat"], 1, str(tmp_path))
    ref = [inline.run(a, s)[0][1] for a, s in zip(agents, (100, 200, 300))]

    ev = PeriodicEval(cfg, env, ["squat"], 1, str(tmp_path), async_=True)
    try:
        for a, s in zip(agents, (100, 200, 300)):
            assert ev.run(a, s) == []                     # returns immediately
        done = ev.poll(wait=True)
    finally:
        ev.close()
    assert [d[0] for d in done] == [100, 300]             # 200 was replaced in the queue
    for (step, res, save_best), r in zip(done, (ref[0], ref[2])):
        assert np.isclose(res["squat"]["effort"], r["squat"]["effort"])
    done[1][2](str(tmp_path / "best.pt"))                 # winner's file becomes best.pt
    ev.discard(done)
    assert sorted(os.listdir(tmp_path)) == ["best.pt"]


def test_randomized_pilot_is_same_with_and_without_exo(cfg):
    cfg0 = {**cfg, "exo_hardware": {**cfg["exo_hardware"], "enabled": False}}
    for c in (cfg, cfg0):
        env = ExoAssistEnv(c, seed=0)
        run_episode(env, None, "squat", "off", seed=3001, nominal=False)
        c["_mass"] = env.ep.mass
    assert cfg["_mass"] == cfg0["_mass"] and cfg["_mass"] != cfg["subject"]["mass"]
