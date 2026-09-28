import csv
import os
import subprocess
import sys

import numpy as np

from exo_rl.envs.exo_env import ExoAssistEnv
from exo_rl.envs.vector import VectorRunner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_parallel_training_smoke(tmp_path):
    run = tmp_path / "par"
    sets = ["train.num_envs=2", "train.total_steps=400", "train.start_steps=100", "train.eval_every=100000",
            "train.updates_per_step=0.5", "sim.episode_seconds=0.5", "tasks.names=[squat]",
            "agent.batch_size=32", "agent.buffer_size=2000"]
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--name", str(run)]
    for s in sets:
        cmd += ["--set", s]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True, timeout=300)
    rows = list(csv.DictReader(open(run / "episodes.csv")))
    assert {r["env_id"] for r in rows} == {"0", "1"}
    assert all(int(r["length"]) == 50 for r in rows)
    assert (run / "checkpoints" / "latest.pt").exists()


def test_autoreset_stores_true_final_observation(cfg):
    cfg["sim"]["episode_seconds"] = 0.05              # 5 steps, then truncation
    ref = ExoAssistEnv(cfg, seed=0, lean_info=True)   # same seed as vector env 0
    ref.reset()
    runner = VectorRunner(cfg, num_envs=2, seed=0)
    try:
        runner.reset()
        for k in range(5):
            ref_obs, _, _, ref_trunc, _ = ref.step(np.zeros(4))
            runner.step_async(np.zeros((2, 4)))
            obs, trans = runner.step_wait()
        assert ref_trunc and trans[0].truncated and not trans[0].terminated
        first_obs_next_ep, _ = ref.reset()
        np.testing.assert_allclose(trans[0].next_obs, ref_obs, rtol=1e-6)          # final obs, stored
        np.testing.assert_allclose(obs[0], first_obs_next_ep, rtol=1e-6)            # policy acts on this
        assert not np.allclose(trans[0].next_obs, obs[0])
        assert trans[0].reset_task is not None and "reward_terms" in trans[0].info
    finally:
        runner.close()
