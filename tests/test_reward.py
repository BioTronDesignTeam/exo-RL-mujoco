"""The reward must reflect assistance: helpful torque lowers human effort, opposing raises it."""
import numpy as np
import pytest

from exo_rl.envs.exo_env import ExoAssistEnv


def mean_effort(env, task, gain, beta):
    env.reset(seed=3, options={"task": task, "exo_mode": "policy"})
    env.human.beta = beta
    last = np.zeros(6)
    eff = []
    for k in range(300):
        a = gain * last[[0, 1, 3, 4]] / env.pipeline.action_scale    # oracle: share of human torque
        *_, info = env.step(a)
        last = info["tau_human"].copy()
        last[[0, 1, 3, 4]] += info["tau_exo_cmd"]
        eff.append(info["reward_terms"]["effort"])
    return np.mean(eff[50:])


@pytest.mark.parametrize("task", ["squat", "high_knees"])
@pytest.mark.parametrize("beta", [1.0, 0.6])
def test_helpful_torque_reduces_effort(cfg, task, beta):
    cfg["randomization"]["enabled"] = False
    env = ExoAssistEnv(cfg, seed=0)
    off = mean_effort(env, task, 0.0, beta)
    helpful = mean_effort(env, task, 0.5, beta)
    opposing = mean_effort(env, task, -0.5, beta)
    assert helpful < 0.95 * off
    assert opposing > 1.05 * off
