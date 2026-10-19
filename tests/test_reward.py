"""The reward must reflect assistance: helpful torque lowers human effort, opposing raises it."""
import numpy as np
import pytest

from exo_rl.envs.exo_env import ExoAssistEnv
from exo_rl.envs.rewards import EffortReward


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


# Unit tests of the reward formula in exo_rl/envs/rewards.py (fast, no simulation).

W = {"alive": 2.0, "w_effort": 2.0, "w_tracking": 0.3, "tracking_sigma_deg": 10, "w_smooth": 0.5,
     "w_magnitude": 0.05}


def make_reward(**overrides):
    return EffortReward({**W, **overrides}, tau_human_max=np.full(6, 100.0), tau_exo_max=np.full(4, 40.0))


def test_reward_is_alive_bonus_when_everything_is_zero():
    r, terms = make_reward()(np.zeros(6), np.zeros(6), np.zeros(4), np.zeros(4))
    assert r == 2.0 and all(v == 0.0 for v in terms.values())


def test_reward_terms_match_formula():
    effort_sq = np.full(6, 0.25)                       # (tau_h / tau_max)^2 = 0.25
    q_err = np.full(6, np.radians(5.0))                # half of sigma -> tracking term 0.25
    tau, tau_prev = np.full(4, 20.0), np.full(4, 10.0)  # magnitude (20/40)^2, smooth (10/40)^2
    r, t = make_reward()(effort_sq, q_err, tau, tau_prev)
    assert np.isclose(t["effort"], 0.25) and np.isclose(t["tracking"], 0.25)
    assert np.isclose(t["magnitude"], 0.25) and np.isclose(t["smooth"], 0.0625)
    expected = 2.0 - (2.0 * 0.25 + 0.3 * 0.25 + 0.5 * 0.0625 + 0.05 * 0.25)
    assert np.isclose(r, expected)


def test_each_penalty_lowers_reward():
    base, _ = make_reward()(np.zeros(6), np.zeros(6), np.zeros(4), np.zeros(4))
    worse = [
        make_reward()(np.full(6, 0.1), np.zeros(6), np.zeros(4), np.zeros(4))[0],          # effort
        make_reward()(np.zeros(6), np.full(6, 0.1), np.zeros(4), np.zeros(4))[0],          # tracking
        make_reward()(np.zeros(6), np.zeros(6), np.full(4, 10.0), np.full(4, 10.0))[0],    # magnitude
        make_reward()(np.zeros(6), np.zeros(6), np.zeros(4), np.full(4, 10.0))[0],         # smoothness
    ]
    assert all(r < base for r in worse)


def test_reward_weights_come_from_config():
    args = (np.full(6, 0.5), np.zeros(6), np.zeros(4), np.zeros(4))
    r1, _ = make_reward(w_effort=1.0)(*args)
    r2, _ = make_reward(w_effort=3.0)(*args)
    assert np.isclose(r1 - r2, 2.0 * 0.5)


def test_default_config_keeps_typical_step_reward_positive(cfg):
    # 'alive' must outweigh a typical step's penalties, or ending the episode early looks attractive.
    rw = EffortReward(cfg["reward"], np.full(6, 100.0), np.full(4, 40.0))
    r, _ = rw(np.full(6, 0.5), np.full(6, np.radians(5.0)), np.full(4, 20.0), np.full(4, 18.0))
    assert r > 0
