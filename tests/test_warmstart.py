import numpy as np
import torch

from exo_rl.agents.td3 import Actor, ReplayBuffer, TD3Agent
from exo_rl.envs.exo_env import EXO_IN_HUMAN, ExoAssistEnv
from exo_rl.evaluation import run_episode
from exo_rl.warmstart import causal_teacher, label_episode, mse, train_bc

# Sum over 50 steps of info["tau_human"], recorded before tau_needed was exposed (see below).
TAU_HUMAN_SUM_REF = np.array([-2666.201628372537, -3949.799064933687, -1898.8467105572,
                              -2679.67258357634, -3932.748805686538, -1885.024523532451])


def test_tau_needed_exposed_and_human_torques_unchanged(cfg):
    env = ExoAssistEnv(cfg, seed=0)
    env.reset(seed=0, options={"task": "squat", "nominal": True})
    tau_sum = np.zeros(6)
    for _ in range(50):
        _, _, _, _, info = env.step(np.array([0.3, -0.2, 0.3, -0.2]))
        tau_sum += info["tau_human"]
        assert info["tau_needed"].shape == (4,) and np.all(np.isfinite(info["tau_needed"]))
    np.testing.assert_allclose(tau_sum, TAU_HUMAN_SUM_REF, rtol=1e-6)
    assert np.allclose(info["tau_needed"], env.tau_needed)

    lean = ExoAssistEnv(cfg, seed=0, lean_info=True)
    lean.reset(seed=0, options={"task": "squat", "nominal": True})
    assert "tau_needed" in lean.step(np.zeros(4))[4]


def test_tau_needed_matches_human_torque_without_exo(cfg):
    # Exo off: nothing to offload, so the human's torque equals what it needs (unless saturated).
    env = ExoAssistEnv(cfg, seed=0, exo_mode="off")
    env.reset(seed=0, options={"task": "squat", "nominal": True})
    for _ in range(20):
        _, _, _, _, info = env.step(np.zeros(4))
    tau_h = info["tau_human"][EXO_IN_HUMAN]
    np.testing.assert_allclose(info["tau_needed"], tau_h, rtol=1e-9)


def test_teacher_lowers_effort_on_squat(cfg):
    env = ExoAssistEnv(cfg, seed=0)
    off = run_episode(env, None, "squat", "off", seed=2000)
    teach = run_episode(env, causal_teacher(env, 0.3), "squat", "policy", seed=2000)
    assert not teach["failed"]
    assert teach["effort"] < 0.9 * off["effort"]


def test_bc_training_reduces_mse(cfg):
    torch.manual_seed(0)
    env = ExoAssistEnv(cfg, seed=0)
    obs, _ = env.reset(seed=0, options={"task": "squat"})
    teacher = causal_teacher(env, 0.3)
    os_, taus = [], []
    for _ in range(200):
        os_.append(obs)
        obs, _, _, _, info = env.step(teacher(obs))
        taus.append(info["tau_needed"])
    x, y = label_episode(os_, taus, 0.3, cfg["exo"]["action_scale"], offset=1)
    assert len(x) == 199 and np.abs(y).max() <= 1.0
    actor = Actor(x.shape[1], 4, [64, 64])
    before = mse(actor, x, y)
    losses = train_bc(actor, x, y, epochs=30, batch_size=32)
    assert losses[-1] < 0.3 * before and mse(actor, x, y) < 0.3 * before


def test_actor_freeze_and_bc_reference(tmp_path, cfg):
    agent = TD3Agent(10, 4, cfg["agent"])
    agent.set_bc_reference()
    buf = ReplayBuffer(10, 4, 500)
    for _ in range(300):
        buf.add(np.random.randn(10), np.random.uniform(-1, 1, 4), 1.0, np.random.randn(10), False)
    w0 = [p.clone() for p in agent.actor.parameters()]
    for _ in range(4):
        agent.update(buf, 32, actor_frozen=True, bc_weight=2.5)
    assert all(torch.equal(a, b) for a, b in zip(w0, agent.actor.parameters()))
    for _ in range(4):
        info = agent.update(buf, 32, actor_frozen=False, bc_weight=2.5)
    assert np.isfinite(info["actor_loss"])
    assert not all(torch.equal(a, b) for a, b in zip(w0, agent.actor.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(w0, agent.actor_bc.parameters()))   # reference frozen
    agent.save(tmp_path / "a.pt")
    loaded = TD3Agent.load(tmp_path / "a.pt")
    assert all(torch.equal(a, b) for a, b in zip(w0, loaded.actor_bc.parameters()))
