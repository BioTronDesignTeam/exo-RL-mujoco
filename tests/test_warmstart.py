import mujoco
import numpy as np
import torch

from exo_rl.agents.td3 import Actor, ReplayBuffer, TD3Agent
from exo_rl.envs.exo_env import EXO_IN_HUMAN, ExoAssistEnv
from exo_rl.evaluation import run_episode
from exo_rl.sim.human import HumanController
from exo_rl.warmstart import causal_teacher, label_episode, mse, train_bc


def compute_before_tau_needed(self, hm, q_ref, qd_ref, qdd_ref, trunk_ref, trunk_d_ref, v_ref,
                              tau_exo_human_order, dt):
    """Frozen copy of HumanController.compute from before tau_needed was exposed."""
    d = hm.data
    q = d.qpos[hm.human_qpos]
    qd = d.qvel[hm.human_dof]
    e = q_ref - q
    ed = qd_ref - qd
    self.integral = np.clip(self.integral + e * dt, -self.i_limit / np.maximum(self.ki, 1e-9),
                            self.i_limit / np.maximum(self.ki, 1e-9))
    bias = d.qfrc_bias[hm.human_dof]
    qdd_des = np.zeros(hm.model.nv)
    qdd_des[hm.human_dof] = qdd_ref
    m_qdd = np.zeros(hm.model.nv)
    mujoco.mj_mulM(hm.model, d, m_qdd, qdd_des)
    tau_ff = m_qdd[hm.human_dof] + bias
    tau_needed = tau_ff + self.kp * e + self.kd * ed + self.ki * self.integral
    tau_h = np.clip(tau_needed - self.beta * tau_exo_human_order, -self.tau_max, self.tau_max)
    r, s = self.residual, self.mass_scale
    pitch = d.qpos[hm.root_qpos[2]]
    xd, pitchd = d.qvel[hm.root_dof[0]], d.qvel[hm.root_dof[2]]
    f_x = s * r["x_kv"] * (v_ref - xd)
    t_pitch = s * (r["pitch_kp"] * (trunk_ref - pitch) + r["pitch_kd"] * (trunk_d_ref - pitchd))
    return tau_h, np.array([f_x, 0.0, t_pitch])


def test_tau_needed_exposed_and_human_torques_unchanged(cfg, monkeypatch):
    # At every substep, run the current compute() and the frozen copy on the same state and
    # compare. Platform independent, unlike stored numbers (floating point differs per OS).
    current = HumanController.compute
    checked = []

    def compare(self, *args):
        integral = self.integral.copy()
        out = current(self, *args)
        after = self.integral.copy()
        self.integral = integral
        ref = compute_before_tau_needed(self, *args)
        assert np.array_equal(out[0], ref[0]) and np.array_equal(out[1], ref[1])
        assert np.array_equal(self.integral, after)
        checked.append(1)
        return out

    monkeypatch.setattr(HumanController, "compute", compare)
    env = ExoAssistEnv(cfg, seed=0)
    env.reset(seed=0, options={"task": "squat", "nominal": True})
    for _ in range(50):
        _, _, _, _, info = env.step(np.array([0.3, -0.2, 0.3, -0.2]))
        assert info["tau_needed"].shape == (4,) and np.all(np.isfinite(info["tau_needed"]))
    assert len(checked) == 50 * env.frame_skip
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


def test_warm_start_schedule():
    from exo_rl.training import WarmStartSchedule
    off = WarmStartSchedule.from_config({"actor_freeze_steps": 100, "bc_weight": 2.0}, active=False)
    assert off.at(1) == (False, 0.0)        # inactive: no freeze, no BC term
    s = WarmStartSchedule(freeze_steps=100, bc_weight=2.0, bc_decay_steps=200)
    assert s.at(50) == (True, 2.0)          # frozen
    assert s.at(200) == (False, 1.0)        # halfway through the decay
    assert s.at(1000) == (False, 0.0)
    floor = WarmStartSchedule(freeze_steps=100, bc_weight=2.0, bc_decay_steps=200, bc_weight_min=0.5)
    assert floor.at(200) == (False, 1.25)   # halfway from 2.0 to the 0.5 floor
    assert floor.at(1000) == (False, 0.5)   # stays at the floor after the decay


def test_collector_handles_episode_ending_at_round_end(cfg):
    from exo_rl.envs.vector import SingleRunner
    from exo_rl.warmstart import Collector
    cfg["sim"]["episode_seconds"] = 0.1                 # 10-step episodes
    buf = ReplayBuffer(ExoAssistEnv(cfg).observation_space.shape[0], 4, 100)
    col = Collector(SingleRunner(cfg, seed=0), cfg["warmstart"], cfg["exo"]["action_scale"], buf)
    x, y = col.collect(20)                              # ends exactly on an episode boundary
    x2, _ = col.collect(10)
    assert len(x) == 2 * 9 and len(x2) == 9 and len(buf) == 30 and y.shape[1] == 4
