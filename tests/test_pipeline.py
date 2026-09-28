import numpy as np

from exo_rl.control.observation import ObservationBuilder
from exo_rl.control.torque_pipeline import ExoTorquePipeline, LimbParams, limb_gravity_torque

LIMB = LimbParams(7.5, 0.43, 0.186, 4.6, 0.25)


def make(**kw):
    cfg = {"max_torque": [40] * 4, "action_scale": [40] * 4, "rate_limit": 1500,
           "direction": ["both"] * 4, "feedforward": "none", "alpha": 0.0}
    cfg.update(kw)
    return ExoTorquePipeline(cfg, LIMB, 0.01)


def test_saturation_and_rate_limit():
    p = make()
    tau = p(np.ones(4), np.zeros(4), 0.0)
    assert np.allclose(tau, 15.0)                  # 1500 Nm/s * 10 ms
    for _ in range(10):
        tau = p(np.ones(4), np.zeros(4), 0.0)
    assert np.allclose(tau, 40.0)                  # saturates at max_torque


def test_direction_constraints():
    p = make(direction=["flexion_only", "extension_only", "both", "both"], rate_limit=1e9)
    tau = p(np.array([-1.0, 1.0, -1.0, 1.0]), np.zeros(4), 0.0)
    assert tau[0] == 0.0 and tau[1] == 0.0 and tau[2] < 0 and tau[3] > 0


def test_gravity_feedforward_signs():
    # Leg swung forward: gravity pulls it back -> positive (flexion) hip compensation.
    th, tk = limb_gravity_torque(np.radians(30), 0.0, 0.0, LIMB)
    assert th > 0 and abs(tk) < 1e-9 + abs(th)
    # Hanging straight down: no gravity torque.
    th, tk = limb_gravity_torque(0.0, 0.0, 0.0, LIMB)
    assert abs(th) < 1e-9 and abs(tk) < 1e-9


def test_observation_history_shape():
    ob = ObservationBuilder({"history": 5, "use_phase": True, "include_prev_action": True}, 0.01)
    s = {"q_exo": np.zeros(4), "qd_exo": np.zeros(4), "trunk_pitch": 0.0}
    for k in ("pelvis", "thigh_l", "thigh_r"):
        s[f"gyro_{k}"] = np.zeros(3)
        s[f"acc_{k}"] = np.zeros(3)
    o = ob(s, np.zeros(4))
    assert o.shape == (ob.dim,) and ob.dim == 5 * ob.frame_dim
