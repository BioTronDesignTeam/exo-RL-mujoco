"""Training and deployment must produce identical torques from identical sensors."""
import numpy as np
import torch

from deploy.controller import DeployController
from exo_rl.agents.td3 import TD3Agent
from exo_rl.envs.exo_env import ExoAssistEnv
from exo_rl.sim.model_builder import EXO_JOINTS


def test_env_and_deploy_controller_match(cfg):
    cfg["randomization"]["enabled"] = True       # include sensor noise, latency, etc.
    cfg["exo"]["alpha"] = 0.5                     # exercise the feedforward path too
    env = ExoAssistEnv(cfg, seed=7)
    obs, _ = env.reset(options={"task": "high_knees"})
    torch.manual_seed(0)
    agent = TD3Agent(env.observation_space.shape[0], 4, cfg["agent"])
    actor = torch.jit.trace(agent.actor.eval(), torch.zeros(1, agent.obs_dim))

    deploy_cfg = {"control_dt": cfg["sim"]["control_dt"], "exo_joints": EXO_JOINTS,
                  "observation": cfg["observation"], "exo": cfg["exo"],
                  "limb": env.limb.to_dict(), "obs_dim": agent.obs_dim}
    ctrl = DeployController(actor, deploy_cfg)

    for _ in range(200):
        sensors = env.sensors                      # exactly what the env's pipeline will use
        tau_deploy = ctrl.step(sensors)
        with torch.no_grad():
            action = actor(torch.as_tensor(obs).unsqueeze(0)).numpy()[0]
        obs, *_, info = env.step(action)
        np.testing.assert_allclose(tau_deploy, info["tau_exo_cmd"], rtol=1e-5, atol=1e-5)
