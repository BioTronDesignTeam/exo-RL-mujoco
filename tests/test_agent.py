import numpy as np

from exo_rl.agents.td3 import ReplayBuffer, TD3Agent


def test_td3_and_ddpg_update_and_save(tmp_path, cfg):
    for algo in ("td3", "ddpg"):
        cfg["agent"]["algo"] = algo
        agent = TD3Agent(10, 4, cfg["agent"])
        buf = ReplayBuffer(10, 4, 1000)
        for _ in range(300):
            buf.add(np.random.randn(10), np.random.uniform(-1, 1, 4), 1.0, np.random.randn(10), False)
        for _ in range(4):
            info = agent.update(buf, 32)
        assert np.isfinite(info["critic_loss"])
        path = tmp_path / f"{algo}.pt"
        agent.save(path)
        agent2 = TD3Agent.load(path)
        o = np.random.randn(10).astype(np.float32)
        assert np.allclose(agent.act(o), agent2.act(o))
