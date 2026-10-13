# Working on exo-rl

How the code fits together, where to make common changes, and the rules that keep training
reproducible and the hardware path safe. Usage (commands, flags) is in [README.md](README.md).

## Setup

```bash
python -m venv venv && venv/Scripts/activate      # Windows; source venv/bin/activate elsewhere
pip install -r requirements.txt
python -m pytest                                  # all tests, ~1.5 min
```

## One control step (10 ms)

```
sensors (sim or hardware)
  -> exo_rl/control/observation.py     ObservationBuilder: scaled sensors, stacked history
  -> policy (TD3 actor)                action in [-1, 1] per exo joint
  -> exo_rl/control/torque_pipeline.py ExoTorquePipeline: scale, limits, rate limit, feedforward
  -> exo_rl/envs/exo_env.py            latency + torque noise, then 5 physics substeps in which
       exo_rl/sim/human.py             the simulated human computes its own joint torques
  -> exo_rl/envs/rewards.py            reward = alive - effort - tracking - smoothness - magnitude
```

`exo_rl/control/` is the only code that also runs on the real exo (`deploy/`). Everything else
is simulation or training.

## Code map

| Path | What it does |
|---|---|
| `exo_rl/envs/exo_env.py` | Gymnasium env: one episode = one pilot + one task (or a chained circuit) |
| `exo_rl/envs/randomization.py` | Per-episode pilot/hardware randomization |
| `exo_rl/envs/vector.py` | `SingleRunner` / `VectorRunner`: one interface for 1 or N envs |
| `exo_rl/sim/` | MuJoCo model (human + CAD exo + payload), reference motions, human controller |
| `exo_rl/agents/td3.py` | TD3/DDPG networks, `ReplayBuffer`, checkpoint save/load |
| `exo_rl/training.py` | `Trainer`: the training loop behind `scripts/train.py` |
| `exo_rl/evaluation.py` | Effort vs exo off / no exo; `PeriodicEval` for training-time evaluation |
| `exo_rl/warmstart.py` | Teacher, BC data `Collector`, BC training (`scripts/pretrain_bc.py`) |
| `exo_rl/viewing.py` | Viewer/recording helpers for `view.py` and `watch_training.py` |
| `scripts/` | Command-line entry points only; logic lives in `exo_rl/` |
| `deploy/` | Hardware loop, motor map, safety monitor |

## Where to change what

| I want to... | Change |
|---|---|
| Reward weights | `reward:` in `configs/default.yaml` |
| Reward terms | `exo_rl/envs/rewards.py` |
| Add a motion | `exo_rl/sim/reference.py` (synthetic) or a CSV in `data/motions/` |
| Randomization ranges | `randomization:` in the config; new parameters in `envs/randomization.py` |
| Observation contents | `exo_rl/control/observation.py`. **Hardware code**: re-export and re-test on hardware |
| Torque limits/shaping | `exo:` in the config; logic in `control/torque_pipeline.py` (**hardware code**) |
| Human model behavior | `exo_rl/sim/human.py` |
| Training algorithm | `exo_rl/agents/td3.py` (update rule) and `exo_rl/training.py` (loop) |
| Evaluation metrics | `exo_rl/evaluation.py` |

## Conventions

- **Joint order.** Human: `hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r`
  (`model_builder.HUMAN_JOINTS`). Exo: `hip_l, knee_l, hip_r, knee_r` (`EXO_JOINTS`);
  `exo_env.EXO_IN_HUMAN` maps between them. Arrays follow these orders everywhere.
- **Signs.** Torques and angles are in human convention: **+ = flexion**. Motor signs are
  converted only in `deploy/joint_map.py`.
- **Config keys.** Every option lives in `configs/default.yaml` with a comment. Code reads new
  keys with a fallback, e.g. `cfg["train"].get("num_envs", 1)`, because each run saves its own
  `config.yaml` and older runs don't have new keys.
- **Checkpoints.** Write through `TD3Agent.save` / `ReplayBuffer.save` / `utils.atomic_write`
  (temp file + rename), so `watch_training.py` or a crash never sees a half-written file.
- **Scripts stay thin.** Argument parsing and printing in `scripts/`; reusable logic in `exo_rl/`.

## Defaults must not change

With the default config, training must produce **bit-identical** results to the original
single-env trainer. New features are opt-in config keys or flags. If you touch
`training.py`, `td3.py`, `exo_env.py` or the human/reward code, check with a short run against
a known-good commit:

```bash
git worktree add ../ref <known-good-commit>
SETS="--set train.total_steps=1500 --set train.start_steps=0 --set train.eval_every=500 \
      --set train.eval_episodes=1 --set tasks.names=[squat,walk] --set sim.episode_seconds=2"
(cd ../ref && python scripts/train.py --name eq $SETS)
python scripts/train.py --name eq $SETS
# compare runs/eq/episodes.csv (ignore the env_id column), eval.csv, and the weights in
# checkpoints/latest.pt with torch.equal
git worktree remove ../ref
```

`start_steps=0` matters: the random warm-up uses an unseeded action space, so it differs
between any two runs.

## Before you commit

- `python -m pytest` passes; add a test for new behavior (tests are fast, small configs).
- CI (`.github/workflows/ci.yml`) runs `pyflakes` and the full test suite on every pull request;
  a PR is merged only when the `tests` check is green.
- Changing the reward? `tests/test_reward.py` pins down the formula; update those tests on purpose.
- New option: config key with a comment and/or a CLI flag, plus a README line.
- Don't commit `runs/` (except `runs/demo/`); training output is large.
