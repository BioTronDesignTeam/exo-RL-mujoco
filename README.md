# exo-rl

Train an exoskeleton controller with deep RL (TD3/DDPG) that **reduces the wearer's effort**.
A simulated human performs the ACE tasks in MuJoCo while wearing the MK1 exo (from your CAD).
The policy is rewarded for lowering the human's joint effort without disturbing their motion.
The same control code then runs on the real hardware.

---

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python -m pytest                # 40 tests, ~1 min — confirms everything works
```

- **Video recording** needs `imageio[ffmpeg]` (`pip install "imageio[ffmpeg]"`).
- **macOS live viewer:** use `mjpython` instead of `python` (`view.py`, `watch_training.py`).
- **Headless Linux recording:** set `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU, `apt install libosmesa6`).

---

## Quick start

```bash
python scripts/view.py --task squat                        # watch pilot + exo (unpowered)
python scripts/train.py --name mk1 --max-minutes 60        # train (stops and saves after 60 min)
python scripts/train.py --name mk1 --resume                # continue training
python scripts/evaluate.py --run runs/mk1                  # effort table + plots
python scripts/view.py --run runs/mk1 --task circuit       # watch the trained policy
python scripts/export_policy.py --run runs/mk1 --height 1.78 --mass 72
python -m deploy.run_hardware --export runs/mk1/export --sim   # deployment dry run
```

`runs/demo/` holds a short pre-trained run, so the commands after training work immediately.

---

## Commands

### `scripts/train.py`: train a policy

| Flag | Default | Description |
|---|---|---|
| `--name NAME` | timestamp | Run folder: `runs/NAME/` |
| `--config PATH` | `configs/default.yaml` | Config file |
| `--set KEY=VALUE` | — | Override any config value (repeatable) |
| `--max-minutes N` | — | Stop after N minutes and save a snapshot |
| `--resume` | off | Continue `runs/NAME/` from its snapshot |
| `--init-from FILE` | — | Start actor + critic from a checkpoint (fresh optimizers, empty buffer); see [Training faster](#training-faster) |

Writes `runs/NAME/`:
- `config.yaml`
- `episodes.csv` (reward terms per episode, with `env_id`)
- `eval.csv` (effort reduction per task)
- `checkpoints/` (`best.pt`, `latest.pt` every `train.latest_every` steps, `step_N.pt`; written atomically)
- `resume/` (snapshot for `--resume`)

Training is headless (no window). Watch progress live with `watch_training.py` in another terminal.

### `scripts/pretrain_bc.py`: warm-start a policy by imitation

| Flag | Default | Description |
|---|---|---|
| `--name NAME` | timestamp | Run folder: `runs/NAME/` |
| `--steps N` | `200000` | Dataset size (transitions) |
| `--epochs N` | `warmstart.epochs` | BC epochs after each collection round |
| `--teacher-fraction K` | `warmstart.teacher_fraction` | Share of the human's needed torque the teacher supplies |
| `--config`, `--set` | | Same as `train.py` |

A privileged teacher (the sim knows the torque the human needs) is imitated from normal sensor
observations, with DAgger-style collection in parallel envs. It writes `runs/NAME/checkpoints/bc.pt`
and prints teacher and BC effort per task. Continue with `train.py --init-from` (see [Training faster](#training-faster)).

### `scripts/view.py`: watch the simulation or record a video

| Flag | Default | Description |
|---|---|---|
| `--task NAME` | `squat` | See [Tasks](#tasks), or `circuit` for the ACE course in order |
| `--run PATH` | — | Drive the exo with this run's policy |
| `--ckpt FILE` | `best.pt` | Checkpoint inside `--run` |
| `--mode {policy,off,feedforward}` | `policy` with `--run`, else `off` | Exo control mode |
| `--camera {track,fixed}` | `track` | Follow the pilot, or a static side view that shows travel |
| `--speed X` | `1.0` | Real-time factor (`0.25` = slow motion) |
| `--seconds N` | until closed | Stop after N simulated seconds |
| `--no-exo` | off | Pilot + payload only, without the exo |
| `--randomize` | off | Random pilot instead of the nominal one |
| `--seed N` | `0` | Random seed (also sets the gait start phase) |
| `--record FILE` | — | Save `.mp4`, `.gif`, or `.png` instead of opening a window |
| `--width`, `--height` | `720`, `528` | Recording size |
| `--save-xml FILE` | — | Write the model (MJCF + meshes) and exit |
| `--config`, `--set` | | Same as `train.py` |

It prints task, gait phase, human effort, and exo torque once per second. It collects no training data.

### `scripts/watch_training.py`: watch a policy while it trains

```bash
python scripts/watch_training.py --run runs/mk1                                 # live viewer
python scripts/watch_training.py --run runs/mk1 --record-dir runs/mk1/videos    # headless videos
```

| Flag | Default | Description |
|---|---|---|
| `--run PATH` | required | Run folder; polls `checkpoints/latest.pt` and reloads the policy when it changes |
| `--poll S` | `10` | Seconds between checks |
| `--tasks T [T ...]` | the run's training tasks | Tasks to cycle, one episode each |
| `--camera {track,fixed}`, `--speed X` | `track`, `1.0` | As in `view.py` |
| `--record-dir DIR` | — | No window: for each new checkpoint save `DIR/step_<N>_<task>.mp4` per task |
| `--seconds N` | `8` | Video length per task (`--record-dir`) |
| `--width`, `--height` | `720`, `528` | Video size |

It prints the checkpoint step on each reload and the mean effort per episode, and shows step and
task as a text overlay in the viewer. It waits if the run or checkpoint doesn't exist yet.

### `scripts/evaluate.py`: measure a trained policy

| Flag | Default | Description |
|---|---|---|
| `--run PATH` | required | Run folder |
| `--ckpt FILE` | `best.pt` | Checkpoint |
| `--episodes N` | `3` | Episodes per task and mode |

It compares exo **off / policy**, plus **not wearing the exo**, which is the ACE comparison.
Output goes to `runs/NAME/eval_<ckpt>/`:
- `summary.csv`
- `effort_by_task.png`
- `gait_<task>.png` (joint angles and torques over the gait cycle)

### `scripts/export_policy.py`: package a policy for the hardware

| Flag | Default | Description |
|---|---|---|
| `--run PATH` | required | Run folder |
| `--ckpt FILE` | `best.pt` | Checkpoint |
| `--height M`, `--mass KG` | nominal | The pilot's actual size (sets the gravity feedforward) |

Writes `runs/NAME/export/`:
- `policy.pt` (TorchScript)
- `deploy_config.json` (observation and torque settings, motor map)

### `scripts/exo_info.py`: check the exo model

| Flag | Default | Description |
|---|---|---|
| `--spec PATH` | `configs/exo_mk1.yaml` | Exo spec |
| `--height M`, `--mass KG` | `1.75`, `75` | Pilot to fit it on |
| `--render FILE` | — | Save an image of the pilot wearing the exo |

It prints the exo mass breakdown, knee-axis mismatch, and motor signs.

### `deploy/run_hardware.py`: real-time control loop

```bash
python -m deploy.run_hardware --export runs/mk1/export --sim
```

| Flag | Default | Description |
|---|---|---|
| `--export PATH` | required | Output folder of `export_policy.py` |
| `--sim` | off | Dry run against the simulator |
| `--duration S` | `10` | Seconds to run |
| `--task NAME` | `squat` | Task for `--sim` |

On real hardware, first implement `HardwareInterface` in this file (see [Hardware](#hardware)).

### Tests

```bash
python -m pytest                      # all
python -m pytest tests/test_exo.py    # one file
```

---

## Training faster

**Recommended fast run** (warm start + 4 parallel envs; ~3 min pretraining, then RL):

```bash
python scripts/pretrain_bc.py --name bc --steps 80000 && python scripts/train.py --name ws --config runs/bc/config.yaml --init-from runs/bc/checkpoints/bc.pt --set train.num_envs=4 --set train.updates_per_step=0.25 --set train.eval_every=50000
```

From scratch, the one-line speed-up is `--set train.num_envs=4 --set train.updates_per_step=0.25`.

### Benchmark (8-thread consumer CPU, no GPU; 4 ACE tasks, otherwise default settings)

Throughput in the learning phase (2 min each, evaluation off):

| `num_envs` | `updates_per_step=1.0` | `updates_per_step=0.25` |
|---|---|---|
| 1 (default) | 70 steps/s | 200 steps/s |
| 4 | 93 steps/s | **294 steps/s** |
| 8 | 60 steps/s | 226 steps/s (260 with `train.torch_threads=2`) |

Here a gradient update costs ~15 ms and an env step ~1.2 ms, so fewer updates per sample matter
most. More envs than physical cores stop paying off.

Equal 30-min budget, then `evaluate.py` (best.pt, 3 episodes/task). "Net vs no exo" is effort
compared with the pilot not wearing the exo; positive = the exo helps:

| Run | Steps | squat | high_knees | quick_steps | backward_walk |
|---|---|---|---|---|---|
| A. scratch, `num_envs=1` (default) | 128k | −16.9% | −10.2% | −10.6% | −14.7% |
| B. scratch, 4 envs, 0.25 upd/step | 320k | +2.3% | −6.3% | −22.5% | −10.1% |
| C. BC warm start (3 min) + RL as B (27 min) | 239k | **+13.3%** | **−1.9%** | −21.1% | **−4.6%** |
| (C's BC policy alone, before RL) | — | +7.3% | −1.9% | −20.4% | −3.6% |

Takeaways (single seed, so treat small differences as noise):
- Parallel envs with fewer updates collect 2.5× more experience in the same time, and B beats A
  on three of four tasks. quick_steps got worse in both B and C, so check per-task results.
- The BC warm start gives most of the gain in 3 minutes. Within 27 min, RL fine-tuning improved
  squat but barely moved the others; longer fine-tuning is where RL has to earn its keep.
- With default evaluation settings, the parallel run spent ~40% of its wall time evaluating (177
  vs 294 steps/s); with 1 env it is negligible. Hence `train.eval_every=50000` above.

### Options

**Parallel simulation.** `train.num_envs=N` runs N envs in worker processes and overlaps
their simulation with the gradient updates. On CPU the updates dominate the cost, so pair
it with `train.updates_per_step` < 1. The default (`num_envs=1`, `updates_per_step=1.0`)
trains exactly as before. `--max-minutes` / `--resume` work as usual; with N > 1 training may
stop mid-episode, and resumed envs get fresh seeds.

**Fine-tune instead of restarting.** After a config change (reward weights, torque limits,
randomization), `--init-from runs/OLD/checkpoints/best.pt` starts from the old actor and
critic. Observation settings and `agent.hidden` must match; otherwise it exits with an error.

**Warm start: teacher, then behavior cloning, then RL.** Instead of starting RL from a random policy:

```bash
python scripts/pretrain_bc.py --name bc                       # teacher -> BC actor (bc.pt)
python scripts/train.py --name ws --config runs/bc/config.yaml --init-from runs/bc/checkpoints/bc.pt
```

With `--init-from`, the `start_steps` warm-up uses the loaded policy (plus exploration noise)
instead of random actions. Only the critic learns for `agent.actor_freeze_steps`. After that the
actor loss is `-Q/mean|Q| + w·‖π − π_init‖²`, with `w = agent.bc_weight` decaying to 0 over
`agent.bc_decay_steps`, so the untrained critic can't wreck the initial policy. These safeguards
are off unless `--init-from` is used.

**Cheaper settings while iterating** (compare final results before adopting them):

| Override | Effect |
|---|---|
| `train.eval_every=50000` / `train.eval_episodes=2` | Evaluation is cheap with 1 env but took ~40% of wall time with 4 envs (see above) |
| `agent.hidden=[128,128]` | Roughly 3× cheaper updates; check that final effort holds up |
| `agent.batch_size=512` | Nearly free on a GPU; on CPU cost grows with batch size |
| `"tasks.names=[squat]"` | Train on fewer tasks while tuning, then fine-tune on all with `--init-from` |

---

## Tasks

| Task | Moves | Trained by default | Notes |
|---|---|---|---|
| `squat` | in place | yes | ACE section 1 |
| `high_knees` | in place | yes | ACE section 2 |
| `quick_steps` | in place | yes | ACE section 3 |
| `backward_walk` | backward | yes | ACE section 4 |
| `walk` | forward | yes | |
| `high_knees_travel` | forward | no | Traveling version of high knees |
| `jog` | forward | no | Grounded running; fails with the unpowered exo worn |

- **Train on a task:** add it to `tasks.names`.
- **Use your own recordings:** put `<task>.csv` files in `data/motions/` and set `tasks.data_dir=data/motions` (format in `data/motions/README.md`).

---

## Common config overrides

Pass any of these with `--set` on `train.py` or `view.py`.

| Override | Effect |
|---|---|
| `agent.algo=ddpg` | DDPG instead of TD3 |
| `train.total_steps=1000000` | Training length (a full run is ~1M steps) |
| `train.num_envs=8` | Simulate in 8 parallel processes (see [Training faster](#training-faster)) |
| `train.updates_per_step=0.25` | Gradient updates per collected transition (summed over envs) |
| `train.latest_every=5000` | How often `latest.pt` is written |
| `exo.max_torque=[60,60,60,60]` | Actuator limits in Nm (hip_l, knee_l, hip_r, knee_r) |
| `exo.alpha=0.5` | Add gravity-compensation feedforward under the policy |
| `exo.direction=[flexion_only,both,flexion_only,both]` | One-way (cable) actuators |
| `exo_hardware.enabled=false` | Simulate without the exo |
| `"tasks.names=[squat, high_knees]"` | Tasks to train on |
| `tasks.circuit_order=ace` | Chained episodes follow the ACE course order |
| `tasks.data_dir=data/motions` | Use recorded motion CSVs |
| `randomization.enabled=false` | Nominal pilot every episode |
| `reward.w_effort=3.0` | Weight on human effort |
| `warmstart.teacher_fraction=0.3` | Warm-start teacher strength (`pretrain_bc.py`) |
| `agent.bc_weight=2.5` | Pull toward the initial policy when fine-tuning (`--init-from` only) |

All settings are commented in `configs/default.yaml`. The exo itself is defined in `configs/exo_mk1.yaml`.

---

## Project layout

```
configs/          default.yaml (all settings), exo_mk1.yaml (your exo)
assets/exo_mk1/   MuJoCo-ready URDF + meshes (hi = viewing, lo = training)
exo_rl/sim/       human + exo model, reference motions, human controller
exo_rl/control/   observation + torque pipeline — shared with hardware, keep identical
exo_rl/envs/      environment, reward, randomization, parallel collection (vector.py)
exo_rl/agents/    TD3 / DDPG learning algorithm
exo_rl/*.py       evaluation, warm start (teacher + behavior cloning), viewing helpers
scripts/          train, pretrain_bc, view, watch_training, evaluate, export, exo_info
deploy/           hardware loop, motor map, safety monitor
data/motions/     your motion recordings (CSV)
tests/            40 tests
runs/demo/        example trained run
```

---

## Hardware

1. **Connect your drivers.** Implement `HardwareInterface` in `deploy/run_hardware.py`: `connect`, `read_sensors`, `send_torque`, `disable`.
2. **Convert signs.** Use `deploy/joint_map.py` (`MotorMap`) between motor and human conventions. The URDF signs are hip_l +1, knee_l −1, hip_r −1, knee_r +1.
3. **Calibrate zeros.** Set encoder zero offsets in `configs/exo_mk1.yaml` (`motor_zero_offsets`).
4. **Match your actuators.** Set `exo.max_torque`, `exo.rate_limit`, and `sim.control_dt`.
5. **Test step by step.** Dry run with `--sim`, then run on hardware at reduced torque. Keep firmware torque limits and a hardware e-stop; the Python safety monitor is only a second layer.

---

## Good to know

- **Read effort as relative.** The simulated human's absolute effort runs high. Compare exo on vs. off, or vs. no exo.
- **The exo has a cost.** Wearing it unpowered raises effort by 15–38%. The policy must repay that before the pilot comes out ahead.
- **Placeholders to replace:** the motions are synthetic until you record real ones, and the URDF torque limit of 10 Nm looks like an exporter default.
- **Speed.** With default settings training runs ~70–115 steps/s depending on the CPU (1M steps take 2.5–4 h); gradient updates, not the simulation, dominate. See [Training faster](#training-faster), and use `--max-minutes` / `--resume` to split long runs.
