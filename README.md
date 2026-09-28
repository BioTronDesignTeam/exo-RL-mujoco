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
python -m pytest                # 30 tests, ~30 s — confirms everything works
```

- **Video recording** needs `imageio[ffmpeg]` (`pip install "imageio[ffmpeg]"`).
- **macOS live viewer:** use `mjpython` instead of `python`.
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

Writes `runs/NAME/`:
- `config.yaml`
- `episodes.csv` (reward terms per episode)
- `eval.csv` (effort reduction per task)
- `checkpoints/` (`best.pt`, `latest.pt`, `step_N.pt`)
- `resume/` (snapshot for `--resume`)

Training is headless (no window). Watch progress with `view.py --ckpt latest.pt` in another terminal.

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
| `exo.max_torque=[60,60,60,60]` | Actuator limits in Nm (hip_l, knee_l, hip_r, knee_r) |
| `exo.alpha=0.5` | Add gravity-compensation feedforward under the policy |
| `exo.direction=[flexion_only,both,flexion_only,both]` | One-way (cable) actuators |
| `exo_hardware.enabled=false` | Simulate without the exo |
| `"tasks.names=[squat, high_knees]"` | Tasks to train on |
| `tasks.circuit_order=ace` | Chained episodes follow the ACE course order |
| `tasks.data_dir=data/motions` | Use recorded motion CSVs |
| `randomization.enabled=false` | Nominal pilot every episode |
| `reward.w_effort=3.0` | Weight on human effort |

All settings are commented in `configs/default.yaml`. The exo itself is defined in `configs/exo_mk1.yaml`.

---

## Project layout

```
configs/          default.yaml (all settings), exo_mk1.yaml (your exo)
assets/exo_mk1/   MuJoCo-ready URDF + meshes (hi = viewing, lo = training)
exo_rl/sim/       human + exo model, reference motions, human controller
exo_rl/control/   observation + torque pipeline — shared with hardware, keep identical
exo_rl/envs/      environment, reward, randomization
exo_rl/agents/    TD3 / DDPG learning algorithm
scripts/          train, view, evaluate, export, exo_info
deploy/           hardware loop, motor map, safety monitor
data/motions/     your motion recordings (CSV)
tests/            30 tests
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
- **Speed.** Training runs ~115 steps/s on one CPU core, so 1M steps take ~2.5 h. Use `--max-minutes` / `--resume` to split it up.
