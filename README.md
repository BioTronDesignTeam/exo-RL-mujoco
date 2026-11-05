# exo-rl

Train an exoskeleton controller with deep RL (TD3/DDPG) that **reduces the wearer's effort**.
A simulated human performs the ACE tasks in MuJoCo while wearing the MkI FA900 exo (from your CAD).
The policy is rewarded for lowering the human's joint effort without disturbing their motion.
The same control code then runs on the real hardware.

---

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python -m pytest                # 76 tests, ~4 min — confirms everything works
```

- **Video recording** uses `imageio[ffmpeg]` (in `requirements.txt`).
- **Musculoskeletal human (experimental):** `pip install -r requirements-myo.txt`, then add `--set human.model=myo` to `pretrain_bc.py` / `train.py` to train on the MyoLeg model (80 muscles, exo CAD links and motors, rigid upper body + payload) instead of the planar torque-driven human. Effort becomes mean muscle activation² (static optimization, `exo_rl/sim/muscle_human.py`, settings under `human.muscle`). Observations and actions are unchanged, so policies transfer between models. About 5x slower per environment (~90 steps/s); the leg size is fixed (height only sets the trunk); high knees end early more often than on the planar model.
- **macOS live viewer:** use `mjpython` instead of `python` (`view.py`, `watch_training.py`).
- **Headless Linux recording:** set `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU, `apt install libosmesa6`).

**Docker** (same environment for everyone; headless, CPU):

```bash
docker build -t exo-rl .
docker run --rm exo-rl                                         # run the tests
docker run --rm -v "$PWD/runs:/app/runs" exo-rl python scripts/train.py --name mk1 --max-minutes 60
```

---

## Quick start

```bash
python scripts/view.py --task squat                        # watch pilot + exo (unpowered)
python scripts/train.py --name mk1 --max-minutes 60        # train (stops and saves after 60 min)
python scripts/train.py --name mk1 --resume                # continue training
python scripts/evaluate.py --run runs/mk1                  # effort table + plots
python scripts/plot_training.py --run runs/mk1             # training curves
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
| `--init-buffer FILE` | — | Prefill the replay buffer (e.g. `bc_buffer.npz` from `pretrain_bc.py`); learning starts at once |

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

### `scripts/plot_training.py`: plot training progress

```bash
python scripts/plot_training.py --run runs/mk1                          # -> runs/mk1/training.png
python scripts/plot_training.py --run runs/mk1 runs/ws --out cmp.png    # compare runs
```

| Flag | Default | Description |
|---|---|---|
| `--run PATH [PATH ...]` | required | Run folder(s); several = compare mode |
| `--smooth N` | `20` | Moving-average window in episodes (`1` = off) |
| `--out FILE` | `<run>/training.png` (several runs: `runs/compare_training.png`) | Output PNG |

One run: smoothed total reward per episode, smoothed human effort, and eval effort reduction per task,
with the `best.pt` step marked. Several runs: one color per run, and the eval graph shows each run's
checkpoint score (★ = `best.pt`). Works while training is still running.

### `scripts/evaluate.py`: measure a trained policy

| Flag | Default | Description |
|---|---|---|
| `--run PATH` | required | Run folder |
| `--ckpt FILE` | `best.pt` | Checkpoint |
| `--episodes N` | `3` | Episodes per task and mode |
| `--randomized N` | `0` | Also test on N random pilots (size, strength, latency, noise), each compared with itself |

It compares exo **off / policy**, plus **not wearing the exo**, which is the ACE comparison.
Output goes to `runs/NAME/eval_<ckpt>/`:
- `summary.csv` (and `summary_randomized.csv` with `--randomized`: mean and worst pilot)
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
| `--spec PATH` | `configs/exo_mk1_fa900.yaml` | Exo spec |
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

### Training on a remote machine

Long runs can go on a cloud VM or lab server (Linux, 8–16 CPU cores, no GPU needed), so your
laptop can be off. After cloning the repo on the machine:

```bash
bash scripts/remote_train.sh --name ws --minutes 120 --seeds "0 1 2" --detach
tail -f runs/ws.log                                   # follow progress; safe to log out
scp -r user@VM_IP:exo-RL-mujoco/runs/ws_s* runs/      # later, from your laptop
```

It creates the Python environment on first use, then for each seed runs the warm start, RL
fine-tuning for `--minutes`, and `evaluate.py` (nominal + 5 random pilots). Stop the VM when done.

The settings used for the reported runs are kept in `configs/remote/` (`planar.set`, `myo.set`:
`agent.bc_weight_min=0.5`, `train.total_steps=500000`, plus `human.model=myo`), so a rerun
doesn't depend on remembering `--set` flags:

```bash
bash scripts/remote_train.sh --name myo --set-file configs/remote/myo.set --detach
```

To run several jobs on one machine at once, pass `--threads N` (`train.torch_threads`). The
auto value assumes the run has the machine to itself: three runs with 8 envs each on 32 cores
got ~16 torch threads apiece, starved their env workers and ran at ~5 steps/s instead of ~200.
Keep `runs × (threads + envs)` near the core count, e.g. `--threads 4` for three 8-env runs.

### Options

**Parallel simulation.** `train.num_envs=N` runs N envs in worker processes and overlaps
their simulation with the gradient updates. On CPU the updates dominate the cost, so pair
it with `train.updates_per_step` < 1. The default (`num_envs=1`, `updates_per_step=1.0`)
is the classic single-env trainer. `--max-minutes` / `--resume` work as usual; with N > 1 training may
stop mid-episode, and resumed envs get fresh seeds.

**Fine-tune instead of restarting.** After a config change (reward weights, torque limits,
randomization), `--init-from runs/OLD/checkpoints/best.pt` starts from the old actor and
critic. Observation settings and `agent.hidden` must match; otherwise it exits with an error.

**Warm start: teacher, then behavior cloning, then RL.** Instead of starting RL from a random policy:

```bash
python scripts/pretrain_bc.py --name bc                       # teacher -> BC actor (bc.pt, bc_buffer.npz)
python scripts/train.py --name ws --config runs/bc/config.yaml --init-from runs/bc/checkpoints/bc.pt     --init-buffer runs/bc/checkpoints/bc_buffer.npz
```

With `--init-from`, the `start_steps` warm-up uses the loaded policy (plus exploration noise)
instead of random actions. Only the critic learns for `agent.actor_freeze_steps`. After that the
actor loss is `-Q/mean|Q| + w·‖π − π_init‖²`, with `w = agent.bc_weight` decaying to
`agent.bc_weight_min` over `agent.bc_decay_steps`, so the untrained critic can't wreck the initial
policy. With the floor at 0, all three 1M-step runs dropped sharply once `w` reached 0;
`agent.bc_weight_min=0.5` keeps a small pull toward the warm start. These safeguards are off
unless `--init-from` is used.

**Cheaper settings while iterating** (compare final results before adopting them):

| Override | Effect |
|---|---|
| `train.eval_async=true` | Evaluate in a background process while training continues (same results) |
| `train.eval_every=50000` / `train.eval_episodes=2` | Evaluation is cheap with 1 env but took ~40% of wall time with 4 envs (see above) |
| `train.best_metric=worst` | Pick `best.pt` by the worst task, so one task can't regress unnoticed |
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
| `"reward.effort_joints=[hip, knee]"` | Count only exo-assisted joints as effort in the reward (evaluation stays whole-body) |
| `warmstart.teacher_fraction=0.3` | Warm-start teacher strength (`pretrain_bc.py`) |
| `agent.bc_weight=2.5` | Pull toward the initial policy when fine-tuning (`--init-from` only) |

All settings are commented in `configs/default.yaml`. The exo itself is defined in `configs/exo_mk1_fa900.yaml` (the previous MK1 FB900 in `configs/exo_mk1.yaml`, still used by older runs). A new SolidWorks URDF export is prepared with `python scripts/prepare_urdf.py <export folder> assets/<name>` (decimated meshes, MuJoCo-ready URDF), then described in a spec like `exo_mk1_fa900.yaml`.

---

## Project layout

Working on the code? Read [CONTRIBUTING.md](CONTRIBUTING.md) first: how a control step flows,
where to change what, conventions, and how to keep default training bit-identical.

```
configs/          default.yaml (all settings), exo_mk1_fa900.yaml (your exo), exo_mk1.yaml (previous exo)
assets/exo_mk1_fa900/  MuJoCo-ready URDF + meshes (hi = viewing, lo = training); assets/exo_mk1/ previous exo
exo_rl/sim/       human + exo model, reference motions, human controller
exo_rl/control/   observation + torque pipeline — shared with hardware, keep identical
exo_rl/envs/      environment, reward, randomization, parallel collection (vector.py)
exo_rl/agents/    TD3 / DDPG learning algorithm
exo_rl/*.py       training loop, evaluation, warm start (teacher + BC), viewing helpers
scripts/          train, pretrain_bc, view, plot_training, watch_training, evaluate, export, exo_info
deploy/           hardware loop, motor map, safety monitor
data/motions/     your motion recordings (CSV)
tests/            76 tests (run by CI on every PR)
runs/demo/        example trained run
```

---

## Hardware

1. **Connect your drivers.** Implement `HardwareInterface` in `deploy/run_hardware.py`: `connect`, `read_sensors`, `send_torque`, `disable`.
2. **Convert signs.** Use `deploy/joint_map.py` (`MotorMap`) between motor and human conventions. The URDF signs are hip_l +1, knee_l −1, hip_r −1, knee_r +1.
3. **Calibrate zeros.** Set encoder zero offsets in `configs/exo_mk1_fa900.yaml` (`motor_zero_offsets`).
4. **Match your actuators.** Set `exo.max_torque`, `exo.rate_limit`, and `sim.control_dt`.
5. **Test step by step.** Dry run with `--sim`, then run on hardware at reduced torque. Keep firmware torque limits and a hardware e-stop; the Python safety monitor is only a second layer.

---

## Good to know

- **Read effort as relative.** The simulated human's absolute effort runs high. Compare exo on vs. off, or vs. no exo.
- **The exo has a cost.** Wearing it unpowered raises effort by 15–38%. The policy must repay that before the pilot comes out ahead.
- **Placeholders to replace:** the motions are synthetic until you record real ones, and the URDF torque limit of 10 Nm looks like an exporter default.
- **Speed.** With default settings training runs ~70–115 steps/s depending on the CPU (1M steps take 2.5–4 h); gradient updates, not the simulation, dominate. See [Training faster](#training-faster), and use `--max-minutes` / `--resume` to split long runs.

---

## TODO

Open tasks for contributors. Pick one, put your name next to it, and see
[CONTRIBUTING.md](CONTRIBUTING.md) for where things live.

### Training
- [ ] **Multi-seed benchmark.** The results in [Training faster](#training-faster) are one seed each; rerun A/B/C with 3 seeds (plugged in, nothing else running).
- [ ] **Fix quick_steps.** It is the weakest task. The warm-start teacher only reaches −27% net vs no exo there; try a better teacher (e.g. per-task `teacher_fraction`) or reward shaping.
- [ ] **Tune warm-start fine-tuning.** In the 30-min run, RL barely improved on the BC policy; sweep `agent.bc_weight`, `bc_decay_steps`, `actor_freeze_steps`.
- [ ] **Smaller networks.** `agent.hidden=[128,128]` is faster to train; check whether final effort holds up.
- [ ] **GPU simulation.** Port the env to MuJoCo MJX / Warp (or Isaac Sim) for hundreds of parallel envs; the human controller in `exo_rl/sim/human.py` must be rewritten as batched tensor code.

### Simulation
- [ ] **Randomization ranges.** Check `randomization:` against real measurements (pilot sizes, actuator latency, torque noise) once hardware data exists.
- [ ] **Real motions.** Replace the synthetic reference motions with recorded CSVs in `data/motions/`.
- [ ] **Exo model.** The URDF torque limit of 10 Nm looks like an exporter default; set the real actuator limits.
- [ ] **Human model.** Upgrade to a musculoskeletal model (e.g. MyoSuite) and use muscle activation as the effort signal.

### Evaluation
- [ ] **Randomized pilots during training.** `evaluate.py --randomized N` exists; add the same check to training-time evaluation.
- [x] **CI.** `.github/workflows/ci.yml` runs lint + tests on every PR and push to main.

### Hardware
- [ ] **Hardware interface.** Implement `HardwareInterface` in `deploy/run_hardware.py` (connect, read sensors, send torque, disable).
- [ ] **Calibration.** Set encoder zero offsets in `configs/exo_mk1_fa900.yaml` and verify motor signs on the real exo.
- [ ] **First tests.** Run the exported policy at reduced torque with firmware limits and an e-stop.
- [ ] **Docs.** `exo_rl/control/observation.py` refers to `deploy/hardware_interface.py`, which doesn't exist (the interface is in `deploy/run_hardware.py`).

### Tooling
- [ ] **macOS.** Test the live viewer with `mjpython` (`view.py`, `watch_training.py`).
- [ ] **Headless Linux.** Test video recording with `MUJOCO_GL=egl` / `osmesa`.
