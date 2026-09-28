# exo-rl: learning exoskeleton assistance that reduces human effort

Deep RL (TD3, with a DDPG mode) for hip/knee assistance on a lower-limb exoskeleton,
trained in MuJoCo against an **active simulated human** and rewarded for **reducing that
human's effort**, which is what the ACE Athletic Test scores.

```
reference motion ──► simulated human (tracks motion, produces its own torques)
                                   ▲             │
               exo torque ─────────┘             ▼
policy ◄── observation (sensors only) ◄──── MuJoCo physics
   │
   └─► torque pipeline (same code on hardware) ─► exo torque
reward = alive − human effort − tracking error − jerk − torque magnitude
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest                                    # 30 tests, ~30 s

python scripts/train.py --name baseline             # full run (1M steps)
python scripts/train.py --name baseline --max-minutes 60   # ...or in chunks
python scripts/train.py --name baseline --resume

python scripts/evaluate.py --run runs/baseline      # effort table + plots
python scripts/export_policy.py --run runs/baseline --height 1.78 --mass 72
python -m deploy.run_hardware --export runs/baseline/export --sim   # dry run in sim

python scripts/view.py --task squat                  # watch the pilot wearing the exo (unpowered)
python scripts/view.py --run runs/demo --task circuit --speed 0.5   # trained policy, ACE course
python scripts/exo_info.py                           # exo mass, fit, and motor sign report
```

Override any config value from the command line, e.g.
`--set agent.algo=ddpg --set exo.alpha=0.5 --set tasks.data_dir=data/motions`.

`runs/demo/` contains a short example run **with the MK1 exo mounted** (75k steps, about
12 minutes on one CPU core), so `evaluate.py`, `view.py`, `export_policy.py`, and the deploy
dry run work immediately. It is a pipeline check, not a result: a real run needs ~1M steps.
Its evaluation (exo worn in every case):

| task | powered vs unpowered exo | net vs not wearing the exo (ACE) |
|---|---|---|
| squat | +30.6% | **+4.7%** |
| quick steps | +12.1% | −16.2% |
| high knees | +7.8% | −9.7% |
| walk | +5.4% | −9.2% |
| backward walk | +3.4% | −16.5% |

The unpowered exo costs 15–38% extra effort, so the policy must repay that before the pilot
comes out ahead. The policy also sits at the 40 Nm cap most of the time.

## Project layout

```
configs/default.yaml        every tunable in one place (copied into each run)
exo_rl/sim/
  model_builder.py          planar human + exo MJCF, scaled to height/mass (Winter ratios)
  reference.py              task motions: synthetic placeholders, CSV loader, circuits
  human.py                  active human model (the effort signal comes from here)
exo_rl/control/             SHARED with hardware; never import sim code here
  torque_pipeline.py        feedforward, blending, direction limits, saturation, rate limit
  observation.py            sensor scaling, history window, gait-phase estimate
exo_rl/envs/
  exo_env.py                Gymnasium env tying it together
  rewards.py                effort-based reward, every term logged
  randomization.py          subject, payload, friction, latency, noise, human adaptation
exo_rl/agents/td3.py        TD3 / DDPG + replay buffer
exo_rl/evaluation.py        effort-with-vs-without-exo metrics
exo_rl/sim/exo_hardware.py  mounts the CAD exo (URDF) on the human; motor sign derivation
configs/exo_mk1.yaml        your MK1 FB900 exo: URDF, left/right mapping, knee alignment, zeros
assets/exo_mk1/             MuJoCo-ready URDF + meshes (hi: viewing, lo: training)
scripts/                    train, evaluate, export, view (live viewer / video), exo_info
deploy/                     controller (same pipeline), motor map, hardware loop skeleton, safety monitor
data/motions/               put recorded CSVs here (format in its README)
tests/                      physics, reward sanity, pipeline, sim/deploy consistency, agent
```

## How this maps to the project TODOs

| TODO | Where | Status |
|---|---|---|
| 1. Effort-based reward | `envs/rewards.py` | Done. The old `r_torque` term is removed. A test checks that helpful torque lowers effort and opposing torque raises it. |
| 2. Active human model | `sim/human.py` | Level 1 (joint torques) done. Musculoskeletal upgrade is future work. |
| 3. Train/deploy consistency | `control/`, `deploy/controller.py` | Done. `test_consistency.py` checks that the env and deploy controller give identical torques from identical sensors. |
| 4. Task context | `control/observation.py`, `tasks.circuit_prob` | History window, phase estimate, and chained-task episodes are in place. |
| 5. Payload | `sim/model_builder.py`, `payload:` in config | 11.34 kg mounted on the trunk (suit), randomized. Set `offset` to your real mount point. The CAD exo itself (19.1 kg) is mounted as well. |
| 6. Data for all ACE tasks | `data/motions/` | Loader is ready. **Currently synthetic placeholders; needs your recordings.** |
| 7. Domain randomization | `envs/randomization.py` | Done. Center the ranges on hardware system ID. |
| 8. Report cleanup | — | Not code. |

## Design decisions you should know about

**The human model.** Each human joint gets torque

τ_h = M·q̈_ref + bias + PD + I − β·τ_exo

The inertial and gravity feedforward removes tracking lag, and the integral term carries
quasi-static loads such as a squat. β ("exo adaptation") says how much of the exo torque
the human offloads. It is randomized in the range 0.6–1.0 so the policy can't count on a
perfectly cooperative user. Effort is `mean((τ_h/τ_max)²)` over all six human joints, so
shifting load onto the ankle doesn't count as help.

**Residual helpers.** A planar leg model has no balance strategy, so two "residual"
forces stand in for one:

- a forward-velocity helper (toward the speed implied by the reference kinematics)
- a trunk-pitch helper

Vertical support is *not* assisted: the legs carry body weight plus payload. Residuals are
logged separately and are not counted as effort.

**Read effort as relative, not absolute.** Absolute human torques run higher than real
biomechanics, because a PD-tracking human co-contracts more than a person does. Compare
exo-on vs exo-off, which is also how ACE measures.

**Overground, normal friction.** An early treadmill-style low-friction floor made the feet
splay in double support. With real friction, the body walks overground and starts with the
reference's implied speed.

**Synthetic references.** Walking and backward walking are generated by solving leg
geometry so the stance foot stays on the floor and the swing foot clears it. Squats keep
the feet flat. High knees and quick steps are hand-shaped. All of them are placeholders:
**replace them with recordings of your pilots** (TODO 6). Real data is kinematically
consistent by nature and will make the effort numbers more meaningful.

**Feedforward baseline.** `exo.feedforward: limb_gravity` computes leg-gravity
compensation from *measured* angles only. The default `alpha: 0` means pure RL. Set
`alpha > 0` to learn a residual on top of it, and evaluate the `feedforward` mode as a
baseline.

**Soft-suit cables.** If an actuator can only pull, set its `exo.direction` entry to
`flexion_only` or `extension_only`. The same constraint then applies in training and on
hardware.

## Your exoskeleton (MK1 FB900)

`exo_hardware.enabled: true` (the default) mounts the CAD exo from `configs/exo_mk1.yaml` on
the human. Each link keeps its CAD mass, center of mass, full inertia tensor, and mesh:
the back plate and hip chassis ride on the pelvis, the thigh and shank links on the human
thigh and shank. The exo hip-flexion axes sit on the human hips. The passive hip
ab/adduction and rotation joints and the width sliders are locked, since the model is
planar. Exo torque is applied at the human hip/knee joints (rigid coupling).

What was changed from the SolidWorks export (`assets/exo_mk1/`):
- **Meshes decimated.** The original back-plate mesh (221k faces) exceeded MuJoCo's 200k limit.
- **Back plate COM corrected.** The exporter wrote it rotated 90° about x, about 2.4 m from
  the part.

Things the model makes explicit, which you should check on the hardware:
- **Thigh length.** The exo's hip→knee axis distance is 0.518 m; a 1.75 m pilot's thigh is
  about 0.43 m. With `knee_mode: align_knee` the sim puts the exo knee on the human knee
  (i.e. assumes the length is adjusted); `rigid` keeps the CAD geometry.
- **Left/right naming.** The CAD names are from the viewer's side: `Right_*` links are on
  the pilot's left. The loader checks the l/r assignment against the geometry.
- **Motor signs.** Signs are derived from the URDF axes: hip_l +1, knee_l −1, hip_r −1,
  knee_r +1. `tests/test_exo.py` verifies them physically, and they are exported in
  `deploy_config.json["motor_map"]` for `deploy/joint_map.py`. Set encoder zeros in
  `motor_zero_offsets`.
- **Torque limits.** The URDF's 10 Nm effort limit looks like an exporter placeholder.
  Training uses `exo.max_torque` (40 Nm); set it from your actuator datasheets.
- **Fixed assumptions.** The gravity feedforward (`exo.alpha > 0`) compensates the human
  *plus exo* leg weight, and exo masses are randomized ±5%.

To use a different exo, copy `configs/exo_mk1.yaml`, point it at your URDF and meshes (binary
STL, ≤200k faces), list which links ride on the pelvis, thigh, and shank, and run
`scripts/exo_info.py` to check the fit and signs. Set `exo_hardware.enabled: false` for the
human + payload only.

## Visualizing the simulation

```bash
python scripts/view.py --task squat --mode off                   # live MuJoCo window
python scripts/view.py --run runs/demo --task backward_walk --speed 0.5
python scripts/view.py --run runs/demo --task circuit --record circuit.mp4   # video, no window
python scripts/view.py --save-xml model.xml     # MJCF + meshes, for python -m mujoco.viewer
```

Extra test motions, not in the training task list (add them to `tasks.names` to train on them):

- **`high_knees_travel`**: high knees that move forward (~0.5 m/s), like the ACE course.
  Works with the exo worn.
- **`jog`**: grounded running (bent knees, fast cadence, one foot always down), ~0.7 m/s.
  True running with a flight phase is beyond this simplified human. The model can jog
  without the exo, but not while wearing it unpowered: the pilot's hips hit their strength
  limit. Training on `jog` would show whether the powered exo can make it possible.

`--camera fixed` gives a static side view, so you can see the pilot travel across the floor:

```bash
python scripts/view.py --task high_knees_travel --camera fixed
python scripts/view.py --task jog --no-exo --camera fixed
```

Lateral motions (e.g. a crab/lateral squat walk) need hip abduction and sideways pelvis
motion, which this sagittal-plane model doesn't have yet.

On macOS the live window needs `mjpython scripts/view.py ...`. For headless recording on
Linux, set `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU, `apt install libosmesa6`).
The terminal prints the task, phase, human effort, and exo torques once per second.

## Connecting the real exoskeleton

1. Implement `HardwareInterface` in `deploy/run_hardware.py` (or wrap it in a ROS 2 node).
   `read_sensors()` must return the dict documented there, in the **human convention**; use
   `deploy.joint_map.MotorMap` to convert encoder angles and torque commands. Check each
   sensor with the exo unpowered.
2. Match `sim.control_dt` to your real loop rate, set `exo.max_torque` and
   `exo.rate_limit` to your actuators, and set `payload.offset` to the mount location.
3. Run system ID (latency, torque tracking, sensor noise) and center the `randomization`
   ranges on the results before training the policy you'll deploy.
4. Export with the pilot's height and mass, dry-run with `--sim`, then start on hardware
   at reduced `max_torque`.
5. The Python `SafetyMonitor` is a second layer only. Firmware torque limits and a
   hardware e-stop are required.

## Training notes

- Throughput is about 1,000 env steps/s without learning and about 115 steps/s with TD3
  updates on one CPU core. A GPU mainly speeds up the updates.
- Watch `episodes.csv` per reward term. If `mean_effort` doesn't fall while `mean_magnitude`
  rises, the exo is pushing without helping.
- `eval.csv` reports per-task effort reduction against exo-off. `best.pt` is chosen by mean
  reduction minus a failure penalty.
- To compare DDPG vs TD3, run the same seed with `--set agent.algo=ddpg`.

## Known limitations and next steps

- **Planar (sagittal) model.** There is no frontal-plane balance, hip ab/adduction, or
  rotation. That's fine for hip/knee flexion assistance, but a 3D model is needed for
  lateral effects.
- **Human fidelity.** Upgrade to a musculoskeletal model (e.g. MyoSuite) and use muscle
  activation² as the effort term, which is closer to metabolic cost. The env and reward
  interfaces stay the same.
- **Rigid coupling.** Exo torque acts directly on the human joints, and exo links are rigidly
  fixed to the body segments. Add compliant strap bodies to capture misalignment (e.g. the
  9 cm knee offset) and interaction forces.
- **Transitions.** The planar human can't instantly reverse from backward to forward walking
  while carrying the exo, so that pair is excluded (`tasks.forbidden_transitions`).
  `tasks.circuit_order: ace` trains on the exact ACE course order.
- **Scale-up.** If training speed becomes the bottleneck, consider MJX / MuJoCo Warp for
  GPU-parallel environments. With many parallel environments, PPO may then be worth
  trying.
- **Phase estimator.** The phase-portrait estimator assumes cyclic motion. Validate it on
  recorded transitions.
