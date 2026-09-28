# Recorded motion data

Put one CSV per task here, named `<task>.csv` (e.g. `squat.csv`, `high_knees.csv`,
`quick_steps.csv`, `backward_walk.csv`), then set `tasks.data_dir: data/motions` in your
config. Any task without a CSV falls back to its synthetic placeholder.

Each file holds **one full cycle**, which is looped:

| column | unit | sign |
|---|---|---|
| `time` | s | starts anywhere; the span is one period |
| `hip_l`, `hip_r` | deg | + = flexion |
| `knee_l`, `knee_r` | deg | + = flexion |
| `ankle_l`, `ankle_r` | deg | + = dorsiflexion |
| `trunk` (optional) | deg | + = forward lean |

`example_walk.csv` shows the format (it is the synthetic walk, not real data).

Tips:
- Average several cycles and make the first and last rows match, or you'll get a jump.
- Low-pass filter (~6 Hz) before exporting; the human model differentiates twice.
- Use joint angles from IMU/mocap inverse kinematics (e.g. OpenSim IK or AddBiomechanics).
- Record the pilot's height and mass alongside each file.
