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

`gatech_ab01/` holds recorded squat, high knees, backward walk and walk prepared this way and
checked on the MyoLeg human; use it with `--set human.model=myo --set tasks.data_dir=data/motions/gatech_ab01`.

`scripts/prepare_motions.py` (logic in `exo_rl/motion_data.py`) turns a long recording from the
collection in `data/motion_dataset/` (git tracks only the Georgia Tech trials it reads; the rest of
the 1.6 GB is not all redistributable and stays local) into such a file: it averages the repeats, smooths and loops the cycle, keeps it inside the human model's joint
range, and checks that the human model can perform it (keeping the feet flat if the recorded ankles
fail). The tips below are what it does.

Tips:
- Average several cycles and make the first and last rows match, or you'll get a jump.
- Low-pass filter (~6 Hz) before exporting; the human model differentiates twice.
- Sample at 500 Hz or more. The table is interpolated linearly and differentiated over 2 ms, so at
  100–200 Hz most of the acceleration fed to the human model is interpolation noise.
- Stay inside the human model's joint range (MyoLeg: knee 0–120, ankle −40–30, hip −30–120 deg).
- Use joint angles from IMU/mocap inverse kinematics (e.g. OpenSim IK or AddBiomechanics).
- Record the pilot's height and mass alongside each file.
