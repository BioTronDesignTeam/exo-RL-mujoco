"""Summarize a CAD exo spec: masses, fit on a pilot, motor sign mapping, optional render.

    python scripts/exo_info.py                               # configs/exo_mk1.yaml, nominal pilot
    python scripts/exo_info.py --height 1.82 --mass 80
    MUJOCO_GL=egl python scripts/exo_info.py --render exo_fit.png   # headless render
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exo_rl.sim.exo_hardware import ExoHardware  # noqa: E402
from exo_rl.sim.model_builder import HumanExoModel, SubjectParams  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/exo_mk1.yaml")
    ap.add_argument("--height", type=float, default=1.75)
    ap.add_argument("--mass", type=float, default=75.0)
    ap.add_argument("--render", help="save a front/side/3-4 image of the pilot wearing the exo")
    args = ap.parse_args()

    exo = ExoHardware(args.spec)
    hm = HumanExoModel(SubjectParams(height=args.height, mass=args.mass, exo=exo, exo_quality="hi"))
    print(exo.report(hm.segments.thigh_len))
    total = hm.model.body_mass.sum()
    print(f"\nPilot {args.height} m / {args.mass} kg: model total {total:.1f} kg "
          f"(pilot {args.mass:.1f} + exo {exo.total_mass:.1f} + payload {total - args.mass - exo.total_mass:.1f})")
    print(f"Exo is {100 * exo.total_mass / args.mass:.0f}% of pilot mass; "
          f"{100 * 2 * (exo.group_mass('thigh', 'l') + exo.group_mass('shank', 'l')) / exo.total_mass:.0f}% of it is on the legs.")
    print("\nMotor map (human + flexion = sign * motor rotation):", exo.motor_map())

    if args.render:
        import imageio
        import mujoco
        m, d = hm.model, hm.data
        m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, 480), max(m.vis.global_.offheight, 360)
        r = mujoco.Renderer(m, 480, 360)
        cam = mujoco.MjvCamera()
        cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, m.body("pelvis").id
        cam.distance, cam.elevation = 2.4, -8
        imgs = []
        for az in (180, 90, 135):
            cam.azimuth = az
            r.update_scene(d, cam)
            imgs.append(r.render())
        imageio.imwrite(args.render, np.concatenate(imgs, axis=1))
        print(f"saved {args.render}")


if __name__ == "__main__":
    main()
