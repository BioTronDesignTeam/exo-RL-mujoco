"""Export a trained policy for the real exoskeleton.

    python scripts/export_policy.py --run runs/demo --height 1.78 --mass 72

Writes runs/<name>/export/:
    policy.pt            TorchScript actor (obs -> action in [-1, 1]); no training code needed
    deploy_config.json   everything the hardware loop needs to rebuild the SAME observation
                         builder and torque pipeline used in training, plus the pilot's limb
                         parameters for the feedforward term.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exo_rl.agents.td3 import TD3Agent  # noqa: E402
from exo_rl.control.torque_pipeline import LimbParams  # noqa: E402
from exo_rl.sim.exo_hardware import ExoHardware  # noqa: E402
from exo_rl.sim.model_builder import EXO_JOINTS, SubjectParams, segment_info  # noqa: E402
from exo_rl.utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--height", type=float, default=None, help="pilot height (m)")
    ap.add_argument("--mass", type=float, default=None, help="pilot mass (kg)")
    args = ap.parse_args()

    cfg = load_config(os.path.join(args.run, "config.yaml"))
    agent = TD3Agent.load(os.path.join(args.run, "checkpoints", args.ckpt))
    out = os.path.join(args.run, "export")
    os.makedirs(out, exist_ok=True)

    actor = agent.actor.cpu().eval()
    example = torch.zeros(1, agent.obs_dim)
    scripted = torch.jit.trace(actor, example)
    scripted.save(os.path.join(out, "policy.pt"))

    height = args.height or cfg["subject"]["height"]
    mass = args.mass or cfg["subject"]["mass"]
    seg = segment_info(SubjectParams(height=height, mass=mass))
    hw_cfg = cfg.get("exo_hardware") or {}
    exo = ExoHardware(hw_cfg["spec"]) if hw_cfg.get("enabled") else None
    limb = LimbParams(**exo.limb_params(seg)) if exo else LimbParams.from_segment_info(seg)
    deploy = {
        "control_dt": cfg["sim"]["control_dt"],
        "exo_joints": EXO_JOINTS,
        "observation": cfg["observation"],
        "exo": cfg["exo"],
        "limb": limb.to_dict(),
        "pilot": {"height": height, "mass": mass},
        "obs_dim": agent.obs_dim,
        "source": {"run": args.run, "ckpt": args.ckpt, "algo": agent.algo},
    }
    if exo is not None:
        deploy["motor_map"] = exo.motor_map()
        deploy["exo_hardware"] = {"spec": hw_cfg["spec"], "total_mass_kg": exo.total_mass}
        print(exo.report(seg.thigh_len))
    with open(os.path.join(out, "deploy_config.json"), "w") as f:
        json.dump(deploy, f, indent=2)
    print(f"Exported to {out} (pilot {height} m, {mass} kg)")
    if args.height is None or args.mass is None:
        print("NOTE: using nominal subject size; pass --height/--mass for the actual pilot.")


if __name__ == "__main__":
    main()
