"""Real-time control loop for the physical exoskeleton.

    python -m deploy.run_hardware --export runs/demo/export --sim     # dry run against the simulator
    python -m deploy.run_hardware --export runs/demo/export           # real hardware (implement driver first)

To connect your hardware, implement HardwareInterface for your motor drivers / IMUs
(or a ROS 2 node that publishes/subscribes the same fields). read_sensors() must return
the same dict layout as HumanExoModel.sensors():

    q_exo[4], qd_exo[4]      rad, rad/s   order hip_l, knee_l, hip_r, knee_r; + = flexion
    trunk_pitch              rad          + = forward lean (from pelvis IMU orientation filter)
    gyro_pelvis/thigh_l/thigh_r[3]  rad/s  in the IMU frame (x forward, z up when standing)
    acc_pelvis/thigh_l/thigh_r[3]   m/s^2  specific force, same frames

Sign conventions and frames MUST match the simulator; verify them one sensor at a time
with the exo unpowered before closing the loop. Use deploy.joint_map.MotorMap (signs are
exported in deploy_config.json) to convert encoder readings to the human convention and
torque commands back to motor convention inside your HardwareInterface:

    mmap = MotorMap.from_config(ctrl.cfg)
    q_exo  = mmap.q_to_human(encoders)        # in read_sensors()
    motor_cmd = mmap.tau_to_motor(tau)        # in send_torque()

Safety: the checks here are a second layer on top of the pipeline's limits. The motor
drivers/firmware must enforce their own torque limits and a hardware e-stop.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from deploy.controller import DeployController


class HardwareInterface:
    """Implement these four methods for your exoskeleton."""

    def connect(self):
        raise NotImplementedError

    def read_sensors(self) -> dict:
        raise NotImplementedError

    def send_torque(self, tau: np.ndarray):
        raise NotImplementedError

    def disable(self):
        """Zero torque / power stage off. Must be safe to call at any time."""
        raise NotImplementedError


class SimHardware(HardwareInterface):
    """Stands in for the real exo using the training simulator (no randomization)."""

    SCALE = 1000.0

    def __init__(self, task: str = "squat"):
        from exo_rl.envs.exo_env import ExoAssistEnv
        from exo_rl.utils import load_config
        cfg = load_config("configs/default.yaml")
        cfg["randomization"]["enabled"] = False
        self.env = ExoAssistEnv(cfg, seed=0)
        self.task = task
        self.pending = np.zeros(4)

    def connect(self):
        self.env.reset(options={"task": self.task, "nominal": True})
        # Drive the env with torques instead of actions: make its own pipeline a pass-through
        # (action = tau / SCALE), since the deploy controller already applied the real pipeline.
        p = self.env.pipeline
        p.alpha, p.action_scale = 0.0, np.full(4, self.SCALE)
        p.tau_max, p.rate_limit, p.directions = np.full(4, 1e6), 1e12, ["both"] * 4

    def read_sensors(self) -> dict:
        return self.env.sensors

    def send_torque(self, tau):
        self.env.step(np.asarray(tau) / self.SCALE)

    def disable(self):
        pass


class SafetyMonitor:
    def __init__(self, deploy_cfg: dict, joint_limits_deg=((-30, 120), (-5, 130)) * 2,
                 max_velocity=12.0):
        self.tau_max = np.asarray(deploy_cfg["exo"]["max_torque"], dtype=float)
        self.limits = np.radians(np.asarray(joint_limits_deg, dtype=float))
        self.max_velocity = max_velocity

    def check(self, s: dict, tau: np.ndarray) -> str:
        q, qd = np.asarray(s["q_exo"]), np.asarray(s["qd_exo"])
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(tau))):
            return "non-finite value"
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            return f"joint limit exceeded: {np.degrees(q).round(1)}"
        if np.any(np.abs(qd) > self.max_velocity):
            return f"joint velocity too high: {qd.round(1)}"
        if np.any(np.abs(tau) > self.tau_max + 1e-6):
            return f"torque above limit: {tau.round(1)}"
        return ""


def run(hw: HardwareInterface, ctrl: DeployController, duration: float, realtime: bool = True):
    dt = float(ctrl.cfg["control_dt"])
    safety = SafetyMonitor(ctrl.cfg)
    hw.connect()
    ctrl.reset()
    overruns, n = 0, int(duration / dt)
    next_tick = time.perf_counter()
    try:
        for k in range(n):
            s = hw.read_sensors()
            tau = ctrl.step(s)
            fault = safety.check(s, tau)
            if fault:
                print(f"[safety] {fault} at tick {k}: disabling")
                break
            hw.send_torque(tau)
            if realtime:
                next_tick += dt
                slack = next_tick - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    overruns += 1
            if k % int(1.0 / dt) == 0:
                print(f"[run] t={k * dt:5.1f}s tau={np.round(tau, 1)}")
    finally:
        hw.disable()
    print(f"[run] finished; {overruns} deadline overruns out of {n} ticks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="folder from scripts/export_policy.py")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--sim", action="store_true", help="dry run against the simulator")
    ap.add_argument("--task", default="squat")
    args = ap.parse_args()
    ctrl = DeployController.from_export(args.export)
    if args.sim:
        hw = SimHardware(args.task)
    else:
        raise SystemExit("Implement HardwareInterface for your exo in deploy/run_hardware.py first.")
    run(hw, ctrl, args.duration, realtime=not args.sim)


if __name__ == "__main__":
    main()
