"""Muscle-driven human for the MyoLeg model (static optimization).

The musculoskeletal counterpart of human.py. Each physics step:

    tau_needed = M(q)·qdd_ref + bias(q, qd) - f_ground + PID
    tau_human  = tau_needed - beta * tau_exo               (exo joints only)
    a          = argmin ||W (A a + tau_passive - tau_human)||^2 + reg * ||a||^2,   0 <= a <= 1

Unlike human.py, f_ground (the joint torques from the last step's contact and constraint forces,
what the body feels through the feet) is part of the inverse dynamics: the muscles carry body
weight from the start instead of waiting for the integral term, which the saturating muscles
can't afford. W weights the six tracked joints above the held ones.

The six joints of the planar model (hip flexion, knee, ankle per side) follow the reference;
hip adduction/rotation, subtalar, and MTP are held at 0. A = moment arms x active muscle force
at the current fiber length and velocity, tau_passive = passive muscle force; both come from
MuJoCo's muscle model. MyoLeg's knee is a coupled mechanism: the tibia translations and the
patella follow the knee angle through equality constraints, and the quadriceps act on the knee
only through the patella. Everything is therefore projected onto the controlled joints with the
coupling Jacobian C = dq / dq_controlled: torques C^T tau, accelerations C qdd.
Activations follow MuJoCo's first-order activation dynamics (10 ms rise, 40 ms fall): the
excitation is chosen to reach the solved activation in one step, clipped to [0, 1]. Torque the muscles can't produce is simply missing: a
weak or badly placed muscle set shows up as tracking error, not as hidden help.

Effort = mean(a^2) over the 80 muscles, the usual static-optimization cost.

Balance: the reference angles were made for the planar body; on MyoLeg they put the whole-body
COM 4-6 cm behind the feet. update_balance() slowly (1 s) shifts the posture until the COM sits
over the middle of the foot contacts: forward by ankle dorsiflexion with equal hip extension,
sideways by opposite hip adduction. Tracking is against the shifted reference.

Root residuals (NOT human effort), as in human.py: a forward-velocity helper and a trunk
orientation helper (pitch to the reference, roll and yaw to 0) standing in for the balance
strategy a leg-only model lacks, plus lateral velocity damping; with com_damping the velocity
helpers act on the COM, so a squat's hip shift isn't resisted. Vertical is unactuated, so the
muscles carry body weight + payload. Helper loads match the planar model's (squat ~50 Nm pitch,
high knees / quick steps ~200 Nm).
"""
from __future__ import annotations

import mujoco
import numpy as np

from .model_builder import HUMAN_JOINTS

TRACKED = {"hip_l": "hip_flexion_l", "knee_l": "knee_angle_l", "ankle_l": "ankle_angle_l",
           "hip_r": "hip_flexion_r", "knee_r": "knee_angle_r", "ankle_r": "ankle_angle_r"}
HELD = [f"{j}_{s}" for s in "lr" for j in ("hip_adduction", "hip_rotation", "subtalar_angle", "mtp_angle")]
EXO_IN_TRACKED = [HUMAN_JOINTS.index(j) for j in ("hip_l", "knee_l", "hip_r", "knee_r")]
DEFAULTS = {"hold_kp": 400.0, "hold_kd": 40.0, "held_weight": 0.3, "activation_reg": 30.0, "solver_iters": 60,
            "activation_dynamics": True, "ground_filter_s": 0.02,
            "balance": True, "com_damping": True, "balance_tau_s": 1.0, "support_tau_s": 0.2, "balance_max_deg": [15.0, 10.0],
            "residual_roll_kp": 2000.0, "residual_roll_kd": 200.0, "residual_y_kv": 500.0}


def solve_activations(A: np.ndarray, b: np.ndarray, reg: float, a0: np.ndarray, iters: int) -> np.ndarray:
    """min ||A a - b||^2 + reg ||a||^2 with 0 <= a <= 1 (accelerated projected gradient, warm-started)."""
    H = A.T @ A + reg * np.eye(A.shape[1])
    g = A.T @ b
    step = 1.0 / np.abs(H).sum(axis=1).max()        # Gershgorin bound on the largest eigenvalue
    x = y = a0
    t = 1.0
    for _ in range(iters):
        x_new = np.clip(y - step * (H @ y - g), 0.0, 1.0)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


class MuscleHumanController:
    def __init__(self, model: mujoco.MjModel, cfg: dict, subject_mass: float):
        self.model = model
        c = {**DEFAULTS, **cfg.get("muscle", {})}
        s = subject_mass / 75.0
        per_joint = lambda d: np.array([d[j.split("_")[0]] for j in HUMAN_JOINTS], dtype=float)  # noqa: E731
        n_held = len(HELD)
        self.kp0 = np.concatenate([per_joint(cfg["kp"]), np.full(n_held, c["hold_kp"])]) * s
        self.kd0 = np.concatenate([per_joint(cfg["kd"]), np.full(n_held, c["hold_kd"])]) * s
        self.ki0 = np.concatenate([per_joint(cfg["ki"]), np.zeros(n_held)]) * s
        self.i_limit = cfg.get("integral_limit_frac", 0.5) * np.concatenate(
            [per_joint(cfg["max_torque"]), np.zeros(n_held)]) * s
        self.residual = {**cfg["residual"], **{k: c[k] for k in ("residual_roll_kp", "residual_roll_kd",
                                                                 "residual_y_kv")}}
        self.mass_scale = s
        self.reg, self.iters = c["activation_reg"], int(c["solver_iters"])
        self.act_dyn, self.ground_tc = bool(c["activation_dynamics"]), float(c["ground_filter_s"])
        self.row_w = np.concatenate([np.ones(len(HUMAN_JOINTS)), np.full(n_held, c["held_weight"])])

        joints = [model.joint(TRACKED[j]) for j in HUMAN_JOINTS] + [model.joint(n) for n in HELD]
        self.qadr = np.array([model.jnt_qposadr[j.id] for j in joints])
        self.dof = np.array([model.jnt_dofadr[j.id] for j in joints])
        self.q_lo, self.q_hi = model.jnt_range[[j.id for j in joints[:len(TRACKED)]]].T
        root = model.joint("root")
        self.root_q, self.root_v = model.jnt_qposadr[root.id], model.jnt_dofadr[root.id]
        self.root_body = model.jnt_bodyid[root.id]
        self.balance, self.bal_tau, self.sup_tau = bool(c["balance"]), c["balance_tau_s"], c["support_tau_s"]
        self.bal_max = np.radians(c["balance_max_deg"])
        self.com_damping = bool(c["com_damping"])
        self.feet = np.array([i for i in range(model.ngeom)
                              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(("foot", "bofoot"))])
        self.ankles = [model.joint(f"ankle_angle_{s}").id for s in "lr"]

        sign = {"ankle": 1.0, "hip": -1.0}
        self.sagittal = np.array([sign.get(j.split("_")[0], 0.0) for j in HUMAN_JOINTS])
        self.lateral = np.array([{"hip_adduction_l": 1.0, "hip_adduction_r": -1.0}.get(n, 0.0) for n in HELD])
        self.muscles = np.where(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
        self.act_adr = model.actuator_actadr[self.muscles]
        self.tau_act, self.tau_deact = model.actuator_dynprm[self.muscles, 0], model.actuator_dynprm[self.muscles, 1]
        self._moment = np.zeros((model.nu, model.nv))
        # Joint couplings q_dep - q_dep0 = poly(q_ind - q_ind0) whose independent joint is controlled.
        col = {int(model.dof_jntid[k]): i for i, k in enumerate(self.dof)}
        self.couplings = [(int(model.jnt_dofadr[model.eq_obj1id[e]]), col[int(model.eq_obj2id[e])],
                           int(model.jnt_qposadr[model.eq_obj2id[e]]), model.eq_data[e][:5].copy())
                          for e in range(model.neq)
                          if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and int(model.eq_obj2id[e]) in col]
        self.set_randomization(gain_scale=1.0, beta=cfg.get("exo_adaptation", 1.0))
        self.reset()

    def set_randomization(self, gain_scale: float, beta: float):
        self.kp, self.kd, self.ki = self.kp0 * gain_scale, self.kd0 * gain_scale, self.ki0 * gain_scale
        self.beta = beta

    def reset(self):
        self.integral = np.zeros(len(self.dof))
        self.a = np.zeros(len(self.muscles))
        self.f_ground = None
        self.support = None
        self.lean = np.zeros(2)              # balance offsets: forward (ankle/hip), sideways (hip adduction), rad

    def coupling(self, d: mujoco.MjData) -> np.ndarray:
        """C = dq / dq_controlled (nv x controlled joints)."""
        C = np.zeros((self.model.nv, len(self.dof)))
        C[self.dof, np.arange(len(self.dof))] = 1.0
        for dep, i, ind_q, c in self.couplings:
            x = d.qpos[ind_q] - self.model.qpos0[ind_q]
            C[dep, i] = c[1] + 2 * c[2] * x + 3 * c[3] * x ** 2 + 4 * c[4] * x ** 3
        return C

    def muscle_torque_model(self, d: mujoco.MjData, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(A, tau_passive) on the controlled joints: tau = A @ a + tau_passive."""
        m = self.model
        mujoco.mju_sparse2dense(self._moment, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
        act = d.act.copy()
        d.act[self.act_adr] = 0.0
        mujoco.mj_fwdActuation(m, d)
        f0 = d.actuator_force[self.muscles].copy()
        d.act[self.act_adr] = 1.0
        mujoco.mj_fwdActuation(m, d)
        gain = d.actuator_force[self.muscles] - f0
        d.act[:] = act
        mujoco.mj_fwdActuation(m, d)
        R = self._moment[self.muscles] @ C                         # muscles x controlled joints
        return R.T * gain, R.T @ f0

    def update_balance(self, d: mujoco.MjData, dt: float):
        """Slowly shifts the posture so the whole-body COM sits over the middle of the foot contacts.
        Forward: ankle dorsiflexion with equal hip extension (shanks and thighs tilt, pelvis keeps its
        pitch). Sideways: opposite hip adduction on the two legs (pelvis slides over the feet)."""
        con = d.contact
        on_floor = np.isin(con.geom1[:d.ncon], self.feet) | np.isin(con.geom2[:d.ncon], self.feet)
        if on_floor.any():
            sup = con.pos[:d.ncon][on_floor, :2].mean(axis=0)
            self.support = sup if self.support is None else self.support + min(dt / self.sup_tau, 1.0) * (sup - self.support)
        if self.support is None:
            return
        R = d.xmat[self.root_body].reshape(3, 3)
        heading = R[:2, 0] / max(np.linalg.norm(R[:2, 0]), 1e-9)
        gap = self.support - d.subtree_com[self.root_body][:2]
        gap = np.array([gap @ heading, gap @ np.array([-heading[1], heading[0]])])   # forward, left
        height = max(d.subtree_com[self.root_body][2] - np.mean(d.xanchor[self.ankles, 2]), 0.3)
        self.lean = np.clip(self.lean + (dt / self.bal_tau) * gap / height, -self.bal_max, self.bal_max)

    def compute(self, d: mujoco.MjData, q_ref, qd_ref, qdd_ref, trunk_ref, trunk_d_ref, v_ref: float,
                tau_exo: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Sets the muscle activations and the root residual (qfrc_applied).
        tau_exo is in exo order (hip_l, knee_l, hip_r, knee_r). Returns (activations, tau_muscle on the
        six tracked joints in HUMAN_JOINTS order)."""
        m = self.model
        n = len(TRACKED)
        if self.balance:
            self.update_balance(d, dt)
        q_bal = np.asarray(q_ref) + self.lean[0] * self.sagittal
        q_des = np.concatenate([np.clip(q_bal, self.q_lo, self.q_hi), self.lean[1] * self.lateral])
        self.q_target = q_des[:n].copy()                         # what the six joints track (for logging)
        qd_des = np.concatenate([qd_ref, np.zeros(len(HELD))])
        e, ed = q_des - d.qpos[self.qadr], qd_des - d.qvel[self.dof]
        lim = self.i_limit / np.maximum(self.ki, 1e-9)
        self.integral = np.clip(self.integral + e * dt, -lim, lim)

        C = self.coupling(d)
        qdd = C[:, :n] @ qdd_ref
        m_qdd = np.zeros(m.nv)
        mujoco.mj_mulM(m, d, m_qdd, qdd)
        raw = d.qfrc_constraint + d.qfrc_passive
        if self.f_ground is None or self.ground_tc <= 0:
            self.f_ground = raw
        else:
            self.f_ground = self.f_ground + min(dt / self.ground_tc, 1.0) * (raw - self.f_ground)
        tau_needed = C.T @ (m_qdd + d.qfrc_bias - self.f_ground) + self.kp * e + self.kd * ed + self.ki * self.integral
        self.tau_needed = tau_needed[EXO_IN_TRACKED].copy()     # read-only export, exo order
        tau_h = tau_needed.copy()
        tau_h[EXO_IN_TRACKED] -= self.beta * np.asarray(tau_exo)

        A, tau_passive = self.muscle_torque_model(d, C)
        w = self.row_w[:, None]
        self.a = solve_activations(w * A, self.row_w * (tau_h - tau_passive), self.reg, self.a, self.iters)
        if self.act_dyn:
            act = d.act[self.act_adr].copy()
            d.ctrl[self.muscles] = self.excitation(act, self.a, dt)
            self.a = act                                         # effort and torque use the real activation
        else:
            d.ctrl[self.muscles] = self.a
            d.act[self.act_adr] = self.a
        tau_muscle = A @ self.a + tau_passive

        r, s = self.residual, self.mass_scale
        v = d.qvel[self.root_v:self.root_v + 6].copy()
        if self.balance and self.com_damping:                  # damp the COM, not the pelvis
            mujoco.mj_subtreeVel(m, d)
            v[:2] = d.subtree_linvel[self.root_body][:2]
        q_des_root = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_des_root, np.array([0.0, 1.0, 0.0]), trunk_ref)
        rot_err = np.zeros(3)
        mujoco.mju_subQuat(rot_err, q_des_root, d.qpos[self.root_q + 3:self.root_q + 7])
        w_des = np.array([0.0, trunk_d_ref, 0.0])
        kp = np.array([r["residual_roll_kp"], r["pitch_kp"], r["residual_roll_kp"]])
        kd = np.array([r["residual_roll_kd"], r["pitch_kd"], r["residual_roll_kd"]])
        self.root_residual = s * np.concatenate([
            [r["x_kv"] * (v_ref - v[0]), r["residual_y_kv"] * -v[1], 0.0], kp * rot_err + kd * (w_des - v[3:])])
        d.qfrc_applied[self.root_v:self.root_v + 6] = self.root_residual
        return self.a, tau_muscle[:n]

    def excitation(self, act: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        """Excitation that brings each activation to target within one step under MuJoCo's first-order
        muscle activation dynamics (rise tau_act*(0.5+1.5a), fall tau_deact/(0.5+1.5a)); clipped to
        [0, 1], so activations still can't change faster than the muscle allows."""
        rising = target > act
        tau = np.where(rising, self.tau_act * (0.5 + 1.5 * act), self.tau_deact / (0.5 + 1.5 * act))
        return np.clip(act + tau * (target - act) / dt, 0.0, 1.0)

    @staticmethod
    def effort(a: np.ndarray) -> float:
        return float(np.mean(a ** 2))


def satisfy_couplings(model: mujoco.MjModel, d: mujoco.MjData):
    """Moves joint-coupled dofs (knee translations, patella) onto their polynomial of the driving joint."""
    for e in range(model.neq):
        if model.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT or model.eq_obj2id[e] < 0:
            continue
        dep, ind = model.jnt_qposadr[model.eq_obj1id[e]], model.jnt_qposadr[model.eq_obj2id[e]]
        x = d.qpos[ind] - model.qpos0[ind]
        d.qpos[dep] = model.qpos0[dep] + np.polyval(model.eq_data[e][4::-1], x)


def stand_on_floor(model: mujoco.MjModel, d: mujoco.MjData, q: np.ndarray, trunk: float):
    """Puts the tracked joints at q (HUMAN_JOINTS order), pitches the trunk, and lowers the body until
    the lowest foot touches the floor."""
    mujoco.mj_resetData(model, d)
    for j, qi in zip(HUMAN_JOINTS, q):
        jid = model.joint(TRACKED[j]).id
        d.qpos[model.jnt_qposadr[jid]] = np.clip(qi, *model.jnt_range[jid])
    satisfy_couplings(model, d)
    adr = model.jnt_qposadr[model.joint("root").id]
    mujoco.mju_axisAngle2Quat(d.qpos[adr + 3:adr + 7], np.array([0.0, 1.0, 0.0]), trunk)
    mujoco.mj_forward(model, d)
    floor = model.geom("floor").id
    feet = [i for i in range(model.ngeom) if i != floor and model.geom_contype[i]
            and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i).startswith(("foot", "bofoot"))]
    gap = min(mujoco.mj_geomDistance(model, d, g, floor, 10.0, None) for g in feet)
    d.qpos[adr + 2] -= gap
    mujoco.mj_forward(model, d)
