"""
FF Master Guardian — balance + choreography controller.

Balance architecture (closed-loop):
  * High-stiffness PD position control on all joints (near-rigid legs).
  * Vestibular reflex on the ankles: torso tilt + tilt-rate feedback from the
    onboard IMU (framequat sensor) — the classic human "ankle strategy".
  * Foot pads + boosted ankle/leg actuator ratings (model engineering, see
    assets/Master/ff_master_ultra.xml).

Choreography:
  * Time-indexed motion script (wave / head scan / salute / bow / standby).
  * Smoothstep-interpolated joint targets for the upper body; the legs always
    track the standing pose while the vestibular loop holds balance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import mujoco


# ---------------------------------------------------------------------------
# Standing pose — joints chosen so the feet are perfectly level:
#   hip_pitch + knee + ankle_pitch = 0   and   hip_roll + ankle_roll = 0
# ---------------------------------------------------------------------------
STAND_POSE: dict[str, float] = {
    "left_hip_pitch_joint": 0.0, "left_hip_roll_joint": 0.08, "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.35, "left_ankle_pitch_joint": -0.35, "left_ankle_roll_joint": -0.08,
    "right_hip_pitch_joint": 0.0, "right_hip_roll_joint": -0.08, "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.35, "right_ankle_pitch_joint": -0.35, "right_ankle_roll_joint": 0.08,
    "waist_yaw_joint": 0.0, "waist_pitch_joint": 0.0, "waist_roll_joint": 0.0,
    "left_shoulder_pitch_joint": 0.1, "left_shoulder_roll_joint": 0.2, "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.3, "left_wrist_yaw_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.1, "right_shoulder_roll_joint": -0.2, "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.3, "right_wrist_yaw_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_roll_joint": 0.0,
    "head_yaw_joint": 0.0, "head_pitch_joint": 0.0,
}

# PD gains by joint category: (kp, kd)
GAINS: dict[str, tuple[float, float]] = {
    "leg":   (2000.0, 80.0),    # hips + knees — near rigid
    "ankle": (1000.0, 50.0),    # high stiffness for balance authority
    "waist": (150.0, 10.0),
    "arm":   (60.0, 5.0),       # softer so choreography is smooth
    "head":  (40.0, 3.0),
}

# Vestibular reflex gains (ankle strategy)
VEST_PITCH_GAIN = 500.0   # N*m per rad of torso pitch tilt
VEST_PITCH_DAMP = 150.0   # N*m per rad/s of tilt rate
VEST_ROLL_SCALE = 0.6


def _category(joint: str) -> str:
    if "ankle" in joint:
        return "ankle"
    if "hip" in joint or "knee" in joint:
        return "leg"
    if "waist" in joint:
        return "waist"
    if "head" in joint:
        return "head"
    return "arm"


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if x <= edge0:
        return 0.0
    if x >= edge1:
        return 1.0
    t = (x - edge0) / (edge1 - edge0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class Motion:
    """A timed pose keyframe: joints -> target values."""
    t_start: float
    t_end: float
    targets: dict[str, float]

    def value(self, joint: str, t: float) -> float | None:
        if joint not in self.targets:
            return None
        s = smoothstep(self.t_start, self.t_end, t)
        return self.targets[joint] * s


@dataclass
class Controller:
    model: mujoco.MjModel
    data: mujoco.MjData

    names: list[str] = field(default_factory=list)
    qposadr: np.ndarray = field(default_factory=lambda: np.array([]))
    dofadr: np.ndarray = field(default_factory=lambda: np.array([]))
    ctrlid: np.ndarray = field(default_factory=lambda: np.array([]))
    kp: np.ndarray = field(default_factory=lambda: np.array([]))
    kd: np.ndarray = field(default_factory=lambda: np.array([]))
    _idx: dict[str, int] = field(default_factory=dict)

    imu_sens: int = -1
    imu_adr: int = 0
    prev_tp: float = 0.0
    prev_tr: float = 0.0

    motions: list[Motion] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    log_every: int = 20
    step_count: int = 0

    # ------------------------------------------------------------------
    def setup(self) -> None:
        qa, da, ci, kp, kd = [], [], [], [], []
        for name in STAND_POSE:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}")
            assert jid >= 0 and aid >= 0, f"missing joint/actuator for {name}"
            self.names.append(name)
            qa.append(self.model.jnt_qposadr[jid])
            da.append(self.model.jnt_dofadr[jid])
            ci.append(aid)
            g = GAINS[_category(name)]
            kp.append(g[0])
            kd.append(g[1])
            self._idx[name] = len(self.names) - 1
        self.qposadr = np.array(qa, dtype=np.int32)
        self.dofadr = np.array(da, dtype=np.int32)
        self.ctrlid = np.array(ci, dtype=np.int32)
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.imu_sens = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "body-orientation")
        self.imu_adr = self.model.sensor_adr[self.imu_sens]

    # ------------------------------------------------------------------
    def spawn_standing(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.qpos[2] = 0.70
        for name in STAND_POSE:
            self.data.qpos[self.qposadr[self._idx[name]]] = STAND_POSE[name]
        mujoco.mj_forward(self.model, self.data)
        # settle so the foot pads just touch the ground
        lf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        pad_bottom = self.data.xpos[lf][2] - 0.074 - 0.006
        self.data.qpos[2] -= pad_bottom
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    def torso_tilt(self) -> tuple[float, float]:
        q = self.data.sensordata[self.imu_adr:self.imu_adr + 4]
        z = np.array([2 * (q[0] * q[2] + q[1] * q[3]),
                      2 * (q[1] * q[2] - q[0] * q[3]),
                      q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2])
        tp = np.arctan2(z[0], z[2])      # pitch tilt (+ = lean forward)
        tr = np.arctan2(-z[1], z[2])     # roll tilt
        return tp, tr

    # ------------------------------------------------------------------
    def modulate(self, t: float, target: np.ndarray) -> None:
        """Hook for demo scripts to override targets per-timestep (no-op)."""

    # ------------------------------------------------------------------
    def step(self, t: float) -> None:
        target = np.array([STAND_POSE[n] for n in self.names])

        # --- choreography: override upper-body targets from active motions ---
        for name in self.names:
            if _category(name) == "leg" or _category(name) == "ankle":
                continue  # legs always hold the standing pose
            val = None
            for mo in self.motions:
                v = mo.value(name, t)
                if v is not None:
                    val = v  # last motion with this joint wins
            if val is not None:
                target[self._idx[name]] = val

        self.modulate(t, target)

        # --- PD position control ---
        cur = self.data.qpos[self.qposadr]
        vel = self.data.qvel[self.dofadr]
        cmd = self.kp * (target - cur) - self.kd * vel

        # --- vestibular ankle reflex ---
        tp, tr = self.torso_tilt()
        tpd = (tp - self.prev_tp) / self.model.opt.timestep
        trd = (tr - self.prev_tr) / self.model.opt.timestep
        self.prev_tp, self.prev_tr = tp, tr
        la, ra = self._idx["left_ankle_pitch_joint"], self._idx["right_ankle_pitch_joint"]
        lar, rar = self._idx["left_ankle_roll_joint"], self._idx["right_ankle_roll_joint"]
        pitch_cmd = VEST_PITCH_GAIN * tp + VEST_PITCH_DAMP * tpd
        roll_cmd = VEST_ROLL_SCALE * (VEST_PITCH_GAIN * tr + VEST_PITCH_DAMP * trd)
        cmd[la] += pitch_cmd
        cmd[ra] += pitch_cmd
        cmd[lar] -= roll_cmd
        cmd[rar] -= roll_cmd

        self.data.ctrl[self.ctrlid] = cmd

        # --- data logging (10 Hz) ---
        self.step_count += 1
        if self.step_count % self.log_every == 0:
            self._log(t)

    # ------------------------------------------------------------------
    def _log(self, t: float) -> None:
        pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        entry = {
            "time_s": round(t, 4),
            "pelvis_pos": self.data.xpos[pelvis].round(5).tolist(),
            "torso_tilt_deg": round(float(np.degrees(self.torso_tilt()[0])), 4),
            "joint_pos": {n: round(float(self.data.qpos[self.qposadr[self._idx[n]]]), 4) for n in self.names},
            "ankle_cmd": round(float(self.data.ctrl[self.ctrlid[self._idx["left_ankle_pitch_joint"]]]), 3),
            "ncon": int(self.data.ncon),
        }
        self.log.append(entry)


def build_motion_script() -> list[Motion]:
    """Time-indexed choreography (seconds)."""
    ms: list[Motion] = []

    def add(t0: float, t1: float, **targets):
        ms.append(Motion(t0, t1, targets))

    # Right-arm wave (2s ramp, then oscillation handled in run loop is complex —
    # we keep the arm raised; the demo script adds oscillation on top).
    add(2.0, 3.5, right_shoulder_pitch_joint=0.35, right_shoulder_roll_joint=-1.35,
        right_elbow_joint=-1.6, right_wrist_yaw_joint=-0.4)
    # Head scan
    add(5.0, 6.5, head_yaw_joint=0.32)
    add(6.5, 8.0, head_yaw_joint=-0.32)
    add(8.0, 9.0, head_yaw_joint=0.0)
    # Salute (right arm up)
    add(9.5, 11.0, right_shoulder_pitch_joint=-1.9, right_shoulder_roll_joint=-0.15,
        right_elbow_joint=-1.95, right_wrist_pitch_joint=0.2)
    add(11.0, 12.5, right_shoulder_pitch_joint=0.1, right_shoulder_roll_joint=-0.2,
        right_elbow_joint=-0.3, right_wrist_pitch_joint=0.0)
    # Bow (waist + head down)
    add(13.0, 14.5, waist_pitch_joint=0.24, head_pitch_joint=-0.3)
    add(14.5, 16.0, waist_pitch_joint=0.0, head_pitch_joint=0.0)
    # Left-arm wave
    add(16.5, 18.0, left_shoulder_pitch_joint=0.35, left_shoulder_roll_joint=1.35,
        left_elbow_joint=-1.6, left_wrist_yaw_joint=0.4)
    return ms
