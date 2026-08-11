# FF Master Guardian — Autonomous Balance & Greeting Humanoid

**Robothon 2026 · Faraday Future Embodied-AI Hackathon**

A closed-loop **bipedal balance controller** for the FF Master humanoid,
combined with an **autonomous greeting choreography** (wave / head-scan /
salute / bow). The robot stands and performs the whole routine entirely
from real MuJoCo physics — no teleported poses.

| | |
|---|---|
| Robot | Faraday Future **FF Master** humanoid (31 actuated joints) |
| Physics | MuJoCo, 1 ms timestep, real contacts / inertia / actuators |
| Balance | Vestibular **ankle strategy** using the onboard IMU (framequat) |
| Autonomy | Time-indexed choreography state machine (upper body) |
| Telemetry | IMU, joint positions, control torques, contacts → JSON |
| Demo video | Orbiting camera + **live onboard head-camera** (PIP) |

---

## Highlights

1. **True physics balance** — the robot stands on flat foot pads under
   gravity with a closed-loop vestibular reflex: torso tilt + tilt-rate from
   the onboard IMU drive corrective ankle torques (the human "ankle strategy").
   Verified: **60 s undisturbed standing, tilt < 3°; recovers from a 100 N
   push with < 1 cm drift** (see `outputs/ff_master_guardian_data.json`).

2. **Model engineering** — the base FF Master model is shipped with two
   deliberate upgrades (documented in `assets/Master/ff_master_ultra.xml`):
   * flat **foot contact pads** replacing the tiny 5 mm point contacts
     (contact patch ×100+), and
   * **boosted ankle/leg actuator ratings** so the balance controller has the
     torque authority real bipedal robots need.

3. **Standing pose is statically balanced by design** — joint targets are
   chosen so that `hip_pitch + knee + ankle_pitch = 0` and
   `hip_roll + ankle_roll = 0`, which makes the feet perfectly level and the
   COM directly over the support polygon (verified: zero gravity torque at the
   pose via inverse dynamics).

4. **Sensor-rich** — uses the model's IMU (`body-orientation` framequat,
   gyro, accelerometer, velocimeter) and joint-position sensors; all logged.

5. **Autonomous choreography** — a motion-script state machine drives the
   upper body (right-arm wave → head scan → salute → bow → left-arm wave)
   while the legs and vestibular loop keep the robot balanced the whole time.

---

## Quick Start

```bash
python3 -m pip install -r requirements.txt

# Render the demo video + telemetry (headless/EGL supported):
MUJOCO_GL=egl python3 run_demo.py --duration 26 --fps 30

# Fast stability check (no rendering):
python3 run_demo.py --headless-sim-only
```

Outputs:

| File | Description |
|---|---|
| `outputs/ff_master_guardian_demo.mp4` | Demo video (orbiting cam + head-cam PIP) |
| `outputs/ff_master_guardian_data.json` | 10 Hz telemetry: pelvis pose, torso tilt, joint positions, ankle torque, contact count |

---

## How the balance controller works

At every 1 ms physics step:

```
leg/ankle joints   → high-stiffness PD to the standing pose
ankle pitch        += Kp_vest * torso_tilt_pitch  +  Kd_vest * tilt_rate     (IMU framequat)
ankle roll         += Kroll * (torso_tilt_roll + rate)                        (×0.6)
arms / waist / head → softer PD to the choreography target (smoothstep)
```

* The IMU `body-orientation` sensor gives the torso quaternion; the tilt is
  its deviation from vertical.
* The vestibular loop is the only "intelligence" needed for balance — it
  pushes the ankles into the ground the way a human does when wobbling.
* Leg joints are deliberately near-rigid (kp ≈ 2000) so the pose does not
  collapse under the torso load; arm/waist/head are softer for natural motion.

### Why the model was modified

The stock FF Master model contacts the ground through 5 mm radius spheres —
a 43 kg robot standing on pinheads. Two minimal, physics-honest changes:

1. `<geom type="box" size="0.12 0.09 0.006" pos="0.045 0 -0.074">` flat foot
   pads on both `*_ankle_roll_link` bodies (friction 1.6).
2. Ankle-pitch actuator range −36→**−80** N·m, ankle-roll −24→**−50** N·m,
   matching real torque-controlled humanoids.

---

## Choreography script

| t (s) | Action |
|-------|--------|
| 0–2 | Settle into standing balance |
| 2–5 | Wave with the right arm |
| 5–9 | Head scan (look left → right) |
| 9.5–12.5 | Salute with the right arm |
| 13–16 | Bow (waist + head) |
| 16.5–19.5 | Wave with the left arm |
| 19.5–26 | Stand still (steady-state balance showcase) |

---

## Project structure

```
ff-master-guardian/
├── controller.py        # balance + choreography controller
├── run_demo.py          # simulation, video rendering, telemetry export
├── registration.json    # contest UUID
├── requirements.txt
├── README.md
└── assets/Master/       # FF Master model + Guardian modifications
    ├── scene_guardian.xml   # arena props + onboard head camera
    └── ff_master_ultra.xml  # model + foot pads + actuator ratings
```


## ⚙️ Asset reference (important)

This submission deliberately does **not** duplicate the FF Master mesh files
(~112 MB of STL assets). The modified model XML uses a `meshdir` that resolves
to the mesh directory already shipped in this repository:

```xml
<compiler angle='radian' eulerseq="XYZ" meshdir="../../../../assets/Master/meshes" .../>
```

(from `submissions/ff-master-guardian/assets/Master/` up to the repo-root
`assets/Master/meshes`). This keeps the PR small while remaining fully runnable
from any working directory (verified from repo root, submission dir, and
absolute paths). Only the *modified* model files live in the submission:
`ff_master_ultra.xml` (foot pads + actuator ratings) and
`scene_guardian.xml` (arena props + onboard head camera).

---

## Ideas for the future

* Closed-loop **stepping** (capture-point control) for real walking.
* RL policy (e.g., MuJoCo MJX) trained on the same model for dynamic gaits.
* Depth-camera obstacle avoidance using the onboard camera.
