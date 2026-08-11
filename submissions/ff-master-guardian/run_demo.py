"""
FF Master Guardian — demo runner.

Runs the balance + choreography controller, renders a cinematic video
(orbiting main camera + onboard head-camera picture-in-picture) and records
a full state/control telemetry JSON.

Usage:
    python3 run_demo.py [--duration 26] [--fps 30] [--width 640] [--height 480]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import mujoco
    import imageio.v3 as iio
except ImportError as exc:
    raise SystemExit("Missing deps: python3 -m pip install -r requirements.txt") from exc

from controller import Controller, build_motion_script, Motion

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "assets" / "Master" / "scene_guardian.xml"
OUT_VIDEO = ROOT / "outputs" / "ff_master_guardian_demo.mp4"
OUT_DATA = ROOT / "outputs" / "ff_master_guardian_data.json"


class GuardianController(Controller):
    """Adds the arm-wave oscillation on top of the base motion script."""

    def __init__(self, model, data):
        super().__init__(model, data)
        self.waves = []

    def add_wave(self, side: str, t0: float, t1: float, base_roll: float, amp: float, freq: float):
        self.waves.append((side, t0, t1, base_roll, amp, freq))

    def modulate(self, t: float, target: np.ndarray) -> None:
        for side, t0, t1, base_roll, amp, freq in self.waves:
            if t0 <= t <= t1:
                roll = base_roll + amp * math.sin(2 * math.pi * freq * (t - t0))
                target[self._idx[f"{side}_shoulder_roll_joint"]] = roll


def orbit_camera(t: float, duration: float, lookat: np.ndarray) -> mujoco.MjvCamera:
    """Cinematic slow orbit around the robot."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    az0, az1 = 215.0, 325.0
    frac = t / max(duration, 1e-6)
    azimuth = az0 + (az1 - az0) * frac
    cam.azimuth = azimuth
    cam.elevation = 12.0
    cam.distance = 2.0
    cam.lookat[:] = [lookat[0], lookat[1], 0.55]
    return cam


def render_pip(main_frame: np.ndarray, pip_frame: np.ndarray) -> np.ndarray:
    """Composite a small picture-in-picture in the bottom-right corner."""
    out = main_frame.copy()
    h, w = pip_frame.shape[:2]
    y0 = out.shape[0] - h - 12
    x0 = out.shape[1] - w - 12
    out[y0:y0 + h, x0:x0 + w] = pip_frame
    # simple dark border
    out[y0 - 2:y0, x0 - 2:x0 + w + 2] = 20
    out[y0 + h:y0 + h + 2, x0 - 2:x0 + w + 2] = 20
    out[y0 - 2:y0 + h + 2, x0 - 2:x0] = 20
    out[y0 - 2:y0 + h + 2, x0 + w:x0 + w + 2] = 20
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=26.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--pip-scale", type=float, default=0.3, help="head-cam PIP scale")
    ap.add_argument("--headless-sim-only", action="store_true", help="no rendering (fast test)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    ctrl = GuardianController(model, data)
    ctrl.setup()
    ctrl.spawn_standing()
    ctrl.motions = build_motion_script()

    # arm-wave oscillations: right arm 2-5s, left arm 16.5-19.5s
    ctrl.add_wave("right", 2.0, 5.0, base_roll=-1.35, amp=0.28, freq=1.6)
    ctrl.add_wave("left", 16.5, 19.5, base_roll=1.35, amp=0.28, freq=1.6)

    dt = model.opt.timestep
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    if args.headless_sim_only:
        # fast stability check over the full duration
        n_steps = int(args.duration / dt)
        min_h, max_tilt = 99.0, 0.0
        for i in range(n_steps):
            t = i * dt
            ctrl.step(t)
            mujoco.mj_step(model, data)
            if i % (10 * ctrl.log_every) == 0:
                h = data.xpos[pelvis_id][2]
                tp, _ = ctrl.torso_tilt()
                min_h = min(min_h, h)
                max_tilt = max(max_tilt, abs(math.degrees(tp)))
        print(f"sim-only: min_h={min_h:.3f} max_tilt={max_tilt:.1f}deg "
              f"{'OK' if min_h > 0.55 and max_tilt < 15 else 'FAIL'}")
        return 0

    # ---- rendering setup ----
    renderer_main = mujoco.Renderer(model, height=args.height, width=args.width)
    pip_w = max(96, int(args.width * args.pip_scale))
    pip_h = max(72, int(args.height * args.pip_scale))
    renderer_head = mujoco.Renderer(model, height=pip_h, width=pip_w)
    head_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")

    frames: list[np.ndarray] = []
    n_frames = int(args.duration * args.fps)
    sim_steps_per_frame = int(round(1.0 / args.fps / dt))
    print(f"渲染 {n_frames} 帧 ({args.duration}s @ {args.fps}fps, {args.width}x{args.height}) ...", flush=True)

    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as iio2
        writer = iio2.get_writer(OUT_VIDEO, fps=args.fps, codec="libx264", quality=8)
        video_path = OUT_VIDEO
        streaming = True
    except Exception as exc:
        print(f"ffmpeg writer unavailable ({exc}); will accumulate frames")
        writer = None
        streaming = False

    for f in range(n_frames):
        t_frame = f / args.fps
        # simulate this frame's slice
        n_sub = sim_steps_per_frame
        for _ in range(n_sub):
            t = t_frame + _ * dt
            ctrl.step(t)
            mujoco.mj_step(model, data)
        # main orbit view
        cam = orbit_camera(t_frame, args.duration, data.xpos[pelvis_id])
        renderer_main.update_scene(data, camera=cam)
        main_img = renderer_main.render().copy()
        # onboard head view (PIP)
        renderer_head.update_scene(data, camera=head_cam_id)
        pip_img = renderer_head.render().copy()
        frame = render_pip(main_img, pip_img)
        if streaming:
            writer.append_data(frame)
        else:
            frames.append(frame)
        if f % 60 == 0:
            print(f"  frame {f}/{n_frames}", flush=True)

    # ---- write outputs ----
    if streaming:
        writer.close()
    else:
        try:
            iio.imwrite(OUT_VIDEO, np.asarray(frames), fps=args.fps, codec="libx264")
            video_path = OUT_VIDEO
        except Exception as exc:  # fallback GIF
            video_path = OUT_VIDEO.with_suffix(".gif")
            iio.imwrite(video_path, np.asarray(frames), fps=args.fps)
            print(f"mp4 failed ({exc}), wrote GIF instead")

    summary = {
        "project": "FF Master Guardian — Autonomous Balance & Greeting Humanoid",
        "uuid": "96d4708c-b89f-49f2-b1ae-37e3c06bb3c0",
        "task": (
            "Closed-loop bipedal standing balance (vestibular ankle strategy on the "
            "onboard IMU) with an autonomous upper-body greeting choreography "
            "(wave / head-scan / salute / bow), rendered from an orbiting camera "
            "plus a live onboard head-camera feed."
        ),
        "video": str(video_path),
        "duration_s": args.duration,
        "fps": args.fps,
        "telemetry_samples": len(ctrl.log),
        "final_pelvis_pos": data.xpos[pelvis_id].round(4).tolist(),
        "telemetry": ctrl.log,
    }
    OUT_DATA.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n✅ 视频: {video_path}")
    print(f"✅ 数据: {OUT_DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
