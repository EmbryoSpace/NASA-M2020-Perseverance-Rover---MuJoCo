#!/usr/bin/env python3
"""
make_enav_3d.py -- record ENav driving, in 3D, with the real physics.

The top-down animation (make_enav_gif.py) shows the decision. This shows the
consequence: the rover actually driving what ENav chose, over the Jezero
terrain, with the rocker-bogie working.

ENav plans in primitives, not waypoints -- a turn in place, then an arc held
at one curvature, because the drive and steer motors share controllers and a
wheel cannot be steered while it rolls. So this executes those primitives
directly instead of reusing autonav's turn-then-drive waypoint follower.

    python scripts/make_enav_3d.py                     # mp4, 1280x720
    python scripts/make_enav_3d.py --gif --cycles 12   # gif for the README
    python scripts/make_enav_3d.py --dry-run           # drive it, render nothing

Rendering needs a working OpenGL context. On macOS that is plain `python`
(offscreen rendering does not need mjpython). On a headless Linux box install
libosmesa6 and set MUJOCO_GL=osmesa.
"""

import argparse
import math
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# the ace package lives at the repo root, one level up from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace import RoverGeometry
from ace_eval import AceGroundTruth
from m2020_drive import Mobility, hold_pose
from autonav import draw_route
from ace import enav
from ace import EnavMaps, GlobalPlanner, select_path, first_leg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(ROOT, "images")

YAW_TOL = math.radians(1.5)     # a TIP is done within this
ARC_TOL = 0.03                  # m, an arc is done within this


def yaw_of(d):
    q = d.qpos[3:7]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return math.atan2(R[3], R[0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------- planning
def plan(model, start, goal, cycles, M, span):
    """Run ENav and return the primitives it decided to execute, in order."""
    g = RoverGeometry.m2020()
    gt = AceGroundTruth(model)
    print("planning with ENav ...", flush=True)
    maps = EnavMaps(gt.ground, centre=tuple(start), span=span)
    gp = GlobalPlanner(maps, tuple(goal))

    def boxes(x, y, yaw, pad):
        b = gt.wheel_boxes(x, y, yaw, pad=pad)
        return None if len(b) < 6 else {k: (v["lo"], v["hi"])
                                        for k, v in b.items()}

    def belly(x, y, yaw, pad):
        return gt.belly_points(x, y, yaw, pad=pad)

    pose = (start[0], start[1], 0.0)
    prev, history, legs, route = None, [], [], [pose[:2]]
    for c in range(cycles):
        sel = select_path(pose, maps, gp, boxes, g, prev_path=prev,
                          history=history, M=M, belly_fn=belly)
        if sel.path is None:
            print(f"  cycle {c}: no feasible path, stopping")
            break
        leg = first_leg(sel.path)
        legs.append(leg)
        for pr in leg:
            pose = pr.apply(pose)
        route.append(pose[:2])
        history.append(leg[-1] if leg else None)
        prev = sel.path
        d = math.dist(pose[:2], tuple(goal))
        print(f"  cycle {c:2d}  ({pose[0]:6.2f},{pose[1]:6.2f})  "
              f"{d:5.1f} m to go", flush=True)
        if d < 1.0:
            print("  goal reached in planning")
            break
    return legs, route, gt


# ---------------------------------------------------------------- driving
def execute(gt, legs, route, goal, on_frame=None, speed=0.042,
            turn_rate=0.08, realtime=60.0, fps=30, max_minutes=240.0):
    """Drive ENav's primitives in the physics, closing the loop on the
    measured pose. Calls on_frame(d, track) every rendered frame."""
    m, d = gt.model, gt.data
    mujoco.mj_resetDataKeyframe(m, d, 1)
    mujoco.mj_forward(m, d)

    boxes = gt.wheel_boxes(route[0][0], route[0][1], 0.0, pad=0.22)
    top = max(b["hi"] for b in boxes.values())
    d.qpos[0:3] = [route[0][0], route[0][1], top + 0.12]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qvel[:] = 0
    hold_pose(m, d)
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)

    mob = Mobility(m)
    hold_pose(m, d, skip_drives=True)
    steps_per_frame = max(1, int(realtime / (fps * m.opt.timestep)))
    track, t0, nframe = [], d.time, 0

    def run(cmd_fn, done_fn):
        nonlocal nframe
        while not done_fn() and (d.time - t0) < max_minutes * 60:
            cmd_fn()
            for _ in range(steps_per_frame):
                mujoco.mj_step(m, d)
            track.append(d.qpos[:3].copy())
            if on_frame is not None:
                on_frame(d, track)
            nframe += 1

    for i, leg in enumerate(legs):
        for pr in leg:
            if pr.is_tip:
                if abs(pr.theta) < 1e-9:
                    continue
                y0 = yaw_of(d)
                target = pr.theta
                # sign is resolved against the measured yaw, not assumed
                sgn = 1.0 if target > 0 else -1.0
                run(lambda: mob.point_turn(d.ctrl, sgn * turn_rate),
                    lambda: abs(wrap(yaw_of(d) - y0)) >= abs(target) - YAW_TOL)
            else:
                p0 = d.qpos[:2].copy()
                radius = None if abs(pr.theta) < 1e-9 else pr.l / pr.theta
                run(lambda: mob.arc(d.ctrl, speed, radius),
                    lambda: np.linalg.norm(d.qpos[:2] - p0)
                    >= abs(pr.l) - ARC_TOL)
        mob.arc(d.ctrl, 0.0, None)
        print(f"  drove leg {i + 1}/{len(legs)}  "
              f"({d.qpos[0]:6.2f},{d.qpos[1]:6.2f})  {nframe} frames",
              flush=True)
    return track, nframe


# ---------------------------------------------------------------- record
def record(gt, legs, route, goal, out, as_gif, width, height, fps, realtime):
    m = gt.model
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, height)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 11.0, -18.0, 150.0
    opt = mujoco.MjvOption()

    try:
        r = mujoco.Renderer(m, height, width)
    except Exception as e:
        raise SystemExit(
            f"no OpenGL context, so nothing can be rendered: {e}\n"
            "On macOS run this with plain `python`. On headless Linux install "
            "libosmesa6 and set MUJOCO_GL=osmesa. Use --dry-run to drive "
            "without rendering.")

    sink, frames = None, []
    if not as_gif:
        import imageio.v2 as iio
        sink = iio.get_writer(out, fps=fps, macro_block_size=None, quality=7)

    def on_frame(d, track):
        cam.lookat[:] = d.qpos[:3] + np.array([0, 0, 0.6])
        cam.azimuth = 150.0 + 10.0 * math.sin(len(track) / 260.0)
        r.update_scene(d, cam, opt)
        # reset=False: update_scene has already put the rover in this scene,
        # and zeroing ngeom here would delete it
        draw_route(r.scene, gt, route, track, goal, reset=False)
        img = r.render()
        if sink is not None:
            sink.append_data(img)
        else:
            from PIL import Image
            frames.append(Image.fromarray(img))

    try:
        execute(gt, legs, route, goal, on_frame=on_frame,
                fps=fps, realtime=realtime)
    finally:
        if sink is not None:
            sink.close()
        r.close()

    if as_gif:
        picks = sorted({0, len(frames) // 3, 2 * len(frames) // 3,
                        len(frames) - 1})
        w_, h_ = frames[0].size
        from PIL import Image
        mont = Image.new("RGB", (w_ * len(picks), h_))
        for k, i in enumerate(picks):
            mont.paste(frames[i], (k * w_, 0))
        pal = mont.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
        q[0].save(out, save_all=True, append_images=q[1:],
                  duration=1000 / fps, loop=0, optimize=True, disposal=1)
    print(f"wrote {os.path.relpath(out, ROOT)} "
          f"({os.path.getsize(out) / 1e6:.2f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="course_jezero.xml")
    ap.add_argument("--start", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--goal", type=float, nargs=2, default=[18.0, 6.0])
    ap.add_argument("--cycles", type=int, default=22)
    ap.add_argument("--M", type=int, default=150)
    ap.add_argument("--span", type=float, default=50.0)
    ap.add_argument("--gif", action="store_true",
                    help="write a GIF instead of an mp4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--realtime", type=float, default=60.0,
                    help="simulated seconds per wall second")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and drive, render nothing (no OpenGL needed)")
    args = ap.parse_args()

    legs, route, gt = plan(args.model, args.start, args.goal,
                           args.cycles, args.M, args.span)
    if not legs:
        raise SystemExit("ENav found nothing to drive")

    if args.dry_run:
        track, n = execute(gt, legs, route, tuple(args.goal),
                           fps=args.fps, realtime=args.realtime)
        end = track[-1]
        print(f"\ndrove {len(legs)} legs, {n} frames would be rendered")
        print(f"  ended at ({end[0]:.2f}, {end[1]:.2f}), "
              f"{math.dist(end[:2], tuple(args.goal)):.2f} m from the goal")
        return

    os.makedirs(IMAGES, exist_ok=True)
    out = args.out or os.path.join(
        IMAGES, "enav_drive.gif" if args.gif else "enav_drive.mp4")
    record(gt, legs, route, tuple(args.goal), out, args.gif,
           args.width, args.height, args.fps, args.realtime)


if __name__ == "__main__":
    main()
