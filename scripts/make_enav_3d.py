#!/usr/bin/env python3
"""
make_enav_3d.py -- the whole ENav + ACE loop, in MuJoCo, in 3D.

Not just the rover driving. Every cycle the scene shows what the planner was
actually looking at, drawn on the terrain itself:

    pale grey   where all 1694 candidate paths end, 6 m out
    grey        the paths ACE checked and passed
    red         the paths ACE rejected
    blue        the path it chose
    green       where the rover has actually been
    yellow      the goal

Then the rover drives the first metre of the blue one with the real physics,
the overlay is rebuilt for the next cycle, and it all happens again.

    python scripts/make_enav_3d.py --gif          # for the README
    python scripts/make_enav_3d.py                # mp4, 1280x720
    python scripts/make_enav_3d.py --dry-run      # drive it, render nothing

Rendering needs a working OpenGL context. On macOS that is plain `python`
from the venv (offscreen rendering does not need mjpython). On headless
Linux install libosmesa6 and set MUJOCO_GL=osmesa.
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

from ace import (RoverGeometry, EnavMaps, GlobalPlanner, select_path,
                 first_leg, fixed_tree)
from ace_eval import AceGroundTruth
from m2020_drive import Mobility, hold_pose

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(ROOT, "images")

YAW_TOL = math.radians(1.5)
ARC_TOL = 0.03

ENDS_RGBA = np.array([0.62, 0.64, 0.66, 0.65], np.float32)   # the reachable set
OK_RGBA = np.array([0.55, 0.57, 0.60, 0.80], np.float32)     # ACE passed
BAD_RGBA = np.array([0.90, 0.25, 0.21, 0.95], np.float32)    # ACE rejected
PICK_RGBA = np.array([0.15, 0.50, 0.90, 1.00], np.float32)   # chosen
TRACK_RGBA = np.array([0.18, 0.72, 0.35, 1.00], np.float32)  # driven
GOAL_RGBA = np.array([0.97, 0.82, 0.12, 1.00], np.float32)


def yaw_of(d):
    q = d.qpos[3:7]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return math.atan2(R[3], R[0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------- planning
def plan(model, start, goal, cycles, M, span, fan_every=6):
    """Run ENav, keeping what each cycle considered as well as what it chose."""
    g = RoverGeometry.m2020()
    gt = AceGroundTruth(model)
    print("planning with ENav ...", flush=True)
    maps = EnavMaps(gt.ground, centre=tuple(start), span=span)
    gp = GlobalPlanner(maps, tuple(goal))
    tree = fixed_tree()

    def boxes(x, y, yaw, pad):
        b = gt.wheel_boxes(x, y, yaw, pad=pad)
        return None if len(b) < 6 else {k: (v["lo"], v["hi"])
                                        for k, v in b.items()}

    def belly(x, y, yaw, pad):
        return gt.belly_points(x, y, yaw, pad=pad)

    pose = (start[0], start[1], 0.0)
    prev, history, out = None, [], []
    for c in range(cycles):
        seen = []
        sel = select_path(pose, maps, gp, boxes, g, prev_path=prev,
                          history=history, M=M, belly_fn=belly,
                          record=lambda p, pc, ac: seen.append((p, ac)))
        if sel.path is None:
            print(f"  cycle {c}: no feasible path, stopping")
            break

        evaluated = [([q[:2] for q in p.sample_poses(pose)],
                      math.isfinite(ac)) for p, ac in seen]
        ends = [t.end_pose(pose)[:2] for t in tree[::fan_every]]
        chosen = [q[:2] for q in sel.path.sample_poses(pose)]

        leg = first_leg(sel.path)
        for pr in leg:
            pose = pr.apply(pose)
        history.append(leg[-1] if leg else None)
        prev = sel.path
        out.append(dict(leg=leg, ends=ends, evaluated=evaluated,
                        chosen=chosen))
        d = math.dist(pose[:2], tuple(goal))
        print(f"  cycle {c:2d}  ({pose[0]:6.2f},{pose[1]:6.2f})  "
              f"{sel.n_considered} ACE'd  {d:5.1f} m to go", flush=True)
        if d < 1.0:
            print("  goal reached in planning")
            break
    return out, gt


# ---------------------------------------------------------------- overlay
def draw_cycle(scn, gt, cyc, track, goal, lift=0.30):
    """Draw one cycle's ENav decision into an already-filled Renderer scene.

    The scene arrives with the model's own geoms in it, so this appends and
    never resets ngeom -- zeroing it here deletes the rover and renders an
    empty landscape with a plan drawn across it.
    """
    def put(fn):
        if scn.ngeom >= scn.maxgeom:
            return False
        fn(scn.geoms[scn.ngeom])
        scn.ngeom += 1
        return True

    def on_ground(p, extra=0.0):
        z = gt.ground(p[0], p[1])
        return np.array([p[0], p[1], (z if z is not None else 0.0)
                         + lift + extra])

    def polyline(pts, rgba, width, step=1):
        w = [on_ground(p) for p in pts[::step]]
        for a, b in zip(w[:-1], w[1:]):
            put(lambda g, a=a, b=b: (
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                    np.zeros(3), np.zeros(3),
                                    np.eye(3).ravel(), rgba),
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                     width, a, b)))

    # the reachable set: a low dot wherever an option ends
    for p in cyc["ends"]:
        put(lambda g, p=p: mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.07, 0, 0]),
            on_ground(p), np.eye(3).ravel(), ENDS_RGBA))

    # what ACE actually looked at, and what it said
    for pts, ok in cyc["evaluated"]:
        polyline(pts, OK_RGBA if ok else BAD_RGBA,
                 0.022 if ok else 0.035, step=2)

    polyline(cyc["chosen"], PICK_RGBA, 0.055)
    if len(track) > 1:
        step = max(1, len(track) // 200)
        tp = [np.array([t[0], t[1], t[2] + 0.04]) for t in track[::step]]
        for a, b in zip(tp[:-1], tp[1:]):
            put(lambda g, a=a, b=b: (
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                    np.zeros(3), np.zeros(3),
                                    np.eye(3).ravel(), TRACK_RGBA),
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                     0.045, a, b)))
    gp = on_ground(goal) + np.array([0, 0, 0.9])
    put(lambda g: mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_BOX, np.array([0.10, 0.10, 1.2]), gp,
        np.eye(3).ravel(), GOAL_RGBA))


# ---------------------------------------------------------------- driving
def execute(gt, cycles, start, on_frame=None, speed=0.042, turn_rate=0.08,
            realtime=60.0, fps=30, max_minutes=240.0):
    """Drive each cycle's first metre with the real physics, closing the loop
    on the measured pose. on_frame(d, track, cyc) runs once per rendered
    frame, with the cycle currently being executed."""
    m, d = gt.model, gt.data
    mujoco.mj_resetDataKeyframe(m, d, 1)
    mujoco.mj_forward(m, d)

    boxes = gt.wheel_boxes(start[0], start[1], 0.0, pad=0.22)
    top = max(b["hi"] for b in boxes.values())
    d.qpos[0:3] = [start[0], start[1], top + 0.12]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qvel[:] = 0
    hold_pose(m, d)
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)

    mob = Mobility(m)
    hold_pose(m, d, skip_drives=True)
    steps_per_frame = max(1, int(realtime / (fps * m.opt.timestep)))
    track, t0, n = [], d.time, 0

    for i, cyc in enumerate(cycles):
        def run(cmd_fn, done_fn, cyc=cyc):
            nonlocal n
            while not done_fn() and (d.time - t0) < max_minutes * 60:
                cmd_fn()
                for _ in range(steps_per_frame):
                    mujoco.mj_step(m, d)
                track.append(d.qpos[:3].copy())
                if on_frame is not None:
                    on_frame(d, track, cyc)
                n += 1

        for pr in cyc["leg"]:
            if pr.is_tip:
                if abs(pr.theta) < 1e-9:
                    continue
                y0, target = yaw_of(d), pr.theta
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
        print(f"  drove cycle {i + 1}/{len(cycles)}  "
              f"({d.qpos[0]:6.2f},{d.qpos[1]:6.2f})  {n} frames", flush=True)
    return track, n


# ---------------------------------------------------------------- record
def record(gt, cycles, start, goal, out, as_gif, width, height, fps, realtime):
    m = gt.model
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, height)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 13.0, -26.0, 150.0
    opt = mujoco.MjvOption()

    try:
        r = mujoco.Renderer(m, height, width, max_geom=20000)
    except Exception as e:
        raise SystemExit(
            f"no OpenGL context, so nothing can be rendered: {e}\n"
            "On macOS use the venv's python. On headless Linux install "
            "libosmesa6 and set MUJOCO_GL=osmesa. --dry-run skips rendering.")

    sink, frames = None, []
    if not as_gif:
        import imageio.v2 as iio
        sink = iio.get_writer(out, fps=fps, macro_block_size=None, quality=7)

    def on_frame(d, track, cyc):
        cam.lookat[:] = d.qpos[:3] + np.array([0, 0, 0.6])
        cam.azimuth = 150.0 + 9.0 * math.sin(len(track) / 300.0)
        r.update_scene(d, cam, opt)
        draw_cycle(r.scene, gt, cyc, track, goal)
        img = r.render()
        if sink is not None:
            sink.append_data(img)
        else:
            from PIL import Image
            frames.append(Image.fromarray(img))

    try:
        execute(gt, cycles, start, on_frame=on_frame, fps=fps,
                realtime=realtime)
    finally:
        if sink is not None:
            sink.close()
        r.close()

    if as_gif:
        from PIL import Image
        picks = sorted({0, len(frames) // 3, 2 * len(frames) // 3,
                        len(frames) - 1})
        w_, h_ = frames[0].size
        mont = Image.new("RGB", (w_ * len(picks), h_))
        for k, i in enumerate(picks):
            mont.paste(frames[i], (k * w_, 0))
        pal = mont.quantize(colors=200, method=Image.Quantize.MEDIANCUT)
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
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--realtime", type=float, default=120.0,
                    help="simulated seconds per wall second")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cycles, gt = plan(args.model, args.start, args.goal, args.cycles,
                      args.M, args.span)
    if not cycles:
        raise SystemExit("ENav found nothing to drive")

    if args.dry_run:
        track, n = execute(gt, cycles, args.start, fps=args.fps,
                           realtime=args.realtime)
        end = track[-1]
        print(f"\ndrove {len(cycles)} cycles, {n} frames would be rendered")
        print(f"  ended at ({end[0]:.2f}, {end[1]:.2f}), "
              f"{math.dist(end[:2], tuple(args.goal)):.2f} m from the goal")
        return

    os.makedirs(IMAGES, exist_ok=True)
    out = args.out or os.path.join(
        IMAGES, "enav_3d.gif" if args.gif else "enav_3d.mp4")
    record(gt, cycles, args.start, tuple(args.goal), out, args.gif,
           args.width, args.height, args.fps, args.realtime)


if __name__ == "__main__":
    main()
