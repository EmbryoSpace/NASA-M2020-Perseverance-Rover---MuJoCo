#!/usr/bin/env python3
"""
ace_eval.py -- physics ground truth for Approximate Clearance Evaluation.

ACE (Abcouwer et al.; McHenry et al., "Mars 2020 Autonomous Rover
Navigation") is the traversability check inside ENav, Perseverance's surface
navigation software:

    "Given a particular rover position and heading, it examines the elevation
     data that lies in the potential footprint of each wheel and determines
     the min and max height of each wheel. From those heights it computes
     bounds on the vehicle attitude and suspension angles from which the
     minimum clearance between the belly pan of the rover and the underlying
     terrain can be derived."

The defining property is that those bounds are *hard*: ACE claims principled
conservatism, so the true settled attitude, suspension angles and clearance
must always lie inside the interval ACE returns. That is exactly what this
module lets you check.

It provides three things:

  1. `heightmap()`   -- the 2.5D terrain grid ACE consumes, sampled from the
                        same terrain the physics uses, so your ACE and the
                        simulator see identical ground.
  2. `evaluate()`    -- drop the rover at (x, y, heading), let it settle, and
                        report the TRUE attitude, suspension angles, per-wheel
                        heights and belly clearance.
  3. `sweep()`       -- do that over a grid of poses to build a truth map.

Then:

    truth = ace.evaluate(x, y, yaw)
    bounds = your_ace(ace.heightmap(x, y, yaw), ...)
    assert bounds.tilt_lo <= truth["tilt"] <= bounds.tilt_hi   # conservatism

    python ace_eval.py --demo          # one pose, full readout
    python ace_eval.py --sweep 12      # 12x12 truth map over the terrain
    python ace_eval.py --dump-terrain  # terrain grid -> ace_terrain.npz
"""

import argparse
import json
import math
import os

import numpy as np
import mujoco

from m2020_drive import hold_pose

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")   # MJCF + assets

# --- rover geometry ACE needs, all from the JPL URDF -----------------------
# Wheel contact centres in the chassis (RNAV) frame: x forward, y left.
WHEELS = {
    "LF": (1.185, 1.0625), "LM": (0.0, 1.1845), "LR": (-1.075, 1.0625),
    "RF": (1.185, -1.0625), "RM": (0.0, -1.1845), "RR": (-1.075, -1.0625),
}
WHEEL_RADIUS = 0.2625
# Belly pan underside, chassis frame. The box is 1.90 x 1.40 x 0.52 centred at
# (-0.05, 0, 0.86), so its underside sits at z = 0.60 -- the published 60 cm
# ground clearance.
BELLY = dict(cx=-0.05, cy=0.0, hx=0.95, hy=0.70, z=0.60)

# Limits ACE checks against. These are EST -- the flight values are not
# published. Override them to match whatever your ACE uses.
LIMITS = dict(tilt_deg=30.0, rocker_deg=25.0, bogie_deg=25.0,
              clearance_m=0.20)

SUSPENSION = ["LEFT_DIFFERENTIAL", "RIGHT_DIFFERENTIAL",
              "LEFT_BOGIE", "RIGHT_BOGIE", "CENTER_DIFFERENTIAL"]


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class AceGroundTruth:
    """Settle the full rocker-bogie model to get truth for ACE bounds."""

    def __init__(self, xml="course_jezero.xml", settle_s=6.0):
        self.model = mujoco.MjModel.from_xml_path(os.path.join(MODELS, xml))
        self.data = mujoco.MjData(self.model)
        self.settle_steps = int(settle_s / self.model.opt.timestep)
        self.jid = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i): i
                    for i in range(self.model.njnt)}
        self.terrain_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        tg = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        if tg < 0:
            raise SystemExit("no geom named 'terrain' in " + xml)
        self.terrain_geom = tg
        # ray origin must start above the tallest terrain
        self.ray_top = float(self.model.stat.extent * 4 + 60.0)
        # mj_ray reads geom_xpos, so the kinematics must be evaluated at least
        # once before any cast -- otherwise every ray silently misses.
        mujoco.mj_forward(self.model, self.data)

    # -- terrain -------------------------------------------------------
    # A ray that lands exactly on a heightfield triangle edge can come back
    # degenerate -- mj_ray reports the full ray length, i.e. an elevation of
    # zero, on grid-aligned samples. It is rare (9 cells in 6561 on a 513-post
    # hfield) but each one punches a fake hole in the terrain, which corrupts
    # a wheel box or a costmap cell. Sampling a hair off the grid and
    # cross-checking removes it; 1 mm is 0.1% of the terrain post spacing, so
    # it costs nothing in accuracy.
    _JITTER = 1.7e-3

    def _cast(self, x, y):
        pnt = np.array([float(x), float(y), self.ray_top])
        vec = np.array([0.0, 0.0, -1.0])
        gid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(self.model, self.data, pnt, vec,
                             self.terrain_group, 1, -1, gid)
        if dist < 0:
            return None
        return self.ray_top - dist

    def ground(self, x, y, tol=0.25):
        """Terrain elevation at world (x, y), or None off the map.

        Ray-cast against the compiled terrain, which is what the physics
        actually collides with."""
        j = self._JITTER
        a = self._cast(x + j, y + j)
        b = self._cast(x - j, y - j)
        if a is None or b is None:
            return a if b is None else b
        if abs(a - b) <= tol:
            return 0.5 * (a + b)
        # disagreement means at least one cast was degenerate -- take the
        # median of a wider sample, which outvotes a single bad hit
        vals = [v for v in (a, b,
                            self._cast(x + j, y - j),
                            self._cast(x - j, y + j),
                            self._cast(x, y)) if v is not None]
        return float(np.median(vals))

    def heightmap(self, x, y, yaw=0.0, size=6.0, res=0.1, body_frame=True):
        """2.5D elevation grid for ACE.

        size : side length in metres, centred on (x, y)
        res  : grid spacing in metres
        body_frame : if True the grid axes are the rover's forward/left, which
                     is how ENav keeps its heightmap; otherwise world-aligned.

        Returns (Z, extent) with Z[i, j] the elevation and extent the grid
        coordinates in metres relative to the rover centre.
        """
        n = int(round(size / res)) + 1
        ax = np.linspace(-size / 2, size / 2, n)
        GX, GY = np.meshgrid(ax, ax, indexing="ij")
        if body_frame:
            c, s = math.cos(yaw), math.sin(yaw)
            WX = x + GX * c - GY * s
            WY = y + GX * s + GY * c
        else:
            WX, WY = x + GX, y + GY
        Z = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                g = self.ground(WX[i, j], WY[i, j])
                if g is not None:
                    Z[i, j] = g
        return Z, ax

    def wheel_boxes(self, x, y, yaw, pad=0.30):
        """The terrain each wheel could sit on -- ACE's 'wheel boxes'.

        Returns per wheel the min and max terrain height inside its box,
        which is the input ACE derives all of its bounds from."""
        c, s = math.cos(yaw), math.sin(yaw)
        out = {}
        for name, (wx, wy) in WHEELS.items():
            px = x + wx * c - wy * s
            py = y + wx * s + wy * c
            hs = []
            k = max(3, int(pad * 2 / 0.1))
            for dx in np.linspace(-pad, pad, k):
                for dy in np.linspace(-pad, pad, k):
                    g = self.ground(px + dx, py + dy)
                    if g is not None:
                        hs.append(g)
            if hs:
                out[name] = dict(centre=(px, py), lo=float(min(hs)),
                                 hi=float(max(hs)), n=len(hs))
        return out

    # Belly-pan footprint, in the rover frame. ACE eq 32 needs the ground
    # UNDER THE BELLY PAN, point by point: the pan and the ground beneath it
    # tilt together, so the pan's lowest corner sits over the lowest ground.
    # Reducing the footprint to a single (min, max) subtracts the slope twice.
    BELLY_LEN = 1.90        # l_p
    BELLY_WID = 1.40        # w_p
    BELLY_CX = -0.05        # footprint centre, forward of the rover origin
    BELLY_STEP = 0.15       # footprint sampling pitch, m

    def belly_points(self, x, y, yaw, pad=0.0):
        """Terrain under the belly footprint as (dx, dy, lo, hi) samples.

        dx is forward and dy is to the RIGHT of the rover origin -- the
        paper's frame, not the model's. Each sample's interval is the terrain
        min/max within `pad` of it, so position and slip uncertainty grow the
        interval at each point instead of smearing the whole footprint into
        one number."""
        c, s = math.cos(yaw), math.sin(yaw)
        hl, hw = self.BELLY_LEN / 2, self.BELLY_WID / 2
        step = self.BELLY_STEP
        # fine grid over the footprint grown by pad, so every sample has
        # neighbours out to pad on all sides
        nx = int(math.ceil((hl + pad) / step))
        ny = int(math.ceil((hw + pad) / step))
        gx = np.arange(-nx, nx + 1) * step
        gy = np.arange(-ny, ny + 1) * step
        Z = np.full((gx.size, gy.size), np.nan)
        for i, dx in enumerate(gx):
            for j, dy in enumerate(gy):
                px = x + (self.BELLY_CX + dx) * c - dy * s
                py = y + (self.BELLY_CX + dx) * s + dy * c
                gz = self.ground(px, py)
                if gz is not None:
                    Z[i, j] = gz
        if np.all(np.isnan(Z)):
            return None
        w = int(math.ceil(pad / step))
        out = []
        for i, dx in enumerate(gx):
            if abs(dx) > hl + 1e-9:
                continue
            for j, dy in enumerate(gy):
                if abs(dy) > hw + 1e-9:
                    continue
                win = Z[max(0, i - w):i + w + 1, max(0, j - w):j + w + 1]
                win = win[~np.isnan(win)]
                if win.size == 0:
                    continue
                out.append((self.BELLY_CX + float(dx), -float(dy),
                            float(win.min()), float(win.max())))
        return out or None

    # -- ground truth --------------------------------------------------
    def evaluate(self, x, y, yaw=0.0, clearance_samples=9):
        """Place the rover, settle it, and measure the truth."""
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, 1)          # stowed, drives at zero
        mujoco.mj_forward(m, d)                       # needed before mj_ray

        # Spawn clear of the highest ground under the wheel footprint --
        # a wheel started inside the terrain gets a mega-newton impulse.
        boxes = self.wheel_boxes(x, y, yaw, pad=0.22)
        if len(boxes) < 6:
            return None                                # off the map
        top = max(b["hi"] for b in boxes.values())
        d.qpos[0:3] = [x, y, top + 0.12]
        d.qpos[3:7] = yaw_quat(yaw)
        d.qvel[:] = 0
        # Latch the position servos onto the stowed keyframe. Leaving ctrl at
        # zero deploys the arm, which moves the CoM forward and changes the
        # settled attitude -- the real rover drives stowed, and so should the
        # pose ACE is being checked against.
        hold_pose(m, d)
        for _ in range(self.settle_steps):
            mujoco.mj_step(m, d)

        q = d.qpos[3:7]
        roll = math.degrees(math.atan2(2 * (q[0] * q[1] + q[2] * q[3]),
                                       1 - 2 * (q[1] ** 2 + q[2] ** 2)))
        pitch = math.degrees(math.asin(
            max(-1, min(1, 2 * (q[0] * q[2] - q[3] * q[1])))))
        # tilt = angle between the body z axis and vertical, which is what the
        # vehicle's own safety check uses -- not max(|roll|, |pitch|)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))

        susp = {j: math.degrees(d.qpos[m.jnt_qposadr[self.jid[j]]])
                for j in SUSPENSION if j in self.jid}

        # belly clearance: cast along the body's own -z from the belly pan
        down = -R[:, 2]
        clear = []
        for fx in np.linspace(-BELLY["hx"], BELLY["hx"], clearance_samples):
            for fy in np.linspace(-BELLY["hy"], BELLY["hy"], clearance_samples):
                local = np.array([BELLY["cx"] + fx, BELLY["cy"] + fy, BELLY["z"]])
                world = d.qpos[:3] + R @ local
                gid = np.zeros(1, dtype=np.int32)
                dist = mujoco.mj_ray(m, d, world, down,
                                     self.terrain_group, 1, -1, gid)
                if dist >= 0:
                    clear.append(dist)
        clearance = float(min(clear)) if clear else float("nan")

        # per-wheel truth
        wheel_z, contacts = {}, 0
        for name in WHEELS:
            gname = "wheel_" + name.lower()
            gi = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gi >= 0:
                wheel_z[name] = float(d.geom_xpos[gi][2] - WHEEL_RADIUS)
        for i in range(d.ncon):
            g1, g2 = d.contact.geom1[i], d.contact.geom2[i]
            if g1 == self.terrain_geom or g2 == self.terrain_geom:
                contacts += 1

        return dict(
            x=float(x), y=float(y), yaw_deg=math.degrees(yaw),
            z=float(d.qpos[2]),
            roll_deg=roll, pitch_deg=pitch, tilt_deg=tilt,
            suspension_deg=susp,
            clearance_m=clearance,
            wheel_bottom_z=wheel_z,
            wheel_box=boxes,
            terrain_contacts=contacts,
            residual_speed=float(np.linalg.norm(d.qvel[:6])),
            safe=bool(tilt <= LIMITS["tilt_deg"]
                      and clearance >= LIMITS["clearance_m"]
                      and all(abs(v) <= LIMITS["rocker_deg"]
                              for k, v in susp.items() if "DIFFERENTIAL" in k)
                      and all(abs(v) <= LIMITS["bogie_deg"]
                              for k, v in susp.items() if "BOGIE" in k)),
        )

    # -- sweep ---------------------------------------------------------
    def sweep(self, n=10, span=40.0, yaw=0.0, centre=(0.0, 0.0), verbose=True):
        ax = np.linspace(-span / 2, span / 2, n)
        keys = ["tilt_deg", "roll_deg", "pitch_deg", "clearance_m", "z"]
        out = {k: np.full((n, n), np.nan) for k in keys}
        out["safe"] = np.zeros((n, n), dtype=bool)
        out["rocker_spread_deg"] = np.full((n, n), np.nan)
        done = 0
        for i, dx in enumerate(ax):
            for j, dy in enumerate(ax):
                r = self.evaluate(centre[0] + dx, centre[1] + dy, yaw)
                done += 1
                if r is None:
                    continue
                for k in keys:
                    out[k][i, j] = r[k]
                out["safe"][i, j] = r["safe"]
                s = r["suspension_deg"]
                out["rocker_spread_deg"][i, j] = abs(
                    s.get("LEFT_DIFFERENTIAL", 0) - s.get("RIGHT_DIFFERENTIAL", 0))
            if verbose:
                print(f"  row {i+1}/{n}  ({done}/{n*n} poses)", flush=True)
        out["axis_m"] = ax
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="course_jezero.xml")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--sweep", type=int, default=0, metavar="N")
    ap.add_argument("--span", type=float, default=40.0)
    ap.add_argument("--yaw", type=float, default=0.0, help="degrees")
    ap.add_argument("--dump-terrain", action="store_true")
    ap.add_argument("--size", type=float, default=6.0)
    ap.add_argument("--res", type=float, default=0.1)
    args = ap.parse_args()

    ace = AceGroundTruth(args.model)
    yaw = math.radians(args.yaw)
    print(f"model: {args.model}")

    if args.dump_terrain:
        Z, ax = ace.heightmap(0, 0, yaw, size=args.size, res=args.res)
        np.savez(os.path.join(ROOT, "ace_terrain.npz"), Z=Z, axis=ax,
                 res=args.res, yaw=yaw,
                 wheels=np.array([WHEELS[k] for k in WHEELS]),
                 wheel_names=np.array(list(WHEELS)),
                 wheel_radius=WHEEL_RADIUS,
                 belly=np.array([BELLY["cx"], BELLY["cy"],
                                 BELLY["hx"], BELLY["hy"], BELLY["z"]]))
        print(f"wrote ace_terrain.npz  grid {Z.shape} at {args.res} m, "
              f"relief {np.nanmax(Z)-np.nanmin(Z):.3f} m")
        print("  arrays: Z, axis, res, yaw, wheels, wheel_names, "
              "wheel_radius, belly")
        return

    if args.sweep:
        print(f"sweeping {args.sweep}x{args.sweep} poses over "
              f"{args.span:.0f} x {args.span:.0f} m ...")
        r = ace.sweep(args.sweep, args.span, yaw)
        np.savez(os.path.join(ROOT, "ace_truth.npz"), **r)
        t, c = r["tilt_deg"], r["clearance_m"]
        print(f"\n  tilt       {np.nanmin(t):.2f} to {np.nanmax(t):.2f} deg "
              f"(mean {np.nanmean(t):.2f})")
        print(f"  clearance  {np.nanmin(c):.3f} to {np.nanmax(c):.3f} m "
              f"(mean {np.nanmean(c):.3f})")
        print(f"  safe       {r['safe'].sum()}/{r['safe'].size} poses "
              f"at limits {LIMITS}")
        print("\nwrote ace_truth.npz")
        return

    # default: one pose, full readout
    res = ace.evaluate(0.0, 0.0, yaw)
    if res is None:
        raise SystemExit("pose is off the terrain")
    print(f"\nGROUND TRUTH at ({res['x']:.1f}, {res['y']:.1f}) "
          f"heading {res['yaw_deg']:.0f} deg\n" + "=" * 58)
    print(f"  attitude     roll {res['roll_deg']:+.2f}  "
          f"pitch {res['pitch_deg']:+.2f}  tilt {res['tilt_deg']:.2f} deg")
    print(f"  clearance    {res['clearance_m']:.3f} m "
          f"(belly pan to terrain, along body -z)")
    short = {"LEFT_DIFFERENTIAL": "rockerL", "RIGHT_DIFFERENTIAL": "rockerR",
             "LEFT_BOGIE": "bogieL", "RIGHT_BOGIE": "bogieR",
             "CENTER_DIFFERENTIAL": "diffBar"}
    print("  suspension   " + "  ".join(
        f"{short.get(k, k)}={v:+.2f}" for k, v in res["suspension_deg"].items()))
    print(f"  contacts     {res['terrain_contacts']}   "
          f"residual speed {res['residual_speed']:.2e}")
    print(f"  SAFE         {res['safe']}   limits {LIMITS}")
    print("\n  ACE wheel boxes (the input ACE bounds everything from):")
    print(f"  {'wheel':>6} {'centre x':>10} {'centre y':>10} "
          f"{'h_min':>9} {'h_max':>9} {'spread':>8}")
    for k, b in res["wheel_box"].items():
        print(f"  {k:>6} {b['centre'][0]:10.2f} {b['centre'][1]:10.2f} "
              f"{b['lo']:9.3f} {b['hi']:9.3f} {b['hi']-b['lo']:8.3f}")
    print("\n  Any correct ACE must return bounds containing the attitude,")
    print("  suspension and clearance values above.")


if __name__ == "__main__":
    main()
