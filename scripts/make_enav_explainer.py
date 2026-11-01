#!/usr/bin/env python3
"""
make_enav_explainer.py -- the whole ENav + ACE loop, step by step.

make_enav_gif.py shows where the rover goes. This shows WHY: alongside the
plan, it plots what ACE predicted at every pose along the chosen path -- the
belly clearance band and the tilt band -- against the published limits that
decide whether the path is allowed at all.

Reading it:

  left    the plan. 1694 options reach 6 m out (the ring). The cheapest few
          get ACE'd: grey passed, red rejected. Blue is the winner, green is
          where the rover has actually been.
  top     belly clearance. ACE returns a RANGE, not a number, so the band is
          what it guarantees. The path is refused if the band's floor ever
          touches the 25 cm limit.
  bottom  tilt, the same way, against the 30 deg limit.

The marker on both plots is where the rover is now, so you can watch it move
along the very curve ACE predicted before it set off.

    python scripts/make_enav_explainer.py
    python scripts/make_enav_explainer.py --cycles 12 --theme dark

Writes images/anim_enav_explainer_{light,dark}.gif. No GPU needed.
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# the ace package lives at the repo root, one level up from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace import (ace, RoverGeometry, DEFAULT_LIMITS, PERCEPTION_EPS,
                 wheel_box_halfsize, EnavMaps, GlobalPlanner, select_path,
                 first_leg, fixed_tree)
from ace_eval import AceGroundTruth
from rover_art import draw_rover, move_rover

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(ROOT, "images")
FPS = 5
SUBFRAMES = 3

THEMES = {
    "light": dict(bg="#FFFFFF", panel="#F6F8FA", text="#1F2328", sub="#57606A",
                  fan="#8C959F", ok="#8C959F", bad="#CF222E", pick="#0969DA",
                  trail="#2DA44E", body="#AEB7C2", wheel="#4B535C", edge="#1F2328", belly="#0969DA",
                  goal="#8250DF", limit="#CF222E", band="#0969DA",
                  terrain="Greys", grid="#D0D7DE"),
    "dark": dict(bg="#0D1117", panel="#161B22", text="#E6EDF3", sub="#8B949E",
                 fan="#6E7681", ok="#6E7681", bad="#F85149", pick="#58A6FF",
                 trail="#3FB950", body="#6E7681", wheel="#C9D1D9", edge="#E6EDF3", belly="#58A6FF",
                 goal="#A371F7", limit="#F85149", band="#58A6FF",
                 terrain="Greys_r", grid="#30363D"),
}


def ace_along(path, start, gt, maps, g):
    """ACE's prediction at every pose on a path: distance, clearance band,
    tilt band. This is literally what the planner saw before committing."""
    poses = path.sample_poses(start)
    out, d, prev = [], 0.0, start
    for q in poses:
        d += math.hypot(q[0] - prev[0], q[1] - prev[1])
        prev = q
        slope = math.radians(maps.tilt(q[0], q[1])) if maps.inside(q[0], q[1]) else 0.0
        pad = wheel_box_halfsize(d_predict=d, slope_rad=slope)
        b = gt.wheel_boxes(q[0], q[1], q[2], pad=pad)
        if len(b) < 6:
            continue
        base = min(v["lo"] for v in b.values())
        tiv = {k: (v["lo"] - base, v["hi"] - base) for k, v in b.items()}
        bp = gt.belly_points(q[0], q[1], q[2], pad=pad)
        pts = None if not bp else [(dx, dy, lo - base, hi - base)
                                   for dx, dy, lo, hi in bp]
        st = ace(tiv, g, eps=PERCEPTION_EPS, belly_pts=pts)
        out.append((d, st.clearance.lo, st.clearance.hi,
                    math.degrees(st.tilt.lo), math.degrees(st.tilt.hi)))
    return np.array(out) if out else np.zeros((0, 5))


def simulate(model, start, goal, cycles, M, span):
    g = RoverGeometry.m2020()
    gt = AceGroundTruth(model)
    print("building maps ...", flush=True)
    maps = EnavMaps(gt.ground, centre=tuple(start), span=span)
    gp = GlobalPlanner(maps, tuple(goal))

    def boxes(x, y, yaw, pad):
        b = gt.wheel_boxes(x, y, yaw, pad=pad)
        return None if len(b) < 6 else {k: (v["lo"], v["hi"])
                                        for k, v in b.items()}

    def belly(x, y, yaw, pad):
        return gt.belly_points(x, y, yaw, pad=pad)

    pose = (start[0], start[1], 0.0)
    trail, prev, history, frames = [pose], None, [], []
    tree = fixed_tree()

    for c in range(cycles):
        seen = []
        sel = select_path(pose, maps, gp, boxes, g, prev_path=prev,
                          history=history, M=M, belly_fn=belly,
                          record=lambda p, pc, ac: seen.append((p, ac)))
        if sel.path is None:
            print(f"  cycle {c}: no feasible path")
            break

        prof = ace_along(sel.path, pose, gt, maps, g)
        ends = np.array([t.end_pose(pose)[:2] for t in tree])
        evaluated = [(np.array([q[:2] for q in p.sample_poses(pose)]),
                      math.isfinite(ac)) for p, ac in seen]
        chosen = np.array([q[:2] for q in sel.path.sample_poses(pose)])

        leg = first_leg(sel.path)
        start_pose = pose
        for pr in leg:
            pose = pr.apply(pose)
        trail.append(pose)
        history.append(leg[-1] if leg else None)
        prev = sel.path
        driven = math.dist(start_pose[:2], pose[:2])

        frames.append(dict(cycle=c, pose=start_pose, driven_to=pose,
                           ends=ends, evaluated=evaluated, chosen=chosen,
                           prof=prof, driven=driven,
                           trail=list(trail[:-1]), n_ace=sel.n_ace,
                           n_opts=sel.n_considered,
                           dist=math.dist(pose[:2], tuple(goal))))
        print(f"  cycle {c:2d}  {sel.n_considered} ACE'd  "
              f"clearance floor {prof[:,1].min():.3f} m  "
              f"tilt ceiling {prof[:,4].max():4.1f} deg  "
              f"{frames[-1]['dist']:5.1f} m to go", flush=True)
        if frames[-1]["dist"] < 1.0:
            print("  goal reached")
            break
    return frames, maps, gt


def terrain_grid(gt, centre, span, n=200):
    ax = np.linspace(-span / 2, span / 2, n)
    Z = np.full((n, n), np.nan)
    for i, dx in enumerate(ax):
        for j, dy in enumerate(ax):
            z = gt.ground(centre[0] + dx, centre[1] + dy)
            if z is not None:
                Z[j, i] = z
    return Z


def make_gif(frames, gt, start, goal, span, theme, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LightSource
    from matplotlib.patches import Polygon
    from matplotlib.gridspec import GridSpec
    from PIL import Image

    T = THEMES[theme]
    Z = terrain_grid(gt, start, span)
    ext = [start[0] - span / 2, start[0] + span / 2,
           start[1] - span / 2, start[1] + span / 2]

    pad = 4.5
    xs = [f["pose"][0] for f in frames] + [goal[0], start[0]]
    ys = [f["pose"][1] for f in frames] + [goal[1], start[1]]
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad

    fig = plt.figure(figsize=(12.4, 5.5), dpi=105)
    fig.patch.set_facecolor(T["bg"])
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.62, 1.0],
                  hspace=0.34, wspace=0.17,
                  left=0.045, right=0.975, top=0.875, bottom=0.165)
    axm = fig.add_subplot(gs[:, 0])
    axc = fig.add_subplot(gs[0, 1])
    axt = fig.add_subplot(gs[1, 1])

    for a in (axm, axc, axt):
        a.set_facecolor(T["panel"])
        for sp in a.spines.values():
            sp.set_color(T["sub"])
        a.tick_params(colors=T["sub"], labelsize=8)

    ls = LightSource(azdeg=315, altdeg=42)
    Zf = np.nan_to_num(Z, nan=float(np.nanmean(Z)))
    axm.imshow(ls.hillshade(Zf, vert_exag=2.5, dx=span / Zf.shape[1],
                            dy=span / Zf.shape[0]),
               extent=ext, origin="lower", cmap=T["terrain"], alpha=0.30,
               interpolation="bilinear", zorder=0)
    axm.set_xlim(x0, x1); axm.set_ylim(y0, y1); axm.set_aspect("equal")
    axm.set_xlabel("metres", color=T["sub"], fontsize=9)
    axm.plot(*goal, marker="*", ms=19, color=T["goal"], zorder=9,
             markeredgecolor=T["bg"], markeredgewidth=0.8)
    axm.annotate("goal", goal, textcoords="offset points", xytext=(11, -4),
                 color=T["goal"], fontsize=9, weight="bold", zorder=9)

    fan = axm.scatter([], [], s=1.3, c=T["fan"], alpha=0.30, zorder=2)
    ev_ok = LineCollection([], colors=T["ok"], linewidths=0.7, alpha=0.5, zorder=3)
    ev_bad = LineCollection([], colors=T["bad"], linewidths=1.0, alpha=0.85, zorder=4)
    axm.add_collection(ev_ok); axm.add_collection(ev_bad)
    pick, = axm.plot([], [], color=T["pick"], lw=2.5, zorder=6)
    trail, = axm.plot([], [], color=T["trail"], lw=3.0, zorder=5,
                      solid_capstyle="round")
    rover = draw_rover(axm, (start[0], start[1], 0.0), T, zorder=8)
    axm.set_title("", color=T["text"], fontsize=12, loc="left",
                  weight="bold", pad=8)

    lim_c = DEFAULT_LIMITS["clearance_m"]
    lim_t = DEFAULT_LIMITS["tilt_deg"]
    for a, lab, lim, txt in ((axc, "belly clearance (m)", lim_c,
                              f"{lim_c * 100:.0f} cm limit"),
                             (axt, "tilt (deg)", lim_t, f"{lim_t:.0f}° limit")):
        a.axhline(lim, color=T["limit"], lw=1.3, ls="--", zorder=4)
        a.set_ylabel(lab, color=T["sub"], fontsize=9)
        a.grid(True, color=T["grid"], lw=0.5, alpha=0.6)
        a.annotate(txt, (0.985, lim), xycoords=("axes fraction", "data"),
                   textcoords="offset points", xytext=(0, 5), ha="right",
                   color=T["limit"], fontsize=8, weight="bold", zorder=5)
    axt.set_xlabel("distance along the chosen path (m)", color=T["sub"],
                   fontsize=9)
    axc.set_ylim(0.0, 0.85)
    axt.set_ylim(0.0, lim_t * 1.12)
    axc.set_title("what ACE promised before setting off", color=T["sub"],
                  fontsize=9.5, loc="left", pad=6)

    band_c = [axc.fill_between([], [], [], color=T["band"], alpha=0.22)]
    band_t = [axt.fill_between([], [], [], color=T["band"], alpha=0.22)]
    flo_c, = axc.plot([], [], color=T["band"], lw=1.6)
    fhi_c, = axc.plot([], [], color=T["band"], lw=0.9, alpha=0.7)
    flo_t, = axt.plot([], [], color=T["band"], lw=0.9, alpha=0.7)
    fhi_t, = axt.plot([], [], color=T["band"], lw=1.6)
    now_c, = axc.plot([], [], marker="o", ms=6, color=T["trail"], zorder=6)
    now_t, = axt.plot([], [], marker="o", ms=6, color=T["trail"], zorder=6)

    caption = fig.text(0.045, 0.030, "", color=T["sub"], fontsize=8.8)
    legend = fig.text(0.045, 0.962, "", color=T["sub"], fontsize=8.5, va="top")

    seq = [(f, (k + 1) / SUBFRAMES) for f in frames for k in range(SUBFRAMES)]

    def update(i):
        f, t = seq[i]
        a, b = f["pose"], f["driven_to"]
        now = tuple(a[k] + t * (b[k] - a[k]) for k in range(3))

        fan.set_offsets(f["ends"])
        ev_ok.set_segments([p for p, ok in f["evaluated"] if ok])
        ev_bad.set_segments([p for p, ok in f["evaluated"] if not ok])
        pick.set_data(f["chosen"][:, 0], f["chosen"][:, 1])
        tr = np.array([q[:2] for q in f["trail"]] + [now[:2]])
        trail.set_data(tr[:, 0], tr[:, 1])
        move_rover(rover, now)
        axm.set_title(f"ENav cycle {f['cycle']}   {f['dist']:.1f} m to goal",
                      color=T["text"], fontsize=12, loc="left", weight="bold")

        p = f["prof"]
        band_c[0].remove(); band_t[0].remove()
        if len(p):
            band_c[0] = axc.fill_between(p[:, 0], p[:, 1], p[:, 2],
                                         color=T["band"], alpha=0.22, zorder=2)
            band_t[0] = axt.fill_between(p[:, 0], p[:, 3], p[:, 4],
                                         color=T["band"], alpha=0.22, zorder=2)
            flo_c.set_data(p[:, 0], p[:, 1]); fhi_c.set_data(p[:, 0], p[:, 2])
            flo_t.set_data(p[:, 0], p[:, 3]); fhi_t.set_data(p[:, 0], p[:, 4])
            axc.set_xlim(0, max(6.2, p[:, 0].max()))
            axt.set_xlim(0, max(6.2, p[:, 0].max()))
            d_now = t * f["driven"]
            j = int(np.argmin(np.abs(p[:, 0] - d_now)))
            now_c.set_data([p[j, 0]], [p[j, 1]])
            now_t.set_data([p[j, 0]], [p[j, 4]])
            caption.set_text(
                f"cycle {f['cycle']}: {len(f['ends'])} options to 6 m · "
                f"{f['n_opts']} reached ACE ({f['n_ace']} pose checks) · "
                f"worst predicted clearance {p[:, 1].min():.2f} m "
                f"against a {lim_c:.2f} m limit · drive 1 m, replan")
        legend.set_text(
            "left  dots: where all 1694 options end   grey: ACE passed   "
            "red: ACE rejected   blue: chosen   green: driven        "
            "right  band: the range ACE guarantees   dot: the rover now")
        return ()

    rgb = []
    for i in range(len(seq)):
        update(i)
        fig.canvas.draw()
        rgb.append(Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())[..., :3]))

    picks = sorted({0, len(rgb) // 3, 2 * len(rgb) // 3, len(rgb) - 1})
    w_, h_ = rgb[0].size
    mont = Image.new("RGB", (w_ * len(picks), h_))
    for k, idx in enumerate(picks):
        mont.paste(rgb[idx], (k * w_, 0))
    pal = mont.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in rgb]
    q[0].save(out, save_all=True, append_images=q[1:], duration=1000 / FPS,
              loop=0, optimize=True, disposal=1)
    plt.close(fig)
    print(f"wrote {os.path.relpath(out, ROOT)} "
          f"({os.path.getsize(out) / 1e6:.2f} MB, {len(q)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="course_jezero.xml")
    ap.add_argument("--start", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--goal", type=float, nargs=2, default=[18.0, 6.0])
    ap.add_argument("--cycles", type=int, default=22)
    ap.add_argument("--M", type=int, default=150)
    ap.add_argument("--span", type=float, default=50.0)
    ap.add_argument("--theme", choices=["light", "dark", "both"],
                    default="both")
    args = ap.parse_args()

    frames, maps, gt = simulate(args.model, args.start, args.goal,
                                args.cycles, args.M, args.span)
    if not frames:
        raise SystemExit("no frames: the planner found nothing to drive")
    os.makedirs(IMAGES, exist_ok=True)
    for th in (["light", "dark"] if args.theme == "both" else [args.theme]):
        make_gif(frames, gt, args.start, args.goal, args.span, th,
                 os.path.join(IMAGES, f"anim_enav_explainer_{th}.gif"))


if __name__ == "__main__":
    main()
