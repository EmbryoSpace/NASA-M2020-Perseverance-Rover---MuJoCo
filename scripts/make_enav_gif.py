#!/usr/bin/env python3
"""
make_enav_gif.py -- top-down animation of ENav choosing where to drive.

A camera view cannot show what ENav does. The interesting part is the
decision: a tree of candidate paths out to a 6 m horizon, ranked by time to
the goal, with ACE run down the ranking until something is safe -- and then
only the first metre of the winner is driven before the whole thing happens
again. So this draws the plan, not the rover's view of the world.

    python scripts/make_enav_gif.py                  # both themes
    python scripts/make_enav_gif.py --cycles 18

Writes images/anim_enav_light.gif and images/anim_enav_dark.gif.
Needs no GPU -- everything is drawn with matplotlib.
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# the ace package lives at the repo root, one level up from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace import RoverGeometry
from ace_eval import AceGroundTruth
from ace import enav
from ace import EnavMaps, GlobalPlanner, select_path, first_leg, fixed_tree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(ROOT, "images")
FPS = 6
SUBFRAMES = 3           # interpolated steps per driven metre

THEMES = {
    "light": dict(bg="#FFFFFF", panel="#F6F8FA", text="#1F2328", sub="#57606A",
                  fan="#8C959F", ok="#8C959F", bad="#CF222E", pick="#0969DA",
                  trail="#2DA44E", body="#57606A", wheel="#1F2328",
                  goal="#8250DF", terrain="Greys"),
    "dark": dict(bg="#0D1117", panel="#161B22", text="#E6EDF3", sub="#8B949E",
                 fan="#6E7681", ok="#6E7681", bad="#F85149", pick="#58A6FF",
                 trail="#3FB950", body="#8B949E", wheel="#E6EDF3",
                 goal="#A371F7", terrain="Greys_r"),
}


# ---------------------------------------------------------------- simulate
def simulate(model, start, goal, cycles, M, span):
    """Drive the goal with ENav, keeping what each cycle considered."""
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
    trail = [pose]
    prev, history, frames = None, [], []
    tree = fixed_tree()

    for c in range(cycles):
        seen = []
        sel = select_path(pose, maps, gp, boxes, g, prev_path=prev,
                          history=history, M=M, belly_fn=belly,
                          record=lambda p, pc, ac: seen.append((p, ac)))
        if sel.path is None:
            print(f"  cycle {c}: no feasible path")
            break

        # the reachable set: where every option in the fixed tree ends up
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

        frames.append(dict(cycle=c, pose=start_pose, driven=pose,
                           ends=ends, evaluated=evaluated, chosen=chosen,
                           trail=list(trail[:-1]), n_ace=sel.n_ace,
                           n_opts=sel.n_considered,
                           dist=math.dist(pose[:2], tuple(goal))))
        print(f"  cycle {c:2d}  ({pose[0]:6.2f},{pose[1]:6.2f})  "
              f"{sel.n_considered} options ACE'd, {sel.n_ace} runs, "
              f"{frames[-1]['dist']:.1f} m to go", flush=True)
        if frames[-1]["dist"] < 1.0:
            print("  goal reached")
            break
    return frames, maps, gt


# ---------------------------------------------------------------- draw
def terrain_grid(gt, centre, span, n=220):
    ax = np.linspace(-span / 2, span / 2, n)
    Z = np.full((n, n), np.nan)
    for i, dx in enumerate(ax):
        for j, dy in enumerate(ax):
            z = gt.ground(centre[0] + dx, centre[1] + dy)
            if z is not None:
                Z[j, i] = z
    return ax, Z


def rover_patch(pose, scale=1.0):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    R = np.array([[c, -s], [s, c]])
    pts = np.array([[-1.0, -0.7], [1.0, -0.7], [1.3, 0.0],
                    [1.0, 0.7], [-1.0, 0.7]]) * scale
    return pts @ R.T + [pose[0], pose[1]]


def make_gif(frames, gt, start, goal, span, theme, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon
    from PIL import Image

    from matplotlib.colors import LightSource

    T = THEMES[theme]
    ax_grid, Z = terrain_grid(gt, start, span)
    ext = [start[0] - span / 2, start[0] + span / 2,
           start[1] - span / 2, start[1] + span / 2]

    # view bounds first: the figure is shaped to the data so that an equal
    # aspect ratio does not letterbox the plot into a thin band
    pad = 4.5
    xs = [f["pose"][0] for f in frames] + [goal[0], start[0]]
    ys = [f["pose"][1] for f in frames] + [goal[1], start[1]]
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    w = 8.2
    h = w * (y1 - y0) / (x1 - x0) + 1.5          # + room for title and caption
    fig, axp = plt.subplots(figsize=(w, h), dpi=110)
    fig.patch.set_facecolor(T["bg"])
    axp.set_facecolor(T["panel"])
    fig.subplots_adjust(left=0.065, right=0.985,
                        bottom=0.9 / h, top=1 - 0.78 / h)

    # proper hillshade, so the background reads as ground rather than a ramp
    Zf = np.nan_to_num(Z, nan=float(np.nanmean(Z)))
    ls = LightSource(azdeg=315, altdeg=42)
    axp.imshow(ls.hillshade(Zf, vert_exag=2.5,
                            dx=span / Zf.shape[1], dy=span / Zf.shape[0]),
               extent=ext, origin="lower", cmap=T["terrain"],
               alpha=0.30, interpolation="bilinear", zorder=0)

    axp.set_xlim(x0, x1)
    axp.set_ylim(y0, y1)
    axp.set_aspect("equal")
    for sp in axp.spines.values():
        sp.set_color(T["sub"])
    axp.tick_params(colors=T["sub"], labelsize=8)
    axp.set_xlabel("metres", color=T["sub"], fontsize=9)

    axp.plot(*goal, marker="*", ms=20, color=T["goal"], zorder=9,
             markeredgecolor=T["bg"], markeredgewidth=0.8)
    axp.annotate("goal", goal, textcoords="offset points", xytext=(12, -4),
                 color=T["goal"], fontsize=9, weight="bold", zorder=9)

    fan = axp.scatter([], [], s=1.4, c=T["fan"], alpha=0.30, zorder=2)
    ev_ok = LineCollection([], colors=T["ok"], linewidths=0.7, alpha=0.55, zorder=3)
    ev_bad = LineCollection([], colors=T["bad"], linewidths=1.0, alpha=0.85, zorder=4)
    axp.add_collection(ev_ok)
    axp.add_collection(ev_bad)
    pick, = axp.plot([], [], color=T["pick"], lw=2.6, zorder=6)
    trail, = axp.plot([], [], color=T["trail"], lw=3.0, zorder=5,
                      solid_capstyle="round")
    body = Polygon(rover_patch((0, 0, 0)), closed=True, fc=T["body"],
                   ec=T["wheel"], lw=1.1, zorder=8)
    axp.add_patch(body)

    title = axp.set_title("", color=T["text"], fontsize=12, loc="left",
                          weight="bold", pad=8)
    caption = fig.text(0.065, 0.30 / h, "", color=T["sub"], fontsize=8.8)
    legend = fig.text(0.065, 1 - 0.30 / h, "", color=T["sub"], fontsize=8.5,
                      va="top")

    seq = []
    for f in frames:
        for k in range(SUBFRAMES):
            seq.append((f, (k + 1) / SUBFRAMES))

    def update(i):
        f, t = seq[i]
        a, b = f["pose"], f["driven"]
        now = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]),
               a[2] + t * (b[2] - a[2]))

        fan.set_offsets(f["ends"])
        ev_ok.set_segments([p for p, ok in f["evaluated"] if ok])
        ev_bad.set_segments([p for p, ok in f["evaluated"] if not ok])
        pick.set_data(f["chosen"][:, 0], f["chosen"][:, 1])
        tr = np.array(f["trail"] + [now])
        trail.set_data(tr[:, 0], tr[:, 1])
        body.set_xy(rover_patch(now))
        title.set_text(f"ENav cycle {f['cycle']}   "
                       f"{f['dist']:.1f} m to goal")
        caption.set_text(
            f"{len(f['ends'])} path options to a 6 m horizon · "
            f"{f['n_opts']} reached ACE ({f['n_ace']} pose checks) · "
            f"first 1 m driven, then replan")
        legend.set_text("dots: where every option ends    "
                        "grey: ACE passed    red: ACE rejected    "
                        "blue: chosen    green: driven")
        return ()

    # Render every frame first, then build ONE palette from a montage of
    # frames spread across the drive. Quantizing from frame 0 alone starves
    # whatever is not on screen yet: the driven trail is a single point there,
    # so green never gets a palette slot and the trail renders grey for the
    # whole animation.
    rgb = []
    for i in range(len(seq)):
        update(i)
        fig.canvas.draw()
        rgb.append(Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())[..., :3]))

    picks = sorted({0, len(rgb) // 3, 2 * len(rgb) // 3, len(rgb) - 1})
    w_, h_ = rgb[0].size
    montage = Image.new("RGB", (w_ * len(picks), h_))
    for k, idx in enumerate(picks):
        montage.paste(rgb[idx], (k * w_, 0))
    pal = montage.quantize(colors=256, method=Image.Quantize.MEDIANCUT)

    out_frames = [im.quantize(palette=pal, dither=Image.Dither.NONE)
                  for im in rgb]
    out_frames[0].save(out, save_all=True, append_images=out_frames[1:],
                       duration=1000 / FPS, loop=0, optimize=True, disposal=1)
    plt.close(fig)
    print(f"wrote {os.path.relpath(out, ROOT)} "
          f"({os.path.getsize(out) / 1e6:.2f} MB, {len(out_frames)} frames)")


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
    themes = ["light", "dark"] if args.theme == "both" else [args.theme]
    for th in themes:
        make_gif(frames, gt, args.start, args.goal, args.span, th,
                 os.path.join(IMAGES, f"anim_enav_{th}.gif"))


if __name__ == "__main__":
    main()
