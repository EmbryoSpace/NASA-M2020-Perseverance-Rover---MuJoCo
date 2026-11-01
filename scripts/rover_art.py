#!/usr/bin/env python3
"""
rover_art.py -- the rover drawn from above, in its real proportions.

A blob with a pointy end tells you where the rover is. It does not tell you
how big it is, which wheel is about to meet the rock, or why a 1.9 m belly
over a 25 cm limit is a tight thing. These are the flight dimensions, so a
drawing made with them is to scale against the terrain beside it.

All measurements are from the JPL URDF, in the chassis frame: x forward,
y left, metres.
"""

import math

import numpy as np

# wheel contact centres (same table as ace_eval.WHEELS)
WHEELS = {
    "LF": (1.185, 1.0625), "LM": (0.0, 1.1845), "LR": (-1.075, 1.0625),
    "RF": (1.185, -1.0625), "RM": (0.0, -1.1845), "RR": (-1.075, -1.0625),
}
WHEEL_LEN = 0.525          # wheel diameter, seen from above
WHEEL_WID = 0.40
BODY_LEN, BODY_WID = 1.90  , 1.40      # belly pan footprint
BODY_CX = -0.05                        # its centre, forward of the origin
MAST_AT = (0.55, 0.42)                 # remote sensing mast, roughly

# The rocker-bogie, seen from above. Rocker pivots on the chassis side at
# y_od; the front wheel hangs off it one way and the bogie the other, with
# the middle and rear wheels on the bogie. Drawing the links is what makes
# six floating rectangles read as a suspension.
ROCKER_PIVOT_X = 0.0
ROCKER_PIVOT_Y = 0.79959
BOGIE_JOINT_X = -0.754


def _rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _rect(cx, cy, length, width, R, origin):
    pts = np.array([[-length, -width], [length, -width],
                    [length, width], [-length, width]]) / 2.0
    return (pts + [cx, cy]) @ R.T + origin


def rover_parts(pose, belly=True):
    """Polygons for one pose, in world xy.

    Returns a dict: 'belly' (the pan footprint, which is what clearance is
    measured under), 'wheels' (six), 'deck' (the chassis box), and 'nose'
    (a short heading tick). Pass belly=False to leave the footprint out.
    """
    x, y, yaw = pose
    R, o = _rot(yaw), np.array([x, y])
    parts = {
        "deck": _rect(BODY_CX, 0.0, 1.78, 1.22, R, o),
        "wheels": [_rect(wx, wy, WHEEL_LEN, WHEEL_WID, R, o)
                   for wx, wy in WHEELS.values()],
        "nose": np.array([[0.86, -0.26], [1.34, 0.0],
                          [0.86, 0.26]]) @ R.T + o,
        "mast": _rect(MAST_AT[0], MAST_AT[1], 0.26, 0.26, R, o),
        "links": [],
    }
    for sgn in (1.0, -1.0):
        piv = np.array([ROCKER_PIVOT_X, sgn * ROCKER_PIVOT_Y])
        front = np.array([WHEELS["LF" if sgn > 0 else "RF"][0],
                          WHEELS["LF" if sgn > 0 else "RF"][1]])
        mid = np.array([WHEELS["LM" if sgn > 0 else "RM"][0],
                        WHEELS["LM" if sgn > 0 else "RM"][1]])
        rear = np.array([WHEELS["LR" if sgn > 0 else "RR"][0],
                         WHEELS["LR" if sgn > 0 else "RR"][1]])
        bog = np.array([BOGIE_JOINT_X, sgn * 1.12])
        for a, b in ((piv, front), (piv, bog), (bog, mid), (bog, rear)):
            parts["links"].append(np.vstack([a, b]) @ R.T + o)
    if belly:
        parts["belly"] = _rect(BODY_CX, 0.0, BODY_LEN, BODY_WID, R, o)
    return parts


def draw_rover(ax, pose, colours, zorder=8, belly=True, alpha_belly=0.13):
    """Add the rover to a matplotlib axes and return the artists, in the
    order they must be updated: belly, deck, mast, nose, then six wheels."""
    from matplotlib.patches import Polygon
    p = rover_parts(pose, belly=belly)
    out = {}
    if belly:
        out["belly"] = ax.add_patch(Polygon(
            p["belly"], closed=True, fc=colours["belly"], ec=colours["belly"],
            lw=1.2, ls=(0, (5, 3)), alpha=1.0, fill=True,
            zorder=zorder))
        out["belly"].set_facecolor("none")
    out["links"] = [ax.plot(l[:, 0], l[:, 1], color=colours["edge"],
                            lw=1.3, alpha=0.75, solid_capstyle="round",
                            zorder=zorder + 1)[0] for l in p["links"]]
    out["wheels"] = [ax.add_patch(Polygon(
        w, closed=True, fc=colours["wheel"], ec=colours["edge"], lw=0.8,
        zorder=zorder + 3)) for w in p["wheels"]]
    out["deck"] = ax.add_patch(Polygon(
        p["deck"], closed=True, fc=colours["body"], ec=colours["edge"],
        lw=1.1, zorder=zorder + 2))
    out["nose"] = ax.add_patch(Polygon(
        p["nose"], closed=True, fc=colours["edge"], ec=colours["edge"],
        lw=0.8, zorder=zorder + 4))
    out["mast"] = ax.add_patch(Polygon(
        p["mast"], closed=True, fc=colours["edge"], ec=colours["edge"],
        lw=0.8, zorder=zorder + 5))
    return out


def move_rover(artists, pose, belly=True):
    """Move an already-drawn rover to a new pose."""
    p = rover_parts(pose, belly=belly)
    if belly and "belly" in artists:
        artists["belly"].set_xy(p["belly"])
    for art, line in zip(artists["links"], p["links"]):
        art.set_data(line[:, 0], line[:, 1])
    for art, poly in zip(artists["wheels"], p["wheels"]):
        art.set_xy(poly)
    artists["deck"].set_xy(p["deck"])
    artists["mast"].set_xy(p["mast"])
    artists["nose"].set_xy(p["nose"])
