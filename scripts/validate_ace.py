#!/usr/bin/env python3
"""
validate_ace.py -- check that ACE's bounds actually bound.

ACE's whole claim is conservatism: the true settled state must lie inside the
interval it returns. The paper validates this against numeric settling in
ROAMS; this does the same thing against the full rocker-bogie physics in
`ace_eval.AceGroundTruth`, on the real Jezero DTM.

    python scripts/validate_ace.py --poses 40
    python scripts/validate_ace.py --poses 40 --eps 0.015

A failure here means either ACE is wrong or the oracle is -- both worth
knowing. Report both the containment rate and the tightness (how much slack
the bounds carry), because a bound that is always true but infinitely wide is
useless.
"""

import argparse
import math

import numpy as np

import os
import sys

# the ace package lives at the repo root, one level up from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace import ace, RoverGeometry, WHEEL_KEYS, PERCEPTION_EPS
from ace_eval import AceGroundTruth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="course_jezero.xml")
    ap.add_argument("--poses", type=int, default=30)
    ap.add_argument("--span", type=float, default=30.0)
    ap.add_argument("--pad", type=float, default=0.22,
                    help="wheel box half-size, m")
    ap.add_argument("--eps", type=float, default=0.0,
                    help="perception margin; the paper used 0.015 m")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    g = RoverGeometry.m2020()
    gt = AceGroundTruth(args.model)
    rng = np.random.default_rng(args.seed)

    checks = {"tilt": [0, 0], "clearance": [0, 0],
              "rocker": [0, 0], "bogie": [0, 0]}
    slack = {"tilt": [], "clearance": [], "rocker": [], "bogie": []}
    width = {"tilt": [], "clearance": []}
    n_eval = 0
    failures = []

    print(f"model {args.model}, {args.poses} poses, wheel box +-{args.pad} m, "
          f"eps {args.eps*1000:.0f} mm\n")
    print(f"{'#':>3} {'x':>7} {'y':>7} {'yaw':>6} | "
          f"{'tilt true':>9} {'ACE tilt bound':>20} | "
          f"{'clear true':>10} {'ACE clear bound':>20}")

    for i in range(args.poses):
        x = rng.uniform(-args.span / 2, args.span / 2)
        y = rng.uniform(-args.span / 2, args.span / 2)
        yaw = rng.uniform(-math.pi, math.pi)

        boxes = gt.wheel_boxes(x, y, yaw, pad=args.pad)
        if len(boxes) < 6:
            continue
        terrain_iv = {k: (b["lo"], b["hi"]) for k, b in boxes.items()}
        # ACE works in terrain heights relative to the pose; the oracle
        # reports absolute elevation, so reference both to the local datum.
        base = min(v[0] for v in terrain_iv.values())
        terrain_iv = {k: (lo - base, hi - base)
                      for k, (lo, hi) in terrain_iv.items()}

        bp = gt.belly_points(x, y, yaw, pad=args.pad)
        belly = (None if not bp else
                 [(dx, dy, lo - base, hi - base) for dx, dy, lo, hi in bp])
        b = ace(terrain_iv, g, eps=args.eps, belly_pts=belly)
        t = gt.evaluate(x, y, yaw)
        if t is None:
            continue
        n_eval += 1

        # --- tilt
        ok = b.tilt.contains(math.radians(t["tilt_deg"]), tol=1e-9)
        checks["tilt"][0] += ok
        checks["tilt"][1] += 1
        slack["tilt"].append(math.degrees(b.tilt.hi) - t["tilt_deg"])
        width["tilt"].append(math.degrees(b.tilt.width))

        # --- clearance (lower bound is the safety-relevant side)
        okc = b.clearance.lo <= t["clearance_m"] + 1e-9
        checks["clearance"][0] += okc
        checks["clearance"][1] += 1
        slack["clearance"].append(t["clearance_m"] - b.clearance.lo)
        width["clearance"].append(b.clearance.width)

        # --- suspension: compare magnitudes, the sign convention between the
        # paper's delta/beta and the model's joint axes is not guaranteed
        sd = t["suspension_deg"]
        tr = abs(sd.get("LEFT_DIFFERENTIAL", 0.0))
        ab = max(abs(math.degrees(b.rocker_l.lo)),
                 abs(math.degrees(b.rocker_l.hi)))
        okr = tr <= ab + 1e-6
        checks["rocker"][0] += okr
        checks["rocker"][1] += 1
        slack["rocker"].append(ab - tr)

        tb = abs(sd.get("LEFT_BOGIE", 0.0))
        abg = max(abs(math.degrees(b.bogie_l.lo)),
                  abs(math.degrees(b.bogie_l.hi)))
        okb = tb <= abg + 1e-6
        checks["bogie"][0] += okb
        checks["bogie"][1] += 1
        slack["bogie"].append(abg - tb)

        flag = "" if (ok and okc and okr and okb) else "   <-- VIOLATION"
        if flag:
            failures.append((i, x, y, yaw, t, b))
        print(f"{i:3d} {x:7.2f} {y:7.2f} {math.degrees(yaw):6.0f} | "
              f"{t['tilt_deg']:9.2f} [{math.degrees(b.tilt.lo):7.2f},"
              f"{math.degrees(b.tilt.hi):7.2f}] | "
              f"{t['clearance_m']:10.3f} [{b.clearance.lo:8.3f},"
              f"{b.clearance.hi:8.3f}]{flag}")

    print(f"\n{'='*74}")
    print(f"  {n_eval} poses evaluated on real Jezero terrain")
    for k in ("tilt", "clearance", "rocker", "bogie"):
        got, tot = checks[k]
        s = np.array(slack[k]) if slack[k] else np.array([0.0])
        unit = "m" if k == "clearance" else "deg"
        print(f"  {k:10s} contained {got}/{tot}"
              f"   slack: median {np.median(s):+.3f} {unit}, "
              f"min {s.min():+.3f} {unit}")
    if width["tilt"]:
        print(f"\n  bound width: tilt {np.median(width['tilt']):.2f} deg, "
              f"clearance {np.median(width['clearance']):.3f} m (medians)")
    allok = all(checks[k][0] == checks[k][1] for k in checks)
    print(f"\n  CONSERVATISM {'HOLDS' if allok else 'VIOLATED'} "
          f"on every metric" if allok else
          f"\n  CONSERVATISM VIOLATED -- {len(failures)} pose(s)")
    print("=" * 74)


if __name__ == "__main__":
    main()
