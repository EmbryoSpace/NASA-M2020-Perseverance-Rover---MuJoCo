#!/usr/bin/env python3
"""
run_enav.py -- drive a goal with the ENav planner on the real Jezero terrain.

    python run_enav.py --goal 18 6
    python run_enav.py --goal 18 6 --cycles 12 --M 60

Receding horizon: each cycle builds the search tree, ranks every option by
path cost, spends ACE down the ranking, drives the first 1 m of the winner,
and replans.
"""
import argparse, math, time
import numpy as np
import mujoco

import os
import sys

# the ace package lives at the repo root, one level up from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace import RoverGeometry
from ace_eval import AceGroundTruth
from ace import enav
from ace import (EnavMaps, GlobalPlanner, select_path, first_leg,
                  DRIVE_PER_CYCLE, PLANNING_HORIZON)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="course_jezero.xml")
    ap.add_argument("--goal", type=float, nargs=2, default=[18.0, 6.0])
    ap.add_argument("--start", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--M", type=int, default=enav.M_ACE_RUNS)
    ap.add_argument("--span", type=float, default=70.0)
    args = ap.parse_args()

    g = RoverGeometry.m2020()
    gt = AceGroundTruth(args.model)
    print("building hierarchical maps (cost map 1 m, height map fine) ...")
    t0 = time.time()
    maps = EnavMaps(gt.ground, centre=tuple(args.start), span=args.span)
    print(f"  {time.time()-t0:.1f}s")
    gp = GlobalPlanner(maps, tuple(args.goal))
    print(f"  cost-to-go at start: {gp.time_to_go(*args.start)/60:.1f} min")

    def boxes(x, y, yaw, pad):
        b = gt.wheel_boxes(x, y, yaw, pad=pad)
        if len(b) < 6:
            return None
        return {k: (v["lo"], v["hi"]) for k, v in b.items()}

    def belly(x, y, yaw, pad):
        return gt.belly_points(x, y, yaw, pad=pad)

    pose = (args.start[0], args.start[1], 0.0)
    prev, history = None, []
    print(f"\n{'cyc':>4} {'pose':>22} {'opts':>6} {'ACE':>6} {'think':>6} "
          f"{'chosen':>52} {'cost':>9}")
    for c in range(args.cycles):
        t0 = time.time()
        sel = select_path(pose, maps, gp, boxes, g,
                          prev_path=prev, history=history, M=args.M,
                          belly_fn=belly)
        if sel.path is None:
            print(f"{c:4d}  NO FEASIBLE PATH")
            break
        leg = first_leg(sel.path)
        for pr in leg:
            pose = pr.apply(pose)
        history.append(leg[-1] if leg else None)
        prev = sel.path
        d = math.dist(pose[:2], tuple(args.goal))
        print(f"{c:4d} ({pose[0]:6.2f},{pose[1]:6.2f},{math.degrees(pose[2]):+5.0f}) "
              f"{sel.n_considered:6d} {sel.n_ace:6d} "
              f"{'yes' if sel.stopped_to_think else 'no':>6} "
              f"{str(sel.path)[8:60]:>52} {sel.cost:9.1f}"
              f"   [{time.time()-t0:.1f}s, {d:.1f} m to goal]")
        if d < 1.0:
            print("\n  GOAL REACHED")
            break


if __name__ == "__main__":
    main()
