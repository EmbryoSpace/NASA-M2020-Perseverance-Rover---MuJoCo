#!/usr/bin/env python3
"""
enav.py -- Enhanced AutoNav, as described in

    O. Toupet, M. Ono, T. S. Del Sesto, M. W. Maimone, M. M. McHenry,
    "Enhanced Autonomous Navigation on the Perseverance Mars Rover",
    IEEE Transactions on Field Robotics 3:83-124, 2026.
    doi:10.1109/TFR.2025.3636366  (CC-BY)

This follows the paper's architecture rather than approximating it. Section
references in comments point at that paper; ACE itself lives in `ace.py` and
follows Otsu et al. 2020.

    Local Planner  -- tree of motion primitives out to a 6 m horizon
    Global Planner -- Dijkstra cost-to-go on the cost map
    ACE            -- kinematic safety check, run every 25 cm along a path

Two-stage selection (Algorithm 1): sort every path option by the cheap path
cost, then run the expensive ACE only down the sorted list, stopping once a
feasible path is found AND at least M ACE runs have been spent. Receding
horizon: only the first 1 m of the chosen path is driven before replanning.

MECHANICAL CONSTRAINTS the paper states, both of which shape the primitives:
    1. "Wheels cannot be steered while driving. This is because the driving
       and steering motors share the same motor controllers." So no path of
       continuously changing curvature.
    2. "Among the six wheels, only the front and rear wheels are steerable.
       Therefore, Perseverance cannot move sideways in a crab-like motion."

PUBLISHED PARAMETERS used here (all from the paper)
    planning horizon 6 m, drive 1 m per cycle, ACE every 25 cm
    TIP pose sampling every 10 deg
    fixed tree 14 x 11 x 11 = 1694 options
    max wheel ground speed 4.2 cm/s, max TIP rate 1.5 deg/s
    cost map 1 m resolution, height map 10 cm, plane fit over a 2 m envelope
    position uncertainty 5% of with-VO odometry, 50% of without-VO
    drive rate requirement 100 m/h benign, 86 m/h complex
    benign terrain: slope < 15 deg and CFA < 7%

EST (the paper gives the structure but not the constant)
    C_tau, C_r penalty weights; tau_th, r_th thresholds; the C_i ACE cost
    weights; M; the slip curve table; t_steer timings.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .core import (ace, ace_cost, RoverGeometry, DEFAULT_LIMITS,
                 wheel_box_halfsize, PERCEPTION_EPS)

# ---- published parameters ------------------------------------------------
PLANNING_HORIZON = 6.0          # m, Local Planner reach
DRIVE_PER_CYCLE = 1.0           # m, executed before replanning (Delta d)
ACE_INTERVAL = 0.25             # m, pose sampling along arcs
GOAL_TOLERANCE = 1.0            # m, arrival radius (EST)
TIP_SAMPLE_DEG = 10.0           # deg, pose sampling during a turn in place
ARC_LEN = 3.0                   # m, fixed-tree arc length
WHEEL_GROUND_SPEED = 0.042      # m/s, maximum
TIP_RATE_DEG = 1.5              # deg/s, maximum
COSTMAP_RES = 1.0               # m
HEIGHTMAP_RES = 0.10            # m
ENVELOPE_RADIUS = 2.0           # m, plane-fit window for tilt/roughness
RETREAT_MEMORY = 3              # steps of retreat stored

# Fixed tree, current Perseverance parameterization (Sec. II-E2)
TIP_OPTIONS_DEG = [90, -90, 60, -60, 45, -45, 30, -30, 20, -20, 10, -10, 0, 180]
ARC_OPTIONS_DEG = [60, -60, 45, -45, 30, -30, 20, -20, 10, -10, 0]

# EST -- structure published, constants not
C_TAU, C_R = 1.0, 1.0
TAU_TH_DEG, R_TH = 20.0, 0.01
M_ACE_RUNS = 120
T_STEER_ALL = 12.0              # s, all four corners together
T_STEER_ONE_BY_ONE = 40.0       # s, above the tilt threshold
STEER_TILT_THRESHOLD_DEG = 15.0


# ---- motion primitives (Sec. II-E1) --------------------------------------
@dataclass(frozen=True)
class Primitive:
    """Arc of constant curvature, or a turn in place (modelled as l = 0).

    l < 0 means driving backward. theta is the heading change in radians.
    """
    l: float
    theta: float

    @property
    def is_tip(self) -> bool:
        return abs(self.l) < 1e-9

    @property
    def radius(self) -> Optional[float]:
        """Signed turn radius; None for a straight arc."""
        if self.is_tip or abs(self.theta) < 1e-9:
            return None
        return self.l / self.theta

    def drive_time(self) -> float:
        if self.is_tip:
            return abs(math.degrees(self.theta)) / TIP_RATE_DEG
        return abs(self.l) / WHEEL_GROUND_SPEED

    def apply(self, pose: Tuple[float, float, float]
              ) -> Tuple[float, float, float]:
        x, y, th = pose
        if self.is_tip:
            return (x, y, _wrap(th + self.theta))
        if abs(self.theta) < 1e-9:
            return (x + self.l * math.cos(th), y + self.l * math.sin(th), th)
        R = self.l / self.theta
        cx, cy = x - R * math.sin(th), y + R * math.cos(th)
        th2 = _wrap(th + self.theta)
        return (cx + R * math.sin(th2), cy - R * math.cos(th2), th2)

    def sample(self, pose: Tuple[float, float, float]
               ) -> List[Tuple[float, float, float]]:
        """Poses along this primitive: every ACE_INTERVAL of drive distance
        for arcs, every TIP_SAMPLE_DEG of rotation for a TIP (Sec. II-E5)."""
        out = []
        # A no-op primitive emits no poses. The 0 deg TIP at depth 1 is there
        # to represent "no initial turn", not a manoeuvre -- counting it would
        # put 25 poses on a straight 6 m path where the paper counts 24.
        if abs(self.theta) < 1e-9 and self.is_tip:
            return out
        if self.is_tip:
            n = max(1, int(round(abs(math.degrees(self.theta)) / TIP_SAMPLE_DEG)))
            for k in range(1, n + 1):
                out.append(Primitive(0.0, self.theta * k / n).apply(pose))
        else:
            n = max(1, int(round(abs(self.l) / ACE_INTERVAL)))
            for k in range(1, n + 1):
                f = k / n
                out.append(Primitive(self.l * f, self.theta * f).apply(pose))
        return out

    def __repr__(self):
        return (f"TIP({math.degrees(self.theta):+.0f}d)" if self.is_tip
                else f"Arc({self.l:+.1f}m,{math.degrees(self.theta):+.0f}d)")


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Path:
    prims: Tuple[Primitive, ...]
    kind: str = "fixed"            # fixed | heritage | retreat

    def end_pose(self, start: Tuple[float, float, float]):
        p = start
        for pr in self.prims:
            p = pr.apply(p)
        return p

    def sample_poses(self, start):
        poses, p = [], start
        for pr in self.prims:
            for q in pr.sample(p):
                poses.append(q)
            p = pr.apply(p)
        return poses

    def drive_distance(self) -> float:
        return sum(abs(pr.l) for pr in self.prims)

    def __repr__(self):
        return f"<{self.kind}: " + " ".join(map(repr, self.prims)) + ">"


# ---- search tree (Sec. II-E) ---------------------------------------------
def fixed_tree() -> List[Path]:
    """14 TIPs x 11 arcs x 11 arcs = 1694 options, the published config."""
    d1 = [Primitive(0.0, math.radians(t)) for t in TIP_OPTIONS_DEG]
    d2 = [Primitive(ARC_LEN, math.radians(t)) for t in ARC_OPTIONS_DEG]
    d3 = [Primitive(ARC_LEN, math.radians(t)) for t in ARC_OPTIONS_DEG]
    return [Path((a, b, c)) for a, b, c in product(d1, d2, d3)]


def heritage_tree(prev: Optional[Path], driven: float = DRIVE_PER_CYCLE
                  ) -> List[Path]:
    """The previously selected path, shortened by what was driven and
    re-extended with l = Delta d arcs (Sec. II-E4).

    Without this the rover keeps changing turn radius -- and every change
    costs a stop, because it cannot steer while driving.
    """
    if prev is None:
        return []
    rem: List[Primitive] = []
    budget = driven
    for pr in prev.prims:
        if pr.is_tip:
            continue                      # already executed
        if budget >= abs(pr.l) - 1e-9:
            budget -= abs(pr.l)
            continue
        if budget > 0:
            rem.append(Primitive(pr.l - math.copysign(budget, pr.l),
                                 pr.theta * (1 - budget / abs(pr.l))))
            budget = 0
        else:
            rem.append(pr)
    if not rem:
        return []
    return [Path(tuple(rem) + (Primitive(driven, math.radians(t)),),
                 kind="heritage") for t in ARC_OPTIONS_DEG]


def retreat_tree(history: Sequence[Primitive]) -> List[Path]:
    """One path that reverses the last executed motion (Sec. II-E3).

    Only selected when nothing in the fixed or heritage trees is feasible.
    """
    if not history:
        return []
    last = history[-1]
    return [Path((Primitive(-abs(DRIVE_PER_CYCLE), -last.theta
                            * min(1.0, DRIVE_PER_CYCLE / max(abs(last.l), 1e-9))
                            if not last.is_tip else -last.theta),),
                 kind="retreat")]


# ---- hierarchical maps (Sec. II-C) ---------------------------------------
class EnavMaps:
    """Cost map at 1 m over a wide area, height map at 10 cm locally.

    Tilt and roughness per cost-map cell come from a plane fit over the
    height-map cells inside the rover's 2 m envelope radius. Roughness is the
    maximum deviation from that plane (the paper allows variance instead).
    """

    def __init__(self, ground_fn, centre=(0.0, 0.0), span=120.0,
                 res=COSTMAP_RES, envelope=ENVELOPE_RADIUS,
                 roughness="maxdev"):
        self.ground = ground_fn
        self.res = res
        self.centre = np.array(centre, float)
        self.n = int(round(span / res)) + 1
        self.ax = np.linspace(-span / 2, span / 2, self.n)
        self.envelope = envelope
        self.roughness_mode = roughness
        self._build()

    def _build(self):
        n, res = self.n, self.res
        # sample the terrain at height-map resolution inside each cell's
        # envelope; do it once on a fine grid and reduce
        fine_res = HEIGHTMAP_RES * 2.5      # 25 cm keeps this tractable
        m = int(round((self.ax[-1] - self.ax[0]) / fine_res)) + 1
        fx = np.linspace(self.ax[0], self.ax[-1], m)
        Zf = np.full((m, m), np.nan)
        for i, dx in enumerate(fx):
            for j, dy in enumerate(fx):
                g = self.ground(self.centre[0] + dx, self.centre[1] + dy)
                if g is not None:
                    Zf[i, j] = g
        self.Zfine, self.fine_ax, self.fine_res = Zf, fx, fine_res

        from scipy.ndimage import correlate, maximum_filter
        w = max(1, int(round(self.envelope / fine_res)))
        k = 2 * w + 1
        ii, jj = np.meshgrid(np.arange(-w, w + 1), np.arange(-w, w + 1),
                             indexing="ij")
        dxg, dyg = ii * fine_res, jj * fine_res
        N = k * k
        Sxx, Syy = float((dxg ** 2).sum()), float((dyg ** 2).sum())
        Z = np.nan_to_num(Zf, nan=float(np.nanmean(Zf)))
        mz = correlate(Z, np.ones((k, k)) / N, mode="nearest")
        a = correlate(Z, dxg / Sxx, mode="nearest")
        b = correlate(Z, dyg / Syy, mode="nearest")
        # tilt of the fitted plane
        self.tilt_fine = np.degrees(np.arctan(np.hypot(a, b)))
        dev = np.abs(Z - mz)
        if self.roughness_mode == "maxdev":
            self.rough_fine = maximum_filter(dev, size=k, mode="nearest")
        else:
            mz2 = correlate(Z ** 2, np.ones((k, k)) / N, mode="nearest")
            self.rough_fine = np.maximum(
                mz2 - mz ** 2 - a ** 2 * Sxx / N - b ** 2 * Syy / N, 0.0)

    def _fine_idx(self, x, y):
        i = int(round((x - self.centre[0] - self.fine_ax[0]) / self.fine_res))
        j = int(round((y - self.centre[1] - self.fine_ax[0]) / self.fine_res))
        m = self.Zfine.shape[0]
        return min(max(i, 0), m - 1), min(max(j, 0), m - 1)

    def tilt(self, x, y) -> float:
        i, j = self._fine_idx(x, y)
        return float(self.tilt_fine[i, j])

    def roughness(self, x, y) -> float:
        i, j = self._fine_idx(x, y)
        return float(self.rough_fine[i, j])

    def inside(self, x, y) -> bool:
        return (abs(x - self.centre[0]) <= self.ax[-1]
                and abs(y - self.centre[1]) <= self.ax[-1])


# ---- global planner (Sec. II-F2) -----------------------------------------
class GlobalPlanner:
    """Dijkstra cost-to-go on an 8-connected graph, in units of TIME.

    The paper's global cost is `t_CTG(x, y) + t_turn(theta)` -- the time to
    drive from the end of a candidate path to the goal, plus the time to turn
    onto the global path's initial heading.
    """

    def __init__(self, maps: EnavMaps, goal: Tuple[float, float],
                 tilt_block_deg: float = DEFAULT_LIMITS["tilt_deg"]):
        self.maps = maps
        self.goal = goal
        self.res = maps.res
        self.n = maps.n
        self.ax = maps.ax
        self.tilt_block = tilt_block_deg
        self._dijkstra()

    def _idx(self, x, y):
        i = int(round((x - self.maps.centre[0] - self.ax[0]) / self.res))
        j = int(round((y - self.maps.centre[1] - self.ax[0]) / self.res))
        return i, j

    def _xy(self, i, j):
        return (self.maps.centre[0] + self.ax[0] + i * self.res,
                self.maps.centre[1] + self.ax[0] + j * self.res)

    def _dijkstra(self):
        n = self.n
        self.t_ctg = np.full((n, n), np.inf)
        gi, gj = self._idx(*self.goal)
        gi, gj = min(max(gi, 0), n - 1), min(max(gj, 0), n - 1)
        self.t_ctg[gi, gj] = 0.0
        pq = [(0.0, (gi, gj))]
        nbrs = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if (di, dj) != (0, 0)]
        seen = set()
        while pq:
            t, (i, j) = heapq.heappop(pq)
            if (i, j) in seen:
                continue
            seen.add((i, j))
            for di, dj in nbrs:
                a, b = i + di, j + dj
                if not (0 <= a < n and 0 <= b < n):
                    continue
                x, y = self._xy(a, b)
                if self.maps.tilt(x, y) > self.tilt_block:
                    continue
                step = math.hypot(di, dj) * self.res
                nt = t + step / WHEEL_GROUND_SPEED
                if nt < self.t_ctg[a, b]:
                    self.t_ctg[a, b] = nt
                    heapq.heappush(pq, (nt, (a, b)))

    def time_to_go(self, x, y) -> float:
        i, j = self._idx(x, y)
        if not (0 <= i < self.n and 0 <= j < self.n):
            return math.inf
        return float(self.t_ctg[i, j])

    def heading_at(self, x, y) -> Optional[float]:
        """Initial heading of the global path out of (x, y) -- the direction
        of steepest cost-to-go descent.

        None when no neighbour is strictly cheaper: at the goal, and in a
        local minimum, there is no onward global path to turn onto. Returning
        a heading anyway -- as taking the arg-min unconditionally does -- bills
        a path that ENDS ON THE GOAL for a turn it never has to make. At the
        goal cell every neighbour costs more, the arg-min lands on whichever
        one the loop saw first, and the resulting 98 s of phantom turn was
        enough to make a 1 m retreat outrank the path that finished the drive.
        """
        i, j = self._idx(x, y)
        if not (0 <= i < self.n and 0 <= j < self.n):
            return None
        here = self.t_ctg[i, j]
        best, bh = here, None
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if (di, dj) == (0, 0):
                    continue
                a, b = i + di, j + dj
                if not (0 <= a < self.n and 0 <= b < self.n):
                    continue
                if self.t_ctg[a, b] < best:
                    best = self.t_ctg[a, b]
                    bh = math.atan2(dj, di)
        return bh


# ---- path cost (Sec. II-F) -----------------------------------------------
def local_cost(path: Path, start, maps: EnavMaps) -> float:
    """Wall-clock time to execute the path: steering plus driving.

    The paper: if tilt is below a threshold the rover steers all four
    steerable wheels at once; above it, one wheel at a time, which is slower
    but induces less slip. So t_steer depends on the tilt where the arc
    begins.
    """
    t, pose = 0.0, start
    for pr in path.prims:
        tilt = maps.tilt(pose[0], pose[1]) if maps.inside(pose[0], pose[1]) else 0.0
        if not pr.is_tip and abs(pr.theta) > 1e-9:
            t += (T_STEER_ONE_BY_ONE if tilt > STEER_TILT_THRESHOLD_DEG
                  else T_STEER_ALL)
        elif pr.is_tip and abs(pr.theta) > 1e-9:
            t += (T_STEER_ONE_BY_ONE if tilt > STEER_TILT_THRESHOLD_DEG
                  else T_STEER_ALL)
        t += pr.drive_time()
        pose = pr.apply(pose)
    return t


def global_cost(path: Path, start, gp: GlobalPlanner) -> float:
    ex, ey, eth = path.end_pose(start)
    t_ctg = gp.time_to_go(ex, ey)
    if not math.isfinite(t_ctg):
        return math.inf
    h = gp.heading_at(ex, ey)
    if h is None:                     # nothing left to drive: no turn to bill
        return t_ctg
    dth = abs(_wrap(h - eth))
    return t_ctg + math.degrees(dth) / TIP_RATE_DEG


def penalty(path: Path, start, maps: EnavMaps) -> float:
    """Hinged tilt and roughness penalty, weighted by drive distance.

        penalty = sum_i  dl * ( C_tau h(tau_i, tau_th) + C_r h(r_i, r_th) )
        h(x, c) = max(0, x - c)
    """
    poses = path.sample_poses(start)
    if not poses:
        return 0.0
    total, prev = 0.0, start
    for q in poses:
        dl = math.hypot(q[0] - prev[0], q[1] - prev[1])
        if dl < 1e-9:                     # TIP: use the distance the wheels run
            dl = abs(_wrap(q[2] - prev[2])) * 1.1845
        tau = maps.tilt(q[0], q[1]) if maps.inside(q[0], q[1]) else 90.0
        r = maps.roughness(q[0], q[1]) if maps.inside(q[0], q[1]) else 1.0
        total += dl * (C_TAU * max(0.0, tau - TAU_TH_DEG)
                       + C_R * max(0.0, r - R_TH))
        prev = q
    return total


def arrival_time(path: Path, start, maps: EnavMaps,
                 goal: Tuple[float, float],
                 tol: float = GOAL_TOLERANCE) -> Optional[float]:
    """Execution time to the first pose on `path` inside the goal radius, or
    None if the path never gets there.

    The drive ENDS on arrival, so a path that reaches the goal must not be
    charged for the rest of its 6 m or for turning onto an onward global path
    that no longer exists.
    """
    t, pose = 0.0, start
    for pr in path.prims:
        if abs(pr.theta) > 1e-9:
            tilt = (maps.tilt(pose[0], pose[1])
                    if maps.inside(pose[0], pose[1]) else 0.0)
            t += (T_STEER_ONE_BY_ONE if tilt > STEER_TILT_THRESHOLD_DEG
                  else T_STEER_ALL)
        samples = pr.sample(pose)
        for k, q in enumerate(samples, start=1):
            if math.hypot(q[0] - goal[0], q[1] - goal[1]) <= tol:
                return t + pr.drive_time() * k / len(samples)
        t += pr.drive_time()
        pose = pr.apply(pose)
    return None


def path_cost(path: Path, start, maps: EnavMaps, gp: GlobalPlanner) -> float:
    """Line 4 of Algorithm 1.

    With one addition the paper's formulation needs but does not spell out:
    the goal is ABSORBING. Charging a candidate for its whole 6 m horizon plus
    the turn onto the onward global path, when it already drove through the
    goal, rewards overshooting and turning around -- near the goal every
    endpoint lies beyond it, the turn term dominates a comparison between
    paths of identical length, and the rover orbits the goal at a couple of
    metres instead of arriving.
    """
    t_arrive = arrival_time(path, start, maps, gp.goal)
    if t_arrive is not None:
        return t_arrive
    return (local_cost(path, start, maps)
            + global_cost(path, start, gp)
            + penalty(path, start, maps))


# ---- ACE over a path (Sec. II-G) -----------------------------------------
def run_ace(path: Path, start, wheel_boxes_fn, g: RoverGeometry,
            d_predict0: float = 0.0, maps: EnavMaps = None,
            limits: dict = None, belly_fn=None) -> Tuple[float, int]:
    """ACE cost of a path and the number of ACE runs it took.

    ACE is evaluated at every sampled pose. The wheel boxes grow with the
    slip prediction, which is proportional to accumulated odometry since the
    imagery behind the current height map was taken -- so boxes further along
    the path are larger.
    """
    poses = path.sample_poses(start)
    total, runs, prev = 0.0, 0, start
    d = d_predict0
    for q in poses:
        d += math.hypot(q[0] - prev[0], q[1] - prev[1])
        prev = q
        slope = math.radians(maps.tilt(q[0], q[1])) if maps else 0.0
        pad = wheel_box_halfsize(d_predict=d, slope_rad=slope)
        boxes = wheel_boxes_fn(q[0], q[1], q[2], pad)
        runs += 1
        if boxes is None or len(boxes) < 6:
            return math.inf, runs            # off the map
        base = min(v[0] for v in boxes.values())
        tiv = {k: (lo - base, hi - base) for k, (lo, hi) in boxes.items()}
        # the ground under the belly pan, point by point, for eq 32
        bpts = None
        if belly_fn is not None:
            b = belly_fn(q[0], q[1], q[2], pad)
            if b:
                bpts = [(dx, dy, lo - base, hi - base)
                        for dx, dy, lo, hi in b]
        st = ace(tiv, g, eps=PERCEPTION_EPS, limits=limits, belly_pts=bpts)
        c = ace_cost(st, limits=limits)
        if not math.isfinite(c):
            return math.inf, runs
        total += c
    return total, runs


# ---- Algorithm 1 ----------------------------------------------------------
@dataclass
class Selection:
    path: Optional[Path]
    cost: float
    n_ace: int
    n_considered: int
    stopped_to_think: bool


def _rank_and_ace(P, start, maps, gp, wheel_boxes_fn, g, M, limits,
                  belly_fn, verbose, label, record=None):
    """Algorithm 1 over one candidate set: rank by the cheap path cost, then
    spend ACE down the ranking until something is feasible AND at least M ACE
    runs have gone by."""
    costs = [(path_cost(p, start, maps, gp), p) for p in P]
    costs = [(c, p) for c, p in costs if math.isfinite(c)]
    costs.sort(key=lambda t: t[0])
    if verbose:
        print(f"    {label}: {len(P)} options, "
              f"{len(costs)} with finite path cost")

    c_min, best, n_ace, considered = math.inf, None, 0, 0
    for ci, p in costs:
        considered += 1
        ac, n = run_ace(p, start, wheel_boxes_fn, g, maps=maps,
                        limits=limits, belly_fn=belly_fn)
        n_ace += n
        if record is not None:          # for plotting what ACE saw
            record(p, ci, ac)
        if ci + ac < c_min:
            c_min, best = ci + ac, p
        if math.isfinite(c_min) and n_ace >= M:
            return Selection(best, c_min, n_ace, considered, False)
    return Selection(best, c_min, n_ace, considered,
                     stopped_to_think=math.isfinite(c_min))


def select_path(start, maps: EnavMaps, gp: GlobalPlanner,
                wheel_boxes_fn, g: RoverGeometry,
                prev_path: Optional[Path] = None,
                history: Sequence[Primitive] = (),
                M: int = M_ACE_RUNS,
                limits: dict = None,
                belly_fn=None,
                record=None,
                verbose: bool = False) -> Selection:
    """ENav path selection, Algorithm 1.

    Stage 1 ranks every option by the cheap path cost, stage 2 spends ACE down
    that ranking and stops once a feasible path is found AND at least M ACE
    runs have been used. If nothing is feasible by then it keeps going -- that
    is the rover stopping to think, which the paper notes happens on complex
    terrain.

    The retreat path is held back as a FALLBACK rather than ranked alongside
    the rest (Sec. II-E3: it is selected only when nothing in the fixed or
    heritage trees is feasible). It has to be: every fixed and heritage option
    spans the full 6 m horizon, so their execution times are comparable, while
    the retreat path is 1 m and costs a sixth as much to execute for reasons
    that have nothing to do with its merit. Ranked together, a 1 m reverse
    outranks a path that drives onto the goal, and the rover oscillates
    forward and back over the same two metres forever instead of arriving.
    """
    sel = _rank_and_ace(fixed_tree() + heritage_tree(prev_path),
                        start, maps, gp, wheel_boxes_fn, g, M, limits,
                        belly_fn, verbose, "fixed+heritage", record)
    if sel.path is not None:
        return sel

    R = retreat_tree([h for h in history if h is not None])
    if not R:
        return sel
    rsel = _rank_and_ace(R, start, maps, gp, wheel_boxes_fn, g, 1, limits,
                         belly_fn, verbose, "retreat", record)
    if rsel.path is None:
        return sel
    return Selection(rsel.path, rsel.cost, sel.n_ace + rsel.n_ace,
                     sel.n_considered + rsel.n_considered,
                     stopped_to_think=True)


def first_leg(path: Path, distance: float = DRIVE_PER_CYCLE
              ) -> List[Primitive]:
    """The portion actually executed this cycle: any leading TIP plus the
    first `distance` of arc (receding horizon, Sec. II-A)."""
    out, budget = [], distance
    for pr in path.prims:
        if pr.is_tip:
            out.append(pr)
            continue
        if budget <= 1e-9:
            break
        take = min(budget, abs(pr.l))
        f = take / abs(pr.l)
        out.append(Primitive(math.copysign(take, pr.l), pr.theta * f))
        budget -= take
        if budget <= 1e-9:
            break
    return out


if __name__ == "__main__":
    t = fixed_tree()
    print(f"fixed tree: {len(t)} path options "
          f"({len(TIP_OPTIONS_DEG)} x {len(ARC_OPTIONS_DEG)} x "
          f"{len(ARC_OPTIONS_DEG)})")
    print(f"  published figure: 14 x 11 x 11 = 1694")
    p = t[0]
    print(f"\nexample path: {p}")
    print(f"  end pose from origin: "
          f"{tuple(round(v,3) for v in p.end_pose((0,0,0)))}")
    print(f"  pose samples: {len(p.sample_poses((0,0,0)))}")
    straight = Path((Primitive(0.0, 0.0), Primitive(3.0, 0.0),
                     Primitive(3.0, 0.0)))
    print(f"\nstraight 6 m path: {len(straight.sample_poses((0,0,0)))} poses"
          f"  (paper: 24)")
    tip30 = Path((Primitive(0.0, math.radians(30)), Primitive(3.0, 0.0),
                  Primitive(3.0, 0.0)))
    print(f"30 deg TIP + 6 m:  {len(tip30.sample_poses((0,0,0)))} poses"
          f"  (paper: 27)")
