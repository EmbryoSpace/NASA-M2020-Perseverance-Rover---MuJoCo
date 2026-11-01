#!/usr/bin/env python3
"""
ace.py -- Approximate Clearance Evaluation.

Clean-room implementation of:

    K. Otsu, G. Matheron, S. Ghosh, O. Toupet, M. Ono,
    "Fast approximate clearance evaluation for rovers with articulated
     suspension systems", Journal of Field Robotics 37:768-785, 2020.
    doi:10.1002/rob.21892  (preprint arXiv:1808.00031)

ACE computes CONSERVATIVE BOUNDS on a rover's clearance, attitude and
suspension angles for a candidate pose, in closed form, without the iterative
inverse-kinematics settling that a physics engine does. Equation numbers in
the comments below refer to the JFR paper.

The method:

    1. For a pose (x, y, psi), find a rectangular wheel box per wheel that
       conservatively covers that wheel's footprint over any attitude and
       suspension state.
    2. Take the min and max terrain height inside each box -> six intervals.
    3. Propagate those intervals through the suspension kinematics. The
       suspension is monotonic in the wheel heights, so the bounds fall out
       of the extreme configurations rather than a search: eight per side,
       the corners of (z_f, z_m, z_r) at their min/max (Fig. 7).
    4. Check the worst case against the safety limits.

FRAME
    The paper uses a forward-right-DOWN body frame in which a greater wheel
    "height" means the wheel has moved DOWNWARD, with the origin at the
    middle wheels' ground-contact height on flat ground. This model is
    forward-left-UP with the origin at the same place (the RNAV datum).

    So the conversion is z_paper = -z_model, and an interval inverts:
    [lo, hi] in model elevation becomes [-hi, -lo] in paper height. That is
    done ONCE, at `from_terrain()`, and never again inside the math -- the
    sign conventions in this project have caused four separate bugs and they
    all came from doing the flip in more than one place.

CALIBRATION
    The paper's figures leave the sign of several constants (phi_f, phi_m,
    x_od, z_od) ambiguous without the original CAD. Rather than guess them,
    `RoverGeometry.from_urdf_offsets()` derives the link lengths from the JPL
    URDF and then solves the remaining constants by requiring that ACE
    reproduce the known flat-ground state exactly: zero roll, zero pitch,
    zero suspension angles, and the published 60 cm belly clearance. Any
    residual sign error shows up immediately as a flat-ground residual.

WHAT IS NOT FROM THE PAPER
    The paper lists four safety metrics but gives no threshold values for
    M2020 ("predefined safety ranges"). Those are EST here. It does give one
    concrete figure: a perception margin of epsilon = 15 mm restored full
    conservatism in their hardware test, which is `PERCEPTION_EPS`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# Published: perception margin that restored 100% conservatism in the paper's
# hardware experiment (Section 4.2).
PERCEPTION_EPS = 0.015          # m

# Safety thresholds. Three of the four are PUBLISHED: Toupet et al. 2026
# (IEEE T-FR 3:83-124) plots them as reference lines in its Monte-Carlo V&V
# figures, consistently across all three terrain types --
#   "ACE Min Clearance = 25.0" cm, "ACE Max Wheeldrop = 35.0" cm,
#   "ACE Max Tilt = 30.0" deg.
# The suspension-angle ranges are still unpublished and remain EST.
DEFAULT_LIMITS = dict(
    clearance_m=0.25,           # published: ACE Min Clearance 25 cm
    tilt_deg=30.0,              # published: ACE Max Tilt 30 deg
    wheel_drop_m=0.35,          # published: ACE Max Wheeldrop 35 cm
    rocker_deg=25.0,            # EST -- "prescribed limits", no value given
    bogie_deg=25.0,             # EST
)

# ENav samples poses along a path every 25 cm for planning and every 10 cm for
# mid-arc reevaluation (Toupet 2026, Sec. II-G). A 6 m arc is ~24 ACE runs.
POSE_INTERVAL_PLAN = 0.25       # m
POSE_INTERVAL_MIDARC = 0.10     # m


# ---------------------------------------------------------------- intervals
class Interval:
    """Closed real interval with the arithmetic of Section 3.1, eqs 16-20."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo: float, hi: float = None):
        if hi is None:
            hi = lo
        if hi < lo:
            lo, hi = hi, lo
        self.lo = float(lo)
        self.hi = float(hi)

    # eq 17 / 18
    def __add__(self, o):
        o = _as_iv(o)
        return Interval(self.lo + o.lo, self.hi + o.hi)

    def __sub__(self, o):
        o = _as_iv(o)
        return Interval(self.lo - o.hi, self.hi - o.lo)

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __mul__(self, k: float):
        k = float(k)
        return Interval(self.lo * k, self.hi * k) if k >= 0 else \
            Interval(self.hi * k, self.lo * k)

    __rmul__ = __mul__
    __radd__ = __add__

    def __truediv__(self, k: float):
        return self * (1.0 / float(k))

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def mid(self) -> float:
        return 0.5 * (self.lo + self.hi)

    def contains(self, x: float, tol: float = 0.0) -> bool:
        return (self.lo - tol) <= x <= (self.hi + tol)

    def grow(self, eps: float) -> "Interval":
        return Interval(self.lo - eps, self.hi + eps)

    def __repr__(self):
        return f"[{self.lo:.4f}, {self.hi:.4f}]"


def _as_iv(x) -> Interval:
    return x if isinstance(x, Interval) else Interval(x)


def iv_hull(values: Sequence[float]) -> Interval:
    return Interval(min(values), max(values))


# ---------------------------------------------------------------- geometry
@dataclass
class RoverGeometry:
    """Suspension link geometry, in the paper's parameterisation.

    Rocker triangle  (A = front axle F, B = bogie joint, C = rocker joint D):
        l_df  D->F,  l_fb  F->B,  phi_f
    Bogie triangle   (A = middle axle M, B = rear axle R, C = bogie joint B):
        l_bm  B->M,  l_mr  M->R,  phi_m
    """
    l_df: float
    l_fb: float
    phi_f: float
    l_bm: float
    l_mr: float
    phi_m: float
    y_od: float                 # lateral body centre -> differential joint
    x_od: float = 0.0
    # wheel lateral offsets, which is what the eq-9 rocker heights carry
    y_wheel_f: float = 1.0625
    y_wheel_m: float = 1.1845
    y_wheel_r: float = 1.0625
    y_roll: float = 0.0         # eq-28 lever, set by calibrate_flat
    z_od: float = 0.0
    wheel_radius: float = 0.2625
    belly_clearance: float = 0.60   # c_0, nominal, published 60 cm
    belly_len: float = 1.90         # l_p
    belly_wid: float = 1.40         # w_p
    kappa_d0: float = 0.0           # flat-ground rocker link angle
    kappa_b0: float = 0.0           # flat-ground bogie link angle
    z_o0: float = 0.0               # flat-ground body height, for calibration
    _z_ref: float = 0.0             # clearance anchor, set by calibrate_flat

    @staticmethod
    def m2020() -> "RoverGeometry":
        """M2020 geometry, derived from the JPL URDF joint origins.

        Side-view (x, z) vertex positions in the model frame. Deriving the
        triangle parameters from the vertices rather than hard-coding them
        avoids the two mistakes that are easy here:

          * phi_a in eq 2 is the INTERNAL angle of the triangle at vertex A,
            not the inclination of link AC. For the bogie triangle the two
            coincide (M and R are level on flat ground, 41.63 deg) which makes
            the error invisible there; for the rocker triangle they differ by
            14 deg (21.82 vs 35.57).
          * kappa_d0 is kappa_d EVALUATED on flat ground, not phi_f. The front
            axle and the bogie joint are not level, so the asin term does not
            vanish. Setting kappa_d0 = phi_f puts a constant 13.75 deg of
            pitch and bogie angle into every result.
        """
        # rocker-local: rocker joint at the origin
        D = np.array([0.0, 0.0])
        F = np.array([0.881, -0.63])        # front axle
        Bj = np.array([-0.754, -0.23])      # bogie joint
        # bogie-local: bogie joint at the origin
        Bo = np.array([0.0, 0.0])
        M = np.array([0.44998, -0.40])      # middle axle
        R = np.array([-0.625, -0.40])       # rear axle

        def internal(a, b, c):
            u, v = b - a, c - a
            return math.acos(_clamp(float(np.dot(u, v) /
                                          (np.linalg.norm(u) * np.linalg.norm(v)))))

        g = RoverGeometry(
            l_df=float(np.linalg.norm(D - F)),
            l_fb=float(np.linalg.norm(Bj - F)),
            phi_f=internal(F, Bj, D),
            l_bm=float(np.linalg.norm(Bo - M)),
            l_mr=float(np.linalg.norm(R - M)),
            phi_m=internal(M, R, Bo),
            y_od=0.79959,
            x_od=0.30402,
            z_od=0.89288,
        )
        g.calibrate_flat()
        return g

    # -- calibration ---------------------------------------------------
    def calibrate_flat(self) -> None:
        """Pin kappa_d0, kappa_b0 and the belly reference on flat ground.

        On flat ground every wheel sits on z = 0, so in the paper's downward
        frame every wheel height is -R_w. Evaluating the triangles there gives
        the reference link angles, which is what eqs 11-14 subtract off.
        """
        zw = -self.wheel_radius
        self.kappa_b0 = 0.0
        self.kappa_d0 = 0.0
        kb = self.kappa_b(zw, zw)
        zb = self.z_bogie(zw, zw)
        kd = self.kappa_d(zw, zb)
        self.kappa_b0 = kb
        self.kappa_d0 = kd

        # Calibrate the eq-28 roll lever.
        #
        # y_od is the lateral offset of the rocker pivot from the body centre,
        # but [z_d] is not the pivot height -- it is built from WHEEL heights
        # (eq 9), and the wheels sit outboard of the pivot. Rolling the body
        # moves the wheel plane by the full wheel offset, not the pivot
        # offset, so dividing by the pivot offset over-reports roll by the
        # ratio between them: a 4 deg lateral slope came out as 5.5 deg.
        #
        # The invariant that pins it: on a uniform plane the rover's attitude
        # IS the plane's attitude. Solve for the lever that satisfies that.
        a = math.radians(10.0)              # mid-range anchor slope
        t = math.tan(a)
        zdl, zdr = [], []
        for side, ys in (("l", 1.0), ("r", -1.0)):
            zf = -(ys * self.y_wheel_f * t) - self.wheel_radius
            zm = -(ys * self.y_wheel_m * t) - self.wheel_radius
            zr = -(ys * self.y_wheel_r * t) - self.wheel_radius
            zb = self.z_bogie(zm, zr)
            zd = zf - self.l_df * math.sin(self.kappa_d(zf, zb))
            (zdl if side == "l" else zdr).append(zd)
        self.y_roll = abs(zdr[0] - zdl[0]) / (2.0 * math.sin(a))

        # Anchor the body-height reference so flat ground yields exactly the
        # published 60 cm clearance.
        self._z_ref = 0.0
        flat = {k: Interval(zw) for k in WHEEL_KEYS}
        st = AceState.propagate(self, flat, None)
        self._z_ref = st.clearance.mid - self.belly_clearance

    # -- triangles -----------------------------------------------------
    def kappa_d(self, z_f: float, z_b: float) -> float:
        """eq 2 for the rocker triangle. Raises if the pose is unreachable."""
        s = (z_f - z_b) / self.l_fb
        if abs(s) > 1.0:
            raise Unreachable(f"rocker triangle: |dz|={abs(z_f-z_b):.3f} "
                              f"> l_fb={self.l_fb:.3f}")
        return self.phi_f + math.asin(s)

    def kappa_b(self, z_m: float, z_r: float) -> float:
        """eq 2 for the bogie triangle."""
        s = (z_m - z_r) / self.l_mr
        if abs(s) > 1.0:
            raise Unreachable(f"bogie triangle: |dz|={abs(z_m-z_r):.3f} "
                              f"> l_mr={self.l_mr:.3f}")
        return self.phi_m + math.asin(s)

    def z_bogie(self, z_m: float, z_r: float) -> float:
        """eq 8 -- bogie joint height from middle and rear wheel heights."""
        return z_m - self.l_bm * math.sin(self.kappa_b(z_m, z_r))

    def z_rocker(self, z_f: float, z_b: float) -> float:
        """eq 9 -- rocker joint height from front wheel and bogie joint."""
        return z_f - self.l_df * math.sin(self.kappa_d(z_f, z_b))


class Unreachable(ValueError):
    """The wheel heights cannot be realised by the linkage (eq 2's domain)."""


WHEEL_KEYS = ("LF", "LM", "LR", "RF", "RM", "RR")
SIDES = {"l": ("LF", "LM", "LR"), "r": ("RF", "RM", "RR")}


# ---------------------------------------------------------------- state
@dataclass
class AceState:
    """The bounds ACE returns."""
    roll: Interval
    pitch: Interval
    tilt: Interval
    rocker_l: Interval
    rocker_r: Interval
    bogie_l: Interval
    bogie_r: Interval
    z_o: Interval
    clearance: Interval
    wheel_drop: float
    unreachable: bool = False
    note: str = ""

    @staticmethod
    def propagate(g: RoverGeometry,
                  z: Dict[str, Interval],
                  belly_pts: Optional[Sequence[Tuple[
                      float, float, float, float]]] = None) -> "AceState":
        """Steps 3 of the algorithm: wheel-height intervals -> state bounds.

        The suspension is monotonic in the wheel heights, so the bounds are
        attained at the extreme configurations -- the eight corners of
        (z_f, z_m, z_r) per side (Fig. 7) -- rather than needing a search.
        """
        unreachable = False
        notes = []

        # -- per-side corner enumeration (Fig. 7): three wheel-height
        # intervals per side, so eight extreme configurations each.
        side_rows: Dict[str, list] = {}
        beta_hull: Dict[str, Interval] = {}
        for side, (kf, km, kr) in SIDES.items():
            rows, betas = [], []
            for ef, em, er in product((0, 1), repeat=3):
                zf = z[kf].hi if ef else z[kf].lo
                zm = z[km].hi if em else z[km].lo
                zr = z[kr].hi if er else z[kr].lo
                try:
                    zb = g.z_bogie(zm, zr)
                    kd = g.kappa_d(zf, zb)
                    kb = g.kappa_b(zm, zr)
                except Unreachable as e:
                    unreachable = True
                    notes.append(str(e))
                    continue
                rows.append((zf - g.l_df * math.sin(kd), kd))
                betas.append(kd - kb - g.kappa_d0 + g.kappa_b0)  # eq 11/12
            if not rows:
                raise Unreachable("no reachable configuration on side " + side)
            side_rows[side] = rows
            beta_hull[side] = iv_hull(betas)

        # -- joint enumeration over both sides: 8 x 8 = 64 extreme
        # configurations, each carried through the whole chain as a POINT
        # before anything is hulled. Hulling each side first and combining the
        # hulls -- as an earlier version did -- pairs a left-side corner with a
        # right-side corner that cannot occur together, so it reports maximum
        # pitch and maximum roll simultaneously and inflates every bound
        # downstream of them.
        cfgs = []
        rock_v, roll_v, pitch_v, zo_v = [], [], [], []
        for zd_l, kd_l in side_rows["l"]:
            for zd_r, kd_r in side_rows["r"]:
                rocker = (kd_r - kd_l) / 2.0                     # eq 25
                roll = math.asin(_clamp((zd_r - zd_l)
                                        / (2 * g.y_roll)))       # eq 28
                pitch = g.kappa_d0 - (kd_l + kd_r) / 2.0         # eq 29
                ap, ar = abs(pitch), abs(roll)
                z_o = ((zd_l + zd_r) / 2.0                       # eq 30
                       - g.z_od * math.cos(ap) * math.cos(ar)
                       + g.x_od * math.sin(pitch) * math.cos(ar))
                rock_v.append(rocker)
                roll_v.append(roll)
                pitch_v.append(pitch)
                zo_v.append(z_o)
                cfgs.append((z_o, pitch, roll))

        rocker_l = iv_hull(rock_v)
        rocker_r = -rocker_l
        roll = iv_hull(roll_v)
        pitch = iv_hull(pitch_v)
        z_o = iv_hull(zo_v)

        # tilt is the angle off vertical; bound it from |roll| and |pitch|
        ar = max(abs(roll.lo), abs(roll.hi))
        ap = max(abs(pitch.lo), abs(pitch.hi))
        tilt = Interval(max(_abs_lo(roll), _abs_lo(pitch)),
                        math.acos(_clamp(math.cos(min(ar, math.pi / 2))
                                         * math.cos(min(ap, math.pi / 2)))))
        abs_roll = Interval(_abs_lo(roll), ar)
        abs_pitch = Interval(_abs_lo(pitch), ap)
        ref = getattr(g, "_z_ref", 0.0)

        # -- eqs 31-32, clearance.
        #
        # Evaluated PER GROUND POINT when the caller can sample the terrain
        # under the belly footprint. The belly pan and the ground beneath it
        # tilt together, so the pan's lowest corner sits over the LOWEST
        # ground, not the highest. Taking the global extreme of each
        # independently subtracts the slope twice -- 26 cm across the 1.9 m
        # belly at 8 deg -- which drives the lower bound under the published
        # 25 cm limit on terrain the real rover crosses without pausing.
        #
        # Clearance is a minimum over the footprint by definition, so both
        # ends of the interval take the min over points: the lower bound
        # against the highest terrain each point could be, the upper against
        # the lowest.
        if belly_pts:
            c_lo, c_hi = math.inf, -math.inf
            for z_oc, pitch_c, roll_c in cfgs:
                ap_c, ar_c = abs(pitch_c), abs(roll_c)
                z_bc = (z_oc - g.belly_clearance
                        * math.cos(ap_c) * math.cos(ar_c))
                lo_here, hi_here = math.inf, math.inf
                for dx, dy, t_lo, t_hi in belly_pts:
                    # dx forward, dy RIGHT (the paper's frame). Nose-up is
                    # pitch > 0, which lifts a forward point: smaller in the
                    # downward frame.
                    z_b = (z_bc - dx * math.sin(pitch_c)
                           + dy * math.sin(roll_c))
                    lo_here = min(lo_here, -t_hi - g.wheel_radius - z_b)
                    hi_here = min(hi_here, -t_lo - g.wheel_radius - z_b)
                c_lo = min(c_lo, lo_here)
                c_hi = max(c_hi, hi_here)
            clearance = Interval(c_lo - ref, c_hi - ref)
        else:
            # Fallback: no footprint samples, so the pan's lowest corner is
            # bounded by the attitude hull and the ground by the wheel boxes.
            # Correct but much looser, for exactly the reason above.
            c_pitch_hi = math.cos(abs_pitch.lo)
            c_pitch_lo = math.cos(abs_pitch.hi)
            c_roll_hi = math.cos(abs_roll.lo)
            c_roll_lo = math.cos(abs_roll.hi)
            z_p = Interval(
                z_o.lo - g.belly_clearance * c_pitch_hi * c_roll_hi
                + 0.5 * g.belly_len * math.sin(abs_pitch.lo) * c_roll_lo
                + 0.5 * g.belly_wid * math.sin(abs_roll.lo),
                z_o.hi - g.belly_clearance * c_pitch_lo * c_roll_lo
                + 0.5 * g.belly_len * math.sin(abs_pitch.hi) * c_roll_hi
                + 0.5 * g.belly_wid * math.sin(abs_roll.hi))
            z_ground = Interval(min(v.lo for v in z.values()),
                                max(v.hi for v in z.values()))
            clearance = Interval(z_ground.lo - z_p.hi - ref,
                                 z_ground.hi - z_p.lo - ref)

        drop = max(v.width for v in z.values())

        return AceState(roll=roll, pitch=pitch, tilt=tilt,
                        rocker_l=rocker_l, rocker_r=rocker_r,
                        bogie_l=beta_hull["l"],
                        bogie_r=beta_hull["r"],
                        z_o=z_o, clearance=clearance, wheel_drop=drop,
                        unreachable=unreachable,
                        note="; ".join(sorted(set(notes))[:2]))

    # -- reporting -----------------------------------------------------
    def safe(self, limits: dict = None) -> bool:
        L = limits or DEFAULT_LIMITS
        if self.unreachable:
            return False
        return (self.clearance.lo >= L["clearance_m"]
                and math.degrees(self.tilt.hi) <= L["tilt_deg"]
                and max(abs(self.rocker_l.lo), abs(self.rocker_l.hi))
                <= math.radians(L["rocker_deg"])
                and max(abs(self.bogie_l.lo), abs(self.bogie_l.hi),
                        abs(self.bogie_r.lo), abs(self.bogie_r.hi))
                <= math.radians(L["bogie_deg"])
                and self.wheel_drop <= L["wheel_drop_m"])

    def report(self) -> str:
        d = math.degrees
        return (
            f"  roll      [{d(self.roll.lo):+7.2f},{d(self.roll.hi):+7.2f}] deg\n"
            f"  pitch     [{d(self.pitch.lo):+7.2f},{d(self.pitch.hi):+7.2f}] deg\n"
            f"  tilt      [{d(self.tilt.lo):7.2f},{d(self.tilt.hi):7.2f}] deg\n"
            f"  rocker L  [{d(self.rocker_l.lo):+7.2f},{d(self.rocker_l.hi):+7.2f}] deg\n"
            f"  bogie  L  [{d(self.bogie_l.lo):+7.2f},{d(self.bogie_l.hi):+7.2f}] deg\n"
            f"  bogie  R  [{d(self.bogie_r.lo):+7.2f},{d(self.bogie_r.hi):+7.2f}] deg\n"
            f"  clearance [{self.clearance.lo:7.3f},{self.clearance.hi:7.3f}] m\n"
            f"  wheel drop {self.wheel_drop:.3f} m"
            + (f"\n  UNREACHABLE: {self.note}" if self.unreachable else ""))


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _abs_lo(iv: Interval) -> float:
    """Smallest absolute value an interval attains."""
    return 0.0 if iv.lo <= 0.0 <= iv.hi else min(abs(iv.lo), abs(iv.hi))


# ---------------------------------------------------------------- the API
def from_terrain(terrain_iv: Dict[str, Tuple[float, float]],
                 g: RoverGeometry = None,
                 eps: float = PERCEPTION_EPS) -> Dict[str, Interval]:
    """Model-frame terrain intervals -> paper-frame wheel height intervals.

    `terrain_iv` maps each wheel name to (min_elevation, max_elevation) in the
    MODEL frame (z up, metres). This is the single place the frame flips:
    z_paper = -z_model, so an interval inverts. `eps` inflates each interval
    for perception error, per the paper's Section 4.2.
    """
    g = g or RoverGeometry.m2020()
    out = {}
    for k, (lo, hi) in terrain_iv.items():
        # z-up elevation -> z-down height, with the axle R_w above the ground
        out[k] = Interval(-hi - g.wheel_radius,
                          -lo - g.wheel_radius).grow(eps)
    return out


def ace(terrain_iv: Dict[str, Tuple[float, float]],
        g: RoverGeometry = None,
        eps: float = PERCEPTION_EPS,
        limits: dict = None,
        belly_pts: Optional[Sequence[Tuple[
                      float, float, float, float]]] = None) -> AceState:
    """Run ACE for one pose.

    terrain_iv: {wheel: (min_elev, max_elev)} in the model frame, as produced
    by the wheel boxes. Returns the state bounds; `.safe(limits)` applies the
    thresholds.
    """
    g = g or RoverGeometry.m2020()
    pts = None
    if belly_pts:
        # the same perception margin the wheel boxes get
        pts = [(dx, dy, lo - eps, hi + eps) for dx, dy, lo, hi in belly_pts]
    return AceState.propagate(g, from_terrain(terrain_iv, g, eps), pts)


def slip_vector(slope_rad: float, d_predict: float,
                d_max: float = 2.0,
                slip_curve=None) -> float:
    """Worst-case slip magnitude, per Toupet et al. 2026 Sec. II-G.3:

        d_slip = r_slip * min(d_predict, d_max)

    r_slip is the slip ratio, a function of slope ("slip curve") held in a
    parameter table fitted to past drive data. That table is not published,
    so `slip_curve` defaults to a monotonic stand-in and is EST -- pass your
    own callable to replace it.

    d_predict is accumulated odometry since the images used for the current
    height map were taken, because position uncertainty against that map
    grows after the last look at the terrain. The flight numbers: image
    processing takes ~10 s, during which the rover moves ~50 cm; replanning
    happens about every 1 m; slip growth stops at the next replan point, so
    d_max is the distance from the imaging point to there (~2 m).

    ACE's slip DIRECTION is taken as the gradient of the local slope.
    """
    if slip_curve is None:
        # EST: ~0 on the flat, rising with slope. Replace with the real table.
        slip_curve = lambda s: 0.10 * math.tan(max(0.0, s)) / math.tan(math.radians(20))
    return float(slip_curve(slope_rad)) * min(d_predict, d_max)


def wheel_box_halfsize(d_predict: float = 0.0,
                       slope_rad: float = 0.0,
                       base: float = 0.22,
                       d_max: float = 2.0) -> float:
    """Wheel box half-size: the base box grown by the slip vector.

    The base wheel box covers the non-slip uncertainties and is itself a
    parameter set; `base` is EST. The growth term is the published slip
    model above. ACE's guarantee is that the rover is safe for any deviation
    within this envelope.
    """
    return base + slip_vector(slope_rad, d_predict, d_max)


def ace_cost(state: "AceState", limits: dict = None,
             weights: dict = None,
             unknown_in_box: bool = False,
             within_replan: bool = True) -> float:
    """ACE cost for one pose, per Toupet et al. 2026 Sec. II-G.1:

        cost = sum_i C_i * b(x_i, c)  +  p_U
        b(x, c) = inf                if the constraint is violated
                  1 / |x - c|        if it is satisfied

    So the cost is ~0 with ample margin, large when a constraint is only just
    satisfied, and infinite when violated. That is what injects ENav's
    preference for paths that keep away from the limits, rather than merely
    passing. The C_i weights are not published and are EST.

    p_U penalises unknown cells in a wheel box: infinite up to the next
    replan point (~1 m), zero beyond it, because a hazard hidden in far
    unknown terrain gets avoided on a later cycle.
    """
    L = limits or DEFAULT_LIMITS
    W = weights or dict(clearance=1.0, tilt=1.0, rocker=1.0,
                        bogie=1.0, wheel_drop=1.0)

    if unknown_in_box and within_replan:
        return math.inf
    if state.unreachable:
        return math.inf

    def b(x, c, upper=True):
        violated = (x > c) if upper else (x < c)
        if violated:
            return math.inf
        d = abs(x - c)
        return math.inf if d < 1e-9 else 1.0 / d

    total = 0.0
    total += W["clearance"] * b(state.clearance.lo, L["clearance_m"], upper=False)
    total += W["tilt"] * b(math.degrees(state.tilt.hi), L["tilt_deg"])
    total += W["wheel_drop"] * b(state.wheel_drop, L["wheel_drop_m"])
    rock = max(abs(math.degrees(state.rocker_l.lo)),
               abs(math.degrees(state.rocker_l.hi)))
    total += W["rocker"] * b(rock, L["rocker_deg"])
    bog = max(abs(math.degrees(state.bogie_l.lo)), abs(math.degrees(state.bogie_l.hi)),
              abs(math.degrees(state.bogie_r.lo)), abs(math.degrees(state.bogie_r.hi)))
    total += W["bogie"] * b(bog, L["bogie_deg"])
    return total


if __name__ == "__main__":
    g = RoverGeometry.m2020()
    print("M2020 geometry (link lengths from the JPL URDF):")
    for k in ("l_df", "l_fb", "l_bm", "l_mr", "y_od", "x_od", "z_od"):
        print(f"  {k:6s} = {getattr(g, k):.5f}")
    print(f"  phi_f  = {math.degrees(g.phi_f):.3f} deg")
    print(f"  phi_m  = {math.degrees(g.phi_m):.3f} deg")

    print("\nflat ground (every wheel box reads 0.000 m, no perception error):")
    flat = {k: (0.0, 0.0) for k in WHEEL_KEYS}
    st = ace(flat, g, eps=0.0)
    print(st.report())

    print("\none wheel box 0.20 m higher under the left front:")
    bumped = dict(flat)
    bumped["LF"] = (0.0, 0.20)
    print(ace(bumped, g, eps=0.0).report())
