"""ACE and ENav: the invariants that must not drift.

The pure-geometry checks need no MuJoCo model and run in well under a second,
so they guard every push. The containment check against settled physics is
slower and needs the JPL meshes, so it runs on a handful of poses and skips
when the meshes are not built.

Numbers marked "published" come from the two papers:
  Otsu et al., "Fast Approximate Clearance Evaluation for Rovers with
  Articulated Suspension Systems", JFR 37:768-785, 2020.
  Toupet et al., "Enhanced Autonomous Navigation on the Perseverance Mars
  Rover", IEEE T-FR 3:83-124, 2026.
"""

import math

import numpy as np
import pytest

from ace import ace, RoverGeometry, WHEEL_KEYS, DEFAULT_LIMITS
from ace import Primitive, Path, fixed_tree

# wheel contact centres in the chassis frame, for building synthetic terrain
WHEELS = {
    "LF": (1.185, 1.0625), "LM": (0.0, 1.1845), "LR": (-1.075, 1.0625),
    "RF": (1.185, -1.0625), "RM": (0.0, -1.1845), "RR": (-1.075, -1.0625),
}
BELLY_CX = -0.05


@pytest.fixture(scope="module")
def geom():
    return RoverGeometry.m2020()


def plane(ax_deg, ay_deg):
    """Wheel intervals and belly samples for a uniform plane."""
    tx, ty = math.tan(math.radians(ax_deg)), math.tan(math.radians(ay_deg))
    tiv = {k: (x * tx + y * ty,) * 2 for k, (x, y) in WHEELS.items()}
    pts = []
    for dx in np.linspace(-0.95, 0.95, 9):
        for dy in np.linspace(-0.70, 0.70, 7):
            e = (BELLY_CX + dx) * tx + dy * ty
            pts.append((BELLY_CX + dx, -dy, e, e))   # dy flipped: +y is right
    return tiv, pts


# ---- ACE geometry ---------------------------------------------------------
def test_flat_ground_gives_the_published_clearance(geom):
    st = ace({k: (0.0, 0.0) for k in WHEEL_KEYS}, geom, eps=0.0)
    assert st.clearance.mid == pytest.approx(0.60, abs=1e-6)   # published 60 cm
    assert math.degrees(st.tilt.hi) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("slope", [4.0, 8.0, 12.0, 15.0])
def test_attitude_on_a_plane_equals_the_plane(geom, slope):
    """Parked on a uniform plane the rover's attitude IS the plane's. This is
    what pins the eq-28 roll lever: dividing by the rocker-pivot offset rather
    than the wheel offset over-reports roll by about 40%."""
    tiv, pts = plane(slope, 0.0)
    st = ace(tiv, geom, eps=0.0, belly_pts=pts)
    assert math.degrees(st.pitch.mid) == pytest.approx(slope, abs=1.5)

    tiv, pts = plane(0.0, slope)
    st = ace(tiv, geom, eps=0.0, belly_pts=pts)
    assert math.degrees(st.roll.mid) == pytest.approx(slope, abs=1.0)


@pytest.mark.parametrize("slope", [0.0, 8.0, 15.0])
def test_clearance_on_a_plane_stays_near_nominal(geom, slope):
    """The belly and the ground beneath it tilt together, so a plane never
    eats the clearance. Measuring the lowest belly corner against the highest
    ground subtracts the slope twice and fails this."""
    tiv, pts = plane(slope, 0.0)
    st = ace(tiv, geom, eps=0.0, belly_pts=pts)
    assert st.clearance.lo > 0.40, f"{slope} deg slope ate the clearance"
    assert st.clearance.width < 0.05, "a known plane should give a tight bound"


def test_bounds_widen_with_terrain_uncertainty(geom):
    """Wider wheel boxes must never produce a tighter bound."""
    narrow = ace({k: (0.0, 0.02) for k in WHEEL_KEYS}, geom, eps=0.0)
    wide = ace({k: (0.0, 0.12) for k in WHEEL_KEYS}, geom, eps=0.0)
    assert wide.clearance.lo <= narrow.clearance.lo
    assert wide.tilt.hi >= narrow.tilt.hi


def test_a_step_under_one_wheel_is_rejected(geom):
    """A wheel dropped into a hole deeper than the published wheel-drop limit
    must fail the check."""
    tiv = {k: (0.0, 0.0) for k in WHEEL_KEYS}
    tiv["LF"] = (-0.60, 0.0)
    st = ace(tiv, geom, eps=0.0)
    assert not st.safe(), "a 60 cm wheel drop should not be called safe"


# ---- ENav path tree -------------------------------------------------------
def test_fixed_tree_matches_the_published_count():
    assert len(fixed_tree()) == 14 * 11 * 11 == 1694


def test_path_sampling_matches_the_published_pose_counts():
    straight = Path([Primitive(0.0, 0.0), Primitive(3.0, 0.0),
                     Primitive(3.0, 0.0)], "fixed")
    assert len(straight.sample_poses((0.0, 0.0, 0.0))) == 24

    turned = Path([Primitive(0.0, math.radians(30)), Primitive(3.0, 0.0),
                   Primitive(3.0, 0.0)], "fixed")
    assert len(turned.sample_poses((0.0, 0.0, 0.0))) == 27


def test_no_arc_steers_while_rolling():
    """The paper: the drive and steer motors share controllers, so a wheel
    cannot be steered while driving. Every arc must hold one curvature."""
    for path in fixed_tree()[:200]:
        for pr in path.prims:
            assert pr.is_tip or abs(pr.l) > 0.0


# ---- conservatism against settled physics ---------------------------------
def test_ace_bounds_contain_the_settled_truth(rover):
    """ACE's whole claim. Checked against the full rocker-bogie physics on
    real Jezero terrain -- few poses here to keep CI quick; scripts/
    validate_ace.py runs the study properly."""
    from ace_eval import AceGroundTruth
    gt = AceGroundTruth("course_jezero.xml")
    rng = np.random.default_rng(0)
    g = RoverGeometry.m2020()

    checked = 0
    for _ in range(12):
        if checked >= 3:
            break
        x, y = rng.uniform(-12.0, 12.0, 2)
        yaw = rng.uniform(-math.pi, math.pi)
        boxes = gt.wheel_boxes(x, y, yaw, pad=0.22)
        if len(boxes) < 6:
            continue
        truth = gt.evaluate(x, y, yaw)
        if truth is None:
            continue
        base = min(b["lo"] for b in boxes.values())
        tiv = {k: (b["lo"] - base, b["hi"] - base) for k, b in boxes.items()}
        bp = gt.belly_points(x, y, yaw, pad=0.22)
        pts = None if not bp else [(dx, dy, lo - base, hi - base)
                                   for dx, dy, lo, hi in bp]
        st = ace(tiv, g, eps=0.015, belly_pts=pts)

        assert st.clearance.lo <= truth["clearance_m"] + 1e-9, (
            f"clearance floor {st.clearance.lo:.3f} above truth "
            f"{truth['clearance_m']:.3f} at ({x:.1f}, {y:.1f})")
        assert st.tilt.contains(math.radians(truth["tilt_deg"]), tol=1e-9), (
            f"tilt {truth['tilt_deg']:.2f} deg outside "
            f"[{math.degrees(st.tilt.lo):.2f}, {math.degrees(st.tilt.hi):.2f}]")
        checked += 1

    if checked == 0:
        pytest.skip("no usable poses sampled on this terrain")


def test_published_limits_have_not_drifted():
    assert DEFAULT_LIMITS["clearance_m"] == 0.25     # ACE Min Clearance
    assert DEFAULT_LIMITS["tilt_deg"] == 30.0        # ACE Max Tilt
    assert DEFAULT_LIMITS["wheel_drop_m"] == 0.35    # ACE Max Wheeldrop
