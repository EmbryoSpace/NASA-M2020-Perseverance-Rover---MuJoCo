"""
ace -- Approximate Clearance Evaluation and the ENav planner that uses it.

Implemented from the published papers:

    N. Otsu et al., "Fast Approximate Clearance Evaluation for Rovers with
    Articulated Suspension Systems", Journal of Field Robotics 37:768-785,
    2020. doi:10.1002/rob.21892

    O. Toupet, M. Ono, T. S. Del Sesto, M. W. Maimone, M. M. McHenry,
    "Enhanced Autonomous Navigation on the Perseverance Mars Rover", IEEE
    Transactions on Field Robotics 3:83-124, 2026.
    doi:10.1109/TFR.2025.3636366

This package is pure kinematics: it imports nothing beyond numpy, and in
particular no simulator. That is the point of ACE -- it predicts the settled
pose of the suspension without settling it. The MuJoCo oracle that the
predictions are *checked* against lives separately, in scripts/ace_eval.py.

    from ace import ace, RoverGeometry
    bounds = ace({"LF": (0.0, 0.05), ...})      # per-wheel terrain intervals
    bounds.safe()                                # against the published limits

    from ace import select_path, first_leg      # the ENav side
"""

from .core import (
    Interval,
    iv_hull,
    RoverGeometry,
    Unreachable,
    AceState,
    from_terrain,
    ace,
    slip_vector,
    wheel_box_halfsize,
    ace_cost,
    PERCEPTION_EPS,
    DEFAULT_LIMITS,
    POSE_INTERVAL_PLAN,
    POSE_INTERVAL_MIDARC,
    WHEEL_KEYS,
    SIDES,
)

from .enav import (
    Primitive,
    Path,
    fixed_tree,
    heritage_tree,
    retreat_tree,
    EnavMaps,
    GlobalPlanner,
    local_cost,
    global_cost,
    penalty,
    arrival_time,
    path_cost,
    run_ace,
    Selection,
    select_path,
    first_leg,
    PLANNING_HORIZON,
    DRIVE_PER_CYCLE,
    ACE_INTERVAL,
    GOAL_TOLERANCE,
    TIP_SAMPLE_DEG,
    ARC_LEN,
    WHEEL_GROUND_SPEED,
    TIP_RATE_DEG,
    COSTMAP_RES,
    HEIGHTMAP_RES,
    ENVELOPE_RADIUS,
    RETREAT_MEMORY,
    TIP_OPTIONS_DEG,
    ARC_OPTIONS_DEG,
    M_ACE_RUNS,
    T_STEER_ALL,
    T_STEER_ONE_BY_ONE,
    STEER_TILT_THRESHOLD_DEG,
)

from . import core, enav  # noqa: F401  (so `from ace import enav` works)

__all__ = [
    # ACE
    "Interval", "iv_hull", "RoverGeometry", "Unreachable", "AceState",
    "from_terrain", "ace", "slip_vector", "wheel_box_halfsize", "ace_cost",
    "PERCEPTION_EPS", "DEFAULT_LIMITS", "POSE_INTERVAL_PLAN",
    "POSE_INTERVAL_MIDARC", "WHEEL_KEYS", "SIDES",
    # ENav
    "Primitive", "Path", "fixed_tree", "heritage_tree", "retreat_tree",
    "EnavMaps", "GlobalPlanner", "local_cost", "global_cost", "penalty",
    "arrival_time", "path_cost", "run_ace", "Selection", "select_path",
    "first_leg", "PLANNING_HORIZON", "DRIVE_PER_CYCLE", "ACE_INTERVAL",
    "GOAL_TOLERANCE", "TIP_SAMPLE_DEG", "ARC_LEN", "WHEEL_GROUND_SPEED",
    "TIP_RATE_DEG", "COSTMAP_RES", "HEIGHTMAP_RES", "ENVELOPE_RADIUS",
    "RETREAT_MEMORY", "TIP_OPTIONS_DEG", "ARC_OPTIONS_DEG", "M_ACE_RUNS",
    "T_STEER_ALL", "T_STEER_ONE_BY_ONE", "STEER_TILT_THRESHOLD_DEG",
    # submodules
    "core", "enav",
]
