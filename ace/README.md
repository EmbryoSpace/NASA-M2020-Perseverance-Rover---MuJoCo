# ACE and ENav

The two algorithms Perseverance navigates with, implemented from the
published papers. This folder is the library; it imports numpy and nothing
else.

| | |
|---|---|
| [`core.py`](core.py) | **ACE** — Approximate Clearance Evaluation |
| [`enav.py`](enav.py) | **ENav** — the planner that uses it |

---

## The problem

A car has a fixed shape: you know where its underside is relative to its
wheels. Perseverance does not. Its six wheels hang off a folding linkage — a
rocker-bogie — so that when one wheel rides up over a rock, the leg folds and
the others stay down. That is what lets it climb obstacles taller than its
own wheels without tipping.

The cost is that the rover's tilt and the height of its belly are not
properties of the rover. They are *consequences* of wherever the six wheels
come to rest.

So "will my belly clear that rock?" cannot be answered by looking at the
rock. It depends on all six wheel positions at once — and those are known
only from a photograph, to within a few centimetres.

## What ACE does

ACE answers that question **without simulating it**. Simulation is far too
slow: the rover considers ~1,700 candidate paths per decision, each with two
dozen poses to check.

Instead, ACE takes the terrain's height *uncertainty* and pushes it through
the suspension geometry, returning a **guaranteed range** for each quantity
it cares about:

```python
from ace import ace

bounds = ace({"LF": (0.00, 0.05), "LM": (0.01, 0.04), "LR": (0.00, 0.03),
              "RF": (0.02, 0.06), "RM": (0.00, 0.05), "RR": (0.01, 0.04)})

bounds.clearance      # [0.45, 0.67] m  -- belly to ground
bounds.tilt           # [0.02, 0.16] rad
bounds.rocker_l       # [-0.06, 0.05] rad
bounds.wheel_drop     # 0.098 m
bounds.safe()         # True, against the published limits
```

It never claims to know the exact outcome. It claims to know what the outcome
cannot be outside of.

### Conservatism is the whole contract

The true settled value must always land inside the range. If ACE is wrong, it
has to be wrong *pessimistically* — reporting less clearance than there
really is. An optimistic range drives the rover into a rock; a pessimistic
one only makes it cautious.

But conservatism alone is worthless. A range reporting "clearance is
somewhere between 8 cm and 73 cm" is never wrong and never useful — it
rejects every path, because 8 cm is below the 25 cm limit. **Both halves
matter: the bound must hold, and it must be tight.**

### How it gets there

```
terrain height intervals under each wheel box
        ↓   from_terrain()  — one frame flip, z_paper = −z_model
wheel height intervals
        ↓   64 extreme configurations, 8 per side (Fig. 7)
per-configuration suspension state
        ↓   rocker, bogie, roll, pitch, body height
hull of the finished states
        ↓   eqs 31–32, per ground point under the belly
clearance, tilt, wheel drop, suspension angles
```

Two details that are easy to get wrong, and both of which produce
plausible-looking output when wrong:

**The configurations are carried through whole before anything is hulled.**
The suspension is monotonic in the wheel heights, so every bound is attained
at one of the extreme configurations. Hulling each side separately and then
combining the hulls pairs a left-side corner with a right-side corner that
cannot occur together — it reports maximum pitch and maximum roll at the same
instant, which no real arrangement produces.

**Clearance is evaluated per ground point.** The belly pan and the ground
beneath it tilt together, so the pan's lowest corner sits over the *lowest*
ground. Taking the global extreme of each independently pairs the lowest
corner with the highest ground anywhere in the footprint, subtracting the
slope twice — 26 cm across a 1.9 m belly at 8°, against a 25 cm limit.

## What ENav does

ENav is the navigator. Each cycle:

1. **Fan out.** Build a tree of 1,694 candidate paths reaching 6 m ahead —
   14 turn-in-place options × 11 first arcs × 11 second arcs.
2. **Rank cheaply.** Score every path by drive time + time-to-goal + a tilt
   and roughness penalty, and sort.
3. **Spend ACE down the ranking.** ACE runs at every pose, one every 25 cm —
   roughly 24 per path. It is the expensive part, which is why the ranking
   exists.
4. **Pick.** The winner minimises `path_cost + ace_cost`.
5. **Drive one metre**, then throw the plan away and start again.

ACE is not a pass/fail gate bolted on the end. It returns either **infinity**,
vetoing the path, or a **finite cost that is added** to the cheap one — so a
path that merely grazes its limits is costed worse than one with room to
spare, and that changes which path wins.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../images/anim_enav_explainer_dark.gif">
    <img src="../images/anim_enav_explainer_light.gif" width="900" alt="The ENav loop: the plan on the left, and what ACE predicted for belly clearance and tilt along the chosen path on the right">
  </picture>
</p>

*Left: the plan. The ring is where all 1,694 options end; grey passed ACE, red
was rejected, blue won, green is where the rover has been. Right: what ACE
promised before setting off, against the limits that decide whether the path
is allowed. The marker is the rover now.*

### Two mechanical constraints that shape everything

Both are stated in the ENav paper, and both are why the primitives look the
way they do:

> "Wheels cannot be steered while driving. This is because the driving and
> steering motors share the same motor controllers."

So there is no path of continuously changing curvature. Every option is a
turn in place followed by arcs, each held at a single curvature.

> "Among the six wheels, only the front and rear wheels are steerable.
> Therefore, Perseverance cannot move sideways in a crab-like motion."

## Running it

```bash
python scripts/run_enav.py --goal 18 6 --cycles 30        # drive a goal
python scripts/validate_ace.py --poses 40 --eps 0.015     # check the bounds
python scripts/make_enav_explainer.py                     # the animation above
```

## Does it actually bound the truth?

The papers validate ACE against ROAMS, JPL's internal simulator, which nobody
outside JPL can run. Here it is validated against the full rocker-bogie
physics in MuJoCo, on terrain built from the real Jezero HiRISE DTM — which
anyone can re-run.

Each pose: sample a random position and heading, ask ACE for its ranges from
the terrain alone, then drop the rover there, let the suspension settle under
Mars gravity, and measure what actually happened.

| Across 78 poses, two seeds, ε = 15 mm | |
|---|---|
| Tilt contained | 78 / 78 |
| Clearance contained | 78 / 78 |
| Rocker contained | 78 / 78 |
| Bogie contained | 78 / 78 |
| Clearance range width (median) | 0.153 m |
| Tilt range width (median) | 4.6° |
| Worst case: floor below true clearance | 5.0 cm |

Conservative on every pose, and tight enough to be useful.

`tests/test_ace.py` guards the invariants in CI, including containment
against settled physics on a few poses, so this cannot rot quietly.

## What matches the papers

| Quantity | Published | Here |
|---|---|---|
| Path options in the fixed tree | 14 × 11 × 11 = 1694 | 1694 |
| Poses along a straight 6 m path | 24 | 24 |
| Poses on a 30° turn-in-place plus 6 m | 27 | 27 |
| ACE clearance limit | 25 cm | 25 cm |
| ACE tilt limit | 30° | 30° |
| ACE wheel-drop limit | 35 cm | 35 cm |
| Planning horizon | 6 m | 6 m |
| Driven per cycle | 1 m | 1 m |
| ACE evaluation interval | 25 cm | 25 cm |
| Max wheel ground speed | 4.2 cm/s | 4.2 cm/s |
| Max turn-in-place rate | 1.5 °/s | 1.5 °/s |
| Cost map resolution | 1 m | 1 m |
| Plane-fit envelope radius | 2 m | 2 m |
| Flat-ground belly clearance | 60 cm | 60 cm |

## What is estimated

The papers give the structure but not the constant for these. They are marked
`EST` where they appear in the source.

| | |
|---|---|
| ACE cost weights (`C_i`) | the barrier weighting per constraint |
| Rocker and bogie angle limits | 25° each |
| `M`, ACE runs spent per cycle | 120 |
| Penalty weights and thresholds | `C_tau`, `C_r`, `tau_th`, `r_th` |
| Steering timings | all-at-once vs one-by-one |
| Slip curve | the lookup that grows the wheel boxes |
| Goal arrival radius | 1 m |

## Where this deviates

**The goal is absorbing.** A path that drives through the goal is not charged
for the rest of its 6 m horizon, nor for turning onto an onward global path
that no longer exists. The papers do not state this. Without it the rover
orbits the goal at about 2.5 m instead of arriving: near the goal every
candidate overshoots, so the end-of-path turn term decides a comparison
between paths of otherwise identical length, and the planner is rewarded for
overshooting and turning around.

**The retreat path is a fallback, not a candidate.** Section II-E3 says it is
selected only when nothing in the fixed or heritage trees is feasible. It has
to be: every other option spans the full 6 m horizon, while the retreat path
is 1 m and costs a sixth as much to execute for reasons unrelated to its
merit. Ranked alongside the rest, a 1 m reverse outranks the path that
finishes the drive, and the rover oscillates over the same two metres
indefinitely.

## Sources

> Otsu et al., *Fast Approximate Clearance Evaluation for Rovers with
> Articulated Suspension Systems*, Journal of Field Robotics 37:768–785, 2020.
> [doi:10.1002/rob.21892](https://doi.org/10.1002/rob.21892)
>
> Toupet, Ono, Del Sesto, Maimone, McHenry, *Enhanced Autonomous Navigation on
> the Perseverance Mars Rover*, IEEE Transactions on Field Robotics 3:83–124,
> 2026. [doi:10.1109/TFR.2025.3636366](https://doi.org/10.1109/TFR.2025.3636366)

Implemented from these papers alone. No JPL source was used.
