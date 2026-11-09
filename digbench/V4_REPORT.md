# v4: teacher first, then a fair closed-loop test, then a per-scoop model

Order followed (from the 2026-11-08 plan): (1) check the teacher, (2) make data cheap, (3) fix the robot and the test,
(4) per-scoop model, (5) tune the simple model. Every number is from a run; files in `results/v4/` and
`app/bench/results/`. Status lines marked PENDING are not finished.

## 1. Teacher (MPM) - see TEACHER.md

* Yield surface replaced: Mohr-Coulomb (`Sim3D(yield_model="mc")`). The legacy cone behaved like phi = asin(mu) in plane
  strain (mu 0.6 -> 37 deg) and was 15-20% too strong on the blade test.
* Verified vs analysis: blade passive force vs Sokolovski +4 / -3 / -7% (grid-converged); vertical-cut cohesion
  collapses at N = 3.5-4.2 vs limit analysis 3.83-4.0.
* Validated vs lab: Coetzee bucket +6% at 800 mm, curve MAE 11 N (legacy +114%); blade bracketed by the two side-wall
  bounds. Published Coetzee MPM -12%, DEM ~20 N low.
* Noise floor: particle-seed spread of a bite's surface change 0.016-0.031 m (not 0.10 m as stated in the v3 report).
* NOT validated: dug-surface shape against a real dig (no data exists).

## 2. Cheap data

* `drive_fast` (one async kernel launch per step) and kernel-cache reuse; identical results within 5e-5.
* Bed experiment: margin 1.0 m at dx 4 cm keeps the surface within 0.040 m of the 1.5 m bed (noise 0.023 m); every
  cheaper bed doubles the error; dx 5 cm changes carried volume by up to 62%. Softer soil / coarser grid rejected earlier.
* Honest cost: ~1-4 min of L40S per MPM bite at the chosen settings; with three clients sharing the GPU the campaign
  runs at ~0.5 bites/min. Thousands of bites would take days, not hours.

## 3. Robot and test

Found and fixed in the test itself:
* The heightfield's dig refused to cut below the design in every cell (a machine-control shortcut inside the soil):
  the deployed baseline's 34.5% measured the soil clamp, not the robot. Its bottom was 95.5% on grade, its slopes 24.6%.
* On the old site the design's slope planes intersected a mound metres away; cells unreachable by any lane counted as
  "trench". Fixed by daylighting (keep design cells connected to the floor). The old scenario's floor was 0.9 m wide on
  sloping ground; the v4 gate uses flat ground, a 4 m x 1.1 m floor and 1.5:1 slopes (sandy gravel holds 36 deg).
* The bench's volume metric summed the masked cells outside the sim; every earlier `volume delta` was invalid.
* Legacy robot in MPM (old scenario, 28 bites): 9 of 28 bites stalled at 12-19x the machine's rated force.

New adaptive controller (`controller: 'adaptive'`): machine-control edge (the cutting edge never goes below the design
anywhere across its width), tiltrotator roll on slope lanes (MPM tool roll added), coordinated straight-line tip motion,
bite length planned from the scan to fill the bucket using a fill efficiency learned from measured payloads, cut depth
backed off after stalls, stalled lanes resumed where the bucket stopped, top-down lane order.

v4 gate scenario (sandy-gravel, yaw +50, 1.5:1, 4 m x 1.1 m floor, 1 m deep):

| soil backend | controller | on grade all / bottom / slopes % | bank removed m^3 |
|---|---|---|---|
| heightfield with design clamp (old deployed) | legacy | 77.6 / 86.2 / 75.9 | 4.96 |
| heightfield, no clamp | adaptive | 56.7 / 93.7 / 49.2 | 3.61 |
| MPM (teacher), seed 1 | adaptive | 8.3 / 0.0 / 9.9 (floor 42 cm high) | 15.2 carried, 6.8 net cut; 47 bites, 6.3 h |

In the teacher, the same robot does NOT finish the trench: the 1.5:1 walls are stable in the finished shape, but every
bite leaves steeper faces that slump back in (15.2 m^3 carried out for 6.8 m^3 of net cut), and after three passes
the controller stops because the out-of-tolerance count stops falling (1144 -> 1088 -> 1081 cells). The heightfield
version of the same robot reports a nearly finished trench (floor within 1 cm). That gap is the soil model, not the
robot, and it is the central v4 finding.

## 4. Per-scoop model

One inference per bite (not per 0.1 s): inputs = app terrain on the bite's bed grid, the bucket's swept envelope, a
cut+relax physics prior, soil parameters; U-Net predicts the residual over the prior plus carried / completion /
peak demand; the result is projected through the relax step (no slope steeper than phi survives). Held-out by run.

| model | train bites | held-out bites | final-surface RMSE: zero / prior / model (m) | carried MAE prior / model (m^3) |
|---|---|---|---|---|
| a1 | 48 | 39 | 0.180 / 0.151 / 0.130 | 0.444 / 0.080 |
| a2 | 72 | 52 | 0.186 / 0.156 / 0.109 | 0.422 / 0.093 |
| a3 (volume loss on net ground change) | 76 | 54 | 0.186 / 0.156 / 0.113 | 0.418 / 0.094 |

| a4 | 103 | 66 | 0.193 / 0.162 / 0.114 | 0.405 / 0.086 |
| a5 (3 seeds, 0.101-0.109 each) | 134 | 79 | 0.201 / 0.168 / 0.101-0.109 | 0.412 / 0.071-0.086 |

Net ground volume per gate bite (34 held-out): truth -0.42 m^3; a2 -0.35 (under-removes), a3 -0.50 (over-removes),
a4 -0.41, prior -0.77.

**Closed loop vs the teacher** (same scenario, same controller; `compare_grids.py`; figure `results/v4/gate/gate_compare.png`):

| soil backend in the loop | on grade % | floor error | net cut m^3 (teacher 6.8) | final-surface RMSE vs teacher (design cells) |
|---|---|---|---|---|
| MPM teacher | 8.3 | 42 cm | 6.8 | - |
| heightfield, no clamp | 56.7 | 1 cm | 16.1 | 0.313 m |
| cut+relax prior | 23.3 | 15 cm | 14.6 | 0.383 m |
| learned a1 (48 bites) | 6.7 | 54 cm | 6.5 | 0.314 m |
| learned a3 (76 bites) | 7.8 | 44 cm | 18.6 | 0.534 m |
| learned a4 (103 bites) | 8.2 | 40 cm | 11.2 | 0.297 m |
| learned a5, 3-seed ensemble (134 bites) | 5.9 | 40 cm | 18.2 | 0.587 m |
| a5 ensemble + envelope floor clamp (10 cm) | 6.7 | 41 cm | 16.3 | 0.492 m |

The a5 ensemble's survey trace is almost the teacher's (mean above grade 23.7 / 21.5 / 20.2 cm vs 23.6 / 21.3 / 24.8 cm;
out-of-tolerance cells ~1100 in both), but it digs a pit at the near end of the trench that MPM never digs (over-dig
5.1 m^3 vs 0.03). Per bite the same models do not over-dig (net-volume bias +0.06 m^3/bite, i.e. slightly UNDER-removing);
the pit grows only in the loop, in states the training data never contained. A per-bite floor clamp halves it but does
not remove it. This is the compounding problem again, now at the scoop level: next step is DAgger (dig with the learned
model, have MPM label those exact states), running on fresh scenarios so the gate stays held out.

Reading: the learned per-scoop models reproduce the teacher's OUTCOME (trench left unfinished, floor ~40 cm high,
~8% on grade) where the heightfield predicts a finished trench. Point-wise the final surfaces still differ by ~0.3 m for
every backend (the runs diverge bite by bite), and model-to-model variance is large (a3 over-digs). Not yet known: the
teacher's own run-to-run spread in closed loop (seed-2 reference running).

## 5. Simple model tuned to the teacher

Two-parameter bite model (carry fraction, repose offset) fitted on 22 MPM bites: RMSE 0.205 -> 0.194 m, carried MAE
0.34 -> 0.16 m^3 (best: carry 50%, repose 6 deg below phi). The simple model cannot represent most of what MPM does in a
bite (stalls, wall slumping), so tuning it buys little.
