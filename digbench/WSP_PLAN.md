# Weight-space programming plan for the per-scoop soil model (theoretical)

Goal: close the gap between the learned scoop model and the MPM teacher using frontier models (LLM, VLM, reasoning)
that read, write and edit the model's weights directly, with the teacher as the only judge. Nothing here has been
built; numbers in "Starting point" are measured, numbers in "Targets" are aims.

## Starting point (measured)

| Metric | Now | Floor (teacher's own noise) |
|---|---|---|
| Per-scoop surface error (held out, right soil) | 8.1-10.7 cm | 2.3 cm |
| Closed-loop final trench vs teacher | 13 cm (sandy gravel), 16 cm (wet clay) | 4.3 cm |
| Heightfield (current app) in closed loop | 31 cm, 45 cm | - |

Where the per-scoop error lives: stalled scoops 58% (37% of scoops), cells the bucket cut 42%, slump zone within 0.5 m
32%, farther spill 26%, teacher noise ~7%. Wrong soil costs 10-35%. Error still falls with more data.

## Principles

1. **The teacher is the only judge.** No edit is kept on a model's say-so. Three held-out tiers: edit-time checks,
   acceptance set, and a final closed-loop gate no agent ever sees.
2. **Physics goes in as weights, not as hope.** Known laws become frozen, named layers; learning only fills the gaps.
3. **Every change is a named, reversible patch** with a written behavioural claim that the judge verifies.
4. **Agents get tools, not access to the answer.** They see error maps on the edit set, never the acceptance or gate
   sets (an earlier weight-space autoresearch run drifted into fitting its own eval pool).

## Architecture after the plan

```
inputs: terrain, actual-or-planned bucket path (time + attitude channels), soil fingerprint z
  -> [P1 cut layer]            frozen: material above the bucket edge is removed
  -> [P2 force/stop module]    frozen formula + small learned correction: where the bucket actually stops
  -> [P3 slope-stability map]  frozen: safety factor per face from the wedge formula
  -> learned U-Net core        FiLM-conditioned on z; predicts residuals only
  -> [P4 mass-transport head]  frozen structure: material moves along predicted fluxes, never created
  -> [P5 relax layer]          frozen: no slope steeper than the friction angle survives
outputs: surface change, carried volume, completion, force
+ patch stack: named low-rank deltas (one per diagnosed failure mode), each with a changelog entry
```

## Roles and tools

| Role | Model | Tools it can call | Writes |
|---|---|---|---|
| Physicist | reasoning model | layer compiler (equation -> conv/elementwise weights), unit tests vs analytical cases | physics modules P1-P5 |
| Observer | VLM | renders of before/after/error maps, bite videos, lidar, site photos | failure taxonomy in words; structured bite descriptors; soil descriptors |
| Surgeon | LLM with weight + activation access | probes, activation patching, low-rank fitting on a named slice of data | patches; merged models; model diffs |
| Planner | reasoning model | ensemble weight disagreement, failure taxonomy, scenario generator | next MPM experiments (DAgger targets) |
| Judge | MPM teacher + scripts | per-scoop eval, closed-loop trench eval | accept / reject, measured deltas |

## Phases

### Phase 0: make the model editable (substrate)
- Restructure the scoop model into named blocks with stable interfaces; expose weights and activations through one API.
- Patch format: low-rank delta on a named block + trigger condition + changelog text + judge result. Patches stack and
  can be toggled.
- Judge harness: three data tiers fixed up front (edit set, acceptance set, gate trenches with two teacher seeds each),
  plus the noise floor recorded per tier.
- Exit check: an identity patch round-trips with zero change; the judge reproduces today's numbers exactly.

### Phase 1: physics library in weights (physicist)
For each module: equation -> compiled layers -> standalone test against an analytical or MPM case -> freeze.
- **P1 cut**: min(terrain, bucket edge) per cell. Test: exact on synthetic paths.
- **P2 force and stop point** (targets the 58%): resistance from the fundamental earthmoving equation integrated along
  the path (cumulative sum), compared with the hydraulic limits (the existing per-actuator model); path truncated where it
  exceeds them. Learned correction allowed only as a bounded scalar per scoop. Test: predicted stop point vs MPM
  completion on the 43 stalled held-out scoops.
- **P3 slope stability** (targets the 32%): per-face factor of safety from Culmann's wedge with the soil's friction and
  cohesion; outputs a failure map. Test: the vertical-cut benchmark already used for the teacher (critical N 3.5-4.2).
- **P4 mass transport** (targets the 26%): output becomes fluxes between neighbouring cells plus a sink into the bucket,
  so volume is conserved by construction. Test: total volume change equals minus carried volume to machine precision.
- **P5 relax**: the settle rule as fixed convolutions. Test: identical to the Python reference.
- Exit check: physics-only model (no learned core) beats today's cut+relax prior on the edit set.

### Phase 2: hybrid assembly and retrain
- Insert P1-P5 around the learned core, freeze them, retrain only the core on existing data.
- Ablation: remove one module at a time; each must earn its place on the acceptance set.
- Exit check: per-scoop error below today's 8.1-10.7 cm on the acceptance set; no module that fails its ablation stays.

### Phase 3: observer and surgeon loop (targets the 42% in cut cells, plus leftovers)
Loop, budgeted per round:
1. Judge produces error maps on the edit set; observer clusters them into named failure modes with example bites.
2. Surgeon finds the internal direction for one mode (probe activations on its examples vs others), fits a low-rank patch
   on that slice only, writes the predicted behaviour change.
3. Judge tests the patch on the acceptance set; kept only if total error drops and no other mode gets worse by more
   than the noise floor; the behavioural claim must hold.
- Exit check: at least three accepted patches with verified claims; closed-loop gate improves on both trenches.

### Phase 4: planner-driven data (targets the data limit)
- Planner reads where ensemble members' weights and outputs disagree, maps that to situations (via observer
  descriptors), and designs MPM scenarios that hit them; DAgger runs those with the current model in the loop.
- Exit check: error per added MPM scoop falls faster than with random scenarios (compare against a random-scenario arm
  of equal MPM cost).

### Phase 5: observation to weights (targets the 10-35% soil share, and real sites)
- Train the core across many continuously varied soils with a learned fingerprint z (auto-decoder: one z per training
  trench).
- Two ways to fill z on a new site: (a) solve for z through the frozen model from the first k dug scoops (lidar before /
  after + payload); (b) the observer reads site photos and the first scoops and emits z (or a small adapter) directly.
- Exit check: on held-out soils, error with k = 1, 3, 5 scoops approaches the error with the true soil given.

### Phase 6: merge, compress, deploy
- Surgeon aligns the ensemble members neuron by neuron and merges them into one network; physics modules stay exact.
- Export one ONNX file (physics layers included) for the browser; verify bit-for-bit against the Python model.
- Exit check: merged model within noise of the ensemble; browser and Python agree.

### Phase 7: changelog and rendering tie-in
- Every accepted patch ships with a judge-verified, physics-level changelog line ("+20% slumping on faces over 45 deg").
- Same observer/surgeon loop applied to the Gaussian-splat scene: when the soil model lowers a cell, a learned edit sets
  the splats' colour and opacity to look like fresh dirt; judged against before/after footage once it exists.

## Targets (aims, not results)

| Error source | Today | Mechanism | Aim |
|---|---|---|---|
| Stalled scoops (58%) | 11.9 cm on stalls | P2 stop module | stalled error near completed-scoop error |
| Slump zone (32%) | 10.7 cm | P3 stability map | halve |
| Farther spill (26%) | 6.8 cm | P4 transport | halve |
| Cut cells (42%) | 15.6 cm | patches (Phase 3) | -30% |
| Soil unknown | +10-35% | Phase 5 fingerprint | within 5% of known-soil error |
| Overall per scoop | 8-11 cm | all | 4-5 cm (2x the noise floor) |
| Closed loop | 13-16 cm | all | under 8 cm (2x the teacher's spread) |

## Risks and kill criteria

- **Overfitting by agents**: enforced by the three tiers; if acceptance gains do not show up on the gate, freeze the loop.
- **Compiled physics is wrong for MPM** (e.g. the earthmoving formula misses bucket shape): every module must pass its
  standalone test first; a module that loses its ablation is removed, not tuned.
- **Patches interfere**: patches are tested in stack order and as a full stack; revert any that change other modes by
  more than the noise floor.
- **Weight reading is unreliable** (permutation symmetry, hallucinated circuits): the surgeon's claims are always
  checked by activation patching and the judge, never trusted directly.
- **Cost**: MPM labels are the bottleneck (1-4 min each); Phase 4 must beat random scenarios per MPM minute or stop.
- Kill the whole approach if, after Phases 1-2, the hybrid model is not better than a same-data black-box retrain on the
  acceptance set; then the effort goes to data and model size instead.

## Order and dependencies

Phase 0 -> Phase 1 (modules in parallel) -> Phase 2 -> {Phase 3, Phase 4, Phase 5 in parallel} -> Phase 6 -> Phase 7.
The single highest-value step is P2 (stop point), because stalls carry the largest share of error and the information
it needs (forces, hydraulic limits) already exists in the codebase.

## Feasibility check (2026-11-09; probes in `wsp/`)

Measured on our data:
* **Stop point from compiled physics** (`wsp/probe_stall_fee.py`, 495 MPM bites): the peak passive-wedge (earthmoving
  equation) resistance along the planned path alone classifies stalls with 91% accuracy held out by run (base rate
  36% stalled; recall 81%, precision 92%), and on stalled bites the first point where it crosses the threshold lands a
  median 3% (mean 8%) of the path from where MPM actually stopped. P2 is feasible and probably the biggest single win.
* **Settle rule written as weights by hand** (`wsp/probe_relax_conv.py`): two fixed difference convolutions, a ReLU and
  two transposed convolutions per iteration reproduce the Python relax to 3e-15 m on 20 bites and conserve mass to
  1e-15 m^3, with no training. P5 (and by the same construction P1, P4) is feasible.

From the literature: hand-writing weights works for algorithmic tasks (agent-written transformers solved 10/10
algorithmic tasks); repairing a failing control policy by weight edits worked once in the original report but only 1 of
3 attempts helped in an independent replication; zero-shot weight READING was found to be local arithmetic plus dataset
priors in an audit. So: let models write physics modules and choose where to patch, let gradients fit patch values, and
never accept a change without the judge.
