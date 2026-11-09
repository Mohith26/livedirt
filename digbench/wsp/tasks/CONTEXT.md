# The network you are working on

A per-scoop soil model for an autonomous excavator. One forward pass predicts what one bucket scoop does to the ground.
It was trained (gradient descent) to imitate a validated 3D material-point-method (MPM) soil simulator.

Architecture (PyTorch, 2,016,900 weights), names as in the state dict:
- Encoder: e1 (7->32), e2 (32->64), e3 (64->128), e4 (128->256). Each block: Conv3x3, GroupNorm(8), SiLU, Conv3x3,
  GroupNorm(8), SiLU. Max-pool 2x2 between blocks.
- Bottleneck head: concat(mean, max) of e4 over space (512) -> Linear 128 -> SiLU -> Linear 3 = scalars
  [carried bank volume m^3, completion fraction of the planned path, peak actuator demand (x rated force)].
- Decoder: d3 (cat(up e4, e3) -> 128), d2 (cat(up d3, e2) -> 64), d1 (cat(up d2, e1) -> 32), out Conv1x1 32->1 =
  residual surface change (m) per cell. Final prediction = prior + residual.
Grid: the bite's bed, 5 cm cells, 108 x 80 (x = along the drag direction toward the machine, y = across the bucket,
bucket centred at y = 40). Input channels:
  0 terrain height minus the mean height under the swept area (m)
  1 swept envelope: lowest bucket point minus terrain, clipped to [-1.5, 0.5]; 0.5 where the bucket never passes
  2 1 where the bucket never passes, else 0
  3 physics prior: surface change from "cut everything above the bucket path (up to bucket capacity 0.95 m^3), then let
    slopes relax to the friction angle" (m)
  4 (friction angle - 30 deg) / 10     5 log10(1 + cohesion Pa) / 3     6 (density - 1800) / 200   (constant planes)

Known physics:
- Digging resistance on a blade at depth d (passive wedge / fundamental earthmoving equation, per metre of width):
  F = 0.5 rho g d^2 Kp + 2 c d sqrt(Kp), Kp = tan^2(45 deg + phi/2). The machine's hydraulics stall at roughly the force
  a 0.5-0.6 m cut produces in sandy gravel; when it stalls the bucket stops part-way along the planned path (completion < 1),
  but channel 1 always shows the WHOLE planned path.
- A cut face of height H and angle beta stands if H <= 4 c sin(beta) cos(phi) / (gamma (1 - cos(beta - phi))) (Culmann);
  otherwise it slumps and the material runs out toward the trench floor.
- Volume: in the teacher, ground lost ~= carried + soil riding on the bucket; nothing is created.

Measured error of this network (held out): RMS 8-11 cm per bite on disturbed cells vs the teacher's own 2.3 cm noise.
Error share: stalled bites 58% (36% of bites stall), cells the bucket cut 42%, cells within 0.5 m of the cut (slumps)
32%, farther cells 26%. A hand-written test showed the peak passive-wedge force along the planned path alone predicts
which bites stall with 91% accuracy and where they stop within 3% of the path (median).

Rules: do not train or fit weights to data with an optimizer, and do not run the network except through the probes
(they run on a PRACTICE set; a separate validation set and two full trenches you never see decide whether your edit is
kept). Prefer small, explainable edits; say what each edit should change, so it can be checked.
