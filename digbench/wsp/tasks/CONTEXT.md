# The network you are working on

A per-scoop soil model for an autonomous excavator. One forward pass predicts what one bucket scoop does to the ground.
It was trained (gradient descent) to imitate a validated 3D material-point-method (MPM) soil simulator (the "teacher").

Architecture (PyTorch; exact widths, parameter count and spare channels of THIS copy are listed at the end):
- Encoder: e1, e2, e3, e4. Each block: Conv3x3 (.0), GroupNorm (.1), SiLU, Conv3x3 (.3), GroupNorm (.4), SiLU.
  Max-pool 2x2 between blocks. Base widths 32, 64, 128, 256; GroupNorm with 8 groups of equal size.
- Bottleneck head: concat(spatial mean, spatial max) of e4 -> Linear 128 -> SiLU -> Linear 3 = scalars
  [carried bank volume m^3, completion fraction of the planned path, peak actuator demand (x rated force)].
- Decoder: d3 (cat(up e4, e3)), d2 (cat(up d3, e2)), d1 (cat(up d2, e1)), out Conv1x1 -> residual surface change (m)
  per cell. Final prediction = prior (input channel 3) + residual. Every decoder conv sees only a 3x3 neighbourhood.
Grid: the bite's bed, 5 cm cells, 108 x 80 (axis 0 = x, along the drag; axis 1 = y, across the bucket, centred at
y = 40, bucket about 18 cells wide). The bucket usually moves toward DECREASING x (69 of 79 practice bites), but some
bites run toward increasing x; `list_bites` gives each bite's direction.
Input channels:
  0 terrain height minus the mean height under the swept area (m)
  1 swept envelope: lowest bucket point minus terrain over the WHOLE planned path, clipped to [-1.5, 0.5]; 0.5 where the
    bucket never passes
  2 1 where the bucket never passes, else 0
  3 physics prior: cut everything above the planned envelope; if that exceeds the bucket capacity (0.95 m^3) the cut is
    scaled down UNIFORMLY along the whole path (it does not stop part-way); then slopes relax to the friction angle (m)
  4 (friction angle - 30 deg) / 10     5 log10(1 + cohesion Pa) / 3     6 (density - 1800) / 200   (constant planes)
  7 (only in copies that have 8 inputs) beyond the predicted stop: 1 on cells the bucket would first reach after the stop
    calculator's predicted stop point, ramping 0->1 over 5% of the path; 0 on cells reached before it and on untouched
    cells. All-zero on bites the calculator predicts will complete.

The stop calculator (hand-written, uses only terrain, planned poses and soil; fitted on the training tier only): walk the
planned poses in time order; stop at the first pose where the soil the bucket would have had to remove so far exceeds
2.75 m^3, or the passive-wedge digging force at the lip exceeds 43 kN per metre of width. On held-out bites it calls
stall vs complete correctly 90.7% of the time (27 of 108 stall) and, on stalled bites, places the stop within a median
1.0% of the path (assuming no stall gives 31%). It is not perfect: some stalls are missed and a few completed bites
are flagged.

Known physics:
- Passive-wedge (fundamental earthmoving equation) resistance per metre of width at depth d:
  F = 0.5 rho g d^2 Kp + 2 c d sqrt(Kp), Kp = tan^2(45 deg + phi/2). When the machine stalls the bucket stops part-way
  (completion < 1); in front of a stalled bucket the teacher often leaves a pile (+0.3 to +0.7 m), not a cut.
- A cut face of height H and angle beta stands if H <= 4 c sin(beta) cos(phi) / (gamma (1 - cos(beta - phi))) (Culmann);
  otherwise it slumps toward the trench floor.
- Volume: in the teacher, ground lost ~= carried + soil riding on the bucket; nothing is created.

Where the error is (measured on held-out stalled bites the calculator flags; practice is similar): 60-67% of their
squared error is in cells the bucket actually dug before the stop (RMS 19-22 cm), 28-37% outside the path (slumps and
piles, 8-10 cm), only 4-5% past the predicted stop (15-18 cm; there the teacher still shows a cut of about 0.2-0.3 m on
average, roughly 70-95% of the prior's). Within the dug zone the worst cells are where the planned envelope is more than
0.6 m deep: RMS 27-30 cm with almost no mean bias (pred - true about -0.03 to +0.03 m), on stalled AND completed bites.
So the network moves about the right amount of soil but misplaces the deep cut's shape (where the floor, walls and edges
end up). Stalled bites score worse mainly because they contain more deep cells.

This network on the PRACTICE tier (the probes' own metric, RMS over disturbed cells): all 11.8 cm; cut cells 20.9,
slump zone 12.9, far 7.5; stalled bites 13.3 vs completed 10.8; 28 of 79 practice bites stall. The teacher's own
bite-to-bite noise is about 2.3 cm.

What earlier editing sessions on this network found (use it, but check it):
- Taking over channels the network already uses costs more than any fix gains (completed-bite error rose ~0.7-1 cm per
  hijacked group). Build new paths only in spare channels, if this copy has them.
- Envelope depth alone does not separate stalls from completions; the stall is mostly a fill/force effect along the path.
- Earlier tasks assumed stalled-bite error comes from predicting the cut past the stop point. Measurement says otherwise
  (see "Where the error is" below); edits that cancelled the cut past the stop made things worse.
- To pass a signal through GroupNorm predictably, keep the group's statistics fixed (e.g. a large constant channel plus
  zero-sum channels), or work in a spare group whose other channels are zero.

Tools: `python` runs code with R (read-only weights) and W (writable); use checkpoint(label) to save good versions and
restore the best one before you submit: the weights you submit are what is judged. Probes run the network on PRACTICE
bites only and cost forward passes (list_bites is free).

Rules: do not train or fit weights to data with an optimizer, and do not run the network except through the probes (a
separate validation set and two full trenches you never see decide whether your edit is kept). Prefer small, explainable
edits; say what each edit should change, so it can be checked.
