IMPROVE (fix-style, budgeted). Reduce the network's error, starting with its largest component, without making any region
worse. Measured facts (see the context, "Where the error is"): the biggest error is in deep planned cells (envelope more
than 0.6 m below terrain), RMS 27-30 cm with almost no mean bias, i.e. a shape/placement error of the deep cut, largest on
stalled bites because they are deeper; the region past the predicted stop is only 4-5% of stalled-bite error. Do not
assume the cause: confirm it with the probes (`error_report` now also reports deep_planned / shallow_planned /
beyond_stop; `path_profile`, `error_map`, `compare_groups`, `feature_maps`), then explain it inside the weights.

This copy has two function-preserving additions (predictions are exactly the original network's):
1. Input channel 7 = "beyond the predicted stop" from the hand-written stop calculator (see the context), wired to
   nothing yet (e1.0.weight[:, 7] is zero). Use it if it helps.
2. One spare GroupNorm group per block (listed at the end of the context), all-zero, for new paths, so you need not take
   over channels the network already uses.

Build the smallest edit that fixes a cause you have confirmed. Check every region and both stalled and completed bites
after each change (the judge rejects an edit that makes any region or bite class worse by more than 2 mm on the
validation set, or that does not improve the overall error by more than 2 mm).

Report: which weights you changed, what each change does, and the practice error before and after (all, cut, slump, far,
stalled, completed, deep_planned, shallow_planned, beyond_stop). Restore your best checkpoint before submitting; if
nothing helps, submit the original.
