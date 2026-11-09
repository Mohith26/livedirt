FIX (Exp-4 style). The network's worst failure: on bites where the machine stalls, it predicts change along the whole
planned path (channel 1 shows the full path) instead of stopping where the bucket really stopped, so its error on
stalled bites is 11.9 cm vs 8.3 cm on completed ones. Use `list_bites` (free) to find stalled practice bites, and
`worst_bites` / `error_map` to confirm, explain the cause inside the weights, then fix only that cause.

SPARE CAPACITY: this copy of the network has been widened without changing its function. Every conv block has ONE extra
GroupNorm group appended after its original channels (GroupNorm now has 9 groups): e1 and d1 channels 32-35, e2 and d2
64-71, e3 and d3 128-143, e4 256-287. All weights into and out of those channels are exactly zero, so they output 0 and
the predictions equal the original network's. In concatenated decoder inputs the spare channels sit after each source's
original channels (d1.0 input: d2 0-63, d2 spare 64-71, e1 72-103, e1 spare 104-107; d2.0 input: d3 0-127, spare 128-143,
e2 144-207, spare 208-215; d3.0 input: e4 0-255, spare 256-287, e3 288-415, spare 416-431); out.weight has 36 inputs
(32-35 spare); head.0 takes [e4 mean 0-287 | e4 max 288-575]. Each spare group is normalised on its own, so a side path
built there does not disturb the existing channels. Build the stall path in the spare channels, e.g. a gated branch that
estimates the stall point from depth, soil and drag distance and suppresses the residual and the prior's effect beyond it,
leaving the network unchanged on bites that complete. Report which weights you changed and the expected change in the
stalled-bite error.
