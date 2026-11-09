FIX (Exp-4 style). The network's worst failure: on bites where the machine stalls, it predicts change along the whole
planned path (channel 1 shows the full path) instead of stopping where the bucket really stopped, so its error on
stalled bites is 11.9 cm vs 8.3 cm on completed ones. Use `worst_bites` and `error_map` on stalled practice bites to
confirm, explain the cause inside the weights, then fix only that cause: e.g. add a small separate path through a few
units per layer that estimates the stall point from depth, soil and drag distance and suppresses the residual and the
prior's effect beyond it, leaving the network unchanged on bites that complete. Report which weights you changed and the
expected change in the stalled-bite error.
