# Continuous data + training loop (active-learning DAgger)

Runs on the GPU box; state in ~/v4/loop. Stop everything with `touch ~/v4/loop/STOP`.
- `gen.sh <learned-port> <seed0> <tag>`: forever: read the current ensemble (~/v4/loop/CURRENT), (re)start its CPU server,
  dig one random scenario (soil preset, slope, depth, bottom width, length, start, heading up to +-60 deg) with the learned
  soil in the loop, then `score_trace.py` scores every bite by ensemble disagreement and drops it into ~/v4/loop/queue.
- `labeler.py <mpm-port>`: forever: claim the highest-disagreement bite in the queue and have the MPM teacher label it
  (the MPM server writes the training sample). Keeps the GPU busy whatever the generators are doing.
- `train_loop.sh`: every +40 labels: train the next ensemble version (3 seeds; warm start from the previous version, a
  from-scratch retrain every 3rd version), evaluate both v4 gate trenches against the teacher, append to the ledger,
  make it CURRENT (DAgger uses the newest policy), track BEST by the gates.
The gate scenarios (and run names starting with `gate` / `c240`) are never generated or trained on.
