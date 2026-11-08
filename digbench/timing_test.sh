#!/bin/bash
# Instance timing test (same on every box): one full-width open-ground MPM episode (gen_replay --standard, dx 4 cm,
# ~4.2M particles) and one v3 training epoch (flux head, unroll 4) on 40 copies of that episode.
# usage: bash digbench/timing_test.sh <label>   -> prints TIMING lines
set -e
LABEL=${1:-box}
export PATH=/opt/pytorch/bin:$PATH
cd "$HOME/livedirt"
nvidia-smi --query-gpu=name --format=csv,noheader
D=$HOME/timing; rm -rf "$D"; mkdir -p "$D/ep" "$D/data"
s=$(date +%s.%N)
python -u -m digbench.gen_replay --arch cuda --standard --out "$D/ep" 2>&1 | grep -v Taichi
e=$(date +%s.%N)
echo "TIMING $LABEL mpm_episode_s $(echo "$e - $s" | bc)"
for i in $(seq -w 1 40); do cp "$D/ep/standard_task3.npz" "$D/data/ep$i.npz"; done
python -u -m digbench.train_v3 --data "$D/data" --heldout "$D/ep" --head flux --floor-train --curriculum 4 --epochs 1 \
  --tag timing --out "$D/res" 2>&1 | grep -E "^epoch|train " || true
python -c "import json;d=json.load(open('$D/res/heldout-timing.json'));print('TIMING $LABEL train_epoch_s', round(d['train_s'],1))"
