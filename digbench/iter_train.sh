#!/bin/bash
# Train every ablation arm of one surrogate iteration on the shared frozen held-out set (~/v3/heldout).
# usage: bash digbench/iter_train.sh <iter> "<data dirs>"   (run from ~/livedirt via the job queue)
# Arms (each scored on the same held-out set by train_v3; the loop gate runs on the Mac):
#   sym      free head, unroll 4, global force pool, training windows mirror-symmetrised (cause 1: v1/v2-like inputs)
#   full     same, no symmetrisation (full-width data only)
#   flux     full + mass-conserving flux/sink head
#   d        flux + floor projection in training + curriculum 4,16,32 + noise = measured rollout error + occ-pooled force
#   d-s1,s2  two more seeds of d (ensemble of 3 with d)
#   walls-open / walls-walled  free head on the SAME t1 traces generated with margin 1.5 m vs margin 0 (cause 2)
set -e
IT=${1:?iter}; DATA=${2:?data dirs}
export PATH=/opt/pytorch/bin:$PATH
OUT=$HOME/v3/models/it$IT; mkdir -p "$OUT"
H=$HOME/v3/heldout
T="python -u -m digbench.train_v3 --heldout $H --out $OUT"
run() { tag=$1; shift; [ -f "$OUT/heldout-$tag.json" ] && { echo "skip $tag"; return; }; echo "=== arm $tag $(date -u +%T)"; $T --tag "$tag" "$@" 2>&1 | grep -E "^epoch (0|5|10|15|20|23)|summary|Error|Traceback|line [0-9]+" | head -40; }
run it$IT-sym   --data $DATA --head free --pool global --curriculum 4 --noise 0.005 --sym --epochs 12 --max-windows 6000
run it$IT-full  --data $DATA --head free --pool global --curriculum 4 --noise 0.005 --epochs 12 --max-windows 6000
run it$IT-flux  --data $DATA --head flux --pool global --curriculum 4 --noise 0.005 --epochs 12 --max-windows 6000
run it$IT-d     --data $DATA --head flux --pool occ --floor-train --curriculum 4,16,32 --noise auto --epochs 24 --max-windows 3000 --seed 0
run it$IT-d-s1  --data $DATA --head flux --pool occ --floor-train --curriculum 4,16,32 --noise auto --epochs 24 --max-windows 3000 --seed 1
run it$IT-d-s2  --data $DATA --head flux --pool occ --floor-train --curriculum 4,16,32 --noise auto --epochs 24 --max-windows 3000 --seed 2
if [ "$(ls $HOME/v3/replay_walled/*.npz 2>/dev/null | wc -l)" -ge 30 ] && [ "$IT" = 1 ]; then
  mkdir -p $HOME/v3/replay_t1_open && ln -sf $HOME/v3/replay/t1_*.npz $HOME/v3/replay_t1_open/ 2>/dev/null || true
  run it$IT-walls-open   --data $HOME/v3/replay_t1_open --head free --pool global --curriculum 4 --noise 0.005 --epochs 12 --max-windows 6000
  run it$IT-walls-walled --data $HOME/v3/replay_walled  --head free --pool global --curriculum 4 --noise 0.005 --epochs 12 --max-windows 6000
fi
echo "=== iter $IT done $(date -u +%T)"
