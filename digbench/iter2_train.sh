#!/bin/bash
# Iteration 2: volume-corrected MPM truth (Tampubolon correction; no spurious bulking), replays t1-t3, chained bites,
# and DAgger replays of the states the iteration-1 model visited in the loop. Scored on the VC frozen held-out set,
# with the deploy floor projection both on and off (train_v3 writes summary and summary_nofloor).
# Arms:
#   sym          free head, unroll 4, symmetrised training inputs (cause 1 on VC truth)
#   full         free head, unroll 4
#   d            flux + floor in training + curriculum 4,16,32 + auto noise + occ-pooled force   (iteration-1 recipe)
#   fv           flux + curriculum + auto noise + occ pool + volume-consistency loss, NO floor   (floor masked over-removal)
#   fv-floor     fv + floor in training                                                          (floor ablation)
#   fv-s1, fv-s2 ensemble members for fv
set -e
DATA=${1:-"$HOME/v3vc/replay $HOME/v3/vc_chain_train $HOME/v3vc/dagger1 $HOME/v3/vc_dagger1"}
export PATH=/opt/pytorch/bin:$PATH
OUT=$HOME/v3vc/models/it2; mkdir -p "$OUT"
T="python -u -m digbench.train_v3 --heldout $HOME/v3vc/heldout --out $OUT"
run() { tag=$1; shift; [ -f "$OUT/heldout-$tag.json" ] && { echo "skip $tag"; return; }; echo "=== arm $tag $(date -u +%T)"; $T --tag "$tag" "$@" 2>&1 | grep -E "^train |summary|Error|Traceback" | head -8; }
CUR="--curriculum 4,16,32 --noise auto --epochs 36 --max-windows 4000"
run it2-sym  --data $DATA --head free --pool global --curriculum 4 --noise 0.005 --sym --epochs 16 --max-windows 6000
run it2-full --data $DATA --head free --pool global --curriculum 4 --noise 0.005 --epochs 16 --max-windows 6000
run it2-d    --data $DATA --head flux --pool occ --floor-train $CUR --seed 0
run it2-fv   --data $DATA --head flux --pool occ --vol-loss 1.0 $CUR --seed 0
run it2-fv-floor --data $DATA --head flux --pool occ --vol-loss 1.0 --floor-train $CUR --seed 0
run it2-fv-s1 --data $DATA --head flux --pool occ --vol-loss 1.0 $CUR --seed 1
run it2-fv-s2 --data $DATA --head flux --pool occ --vol-loss 1.0 $CUR --seed 2
echo "=== iter 2 done $(date -u +%T)"
