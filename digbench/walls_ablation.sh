#!/bin/bash
# Cause-2 ablation (walls vs open ground), volume-corrected data: the SAME t1 traces generated with margin 1.5 m (open)
# and margin 0 (bed = window, walls in view), same model recipe, scored on the volume-corrected frozen held-out set.
# usage: bash digbench/walls_ablation.sh   (needs ~/v3vc/replay/t1_*.npz, ~/v3/vc_replay_walled/*.npz, ~/v3vc/heldout)
set -e
export PATH=/opt/pytorch/bin:$PATH
OUT=$HOME/v3vc/models/walls; mkdir -p "$OUT" $HOME/v3vc/replay_t1_open
ln -sf $HOME/v3vc/replay/t1_*.npz $HOME/v3vc/replay_t1_open/
nO=$(ls $HOME/v3vc/replay_t1_open/*.npz | wc -l); nW=$(ls $HOME/v3/vc_replay_walled/*.npz | wc -l)
echo "open t1 episodes: $nO   walled t1 episodes: $nW"
T="python -u -m digbench.train_v3 --heldout $HOME/v3vc/heldout --out $OUT --head free --pool global --curriculum 4 --noise 0.005 --epochs 12 --max-windows 6000"
[ -f $OUT/heldout-walls-open.json ] || $T --tag walls-open --data $HOME/v3vc/replay_t1_open 2>&1 | grep -E "^epoch (0|11)|summary|Error|Traceback" | head
[ -f $OUT/heldout-walls-walled.json ] || $T --tag walls-walled --data $HOME/v3/vc_replay_walled 2>&1 | grep -E "^epoch (0|11)|summary|Error|Traceback" | head
