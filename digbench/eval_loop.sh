#!/bin/bash
# Closed-loop fidelity: run one scenario with the adaptive controller on several soil backends and compare each final
# terrain with the MPM-in-the-loop reference (same scenario, same controller).
#   bash digbench/eval_loop.sh <name> <soil> <slope> "<scenario flags>" <ref-grid.json> [scoop-model.pt]
# Backends: heightfield (no design clamp), prior (cut+relax on the bite's swept envelope), scoop (learned, residual on
# the prior). prior/scoop run through mpm_server.py --backend {prior,scoop} on CPU ports 8781/8782.
set -e
NAME=$1; SOIL=$2; SLOPE=$3; FLAGS=$4; REF=$5; MODEL=${6:-}
OUT=$HOME/v4/eval/$NAME; mkdir -p "$OUT"
export PATH=$HOME/node/bin:$PATH
cd "$HOME/livedirt"
PY=${PY:-/opt/pytorch/bin/python}
start_srv() { (setsid nohup env CUDA_VISIBLE_DEVICES= $PY -u -m digbench.mpm_server --port $1 --arch cpu --backend $2 ${3:+--model $3} > "$OUT/srv-$2.log" 2>&1 < /dev/null &); sleep 4; }
start_srv 8781 prior
[ -n "$MODEL" ] && start_srv 8782 scoop "$MODEL"
cd app
REQ="{\"soil\":{\"preset\":\"$SOIL\"},\"margin\":1.0}"
npx tsx bench/autonomy.ts $SOIL $SLOPE $FLAGS --controller adaptive --no-clamp --tag "eval-$NAME-hf" --dump-grid "$OUT/hf.json" > "$OUT/hf.log" 2>&1
npx tsx bench/autonomy.ts $SOIL $SLOPE $FLAGS --controller adaptive --soil-backend mpm --mpm-url http://127.0.0.1:8781 --mpm-req "$REQ" --tag "eval-$NAME-prior" --dump-grid "$OUT/prior.json" > "$OUT/prior.log" 2>&1
[ -n "$MODEL" ] && npx tsx bench/autonomy.ts $SOIL $SLOPE $FLAGS --controller adaptive --soil-backend mpm --mpm-url http://127.0.0.1:8782 --mpm-req "$REQ" --tag "eval-$NAME-scoop" --dump-grid "$OUT/scoop.json" > "$OUT/scoop.log" 2>&1
curl -s 127.0.0.1:8781/shutdown >/dev/null || true; curl -s 127.0.0.1:8782/shutdown >/dev/null || true
cd "$HOME/livedirt"
$PY -m digbench.compare_grids "$REF" "$OUT/hf.json" "$OUT/prior.json" ${MODEL:+"$OUT/scoop.json"} | tee "$OUT/compare.txt"
