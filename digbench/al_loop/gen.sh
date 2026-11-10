#!/bin/bash
# forever: dig one random scenario with the CURRENT learned ensemble, score its bites, enqueue them
LP=$1; SEED=$2; TAG=$3; L=$HOME/v4/loop; export PATH=$HOME/node/bin:$PATH; PY=/opt/pytorch/bin/python
mkdir -p $L/traces; cur=""
while [ ! -f $L/STOP ]; do
  new=$(cat $L/CURRENT); ver=$(cat $L/CURRENT_TAG)
  if [ "$new" != "$cur" ]; then
    curl -s -m 5 127.0.0.1:$LP/shutdown >/dev/null; sleep 3
    (cd $HOME/livedirt; setsid nohup env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 $PY -u -m digbench.mpm_server --port $LP --arch cpu --backend scoop --model $new > $L/srv-$LP.log 2>&1 < /dev/null &)
    sleep 10; cur=$new
  fi
  read soil slope depth bw len start yaw < <(python3 - "$SEED" <<'PYEOF'
import random, sys
r = random.Random(int(sys.argv[1]))
soil = r.choice(["wet-clay", "common-earth", "sandy-gravel", "saturated-silt"])
slope = {"wet-clay": r.choice([0.75, 1.0]), "common-earth": r.choice([1.0, 1.5]), "sandy-gravel": r.choice([1.5, 2.0]), "saturated-silt": 2.0}[soil]
print(soil, slope, round(r.uniform(0.6, 1.4), 2), round(r.uniform(0.6, 1.2), 2), round(r.uniform(2.5, 4.5), 2), round(r.uniform(3.6, 5.0), 2), round(r.uniform(-60, 60), 1))
PYEOF
)
  run="${TAG}${ver}-${SEED}-${soil}"
  echo "=== $run slope $slope depth $depth bw $bw len $len start $start yaw $yaw $(date -u +%T)"
  (cd $HOME/livedirt/app && npx tsx bench/autonomy.ts "$soil" "$slope" --controller adaptive --soil-backend mpm --mpm-url "http://127.0.0.1:$LP" \
    --mpm-req "{\"soil\":{\"preset\":\"$soil\"},\"margin\":1.0}" --depth "$depth" --bottom "$bw" --length "$len" --start "$start" \
    --yaw-offset "$yaw" --tag "al-$run" --trace "$L/traces/$run.json" > "$L/traces/$run.log" 2>&1)
  (cd $HOME/livedirt; OMP_NUM_THREADS=1 $PY -m digbench.al_loop.score_trace "$L/traces/$run.json" "$soil" "$run" "$new" 2>&1 | grep -c queued | sed "s/^/queued bites: /")
  SEED=$((SEED + 1))
done
