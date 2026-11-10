#!/bin/bash
# both v4 gate trenches with ensemble $2 (CPU servers on ports $3 $4); writes ~/v4/eval/$1/gate{1,2}.json
TAG=$1; E=$2; P1=$3; P2=$4; OUT=$HOME/v4/eval/$TAG; mkdir -p $OUT; PY=/opt/pytorch/bin/python; export PATH=$HOME/node/bin:$PATH
cd $HOME/livedirt
for p in $P1 $P2; do (setsid nohup env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY -u -m digbench.mpm_server --port $p --arch cpu --backend scoop --model $E > $OUT/srv$p.log 2>&1 < /dev/null &); done
sleep 10; cd app; F="--yaw-offset 50 --start 4.0 --length 4.0 --bottom 1.1 --controller adaptive"
npx tsx bench/autonomy.ts sandy-gravel 1.5 $F --soil-backend mpm --mpm-url http://127.0.0.1:$P1 --mpm-req '{"soil":{"preset":"sandy-gravel"},"margin":1.0}' --tag eval-gate-$TAG --dump-grid $OUT/gate1.json > $OUT/gate1.log 2>&1 &
npx tsx bench/autonomy.ts wet-clay 0.75 $F --soil-backend mpm --mpm-url http://127.0.0.1:$P2 --mpm-req '{"soil":{"preset":"wet-clay"},"margin":1.0}' --tag eval-gate2-$TAG --dump-grid $OUT/gate2.json > $OUT/gate2.log 2>&1 &
wait
curl -s -m 5 127.0.0.1:$P1/shutdown >/dev/null; curl -s -m 5 127.0.0.1:$P2/shutdown >/dev/null
