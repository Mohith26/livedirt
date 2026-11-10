#!/bin/bash
# every +40 labelled bites: train the next ensemble, gate-evaluate it, ledger it, make it CURRENT
L=$HOME/v4/loop; M=$HOME/v4/models; PY=/opt/pytorch/bin/python; cd $HOME/livedirt
v=$(cat $L/NEXT_VER); last=$(cat $L/LAST_COUNT); per=${PER:-40}
count() { ls $HOME/v4/scoops/campaign/*.npz $HOME/v4/scoops/loop-adaptive/*.npz 2>/dev/null | wc -l; }
while [ ! -f $L/STOP ]; do
  n=$(count)
  if (( n - last < per )); then sleep 300; continue; fi
  prev=$(cat $L/CURRENT_TAG); echo "=== a$v: $n samples (+$((n-last))) $(date -u +%T)"
  for sd in 0 1 2; do
    if (( v % 3 == 0 )); then init=""; ep=100; lr=2e-3; else init="--init $M/scoop-$prev-s$sd.pt"; ep=30; lr=5e-4; fi
    ($PY -u -m digbench.train_scoop --data $HOME/v4/scoops/campaign $HOME/v4/scoops/loop-adaptive --heldout-runs c240,gate \
        --epochs $ep --lr $lr --ch 32 --seed $sd --out $M --tag a$v-s$sd $init > $L/train-a$v-s$sd.log 2>&1) &
  done; wait
  E=$M/scoop-a$v-s0.pt,$M/scoop-a$v-s1.pt,$M/scoop-a$v-s2.pt
  if [ ! -f $M/scoop-a$v-s2.pt ]; then echo "train failed a$v"; sleep 300; continue; fi
  echo "$E" > $L/CURRENT; echo "a$v" > $L/CURRENT_TAG; echo $((v + 1)) > $L/NEXT_VER; echo $n > $L/LAST_COUNT; last=$n
  bash $HOME/livedirt/digbench/al_loop/gate_eval.sh a$v "$E" 8811 8812
  $PY -m digbench.al_loop.ledger a$v $n
  v=$((v + 1))
done
