#!/bin/bash
# every +PER labelled bites: train the next ensemble (NSEED members), re-rank the queue with it, make it CURRENT
# (DAgger: diggers use the newest policy), gate-evaluate, ledger (with the target check), track BEST
L=$HOME/v4/loop; M=$HOME/v4/models; PY=/opt/pytorch/bin/python; cd $HOME/livedirt
v=$(cat $L/NEXT_VER); last=$(cat $L/LAST_COUNT); per=${PER:-30}; NSEED=${NSEED:-5}
count() { ls $HOME/v4/scoops/campaign/*.npz $HOME/v4/scoops/loop-adaptive/*.npz 2>/dev/null | wc -l; }
while [ ! -f $L/STOP ]; do
  n=$(count)
  if (( n - last < per )); then sleep 300; continue; fi
  prev=$(cat $L/CURRENT_TAG); echo "=== a$v: $n samples (+$((n-last))) $(date -u +%T)"
  E=""
  for ((sd = 0; sd < NSEED; sd++)); do
    if (( v % 3 == 0 )) || [ ! -f $M/scoop-$prev-s$sd.pt ]; then init=""; ep=100; lr=2e-3; else init="--init $M/scoop-$prev-s$sd.pt"; ep=30; lr=5e-4; fi
    ($PY -u -m digbench.train_scoop --data $HOME/v4/scoops/campaign $HOME/v4/scoops/loop-adaptive --heldout-runs c240,gate \
        --epochs $ep --lr $lr --ch 32 --seed $sd --out $M --tag a$v-s$sd $init > $L/train-a$v-s$sd.log 2>&1) &
    E="$E${E:+,}$M/scoop-a$v-s$sd.pt"
  done; wait
  ok=1; for f in ${E//,/ }; do [ -f $f ] || ok=0; done
  if (( ok == 0 )); then echo "train failed a$v"; sleep 300; continue; fi
  echo "$E" > $L/CURRENT; echo "a$v" > $L/CURRENT_TAG; echo $((v + 1)) > $L/NEXT_VER; echo $n > $L/LAST_COUNT; last=$n
  OMP_NUM_THREADS=2 $PY -m digbench.al_loop.rescore "$E" 2>&1 | grep rescored
  bash $HOME/livedirt/digbench/al_loop/gate_eval.sh a$v "$E" 8811 8812
  $PY -m digbench.al_loop.ledger a$v $n
  v=$((v + 1))
done
