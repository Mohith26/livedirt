#!/bin/bash
# Second data worker (e.g. Colab) driven entirely by the AWS box: pulls ~/colab_jobs.txt from $HOST, runs each line
# once (with $V3 = local data root and traces synced from $HOST:~/traces), pushes $V3 back to $HOST:~/v3 after each
# job, and records finished lines in $HOST:~/colab_jobs.done. Re-reads the job list every 60 s.
# usage: setsid nohup bash digbench/remote_worker.sh ubuntu@HOST ~/.ssh/aws > /content/worker.log 2>&1 < /dev/null &
HOST=$1; KEY=$2
export V3=${V3:-/content/v3}
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
mkdir -p "$V3" "$HOME/traces"
cd "$(dirname "$0")/.." || exit 1
while true; do
  $SSH "$HOST" 'touch ~/colab_jobs.txt ~/colab_jobs.done; cat ~/colab_jobs.txt' > /tmp/cj.txt 2>/dev/null || { sleep 60; continue; }
  $SSH "$HOST" 'cat ~/colab_jobs.done' > /tmp/cd.txt 2>/dev/null
  job=""
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    grep -qxF "$line" /tmp/cd.txt && continue
    job="$line"; break
  done < /tmp/cj.txt
  if [ -z "$job" ]; then sleep 60; continue; fi
  git pull -q
  rsync -az -e "$SSH" "$HOST:traces/" "$HOME/traces/"
  echo "=== START $(date -u +%H:%M:%S) $job"
  bash -c "$job" 2>&1 | grep -v '^\[Taichi\]'
  rc=${PIPESTATUS[0]}
  rsync -az -e "$SSH" "$V3/" "$HOST:v3/"
  echo "=== END $(date -u +%H:%M:%S) rc=$rc $job"
  printf '%s\n' "$job" | $SSH "$HOST" 'cat >> ~/colab_jobs.done'
done
