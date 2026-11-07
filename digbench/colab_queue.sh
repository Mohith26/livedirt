#!/bin/bash
# Background job queue for Colab: runs each line of a job file once (skips jobs already marked done).
# usage: nohup bash digbench/colab_queue.sh jobs.txt > /content/queue.log 2>&1 &
JOBS=${1:-/content/jobs.txt}
DONE=/content/jobs.done
touch "$DONE"
cd /content/livedirt
while IFS= read -r job; do
  [ -z "$job" ] && continue
  grep -qxF "$job" "$DONE" && continue
  echo "=== START $(date -u +%H:%M:%S) $job"
  bash -c "$job" 2>&1 | grep -v '^\[Taichi\]'
  echo "=== END $(date -u +%H:%M:%S) rc=${PIPESTATUS[0]} $job"
  echo "$job" >> "$DONE"
done < "$JOBS"
echo "=== QUEUE EMPTY $(date -u +%H:%M:%S)"
