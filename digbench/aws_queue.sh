#!/bin/bash
# Persistent job queue for the AWS GPU box (and any SSH-driven worker). Each line of $JOBS is a shell command run
# once, in order, from ~/livedirt; finished lines go to $DONE. The file is re-read every 20 s, so new jobs can be
# appended while it runs. Pauses (does not start new jobs) when free disk is under MIN_FREE_GB.
# usage: setsid nohup bash digbench/aws_queue.sh ~/jobs.txt > ~/queue.log 2>&1 < /dev/null &
JOBS=${1:-$HOME/jobs.txt}
DONE=${JOBS%.txt}.done
MIN_FREE_GB=${MIN_FREE_GB:-15}
touch "$JOBS" "$DONE"
cd "$HOME/livedirt" || exit 1
export PATH=/opt/pytorch/bin:$PATH
while true; do
  job=""
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue ;; esac
    grep -qxF "$line" "$DONE" && continue
    job="$line"; break
  done < "$JOBS"
  if [ -z "$job" ]; then
    [ -z "$idle_since" ] && idle_since=$(date +%s) && echo "=== QUEUE EMPTY $(date -u +%H:%M:%S)"
    sleep 20; continue
  fi
  idle_since=""
  free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  if [ "$free" -lt "$MIN_FREE_GB" ]; then echo "=== PAUSED disk ${free}G free $(date -u +%H:%M:%S)"; sleep 60; continue; fi
  echo "=== START $(date -u +%H:%M:%S) $job"
  bash -c "$job" 2>&1 | grep -v '^\[Taichi\]'
  echo "=== END $(date -u +%H:%M:%S) rc=${PIPESTATUS[0]} $job"
  echo "$job" >> "$DONE"
done
