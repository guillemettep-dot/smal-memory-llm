#!/bin/bash
# Respawn + stall watchdog for the sh21 arm (sh20 + learned key share in the write query) (sh19c + window in every layer + per-token free-row preference) on the 3060.  Copied from guard_gqa2k.sh.
#
# HISTORY THIS GUARD EXISTS TO NOT REPEAT: the first attempt OOM'd at step 480 (deterministically,
# same loss both times, at T=128 -- NOT a sequence-length transition), wrote no checkpoint because
# ckpt_every was 1000, and looped for 6 hours making zero progress because the only stop condition
# was exits under 120s and each failure took 3 hours.
#
#  - CKPT_EVERY IS 250, BELOW THE KNOWN FAILURE POINT. The trainer writes ONE rolling file and
#    overwrites it, so this costs no disk -- there is no reason to checkpoint less often.
#  - OOM STOP: two consecutive exits whose log ends in OutOfMemory means the config does not fit.
#    Stop and say so; do not respawn into it for a fortnight.
#  - FAST-FAIL kept for genuinely broken launches (3 exits inside 120s).
#  - DISK GUARD on /mnt/ssd; / is 96% full which is why checkpoints do not live there.
#  - kills are BY PID from nvidia-smi, never by pattern: a pattern can match this shell.
LOG=/home/paul/memprobe/sh21.log
CKPT_DIR=/mnt/ssd/memprobe_ckpt/sh21
STALL=2400
unset TAG
fast=0; ooms=0
mkdir -p "$CKPT_DIR"
while true; do
    avail=$(df -BG --output=avail /mnt/ssd | tail -1 | tr -dc '0-9')
    if [ "${avail:-0}" -lt 20 ]; then
        echo "=== $(date -u +%FT%TZ) STOP: /mnt/ssd has ${avail}G left (<20G) ===" >> "$LOG"; exit 3
    fi
    t0=$(date +%s)
    mark=$(wc -l < "$LOG" 2>/dev/null || echo 0)
    bash /home/paul/memprobe/gqa21/run_sh21.sh &
    TP=$!
    while kill -0 $TP 2>/dev/null; do
        sleep 60
        now=$(date +%s); mt=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now")
        if [ $((now - mt)) -gt $STALL ]; then
            echo "=== $(date -u +%FT%TZ) STALLED ${STALL}s with no log write, killing $TP ===" >> "$LOG"
            kill $TP 2>/dev/null; sleep 20
            pkill -9 -P $TP 2>/dev/null; kill -9 $TP 2>/dev/null; sleep 5
            for gp in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
                echo "=== killing leftover GPU process $gp ===" >> "$LOG"; kill -9 "$gp" 2>/dev/null
            done
            sleep 5; break
        fi
    done
    wait $TP 2>/dev/null; rc=$?
    dt=$(( $(date +%s) - t0 ))
    if tail -n +$((mark+1)) "$LOG" 2>/dev/null | grep -q OutOfMemory; then
        ooms=$((ooms+1))
        echo "=== $(date -u +%FT%TZ) exited rc=$rc after ${dt}s -- OUT OF MEMORY (${ooms} in a row) ===" >> "$LOG"
        if [ $ooms -ge 2 ]; then
            echo "=== $(date -u +%FT%TZ) STOP: OOM twice running. This config does not fit; stopping rather than looping. ===" >> "$LOG"
            exit 4
        fi
    else
        ooms=0
        echo "=== $(date -u +%FT%TZ) exited rc=$rc after ${dt}s, respawning ===" >> "$LOG"
    fi
    if [ $dt -lt 120 ]; then fast=$((fast+1)); else fast=0; fi
    if [ $fast -ge 3 ]; then
        echo "=== $(date -u +%FT%TZ) STOP: 3 fast exits in a row, a failure not a crash ===" >> "$LOG"; exit 2
    fi
    sleep 10
done
