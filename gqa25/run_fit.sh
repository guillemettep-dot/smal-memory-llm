#!/bin/bash
# fit + speed test, from scratch at T=1024 only.  args: TAG HEADS DIM TOK_PER_MICRO NSTEPS
# The sh21 command line with only: tag, memory shape, micro-batch (accum keeps 65,536 tok/step), stages 1024,
# no evals / banks / real checkpoints, its own log and ckpt dir.  Stopped BY PID after NSTEPS logged steps.
set -u
TAG=$1; H=$2; D=$3; TPM=$4; NS=$5
ACC=$((65536 / TPM))
LOG=/home/paul/memprobe/fit_$TAG.log
CMD=$(sed -n "/^exec /,/>> /p" /home/paul/memprobe/gqa21/run_sh21.sh | sed -e "s/^exec //" -e "s/ *>> .*//" -e "s/\\\\$//" | tr "\n" " ")
CMD=$(echo "$CMD" | sed -e "s/--tag sh21 /--tag $TAG /" -e "s/--sh_heads 12 /--sh_heads $H /" -e "s/--sh_kdim 64 /--sh_kdim $D /" \
  -e "s/--sh_dv 64 /--sh_dv $D /" -e "s/--tok_per_micro 2048 /--tok_per_micro $TPM /" -e "s/--accum 32 /--accum $ACC /" \
  -e "s/--stages 128,256,512,1024 /--stages 1024 /" -e "s/--log_every 20 /--log_every 2 /" -e "s/--eval_every 5000 /--eval_every 1000000 /" \
  -e "s#--ckpt_every 250 --ckpt_dir /mnt/ssd/memprobe_ckpt/sh21 #--ckpt_every 1000000 --ckpt_dir /mnt/ssd/memprobe_ckpt/$TAG #" \
  -e "s#--bank_every 1000 --bank_dir /home/paul/memprobe/bank_sh21 #--bank_every 0 --bank_dir /tmp/nobank #" \
  -e "s/\${TOKENS:-1e10}/1e10/" -e "s/\${LR:-5e-5}/5e-5/")
for want in "--tag $TAG " "--sh_heads $H " "--sh_kdim $D " "--sh_dv $D " "--tok_per_micro $TPM " "--accum $ACC " "--stages 1024 " "--ckpt_dir /mnt/ssd/memprobe_ckpt/$TAG " "--bank_every 0 "; do
  case "$CMD" in *"$want"*) ;; *) echo "SUBSTITUTION MISSING: $want"; exit 2;; esac
done
case "$CMD" in *sh21*) echo "sh21 still in command: $CMD"; exit 2;; esac
[ "${DRY:-0}" = 1 ] && { echo "$CMD"; exit 0; }
[ -e /mnt/ssd/memprobe_ckpt/$TAG ] && { echo "ckpt dir exists, refusing"; exit 2; }
export KV_CUDA=${KV_CUDA:-0}
export MEMPROBE_ROOT=/home/paul/memprobe MEMLM_MOD=memlm_sh TIKTOKEN_CACHE_DIR=/home/paul/memprobe/tiktoken_cache
export KV_DETACH=1 KV_WQ=0 SLOT_ROPE=1 SLOT_ROPE_FRAC=0.5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
cd /home/paul/memprobe/gqa22
echo "$CMD" > $LOG
$CMD >> $LOG 2>&1 &
P=$!
peak=0
while kill -0 $P 2>/dev/null; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$m" -gt "$peak" ] && peak=$m
  n=$(grep -c "^step " $LOG)
  [ "$n" -ge "$NS" ] && { kill $P; break; }
  sleep 1
done
wait $P 2>/dev/null
echo "=== $TAG: peak GPU memory ${peak} MiB (of 12288); logged steps $(grep -c "^step " $LOG)" | tee -a $LOG
grep -m3 -i "out of memory\|Traceback\|Error" $LOG
grep "^step " $LOG | tail -4 | cut -c1-60,190-
