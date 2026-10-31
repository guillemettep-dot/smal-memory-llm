#!/bin/bash
# kv5 = gqa2k's ARCHITECTURE (memlm_kv: 12 per-layer slot memories, 2 heads x 96 slots x (64+64), 4 queries
# per head, per-token reads, original MoE path, no compile) with sh3/sh4's TRAINING (lr 5e-5 wsd, 30% recall
# at answer weight 5, the 16-family task mix, same curriculum, 65,536 tokens/step).  From scratch.
# WHY: sh1-sh4 (shared memory) never used their memory by step 3000 (needle ~0, gap flat) while gqa2k had
# +0.6 nats at 3000 and a falling gap from 2500.  This splits the remaining causes in two:
#   kv5 lifts by ~3000  -> the shared-memory ARCHITECTURE is the cause;
#   kv5 stays flat      -> the training (data mix / answer weight / lr) is.
set -u
export MEMPROBE_ROOT=/home/paul/memprobe MEMLM_MOD=memlm_kv
export KV_DETACH=1 KV_WQ=0 SLOT_ROPE=1 SLOT_ROPE_FRAC=0.5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
cd /home/paul/memprobe/gqa
exec /home/paul/memprobe/.venv/bin/python3 -u train_sh.py --tag ${TAG:-kv5} --amp bf16 \
  --moe 1 --n_exp 4 --top_k 2 --compile 0 \
  --d_model 2048 --layers 12 --heads 2 --dk 48 --dv 64 --lanes 1 --d_ff 1408 \
  --n_slots 96 --slot_hard 0 --slot_hard_gate 0 --gate_init 0.42 --scan_chunk 256 --buf_stride 1 \
  --kv_window 16 --kv_dim 128 --kv_heads 4 --tk_nq 4 \
  --comp_nb 5 --xbus 0 --comp_off 1 --tksel 1 --tksel_dk 32 --sel_heads 0 --gist_seed 20260905 \
  --pre 2 --vocab 50304 --emit_ckpt 1 \
  --kdecor 0 --krep 1.0 --krep_tau 0.85 --wbud 0 --lane_clamp 2.0 --lane_clamp_until 25000 \
  --tokens ${TOKENS:-1e10} --lr ${LR:-5e-5} --warmup 2000 --sched wsd --decay_frac 0.2 \
  --stages 128,256,512,1024 --ramp_frac 0.03 \
  --p_synth 0.30 --p_synth_lo 0.30 --p_synth_hi 0.30 --p_synth_start 0.30 --p_synth_ramp 0.15 --ans_w 5 \
  --stage_weight 1 --p_free 0.05 --tok_per_micro 1024 --accum 64 --cap 2.0 --ce_fused 1 \
  --log_every 20 --eval_every 5000 --eval_bs 1 --opt8bit 1 \
  --ckpt_every 250 --ckpt_dir /mnt/ssd/memprobe_ckpt/kv5 \
  --bank_every 1000 --bank_dir /home/paul/memprobe/bank_kv5 \
  --tbias 2.0 --p_broad 0.10 --long_frac 0.50 --p_tasks 0.6 --p_mqar 0.2 \
  --task_mix keyed_span:1,updated:1,tagged:1,list:1,multihop:1,reverse:1,pred:1,factgen:4,chat:2,record:2,codeconst:2,schedule:1,inventory:1,glossary:1,moves:1,quote:1 \
  >> /home/paul/memprobe/kv5.log 2>&1
