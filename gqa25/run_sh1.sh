#!/bin/bash
# sh1: ONE shared slot memory (memlm_sh), from scratch.  Asked for 2026-10-02:
#   * all-decoder; one memory written at layer 4, read by every layer >= 4 (DeepSeek-style single cache)
#   * memory 12 heads x 192 slots x (64 key + 64 value) = 294,912 numbers, the SAME total as the
#     per-layer design (12 layers x 2 heads x 96 x 128): 2x the slots per head, 6x the heads per memory
#   * readers: 2 queries per head; output stride 16 (= the exact window, so no write is ever invisible)
#   * recall share 30% (was 50-60%), answer weight 5 (was 15), 16 task families incl. 8 natural formats
#   * 3-bit storage is an EVAL setting (SH_QBITS=3), training is fp
#   * speed: compiled blocks, padding-free exact MoE, stride-16 scan, micro-batch 4 -- 2.49x gqa2k measured
#     (B=2 4914 tok/s, B=4 5338, peak 8.4 GB: fewer bf16 grad-accumulation passes per token)
#   * wsd schedule: constant lr after warmup, decay over the last 20% -- the token budget can be cut
#     later by resuming with a smaller TOKENS, at any point before the decay starts
# tok_per_micro x accum = 65,536 tokens/step, the same update size as gqa2k, so lr 2.5e-5 carries over.
set -u
export MEMPROBE_ROOT=/home/paul/memprobe MEMLM_MOD=memlm_sh
export KV_DETACH=1 KV_WQ=0 SLOT_ROPE=1 SLOT_ROPE_FRAC=0.5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
cd /home/paul/memprobe/gqa
exec /home/paul/memprobe/.venv/bin/python3 -u train_sh.py --tag ${TAG:-sh1} --amp bf16 \
  --moe 1 --n_exp 4 --top_k 2 --compile 1 \
  --d_model 2048 --layers 12 --heads 2 --dk 48 --dv 64 --lanes 1 --d_ff 1408 \
  --n_slots 96 --slot_hard 0 --slot_hard_gate 0 --gate_init 0.42 --scan_chunk 256 --buf_stride 1 \
  --kv_window 16 --kv_dim 128 --kv_heads 4 --tk_nq 1 \
  --sh_write 4 --sh_heads 12 --sh_slots 192 --sh_kdim 64 --sh_dv 64 --sh_nq 2 --sh_stride 16 \
  --comp_nb 5 --xbus 0 --comp_off 1 --tksel 1 --tksel_dk 32 --sel_heads 0 --gist_seed 20260905 \
  --pre 2 --vocab 50304 --emit_ckpt 1 \
  --kdecor 0 --krep 1.0 --krep_tau 0.85 --wbud 0 --lane_clamp 2.0 --lane_clamp_until 25000 \
  --tokens ${TOKENS:-1e10} --lr ${LR:-2.5e-5} --warmup 2000 --sched wsd --decay_frac 0.2 \
  --stages 128,256,512,1024 --ramp_frac 0.03 \
  --p_synth 0.30 --p_synth_lo 0.30 --p_synth_hi 0.30 --p_synth_start 0.30 --p_synth_ramp 0.15 --ans_w 5 \
  --stage_weight 1 --p_free 0.05 --tok_per_micro 2048 --accum 32 --cap 2.0 --ce_fused 1 \
  --log_every 20 --eval_every 5000 --eval_bs 1 --opt8bit 1 \
  --ckpt_every 250 --ckpt_dir /mnt/ssd/memprobe_ckpt/sh1 \
  --bank_every 1000 --bank_dir /home/paul/memprobe/bank_sh1 \
  --tbias 2.0 --p_broad 0.10 --long_frac 0.50 --p_tasks 0.6 --p_mqar 0.2 \
  --task_mix keyed_span:1,updated:1,tagged:1,list:1,multihop:1,reverse:1,pred:1,factgen:4,chat:2,record:2,codeconst:2,schedule:1,inventory:1,glossary:1,moves:1,quote:1 \
  >> /home/paul/memprobe/sh1.log 2>&1
