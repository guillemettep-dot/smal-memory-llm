#!/bin/bash
# sh4 = sh3 with PER-TOKEN memory reads (--sh_stride 1), everything else identical.  WHY: at stride 16 the
# memory can never help with a dependency closer than ~16 tokens (the window always covers those), so it
# loses the plentiful short-range repeats of plain text that gave gqa2k's per-token memory (G=1) its early
# gradient.  sh1-sh3 showed ZERO needle lift at step 3000 where gqa2k had +0.6 nats.  Test: lift by 3000?
# Micro-batch 1 (1024 x 64): stride 1 stores 16x more snapshots; B=2 OOMs (measured), B=1 2529 tok/s peak 8.0 GB.
# sh3 = sh2 + two fixes from the DeepSeek-V4.1-Flash paper (arXiv 2609.19969), from scratch, lr 5e-5:
#   1. the memory trains at FULL PRECISION and switches to 3-bit QAT at 80% of the run (the start of the
#      decay): DeepSeek pretrains its global KV in full precision and adds FP4 QAT only in post-training.
#   2. an ENCODER memory (12h x 96 slots, written at layer 2, read by layers 2-5): their encoder layers
#      have long-range attention of their own, ours had only the 16-token window below the main writer.
#      Total memory 442,368 numbers = 1.5x the original budget.
# sh1 notes follow.  Asked for 2026-10-02:
#   * all-decoder; one memory written at layer 4, read by every layer >= 4 (DeepSeek-style single cache)
#   * memory 12 heads x 192 slots x (64 key + 64 value) = 294,912 numbers, the SAME total as the
#     per-layer design (12 layers x 2 heads x 96 x 128): 2x the slots per head, 6x the heads per memory
#   * readers: 2 queries per head; output stride 16 (= the exact window, so no write is ever invisible)
#   * recall share 30% (was 50-60%), answer weight 5 (was 15), 16 task families incl. 8 natural formats
#   * 3-bit storage is TRAINED (sh_qbits 3), as DeepSeek trains FP4
#   * speed: compiled blocks, padding-free exact MoE, stride-16 scan, micro-batch 2 -- 2.3x gqa2k measured
#     (B=4 measured 5338 tok/s but OOMs in the real trainer: optimizer states are not in the bench)
#   * wsd schedule: constant lr after warmup, decay over the last 20% -- the token budget can be cut
#     later by resuming with a smaller TOKENS, at any point before the decay starts
# tok_per_micro x accum = 65,536 tokens/step, lr 5e-5 from the start: sh2 at 2.5e-5 learned ~2.5x slower per
# step than gqa2k (1e-4; plain CE 6.55 vs 4.94 at step 3740), and 1e-4 is ~3x too high for this update size.
set -u
export MEMPROBE_ROOT=/home/paul/memprobe MEMLM_MOD=memlm_sh
export KV_DETACH=1 KV_WQ=0 SLOT_ROPE=1 SLOT_ROPE_FRAC=0.5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
cd /home/paul/memprobe/gqa
exec /home/paul/memprobe/.venv/bin/python3 -u train_sh.py --tag ${TAG:-sh4} --amp bf16 \
  --moe 1 --n_exp 4 --top_k 2 --compile 1 \
  --d_model 2048 --layers 12 --heads 2 --dk 48 --dv 64 --lanes 1 --d_ff 1408 \
  --n_slots 96 --slot_hard 0 --slot_hard_gate 0 --gate_init 0.42 --scan_chunk 256 --buf_stride 1 \
  --kv_window 16 --kv_dim 128 --kv_heads 4 --tk_nq 1 \
  --sh_write 6 --sh_heads 12 --sh_slots 192 --sh_kdim 64 --sh_dv 64 --sh_nq 2 --sh_stride 1 --sh_qbits 0 --sh_qgroup 16 --qat_bits 3 --qat_start_frac 0.8 --sh_enc_write 2 --sh_enc_slots 96 \
  --comp_nb 5 --xbus 0 --comp_off 1 --tksel 1 --tksel_dk 32 --sel_heads 0 --gist_seed 20260905 \
  --pre 2 --vocab 50304 --emit_ckpt 1 \
  --kdecor 0 --krep 1.0 --krep_tau 0.85 --wbud 0 --lane_clamp 2.0 --lane_clamp_until 25000 \
  --tokens ${TOKENS:-1e10} --lr ${LR:-5e-5} --warmup 2000 --sched wsd --decay_frac 0.2 \
  --stages 128,256,512,1024 --ramp_frac 0.03 \
  --p_synth 0.30 --p_synth_lo 0.30 --p_synth_hi 0.30 --p_synth_start 0.30 --p_synth_ramp 0.15 --ans_w 5 \
  --stage_weight 1 --p_free 0.05 --tok_per_micro 1024 --accum 64 --cap 2.0 --ce_fused 1 \
  --log_every 20 --eval_every 5000 --eval_bs 1 --opt8bit 1 \
  --ckpt_every 250 --ckpt_dir /mnt/ssd/memprobe_ckpt/sh4 \
  --bank_every 1000 --bank_dir /home/paul/memprobe/bank_sh4 \
  --tbias 2.0 --p_broad 0.10 --long_frac 0.50 --p_tasks 0.6 --p_mqar 0.2 \
  --task_mix keyed_span:1,updated:1,tagged:1,list:1,multihop:1,reverse:1,pred:1,factgen:4,chat:2,record:2,codeconst:2,schedule:1,inventory:1,glossary:1,moves:1,quote:1 \
  >> /home/paul/memprobe/sh4.log 2>&1
