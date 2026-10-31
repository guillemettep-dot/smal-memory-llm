#!/bin/bash
# sh13 = sh12 with NO WRITE GATE (--sh_nogate 1: every token writes, g=1), by request 2026-10-12.  Started from sh12s
#   step-1000 WEIGHTS (gate still fully open there, so removing it changes nothing: logits equal to 5e-6) + the data
#   stream position of sh12 @1782 (no replay) + FRESH optimizer moments with --relift 200 from lr 1e-7.
#   WHY: the learnable gate SHUT in every run between steps 1000 and 2000 (sh12 hard gate: main 0.0% of tokens writing
#   at 1750) before the memory was useful.  sh12 @1782 was NOT used: forcing writes there cost +0.19 nats (reads had
#   learned the memory was empty).  With --sh_fr 16 every token takes its own row: at T=128 the main memory holds the
#   whole sequence (0% overwrites).  sh12 stopped at 1782 (resumable, gqa/guard_sh12.sh).
# sh12 = sh11 config FROM SCRATCH with the write query initialised as MINUS the key projection (--sh_wq_std 0 --sh_wq_sign -1),
#   by request 2026-10-11.  WHY: random-direction init still overwrote ~1/3 of writes at T=128 (sh11 @500: main 33%,
#   a used row wins when cos > fr/temp = 0.28 and a random 64-dim direction clears that by chance).  The keys share a
#   common direction, so -key points away from them: fresh model main 3.3% into a used row (119-128 rows / 128 writes).
#   sh11 stopped at 1046 (resumable, gqa/guard_sh11.sh).
#   PLUS --sh_fr 16 (free-row preference FROZEN at 16, by request): fr/temp ~1.0 > any cosine, so every write takes a new row
#   until the memory is full (fresh model: main 0.0% into a used row, enc 25.0% = the minimum with 96 rows for 128 tokens).
# sh11 = sh10 config FROM SCRATCH with a HARD write gate (--slot_hard_gate 1, by request 2026-10-11): a token writes
#   fully (g=1) iff gate>0.5, else nothing; gradient straight-through.  WHY: sh10 soft gate closed 0.998 -> 0.09 by step
#   3000 and occupancy o follows g, so fr*o vanished and ~80% of main writes overwrote a used row with most rows empty;
#   battery 0 at 1k/2k/3k.  Fresh-model test at gate 0.1: soft 78% into a used row, hard 0%.  Kernel = ref (hardgate_kcheck.py).
# sh10 = sh9 config FROM SCRATCH with two init changes (2026-10-10, by request):
#   --gate_init 0.999   write gate starts open (sh9 0.42 collapsed to mean 0.013 by step 1000: 81% of writes under the floor)
#   --sh_wq_std 0.01    write query starts as an independent N(0,0.01) matrix, not a copy of the key.  Fresh-model test
#                       (gate 0.999, T=128): copy init -> 87% of main writes into a used row, 9-25 rows/head; random -> 22%, 88-112.
# WHY: sh9 @1000 used 2 of 192 main rows (every write into one row); flat battery/needle through 3730.
# sh9 = sh8's exact config FROM SCRATCH (step 0), lr 5e-5 (halve later if it plateaus, as sh8 did at 10750).
# The separate write key exists from step 0 instead of being added at 8210 by convert_wq.py.
# WHY: at sh8 step 14000 the write key had diverged from the stored key in only 6 of 12 heads (those use all
# 192 slots); in heads 2/9/11 it was still a copy (cos 0.95) and they used 27-40 slots.  Does it learn in every
# head when present from the start?  RISK: this light training (30% recall, ans_w 5) never started a memory from
# scratch (sh2-sh4, kv5); judge at 3000 (gap + battery at 1k/2k/3k), switch to the sh5 recipe if no lift.
# sh8 notes follow.
# sh8 = sh7 (top-1 writes + gate floor 0.05) + a SEPARATE WRITE-QUERY KEY (--sh_wq 1): the slot is chosen by its own
# projection; the stored key stays the key projection.  Branched from sh7's latest checkpoint by convert_wq.py:
# the write query starts as an exact copy of the key projection (same function, verified bit-identical) and the
# optimizer state is remapped so every existing parameter keeps its Adam moments.
# WHY: sh7 @8000 used 13.5 of 192 slots, 96.6% of writes overwrote a used slot, median life of a stored token 6.
# An empty slot always scores 0 while any used slot scores temp*cos - fr*o, so one key cannot both find a
# good slot and be the thing stored.  sh7 notes follow.
# sh7 = sh6 with TOP-1 WRITES (--slot_hard 1): forward writes each token into its single best-matching slot,
# backward passes the softmax (cosine-address) gradient straight through.  Everything else identical to sh6,
# and it starts from the SAME point (sh5 step 5048) on the SAME data stream (rng restored), so sh6's batteries at
# 5456 / 6000 / 6427 / 7000 are the exact A/B.
# WHY: measured on sh6 @7000, only 66% of a write lands on its top row (27% of writes >90%); the rest smears
# into other slots = interference.  sh6 under the light training then lost code recall 90 -> 56-64%.
# Kernel hard path at stride 8, detach 1: fwd + all 5 grads match fp64 to ~1e-6 (hardcheck.py).
# sh6 notes follow.
# sh6 = a BRANCH of sh5, resumed from sh5's step ~4950 checkpoint (copied in as sh6_ckpt.pt), switched to the
# NEW TRAINING: recall 30% at answer weight 5, the 16-family task mix (8 natural formats), lr 5e-5.
# WHY: the light training failed to START the memory (sh2-sh4, kv5: zero lift by 3000), but sh5 started it under
# gqa2k's heavy training (needle 92-94% exact at gaps 64-128, code battery 83-88% at step 4938).  Question:
# does a memory that already works hold, or generalise further, under the lighter, more varied training?
# Watch: needle exact + the loss-CE gap (now at answer weight 5, so compare against sh5's trend, not its value).
# sh5 itself is kept (resume: gqa/guard_sh5.sh) as the control.
# sh5 notes follow.
# sh5 = NEW ARCHITECTURE (sh3: shared memory written at layer 6 + encoder memory at layer 2, fp until 80% then
# 3-bit QAT) with the memory snapshot every 8 tokens (--sh_stride 8; the 16-token window covers the gap), and the
# OLD TRAINING that made gqa2k's memory work: recall 60% -> floor 50-60%, answer weight 15, the original 8-task
# mix, lr 1e-4.  WHY: kv5 (gqa2k's architecture with the sh runs' training) was as flat as sh4 to 3 decimals
# (gap +0.080/+0.074/+0.080 over 2-3.5k, battery 0 at 1k/2k/3k), so the training was the cause, not the design.
# lr 1e-4 is what gqa2k had while its memory formed; it later caused recall swings, so it is lowered by resume
# (wsd: constant phase) once the memory works.
# sh3 notes follow.
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
# tok_per_micro x accum = 65,536 tokens/step (sh5: lr 1e-4, gqa2k's value; see the header).
set -u
export MEMPROBE_ROOT=/home/paul/memprobe MEMLM_MOD=memlm_sh
export KV_DETACH=1 KV_WQ=0 SLOT_ROPE=1 SLOT_ROPE_FRAC=0.5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
cd /home/paul/memprobe/gqa
exec /home/paul/memprobe/.venv/bin/python3 -u train_sh.py --tag ${TAG:-sh13} --amp bf16 \
  --moe 1 --n_exp 4 --top_k 2 --compile 1 \
  --d_model 2048 --layers 12 --heads 2 --dk 48 --dv 64 --lanes 1 --d_ff 1408 \
  --n_slots 96 --slot_hard 1 --slot_hard_gate 1 --gate_init 0.999 --scan_chunk 256 --buf_stride 1 \
  --kv_window 16 --kv_dim 128 --kv_heads 4 --tk_nq 1 \
  --sh_write 6 --sh_heads 12 --sh_slots 192 --sh_kdim 64 --sh_dv 64 --sh_nq 2 --sh_stride 8 --sh_qbits 0 --sh_qgroup 16 --qat_bits 3 --qat_start_frac 0.8 --sh_enc_write 2 --sh_enc_slots 96 --sh_gate_floor 0.05 --sh_wq 1 --sh_wq_std 0 --sh_wq_sign -1 --sh_fr 16 --sh_nogate 1 \
  --comp_nb 5 --xbus 0 --comp_off 1 --tksel 1 --tksel_dk 32 --sel_heads 0 --gist_seed 20260905 \
  --pre 2 --vocab 50304 --emit_ckpt 1 \
  --kdecor 0 --krep 1.0 --krep_tau 0.85 --wbud 0 --lane_clamp 2.0 --lane_clamp_until 25000 \
  --tokens ${TOKENS:-1e10} --lr ${LR:-5e-5} --warmup 2000 --sched wsd --decay_frac 0.2 --relift 200 --relift_lr0 1e-7 \
  --stages 128,256,512,1024 --ramp_frac 0.03 --stage_hold 5000 \
  --p_synth 0.30 --p_synth_lo 0.30 --p_synth_hi 0.30 --p_synth_start 0.30 --p_synth_ramp 0.15 --ans_w 5 \
  --stage_weight 1 --p_free 0.05 --tok_per_micro 2048 --accum 32 --cap 2.0 --ce_fused 1 \
  --log_every 20 --eval_every 5000 --eval_bs 1 --opt8bit 1 \
  --ckpt_every 250 --ckpt_dir /mnt/ssd/memprobe_ckpt/sh13 \
  --bank_every 1000 --bank_dir /home/paul/memprobe/bank_sh13 \
  --tbias 2.0 --p_broad 0.10 --long_frac 0.50 --p_tasks 0.6 --p_mqar 0.2 \
  --task_mix keyed_span:1,updated:1,tagged:1,list:1,multihop:1,reverse:1,pred:1,factgen:4,chat:2,record:2,codeconst:2,schedule:1,inventory:1,glossary:1,moves:1,quote:1 \
  >> /home/paul/memprobe/sh13.log 2>&1
