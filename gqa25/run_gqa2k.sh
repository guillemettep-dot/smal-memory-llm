#!/bin/bash
# d_model 2048, 12 layers, 2 memory heads of (64 key + 64 value), 4 query heads each, 96 rows.
# State per layer 24,576 slot + 16,384 window = exactly 2x the h1/gqa4 budget at exactly 2x width,
# so the compression ratio against a KV cache is unchanged and each fact gets 2x the numbers.
# tok_per_micro x accum is held at 65,536 tokens/step so the schedule matches the pod arms exactly.
# NOTE: no --kdim here. Cfg does `kdim = int(c.kdim) or dv`, so with --dv 64 the stored key is
# 64 wide already -- identical to the pod's explicit --kdim 64. --dk is vestigial for slots.
set -u
export MEMPROBE_ROOT=/home/paul/memprobe MEMLM_MOD=memlm_kv
export KV_DETACH=1 KV_WQ=0 SLOT_ROPE=1 SLOT_ROPE_FRAC=0.5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
cd /home/paul/memprobe/gqa
exec /home/paul/memprobe/.venv/bin/python3 -u train_memlm.py --tag ${TAG:-gqa2k} --amp bf16 \
  --moe 1 --n_exp ${NEXP:-12} --top_k 2 --compile 0 \
  --d_model ${DMODEL:-2048} --layers 12 --heads ${HEADS:-2} --dk 48 --dv 64 --lanes 1 --d_ff ${DFF:-1408} \
  --n_slots ${NSLOTS:-96} --slot_hard 0 --slot_hard_gate 0 --gate_init 0.42 --scan_chunk 256 --buf_stride 1 \
  --kv_window 16 --kv_dim ${KVDIM:-128} --kv_heads ${KVHEADS:-4} --tk_nq ${TKNQ:-4} \
  --comp_nb 5 --xbus 0 --comp_off 1 --tksel 1 --tksel_dk 32 --sel_heads 0 --gist_seed 20260905 \
  --pre 2 --vocab 50304 --emit_ckpt 1 \
  --kdecor 0 --krep 1.0 --krep_tau 0.85 --wbud 0 --lane_clamp 2.0 --lane_clamp_until 25000 \
  --tokens ${TOKENS:-1e10} --lr ${LR:-1e-4} --warmup 2000 --stages ${STAGES:-128,256,512,1024} --ramp_frac ${RAMPFRAC:-0.03} \
  --p_synth 0.08 --p_synth_lo 0.50 --p_synth_hi 0.60 --p_synth_start 0.60 --p_synth_ramp 0.15 --ans_w ${ANSW:-15} \
  --stage_weight 1 --p_free 0.05 --tok_per_micro ${TPM:-2048} --accum ${ACCUM:-32} --cap 2.0 --ce_fused 1 \
  --log_every ${LOGEVERY:-5} --eval_every 5000 --eval_bs 1 --opt8bit 1 \
  --ckpt_every ${CKPTEVERY:-2000} --ckpt_dir ${CKPT_DIR_OVERRIDE:-/home/paul/memprobe/ckpt_${TAG:-gqa2k}} \
  --tbias 2.0 --p_broad 0.10 --long_frac 0.50 --p_tasks 0.6 --p_mqar 0.2 \
  --task_mix keyed_span:2,updated:1,tagged:1,list:1,multihop:1,reverse:1,pred:1,factgen:4 \
  >> /home/paul/memprobe/${TAG:-gqa2k}.log 2>&1
