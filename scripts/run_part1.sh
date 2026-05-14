#!/usr/bin/env bash
set -euo pipefail

K=50
NUM_GPUS=2
EPOCHS=3
BATCH_SIZE=64
GRAD_ACCUM=1

MODEL_PATH="<model-path>"
DATA_PATH="<data-path>"
OUTPUT_DIR="./part1_output/llama-7b-p1-k50"

BF16=True
FP16=False
TF32=False
ADDITIONAL_DIM=32
KEEPING_EPOCH=1
BS_SHRINKING_STEP=100
LR="0.00002"

FACTOR=$(python3 -c "import math; K=$K; EPOCHS=$EPOCHS; print((math.sqrt(100 / K) - 1) / math.log(100 / K) * 2 * (1 - 1 / EPOCHS) + 1 / EPOCHS)")
USED_LR=$(python3 -c "LR=$LR; FACTOR=$FACTOR; print(LR * float(FACTOR))")

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node="$NUM_GPUS" train_bs_p1.py \
  --model_name_or_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --data_length 100000000 \
  --output_directory "$OUTPUT_DIR" \
  --use_fast_tokenizer False \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000000 \
  --save_total_limit 2 \
  --learning_rate "$USED_LR" \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 500 \
  --gradient_checkpointing True \
  --bf16 "$BF16" \
  --tf32 "$TF32" \
  --fp16 "$FP16" \
  --basis_selection_threshold "$K" \
  --bs_additional_dim "$ADDITIONAL_DIM" \
  --bs_keeping_epoch "$KEEPING_EPOCH" \
  --bs_shrinking_step "$BS_SHRINKING_STEP" \
  --report_to "none"
