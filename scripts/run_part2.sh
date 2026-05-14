#!/usr/bin/env bash
set -euo pipefail

K=50
NUM_GPUS=2
EPOCHS=3
BATCH_SIZE=64
GRAD_ACCUM=1

MODEL_PATH="<model-path>"
DATA_PATH="<data-path>"
PART1_OUTPUT_DIR="./part1_output/llama-7b-p1-k50"
PART2_OUTPUT_DIR="./part2_output/llama-7b-p2-k50"
DECOMPRESSED_MODEL_DIR="./part2_output/llama-7b-p2-k50-decompressed"

BF16=True
FP16=False
TF32=False
LR="0.00002"

USED_LR=$(python3 -c "import math; print(${LR} * math.sqrt(100 / ${K}))")

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node="$NUM_GPUS" train_bs_p2.py \
  --model_name_or_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --data_length 100000000 \
  --part_1_output_path "$PART1_OUTPUT_DIR" \
  --output_dir "$PART2_OUTPUT_DIR" \
  --decompressed_model_path "$DECOMPRESSED_MODEL_DIR" \
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
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
  --gradient_checkpointing True \
  --bf16 "$BF16" \
  --tf32 "$TF32" \
  --fp16 "$FP16" \
  --report_to "none"
