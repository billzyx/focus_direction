#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate focus_direction

cache_dir="./"

echo "Cache directory: $cache_dir"
export HF_HUB_CACHE=$cache_dir

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
#MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
#MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
#MODEL_NAME="mistralai/Ministral-8B-Instruct-2410"
TOTAL_LAYERS=28

INITIAL_DATASET="focus_data/qa_prompt_needle/original"
ACTIVATION_SAVE_DIR=${cache_dir}/head_features
ATTENTION_DATASET_DIR=${cache_dir}/head_features/${MODEL_NAME//\//_}/train
ATTENTION_HEADS_JSON_PATH=contextual_heads_scores/${MODEL_NAME//\//_}/distraction_heads.json
MODEL_SAVE_DIR=head_features

BATCH_SIZE=3  # number of layers to cache at once

for ((START_LAYER=0; START_LAYER<${TOTAL_LAYERS}; START_LAYER+=${BATCH_SIZE})); do

  # END_LAYER is typically (START_LAYER + 4), but we clamp it so as not to exceed (TOTAL_LAYERS - 1)
  END_LAYER=$((START_LAYER + BATCH_SIZE - 1))
  if [ ${END_LAYER} -ge $((TOTAL_LAYERS - 1)) ]; then
    END_LAYER=$((TOTAL_LAYERS - 1))
  fi

  rm -rf "${ATTENTION_DATASET_DIR}"

  # Create an array of layers to save
  LAYERS=()
  for ((L=${START_LAYER}; L<=${END_LAYER}; L++)); do
    LAYERS+=( ${L} )
  done

  echo "==> Caching activations for layers: ${LAYERS[@]}"

  # 1) Capture multiple layers in one go:
  python3 get_kq_activation.py \
    --model_name "${MODEL_NAME}" \
    --layers_to_save "${LAYERS[@]}" \
    --dataset_name "${INITIAL_DATASET}" \
    --split train \
    --save_dir "${ACTIVATION_SAVE_DIR}"

  # 2) Train each layer individually:
  for LAYER in "${LAYERS[@]}"; do
    echo "==> Training attention-only for layer ${LAYER}"

    python3 train_attn_only.py \
      --model_name "${MODEL_NAME}" \
      --layer_to_train "${LAYER}" \
      --dataset_root_dir "${ATTENTION_DATASET_DIR}" \
      --save_dir "${MODEL_SAVE_DIR}"\
      --target_type relevant
  done

  rm -rf "${ATTENTION_DATASET_DIR}"

done





