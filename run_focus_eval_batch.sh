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


NUM_GPUS=2

#MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
#MODEL_NAME="edit_models/attn_span/meta-llama/Llama-3.2-3B-Instruct"

#base_model_name="meta-llama/Llama-3.2-3B-Instruct"
#base_model_name="Qwen/Qwen2.5-7B-Instruct"
base_model_name="mistralai/Ministral-8B-Instruct-2410"
MODEL_NAME="edit_models_w/head_kq_directions/${base_model_name}"


TOP_K_HEADS_L=(1 5 10 20 30 50 100)
DIRECTION_FACTOR_L=(0.1 0.2 0.3 0.4 0.5 -0.1 -0.2 -0.5 0.7 1.0)
DIRECTION_TYPE="direction_relevant"


INITIAL_DATASET="focus_data/qa_prompt_needle/original"
SPLIT="test"

for TOP_K_HEADS in "${TOP_K_HEADS_L[@]}"; do
    for DIRECTION_FACTOR in "${DIRECTION_FACTOR_L[@]}"; do
      OUTPUT_DIR="eval_main/qa_prompt_needle/${DIRECTION_TYPE}/${TOP_K_HEADS}_${DIRECTION_FACTOR}/${MODEL_NAME//\//_}"

      python3 init_edit_models.py --model_name ${base_model_name} --target_dir edit_models/head_kq_directions
      python3 edit_kq_direction.py --top_k_heads ${TOP_K_HEADS}\
       --direction_factor ${DIRECTION_FACTOR}\
        --direction_type ${DIRECTION_TYPE}\
        --model_name ${base_model_name}

      accelerate launch\
       --main_process_port "${MASTER_PORT}"\
       run_eval.py\
        --model_name ${MODEL_NAME}\
         --test_dataset_name ${INITIAL_DATASET}\
         --output_dir ${OUTPUT_DIR}\
         --split ${SPLIT}
    done
done



