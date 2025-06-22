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
#MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
#MODEL_NAME="mistralai/Ministral-8B-Instruct-2410"
#MODEL_NAME="edit_models/attn_span/meta-llama/Llama-3.2-3B-Instruct"
#MODEL_NAME="edit_models_w/head_kq_directions/meta-llama/Llama-3.2-3B-Instruct"

TOP_K_HEADS=20
DIRECTION_FACTOR=0.3

INITIAL_DATASET="focus_data/qa_prompt_needle/original"
OUTPUT_DIR="eval_baseline/qa_prompt_needle/${MODEL_NAME//\//_}"
#OUTPUT_DIR="eval/qa_prompt_needle/top_k_heads_${TOP_K_HEADS}/${MODEL_NAME//\//_}"
SPLIT="test"

#python3 init_edit_models.py --model_name meta-llama/Llama-3.2-3B-Instruct --target_dir edit_models/head_kq_directions
#python3 edit_kq_direction.py --top_k_heads ${TOP_K_HEADS} --direction_factor ${DIRECTION_FACTOR}
#python3 edit_head_temperature.py --top_k_heads ${TOP_K_HEADS}

accelerate launch\
 --main_process_port "${MASTER_PORT}"\
 run_eval.py\
  --model_name ${MODEL_NAME}\
   --test_dataset_name ${INITIAL_DATASET}\
   --output_dir ${OUTPUT_DIR}\
   --split ${SPLIT}



