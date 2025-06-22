#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate focus_direction

base_model_name="meta-llama/Llama-3.2-3B-Instruct"

python3 init_edit_models.py --model_name ${base_model_name} --target_dir edit_models/head_kq_directions
python3 edit_kq_direction.py --top_k_heads 20 \
                --direction_factor 0.2 \
                --direction_type direction_relevant