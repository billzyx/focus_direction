import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import shutil
import json
import argparse

def edit_kq_direction(
        model_name,
        top_k_heads,
        direction_factors,
        direction_types,
        save_path,
        edit_path="edit_models/head_kq_directions",
):
    model_path = f'{edit_path}/{model_name}'
    model_name_no_slash = model_name.replace('/', '_')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True)

    attention_heads_json_path = f"contextual_heads_scores_full_seq/{model_name_no_slash}/relevant_heads.json"
    with open(attention_heads_json_path, "r") as f:
        attention_heads = json.load(f)
    attention_heads = attention_heads[:top_k_heads]

    for layer in range(model.config.num_hidden_layers):
        with torch.no_grad():
            model.model.layers[layer].self_attn.head_k_directions.copy_(torch.zeros_like(model.model.layers[layer].self_attn.head_k_directions))
            model.model.layers[layer].self_attn.head_q_directions.copy_(torch.zeros_like(model.model.layers[layer].self_attn.head_q_directions))

    for direction_type, direction_factor in zip(direction_types, direction_factors):
        direction_based_dir = f'head_features/{model_name_no_slash}/{direction_type}'
        for layer in range(model.config.num_hidden_layers):
            with torch.no_grad():
                k_path = os.path.join(direction_based_dir, f'head_k_directions_{layer}.pth')
                if os.path.exists(k_path):
                    k_directions = torch.load(k_path, map_location=next(model.model.layers[layer].parameters()).device)
                    model.model.layers[layer].self_attn.head_k_directions += k_directions * direction_factor
                else:
                    print('Missing k_path:', k_path)

                q_path = os.path.join(direction_based_dir, f'head_q_directions_{layer}.pth')
                if os.path.exists(q_path):
                    q_directions = torch.load(q_path, map_location=next(model.model.layers[layer].parameters()).device)
                    model.model.layers[layer].self_attn.head_q_directions += q_directions * direction_factor
                else:
                    print('Missing q_path:', q_path)

    for layer in range(model.config.num_hidden_layers):
        heads_to_train = [head[1] for head in attention_heads if head[0] == layer]
        head_not_to_train = list(set(range(model.config.num_attention_heads)) - set(heads_to_train))
        for head in head_not_to_train:
            with torch.no_grad():
                model.model.layers[layer].self_attn.head_k_directions[head] = torch.zeros_like(
                    model.model.layers[layer].self_attn.head_k_directions[head]
                )
                model.model.layers[layer].self_attn.head_q_directions[head] = torch.zeros_like(
                    model.model.layers[layer].self_attn.head_q_directions[head]
                )

    save_folder = os.path.join(save_path, model_name)
    model.save_pretrained(save_folder)
    tokenizer.save_pretrained(save_folder, safe_serialization=False)

    if 'llama' in model_name.lower():
        modeling_file = "modeling_llama.py"
    elif 'qwen' in model_name.lower():
        modeling_file = "modeling_qwen2.py"
    elif 'mistral' in model_name.lower():
        modeling_file = "modeling_mistral.py"
    else:
        raise ValueError("Invalid model name.")

    shutil.copy(os.path.join(model_path, modeling_file), os.path.join(save_folder, modeling_file))
    shutil.copy(os.path.join(model_path, "config.json"), os.path.join(save_folder, "config.json"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k_heads', type=int, required=False, default=20,
                        help="Top k heads to use.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Model name or path")
    parser.add_argument('--direction_factor', type=float, nargs='+', required=False, default=[0.2],
                        help="Factors (multipliers) for directional intervention. One per direction type.")
    parser.add_argument('--direction_type', type=str, nargs='+', required=False, default=["direction_relevant"],
                        help="Types of direction for directional intervention. Multiple allowed.")
    parser.add_argument('--save_path', type=str, default="edit_models_w/head_kq_directions")
    parser.add_argument('--edit_path', type=str, default="edit_models/head_kq_directions")
    args = parser.parse_args()
    print(args)
    edit_kq_direction(args.model_name, args.top_k_heads, args.direction_factor, args.direction_type, args.save_path, args.edit_path)

if __name__ == '__main__':
    main()
