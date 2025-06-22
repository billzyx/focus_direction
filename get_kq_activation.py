import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import shutil

from decoding_methods import get_text_idx_range
import decoding_methods

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hidden_state_outputs = []


def hidden_state_hook(module, inputs, output):
    hidden_state_outputs.append(inputs[0].detach().cpu())


def register_hooks(model):
    """
    Loops over each LLaMA layer and registers forward hooks on q_proj and k_proj.
    Returns a list of hook handle objects, so you can remove them later if needed.
    """
    handles = []
    # For LLaMA-based models in transformers, the layers are typically in:
    # model.model.layers[i].self_attn.(q_proj / k_proj)
    for layer in model.model.layers:
        # Register q_proj hook
        # handles.append(layer.self_attn.q_proj.register_forward_hook(q_proj_hook))
        # Register k_proj hook
        # handles.append(layer.self_attn.k_proj.register_forward_hook(k_proj_hook))
        handles.append(layer.self_attn.k_proj.register_forward_hook(hidden_state_hook))

    return handles


def check_len_pair(gold_data, long_data, key):
    if gold_data[key][1] - gold_data[key][0] != long_data[key][1] - long_data[key][0]:
        return False
    return True


def check_length(data_idx, save_dir):
    data_save_dir = os.path.join(save_dir, "{:05}".format(data_idx))
    with open(os.path.join(data_save_dir, 'long.json'), 'r') as f:
        long_data = json.load(f)
    with open(os.path.join(data_save_dir, 'gold.json'), 'r') as f:
        gold_data = json.load(f)
    if not check_len_pair(gold_data, long_data, 'needle_idx_range'):
        return False
    if not check_len_pair(gold_data, gold_data, 'response_idx_range'):
        return False
    if not check_len_pair(gold_data, gold_data, 'question_idx_range'):
        return False
    return True

def del_length_error_dir(data_idx, save_dir):
    data_save_dir = os.path.join(save_dir, "{:05}".format(data_idx))
    # Remove the directory and its contents
    try:
        shutil.rmtree(data_save_dir)
        print(f"Directory '{data_save_dir}' and its contents deleted successfully.")
    except OSError as e:
        print(f"Error: {e}")


def process_dataset_and_save(model, tokenizer, dataset, layers_to_save, save_dir: str):
    """
    1) Goes through each example in the dataset.
    2) Captures the outputs of q_proj and k_proj (via hooks).
    3) Saves them (and needle_idx_range) in separate files at the end.
    """

    # Lists to hold final data (we'll have one entry per prompt)
    all_needle_ranges = []

    for data_idx, data in enumerate(tqdm(dataset, desc="Processing dataset")):
        response = decoding_methods.original_decoding_with_cache(data['gold_prompt'], model, tokenizer)

        prompt = data['prompt']

        process_prompt(
            prompt, response, data, data_idx, model, tokenizer, layers_to_save, save_dir, save_file_postfix='long'
        )
        # process_prompt(
        #     data['gold_prompt'], response, data, data_idx, model, tokenizer, layers_to_save, save_dir, save_file_postfix='gold'
        # )

        # if not check_length(data_idx, save_dir):
        #     print(f"Error: {data_idx}")
        #     print(data)
        #     del_length_error_dir(data_idx, save_dir)


def process_prompt(prompt, response, data, data_idx, model, tokenizer, layers_to_save, save_dir, save_file_postfix='data'):
    # 1) Build the prompt
    prompt = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
    # 2) Tokenize
    prompt = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    needle_idx_range = get_text_idx_range(data['needle'], prompt, tokenizer)
    question_idx_range = get_text_idx_range(data['question'], prompt, tokenizer)
    response_idx_range = get_text_idx_range(response, prompt, tokenizer)
    irrelevant_idx_range_list = [get_text_idx_range(x, prompt, tokenizer) for x in data['irrelevant_docs']]

    # Clear out old hook captures to ensure we only store the output for this sample
    hidden_state_outputs.clear()
    # 4) Forward pass
    with torch.no_grad():
        output = model(
            input_ids=prompt,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
    del output
    data["needle_idx_range"] = needle_idx_range
    data["response_idx_range"] = response_idx_range
    data["question_idx_range"] = question_idx_range
    data["irrelevant_idx_range_list"] = irrelevant_idx_range_list

    hidden_states_np = torch.concatenate(hidden_state_outputs, dim=0).detach().cpu().float().numpy()
    # print(np.shape(hidden_states_np))
    data_save_dir = os.path.join(save_dir, "{:05}".format(data_idx))
    os.makedirs(data_save_dir, exist_ok=True)
    for layer_to_save in layers_to_save:
        hidden_states_np_layer = hidden_states_np[layer_to_save]
        np.save(os.path.join(data_save_dir, "hidden_state_layer_{}_{}.npy".format(layer_to_save, save_file_postfix)), hidden_states_np_layer)
    with open(os.path.join(data_save_dir, "{}.json".format(save_file_postfix)), "w") as f:
        json.dump(data, f)
    hidden_state_outputs.clear()


##############################################################################
# 5. MAIN FUNCTION / ENTRY POINT
##############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                        help='Name or path of the LLaMA-based model.')
    parser.add_argument('--layers_to_save', type=int, nargs='+', default=[12,],
                        help='Which layers to save, e.g. "--layers_to_save 0 1 2 3 4".')
    parser.add_argument('--dataset_name', type=str, default='focus_data/qa_prompt_needle/original',
                        help='Path or identifier for the dataset.')
    parser.add_argument('--split', type=str, default='train',
                        help='Split of the dataset.')
    parser.add_argument('--save_dir', type=str, default='head_features',
                        help='Directory where the output files (q_proj, k_proj, needle indices) will be saved.')
    args = parser.parse_args()

    # 1. Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    config._attn_implementation = 'eager'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print(model.config._attn_implementation)

    # 2. Register hooks
    handles = register_hooks(model)

    # 3. Load dataset
    dataset = load_dataset(args.dataset_name, "default", split=args.split)

    # 4. Process dataset & save
    save_dir = os.path.join(args.save_dir, args.model_name.replace("/", "_"), args.split)
    process_dataset_and_save(model, tokenizer, dataset, args.layers_to_save, save_dir)

    # 5. (Optional) Remove hooks to avoid memory usage in subsequent code
    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
