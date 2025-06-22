import os
import random

import torch
import pandas as pd
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import gather_object, broadcast_object_list
import logging
from tqdm import tqdm
import argparse
import json
from datetime import timedelta
from xopen import xopen

import decoding_methods
from lost_in_the_middle import best_subspan_em


def initialize_logger():
    """Initialize the logger with a basic configuration."""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def initialize_accelerator():
    """Initialize the Accelerator for distributed training."""
    kwargs = [InitProcessGroupKwargs(timeout=timedelta(days=1))]
    accelerator = Accelerator(kwargs_handlers=kwargs)
    return accelerator


def initialize_models_and_tokenizer(model_name, accelerator):
    """
    Load the tokenizer and models, and prepare them with the accelerator.

    Args:
        model_name (str): The name of the pretrained model.
        accelerator (Accelerator): The accelerator instance.

    Returns:
        tuple: The prepared model, reference model, and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map={"": accelerator.process_index},
        trust_remote_code=True,
    )
    return model, tokenizer


def load_prompt_dataset(dataset_name, split="train"):
    dataset = load_dataset(dataset_name, split=split)
    # dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'prompt'])
    data_list = [sample for sample in dataset]
    return data_list


def process_prompt_format(prompt):
    if not isinstance(prompt, str):
        prompt = prompt[0]['content']
    return prompt


def run_eval(model, tokenizer, data_list, accelerator, output_dir, split):
    """
    Generate responses and filter the dataset based on the presence of "aaa".

    Args:
        model (AutoModelForCausalLM): The model for generating responses.
        tokenizer (AutoTokenizer): The tokenizer for encoding and decoding.
        data_list (list): The list of data samples to filter.
        accelerator (Accelerator): The accelerator instance.
        filter_batch_size (int): The batch size for filtering.

    Returns:
        Dataset: The filtered dataset, or None if no samples remain.
    """
    model.eval()
    accelerator.wait_for_everyone()

    random.seed(42)
    random.shuffle(data_list)

    with accelerator.split_between_processes(data_list) as data_list_per_gpu:

        metrics_list_per_gpu = []

        # have each GPU do inference, prompt by prompt
        progress_bar_desc = f"GPU {accelerator.process_index} - Processing prompts"
        for data in tqdm(data_list_per_gpu, desc=progress_bar_desc, leave=False):
            prompt = process_prompt_format(data['prompt'])
            response_regular = decoding_methods.original_decoding_with_cache(
                prompt,
                model,
                tokenizer,
                max_length=100,
                use_chat_template=True
            )
            regular_eval = int(best_subspan_em(response_regular, data['answers']))
            data['result'] = regular_eval
            data['response'] = response_regular
            metrics_list_per_gpu.append(data)

        metrics_list_per_gpu = [metrics_list_per_gpu]

    metrics_list = gather_object(metrics_list_per_gpu)

    # On main process, proceed to create the filtered dataset
    if accelerator.is_main_process:
        # Flatten the list (since it's a list of lists)
        metrics_list = [item for sublist in metrics_list for item in sublist]
        result = {'em': sum([x['result'] for x in metrics_list]) / len(metrics_list)}

        logging.info(result)

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'result_{split}.json'), 'w') as f:
            json.dump(result, f)

        # output_dir = os.path.join(output_dir, 'data')
        # os.makedirs(output_dir, exist_ok=True)
        # with xopen(os.path.join(output_dir, f'{split}.jsonl'), "w") as f:
        #     for example in metrics_list:
        #         f.write(json.dumps(example) + "\n")
        # with open(os.path.join(output_dir, 'data.json'), 'w') as f:
        #     json.dump(metrics_list, f)


def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description="Dataset Eval Script")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Name of the model to use")
    parser.add_argument("--test_dataset_name", type=str, default="trl-lib/ultrafeedback-prompt",
                        help="Name of the dataset to load")
    parser.add_argument("--split", type=str, default="test",
                        help="train/test of the dataset")
    parser.add_argument("--output_dir", type=str, default="output_data", help="Directory to save the filtered dataset")

    args = parser.parse_args()

    initialize_logger()
    accelerator = initialize_accelerator()

    # Initialize models and tokenizer
    model, tokenizer = initialize_models_and_tokenizer(args.model_name, accelerator)

    data_list_test = load_prompt_dataset(args.test_dataset_name, split=args.split)
    run_eval(model, tokenizer, data_list_test, accelerator, args.output_dir, args.split)


if __name__ == "__main__":
    main()
