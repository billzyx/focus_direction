import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
from huggingface_hub import snapshot_download
import sys
import re
import importlib.util


def create_symlinks(source_dir, target_dir):
    # Ensure source and target directories exist
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)  # Create target directory if it doesn't exist

    # Iterate through files in the source directory
    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        target_path = os.path.join(target_dir, item)

        # Only create symlinks for files
        if os.path.isfile(source_path):
            try:
                os.symlink(source_path, target_path)
                print(f"Created symlink: {target_path} -> {source_path}")
            except FileExistsError:
                print(f"Symlink already exists for: {target_path}")
                os.unlink(target_path)
                os.symlink(source_path, target_path)
                print(f"Recreated symlink: {target_path} -> {source_path}")
            except Exception as e:
                print(f"Error creating symlink for {source_path}: {e}")


def remove_symlink(file_path):
    """
    Remove a symbolic link if it exists.

    :param file_path: Path to the file
    """
    if os.path.islink(file_path):
        os.unlink(file_path)
        print(f"Removed symbolic link: {file_path}")
    else:
        print(f"{file_path} is not a symbolic link or does not exist.")


def find_model_classes(directory):
    # Find the file that starts with "modeling"
    modeling_file = next((f for f in os.listdir(directory) if f.startswith("modeling") and f.endswith(".py")), None)
    assert modeling_file, "No file starting with 'modeling' found in the directory."

    # Construct the file path
    file_path = os.path.join(directory, modeling_file)

    # Load the file as a module
    spec = importlib.util.spec_from_file_location("modeling_module", file_path)
    modeling_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modeling_module)

    # Find all class names in the module
    classes = [name for name in dir(modeling_module) if
               re.match(r".*Model$", name) or re.match(r".*ForCausalLM$", name)]

    # Extract the desired classes
    model_class = next((cls for cls in classes if cls.endswith("Model")), None)
    for_causal_lm_class = next((cls for cls in classes if cls.endswith("ForCausalLM")), None)

    modeling_name = modeling_file.split(".")[0]
    model_class = "{}.{}".format(modeling_name, model_class)
    for_causal_lm_class = "{}.{}".format(modeling_name, for_causal_lm_class)

    return model_class, for_causal_lm_class


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name or path")
    parser.add_argument("--target_dir", type=str, default="edit_models/head_kq_directions", help="Path to the editing model")
    parser.add_argument("--eager_attention", action="store_true", help="Use Eager Attention")
    args = parser.parse_args()

    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name,
    #     torch_dtype=torch.bfloat16,
    # )
    model_cache_dir = snapshot_download(repo_id=args.model_name)
    print("Model absolute path:", model_cache_dir)

    output_model_path = os.path.join(args.target_dir, args.model_name)
    # model.save_pretrained(output_model_path)
    # tokenizer.save_pretrained(output_model_path)
    create_symlinks(model_cache_dir, output_model_path)
    print("Output path:", output_model_path)

    # edit config
    with open(os.path.join(model_cache_dir, 'config.json'), "r") as f:
        config = json.load(f)

    if 'llama' in args.model_name.lower():
        modeling_file = "modeling_llama.py"
    elif 'qwen' in args.model_name.lower():
        modeling_file = "modeling_qwen2.py"
    elif 'mistral' in args.model_name.lower():
        modeling_file = "modeling_mistral.py"
    else:
        raise ValueError("Invalid model name.")

    shutil.copyfile(
        os.path.join(args.target_dir, modeling_file), os.path.join(output_model_path, modeling_file)
    )

    model_class, for_causal_lm_class = find_model_classes(output_model_path)
    print(f"Model class name: {model_class}")
    print(f"ForCausalLM class name: {for_causal_lm_class}")

    config["auto_map"] = {
        "AutoModel": model_class,
        "AutoModelForCausalLM": for_causal_lm_class,
    }
    if args.eager_attention:
        config["_attn_implementation"] = "eager"

    config_path = os.path.join(output_model_path, "config.json")
    remove_symlink(config_path)
    with open(config_path, "w") as f:
        json.dump(config, f)

    # tokenizer = AutoTokenizer.from_pretrained(output_model_path)
    # model = AutoModelForCausalLM.from_pretrained(
    #     output_model_path,
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    # )


if __name__ == '__main__':
    main()
