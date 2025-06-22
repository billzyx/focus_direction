import os
import json
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, LlamaConfig
import re
from tqdm import tqdm
import argparse
import torch.nn as nn
import torch.nn.functional as F

from hidden_state_loader import MyHiddenStatesDataset
from edit_models.head_kq_directions.modeling_llama import LlamaAttention
from edit_models.head_kq_directions.modeling_qwen2 import Qwen2Attention
from edit_models.head_kq_directions.modeling_mistral import MistralAttention


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_attention_directions(root_dir,
                               model_name,
                               layer_to_train=12,
                               num_epochs=3,
                               lr=1e-3,
                               target_type='relevant',
                               save_dir=None,
                               ):
    """
    Example training function to maximize the attention on
    the needle_idx_range for the last token in each sample.
    """
    # 1) Build dataset & data loader
    dataset = MyHiddenStatesDataset(
        root_dir=root_dir,
        hidden_state_filename_long='hidden_state_layer_{}_long.npy'.format(layer_to_train),
        # hidden_state_filename_gold='hidden_state_layer_{}_gold.npy'.format(layer_to_train)
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 2) Build the attention layer (or load from a checkpoint, etc.)
    if 'llama' in model_name.lower():
        attention_layer_class = LlamaAttention
    elif 'qwen' in model_name.lower():
        attention_layer_class = Qwen2Attention
    elif 'mistral' in model_name.lower():
        attention_layer_class = MistralAttention
    else:
        raise ValueError("Attention for model name {} is not supported.".format(model_name))
    attention_layer = load_attention_layer(
        model_name=model_name, desired_layer_idx=layer_to_train, attention_layer_class=attention_layer_class
    )
    attention_layer.to(device)

    # attention_layer_original = load_attention_layer(
    #     model_name=model_name, desired_layer_idx=layer_to_train, attention_layer_class=LlamaAttention
    # )
    # attention_layer_original.to(device)
    # attention_layer_original.eval()

    # 3) Freeze all params except head_k_directions & head_q_directions
    for name, param in attention_layer.named_parameters():
        if "head_k_directions" in name or "head_q_directions" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 4) Create optimizer
    params_to_optimize = [
        attention_layer.head_k_directions,
        attention_layer.head_q_directions,
    ]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=lr)

    mse_loss = nn.MSELoss()

    # 5) Train loop
    attention_layer.train()
    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch_idx, (hidden_states_long, meta_long)\
                in enumerate(pbar := tqdm(dataloader, desc="Training")):

            attn_weights_long = get_attention_weights(hidden_states_long, attention_layer)

            # with torch.no_grad():
            #     attn_weights_gold = get_attention_weights(hidden_states_gold, attention_layer_original)

            needle_start, needle_end = meta_long["needle_idx_range"]  # e.g. [25, 174]
            response_start, response_end = meta_long["response_idx_range"]
            attn_slice_score_relevant = attn_weights_long[:, :, response_start:response_end, needle_start:needle_end].sum(dim=-1)
            attn_slice_score_sink = attn_weights_long[:, :, response_start:response_end, 0:2].sum(dim=-1)

            # irrelevant_attn_slice_score_list = []
            # for irrelevant_start, irrelevant_end in meta_long["irrelevant_idx_range_list"]:
            #     irrelevant_attn_slice_score = attn_weights_long[:, :, response_start:response_end, irrelevant_start:irrelevant_end].sum(dim=-1)
            #     irrelevant_attn_slice_score_list.append(irrelevant_attn_slice_score)

            # irrelevant_attn_slice_score = torch.cat(irrelevant_attn_slice_score_list, dim=0)
            # irrelevant_attn_slice_score = torch.max(irrelevant_attn_slice_score, dim=0, keepdim=True)[0]

            if target_type == 'relevant':
                loss = -attn_slice_score_relevant.mean()
            else:
                raise ValueError("Unknown target_type")

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_description(f"Training (batch={batch_idx}, loss={loss.item():.4f})")

        print(f"Epoch {epoch + 1} / {num_epochs}, Avg Loss: {total_loss / len(dataloader):.4f}")

    # 6) (Optional) Save or return the layer
    print("Training done.")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(attention_layer.head_k_directions, os.path.join(save_dir, f"head_k_directions_{layer_to_train}.pth"))
        torch.save(attention_layer.head_q_directions, os.path.join(save_dir, f"head_q_directions_{layer_to_train}.pth"))

    return attention_layer


def get_attention_weights(hidden_states, attention_layer):
    # hidden_states => shape e.g. [bsz, seq_len, hidden_size], but bsz=1 here
    hidden_states = hidden_states.to(device)
    bsz, seq_len, hidden_size = hidden_states.shape
    # 1) Create position_ids: [bsz, seq_len].
    #    Typically just a range from 0..seq_len-1 for each batch element.
    position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
    position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
    # shape => [bsz, seq_len]
    # 2) Create a causal attention mask of shape [bsz, 1, seq_len, seq_len].
    #    We'll fill it with 0.0 where we can attend, and -inf where we cannot.
    mask = torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device)
    mask = torch.triu(mask, diagonal=1)  # mask out future tokens
    # shape => [seq_len, seq_len], with 0 on diagonal and lower tri, -inf in upper tri
    # Broadcast to [bsz, 1, seq_len, seq_len]
    attention_mask = mask.unsqueeze(0).unsqueeze(1).expand(bsz, 1, seq_len, seq_len).clone()
    # Forward pass
    attn_output, attn_weights, _ = attention_layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_attentions=True,
    )
    if attn_weights is None:
        raise RuntimeError("You must return attn_weights to compute the loss.")
    # attn_weights => [bsz, num_heads, seq_len, seq_len]
    bsz, num_heads, q_len, kv_len = attn_weights.shape
    # We want the last query token => q_len - 1
    last_q_idx = q_len - 1
    # Now slice the key dimension: [needle_start:needle_end]
    # shape of the slice => [bsz, num_heads, (needle_end - needle_start)]
    # heads_slice = heads_to_train if len(heads_to_train) > 0 else slice(None)
    # attn_slice = attn_weights[:, heads_slice, last_q_idx, needle_start:needle_end]
    return attn_weights


def load_attention_layer(model_name, desired_layer_idx=12, attention_layer_class=LlamaAttention):
    # 1. Load the full pre-trained model (example: Llama-2-7B)
    full_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    # 2. Extract the attention parameters from layer 12 (0-based: layers[11])
    #    The official huggingface LlamaDecoderLayer has "self_attn" submodule
    prefix = f"model.layers.{desired_layer_idx}.self_attn."

    # Filter out only the keys for that layer’s self-attention
    attention_state_dict = {}
    for k, v in full_model.state_dict().items():
        if k.startswith(prefix):
            # Remove the "model.layers.XX.self_attn." prefix so it matches your custom LlamaAttention param names
            new_k = k[len(prefix):]
            attention_state_dict[new_k] = v

    # 3. Instantiate your custom LlamaAttention. Make sure the config matches the original model config.
    #    You can pass `layer_idx=desired_layer_idx` if needed for caching or other features.
    config: LlamaConfig = full_model.config
    custom_attention = attention_layer_class(config, layer_idx=desired_layer_idx)

    # 4. Load the extracted parameters
    custom_attention.load_state_dict(attention_state_dict, strict=False)

    # (Optional) Move to GPU / half-precision if you want:
    custom_attention.to("cuda")

    # Now custom_attention is layer 12’s attention module from the pretrained LLaMA weights!
    del full_model
    return custom_attention


def main():
    parser = argparse.ArgumentParser(description="Train attention directions for specified model layer and heads.")
    parser.add_argument('--layer_to_train', type=int, required=False, default=12,
                        help="Layer to train.")
    parser.add_argument('--target_type', type=str, required=False, default="relevant",
                        help="Training targets: relevant.")
    parser.add_argument('--epoch', type=int, required=False, default=10,
                        help="Training epochs.")
    parser.add_argument('--dataset_root_dir', type=str, required=False,
                        default="head_features/meta-llama_Llama-3.2-3B-Instruct/train",
                        help="Root directory of the training dataset.")
    parser.add_argument('--save_dir', type=str, required=False, default="head_features",
                        help="Directory to save the trained model.")
    parser.add_argument('--model_name', type=str, required=False,
                        default="meta-llama/Llama-3.2-3B-Instruct", help="Name of the model to train.")

    # Parse the arguments
    args = parser.parse_args()
    # Define the input variables

    layer_to_train = args.layer_to_train

    save_dir = os.path.join(args.save_dir, args.model_name.replace("/", "_"), f"direction_{args.target_type}")

    trained_attention_layer = train_attention_directions(
        root_dir=args.dataset_root_dir,
        model_name=args.model_name,
        num_epochs=args.epoch,
        lr=1e-3,
        target_type=args.target_type,
        layer_to_train=layer_to_train,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
