import os
import torch
import transformers
import re


def original_decoding_with_cache(input_text, model, tokenizer, max_length=100, use_chat_template=True):
    input_dict = to_input_dict(input_text, tokenizer, use_chat_template).to(model.device)

    batch_size, length = input_dict.input_ids.shape

    if not model.generation_config.pad_token_id:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        gen_tokens = model.generate(
            **input_dict,
            max_new_tokens=max_length,
            do_sample=False,
            # output_attentions=True,
        )

    gen_text = tokenizer.decode(gen_tokens[0, length:], skip_special_tokens=True)
    return gen_text


def get_text_idx_range(needle, prompt, tokenizer):
    needle = tokenizer(needle, return_tensors='pt', add_special_tokens=False).input_ids
    needle_idx_range = find_needle_idx(prompt[0], needle[0])
    return needle_idx_range


def find_needle_idx(prompt_ids, needle_ids):
    needle_ids = needle_ids.tolist()
    span_len = len(needle_ids)
    max_overlap = 0
    best_start_idx, best_end_idx = -1, -1

    for i in range(len(prompt_ids)):
        token_span = prompt_ids[i: i + span_len]
        span_ids = set(token_span.tolist())
        overlap = float(len(span_ids.intersection(set(needle_ids)))) / len(set(needle_ids))

        if overlap > max_overlap:
            max_overlap = overlap
            best_start_idx, best_end_idx = i, i + span_len

    if max_overlap > 0.7:
        return best_start_idx, best_end_idx
    return -1, -1


def to_input_dict(input_text, tokenizer, use_chat_template):

    # Convert input_text to a list if it's a string
    if isinstance(input_text, str):
        input_texts = [input_text]
    elif isinstance(input_text, list):
        input_texts = input_text
    else:
        raise ValueError("input_text must be a string or a list of strings")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if use_chat_template:
        messages_list = []
        for text in input_texts:
            messages_list.append([{"role": "user", "content": text}])
        input_texts = tokenizer.apply_chat_template(
            messages_list, tokenize=False, add_generation_prompt=True,
        )

    input_dict = tokenizer(
        input_texts, return_tensors='pt', padding=True, truncation=True
    )
    return input_dict
