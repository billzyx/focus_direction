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


def span_attention_decoding_with_cache(input_text, needle, model, tokenizer, max_length=100,
                                       use_chat_template=True, split_softmax_exponent=1.0, attention_heads=[]):
    input_dict = to_input_dict(input_text, tokenizer, use_chat_template).to(model.device)

    batch_size, length = input_dict.input_ids.shape

    attention_spans = get_text_idx_range(needle, input_dict.input_ids, tokenizer)
    attention_spans = [attention_spans]

    if not model.generation_config.pad_token_id:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    input_ids = input_dict['input_ids']
    attention_mask = input_dict['attention_mask']

    batch_size, length = input_ids.shape

    # Initialize the tensor for generated ids, attention mask, and past key-value states
    generated_ids = input_ids
    past_key_values = transformers.DynamicCache()

    # We will generate one token at a time until we reach max_length
    for _ in range(max_length):

        if len(past_key_values) == 0:
            # For the first step, input the entire sequence
            inputs = {
                'input_ids': generated_ids,
                'attention_mask': attention_mask
            }
        else:
            # For subsequent steps, input only the last generated token
            inputs = {
                'input_ids': next_token_id,
                'attention_mask': attention_mask,
                'past_key_values': past_key_values
            }

        with torch.no_grad():
            outputs = model(
                **inputs,
                attention_heads=attention_heads,
                attention_spans=attention_spans,
                split_softmax_exponent=split_softmax_exponent,
                # output_attentions=True,
            )

        logits = outputs.logits
        past_key_values = outputs.past_key_values

        # Select the last token from the sequence
        next_token_logits = logits[:, -1, :].contiguous()

        # Greedily select the next token (highest logit/probability)
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Append the predicted token id to the generated sequence
        generated_ids = torch.cat((generated_ids, next_token_id), dim=-1)

        # Update the attention mask
        attention_mask = torch.cat((attention_mask, torch.ones_like(next_token_id)), dim=-1)

        # Check if the last token is the end of sequence token
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # Decode the generated ids to a sequence
    generated_sequence = tokenizer.decode(generated_ids[0, length:], skip_special_tokens=True)
    return generated_sequence


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
