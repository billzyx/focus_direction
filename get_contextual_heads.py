import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from decoding_methods import get_text_idx_range
import decoding_methods

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_head_score(args):
    # model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]
    model_name_or_path = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True)
    model.eval()
    dataset = load_dataset(args.dataset_name, "default", split='train')

    process_long_context_prompt(model, tokenizer, dataset, args)
    process_gold_prompt(model, tokenizer, dataset, args)


def process_long_context_prompt(model, tokenizer, dataset, args):
    head_relevant_scores = [[[] for _ in range(model.config.num_attention_heads)] for _ in
                            range(model.config.num_hidden_layers)]
    head_irrelevant_scores = [[[] for _ in range(model.config.num_attention_heads)] for _ in
                              range(model.config.num_hidden_layers)]
    head_irrelevant_scores_max = [[[] for _ in range(model.config.num_attention_heads)] for _ in
                                  range(model.config.num_hidden_layers)]
    head_sink_scores = [[[] for _ in range(model.config.num_attention_heads)] for _ in
                        range(model.config.num_hidden_layers)]
    for data in tqdm(dataset):
        prompt = data['prompt']

        response = decoding_methods.original_decoding_with_cache(prompt, model, tokenizer)
        prompt = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False)

        # prompt = [{"role": "user", "content": prompt}]
        # prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt = tokenizer(prompt, return_tensors='pt').input_ids

        needle_idx_range = get_text_idx_range(data['needle'], prompt, tokenizer)
        question_idx_range = get_text_idx_range(data['question'], prompt, tokenizer)
        response_idx_range = get_text_idx_range(response, prompt, tokenizer)

        irrelevant_idx_range_list = [get_text_idx_range(x, prompt, tokenizer) for x in data['irrelevant_docs']]

        prompt_prefix = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant)."
        prefix_idx_range = get_text_idx_range(prompt_prefix, prompt, tokenizer)

        with torch.no_grad():
            prompt = prompt.to(device)
            output = model(input_ids=prompt, return_dict=True, output_attentions=True, output_hidden_states=False)

        attention_weights = output.attentions

        if -1 in prefix_idx_range + needle_idx_range + question_idx_range:
            print("Error in: ", data['question'])
            print(prefix_idx_range, needle_idx_range, question_idx_range)
            del output
            del attention_weights
            continue

        for layer in range(model.config.num_hidden_layers):
            for head in range(model.config.num_attention_heads):
                relevant_score = extract_attention_with_range(
                    attention_weights, layer, head, needle_idx_range[0], needle_idx_range[1], response_idx_range
                )
                head_relevant_scores[layer][head].append(relevant_score)

                sink_score = extract_attention_with_range(
                    attention_weights, layer, head, 0, 2, response_idx_range
                )
                head_sink_scores[layer][head].append(sink_score)

                irrelevant_score_list = []
                for irrelevant_idx_range in irrelevant_idx_range_list:
                    irrelevant_score = extract_attention_with_range(
                        attention_weights, layer, head, irrelevant_idx_range[0], irrelevant_idx_range[1],
                        response_idx_range
                    )
                    irrelevant_score_list.append(irrelevant_score)
                head_irrelevant_scores[layer][head].append(float(np.sum(irrelevant_score_list)))
                head_irrelevant_scores_max[layer][head].append(float(np.max(irrelevant_score_list)))

        del output
        del attention_weights
    show_and_save_scores(args, head_relevant_scores, "head_relevant_scores")
    show_and_save_scores(args, head_irrelevant_scores, "head_irrelevant_scores")
    show_and_save_scores(args, head_irrelevant_scores_max, "head_irrelevant_scores_max")
    show_and_save_scores(args, head_sink_scores, "head_sink_scores")


def process_gold_prompt(model, tokenizer, dataset, args):
    head_relevant_scores = [[[] for _ in range(model.config.num_attention_heads)] for _ in
                            range(model.config.num_hidden_layers)]
    head_sink_scores = [[[] for _ in range(model.config.num_attention_heads)] for _ in
                        range(model.config.num_hidden_layers)]

    for data in tqdm(dataset):
        prompt = data['gold_prompt']
        response = decoding_methods.original_decoding_with_cache(prompt, model, tokenizer)
        prompt = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
        # prompt = [{"role": "user", "content": prompt}]
        # prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt = tokenizer(prompt, return_tensors='pt').input_ids

        needle_idx_range = get_text_idx_range(data['needle'], prompt, tokenizer)
        response_idx_range = get_text_idx_range(response, prompt, tokenizer)

        with torch.no_grad():
            prompt = prompt.to(device)
            output = model(input_ids=prompt, return_dict=True, output_attentions=True, output_hidden_states=False)

        attention_weights = output.attentions

        if -1 in needle_idx_range:
            print("Error in: ", data['question'])
            print(needle_idx_range)
            del output
            del attention_weights
            continue

        for layer in range(model.config.num_hidden_layers):
            for head in range(model.config.num_attention_heads):
                relevant_score = extract_attention_with_range(
                    attention_weights, layer, head, needle_idx_range[0], needle_idx_range[1], response_idx_range
                )
                head_relevant_scores[layer][head].append(relevant_score)

                sink_score = extract_attention_with_range(
                    attention_weights, layer, head, 0, 2, response_idx_range
                )
                head_sink_scores[layer][head].append(sink_score)

        del output
        del attention_weights

    show_and_save_scores(args, head_relevant_scores, "head_relevant_scores_gold")
    show_and_save_scores(args, head_sink_scores, "head_sink_scores_gold")


def show_and_save_scores(args, head_contextual_scores, save_name):
    print(save_name)
    head_contextual_scores = np.array(head_contextual_scores)
    head_contextual_scores = np.mean(head_contextual_scores, axis=-1)
    top_heads = np.argsort(head_contextual_scores.flatten())[::-1][:30]
    top_head_list = []
    for head in top_heads:
        layer = head // head_contextual_scores.shape[1]
        head_idx = head % head_contextual_scores.shape[1]
        print(f"Layer {layer}, Head {head_idx}, Score {head_contextual_scores[layer, head_idx]}")
        top_head_list.append([int(layer), int(head_idx)])
    print(top_head_list)
    save_dir = os.path.join(args.save_dir, args.model_name.replace("/", "_"))
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'{save_name}.npy'), head_contextual_scores)


def extract_attention_with_range(attention_weights, layer, head, start, end, response_range):
    relevant_score = attention_weights[layer][0, head, response_range[0]:response_range[1], start:end]
    relevant_score = torch.sum(relevant_score, dim=-1).mean().detach().float().cpu().numpy()
    return relevant_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--dataset_name', type=str, default='focus_data/qa_prompt_needle/original')
    parser.add_argument('--save_dir', type=str, default='contextual_heads_scores_full_seq')
    args = parser.parse_args()

    get_head_score(args)


if __name__ == '__main__':
    main()
