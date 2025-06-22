import os
from xopen import xopen
from tqdm import tqdm
import json
from copy import deepcopy
import random

from lost_in_the_middle.prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
    get_qa_prompt_and_gold_prompt,
)


def preprocess_and_generate_prompt_and_needle(input_path, output_path, num_total_documents=20, ):
    with xopen(input_path) as fin:
        example_list = []
        for line in tqdm(fin):
            qa_retrieval_result = json.loads(line)

            # for gold_idx in range(num_total_documents):
            for gold_idx in [0, 4, 9, 14, 19]:
                content_selection_example_side = process_single_sample(
                    qa_retrieval_result, num_total_documents, gold_index=gold_idx, use_random_ordering=True
                )

                example = generate_example(content_selection_example_side, gold_idx)
                example_list.append(example)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with xopen(output_path, "w") as f:
            for example in example_list:
                f.write(json.dumps(example) + "\n")


def generate_example(content_selection_example_side, gold_idx):
    question = content_selection_example_side["question"]
    documents = []
    for ctx in deepcopy(content_selection_example_side['ctxs']):
        documents.append(Document.from_dict(ctx))
    prompt, negative_prompt = get_qa_prompt(
        question,
        documents,
        mention_random_ordering=False,
        query_aware_contextualization=False,
    )
    prompt, gold_prompt = get_qa_prompt_and_gold_prompt(
        question,
        documents,
        mention_random_ordering=False,
        query_aware_contextualization=False,
    )
    needle = content_selection_example_side['ctxs'][gold_idx]
    needle = f"(Title: {needle['title']}) {needle['text']}"

    irrelevant_list = []
    for idx in range(len(content_selection_example_side['ctxs'])):
        if idx != gold_idx:
            irrelevant_doc = content_selection_example_side['ctxs'][idx]
            irrelevant_list.append(f"(Title: {irrelevant_doc['title']}) {irrelevant_doc['text']}")
    example = {
        "prompt": prompt,
        "answers": content_selection_example_side["answers"],
        "question": content_selection_example_side["question"],
        "needle": needle,
        "gold_prompt": gold_prompt,
        "negative_prompt": negative_prompt,
        "irrelevant_docs": irrelevant_list,
    }
    return example


def preprocess_and_generate_test_set(input_path, output_path):
    with xopen(input_path) as fin:
        example_list = []
        for line in tqdm(fin):
            qa_retrieval_result = json.loads(line)

            num_total_documents = 20

            for gold_idx in [0, 4, 9, 14, 19]:
                content_selection_example_side = process_single_sample(
                    qa_retrieval_result, num_total_documents, gold_index=gold_idx, use_random_ordering=True
                )

                example = generate_example(content_selection_example_side, gold_idx)
                example_list.append(example)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with xopen(output_path, "w") as f:
            for example in example_list:
                f.write(json.dumps(example) + "\n")


def get_prompt(question, ctxs):
    documents = []
    for ctx in deepcopy(ctxs):
        documents.append(Document.from_dict(ctx))
    prompt, negative_prompt = get_qa_prompt(
        question,
        documents,
        mention_random_ordering=False,
        query_aware_contextualization=False,
    )
    return prompt


def train_test_split(input_path, train_output_path, test_output_path):
    random.seed(42)

    with xopen(input_path) as fin:
        # Read all lines into a list
        lines = list(fin)

    # Shuffle the lines to randomize the selection
    random.shuffle(lines)

    # Calculate the split index
    split_index = len(lines) // 2

    # Write to train and test files
    with xopen(train_output_path, "w") as train_fout, xopen(test_output_path, "w") as test_fout:
        # Write the first half to train file
        for line in tqdm(lines[:split_index], desc="Writing train data"):
            content_selection_example = json.loads(line)
            train_fout.write(json.dumps(content_selection_example) + "\n")

        # Write the second half to test file
        for line in tqdm(lines[split_index:], desc="Writing test data"):
            content_selection_example = json.loads(line)
            test_fout.write(json.dumps(content_selection_example) + "\n")


def process_single_sample(qa_retrieval_result, num_total_documents, gold_index, use_random_ordering=True):
    # Get documents that don't contain the answer
    valid_distractors_with_retrieval_indices = [
        (idx, doc) for idx, doc in enumerate(qa_retrieval_result["ctxs"]) if doc["hasanswer"] is False
    ]
    # Take the top `num_total_documents - 1` distractors
    distractor_docs_with_retrieval_indices = deepcopy(
        valid_distractors_with_retrieval_indices[: num_total_documents - 1]
    )
    for original_retrieval_index, distractor_doc in distractor_docs_with_retrieval_indices:
        distractor_doc["original_retrieval_index"] = original_retrieval_index
        distractor_doc["isgold"] = False
    distractor_docs = [x[1] for x in distractor_docs_with_retrieval_indices]
    content_selection_example = deepcopy(qa_retrieval_result)
    gold_chunk = {
        "title": qa_retrieval_result["nq_annotated_gold"]["title"],
        "text": qa_retrieval_result["nq_annotated_gold"]["chunked_long_answer"],
        "hasanswer": True,
        "isgold": True,
    }
    ctxs = distractor_docs
    if use_random_ordering:
        random.shuffle(ctxs)
    # Insert the gold chunk at thet specific index
    ctxs.insert(gold_index, gold_chunk)
    content_selection_example["ctxs"] = ctxs
    return content_selection_example


def generate_train_test_split():
    qa_data_path = 'qa_data/nq-open-contriever-msmarco-retrieved-documents.jsonl.gz'
    train_split_path = 'qa_data/nq-open-contriever-msmarco-retrieved-documents_train.jsonl.gz'
    test_split_path = 'qa_data/nq-open-contriever-msmarco-retrieved-documents_test.jsonl.gz'
    train_test_split(qa_data_path, train_split_path, test_split_path)
    return train_split_path, test_split_path


def generate_prompt_and_needle():
    train_split_path, test_split_path = generate_train_test_split()
    train_dataset_path = 'focus_data/qa_prompt_needle/original/data/train.jsonl'
    preprocess_and_generate_prompt_and_needle(train_split_path, train_dataset_path, num_total_documents=20, )

    test_dataset_path = 'focus_data/qa_prompt_needle/original/data/test.jsonl'
    preprocess_and_generate_test_set(test_split_path, test_dataset_path)


def main():
    generate_prompt_and_needle()


if __name__ == '__main__':
    main()
