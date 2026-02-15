"""
Script to load datasets from HuggingFace parquet links, filter by token length,
and save as a HuggingFace dataset in arrow format.
"""

import argparse
import os
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Dataset links
DATASET_LINKS = [
    "https://huggingface.co/datasets/Xnhyacinth/LongBench/resolve/main/multifieldqa_en/test-00000-of-00001.parquet",
    "https://huggingface.co/datasets/Xnhyacinth/LongBench/resolve/main/2wikimqa/test-00000-of-00001.parquet"
]

def main():
    parser = argparse.ArgumentParser(description="Download and filter LongBench dataset")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to tokenizer")
    parser.add_argument("--output_dir", type=str, default="./data/longbench_filtered",
                        help="Output directory for the dataset")
    parser.add_argument("--max_tokens", type=int, default=32768,
                        help="Maximum token length (default: 32768)")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Lists to store filtered samples
    filtered_inputs = []
    filtered_outputs = []

    # Process each dataset link
    for link in DATASET_LINKS:
        print(f"\nProcessing: {link}")

        # Load parquet file directly from URL
        ds = load_dataset('parquet', data_files=link, split='train', num_proc=32)
        print(f"Loaded {len(ds)} samples")

        # Process each sample
        for row in tqdm(ds, total=len(ds), desc="Filtering samples"):
            context = row['context']
            question = row['question']
            answers = row['answers']

            input_text = context + question
            output_text = answers[0] if isinstance(answers, list) else answers
            tot_text = input_text + output_text

            tokens = tokenizer.encode(tot_text, add_special_tokens=True)
            token_length = len(tokens)

            if token_length < args.max_tokens:
                filtered_inputs.append(input_text)
                filtered_outputs.append(output_text)

    print(f"\nTotal filtered samples: {len(filtered_inputs)}")

    # Create and save HuggingFace dataset
    dataset_dict = {
        "input": filtered_inputs,
        "output": filtered_outputs
    }

    hf_dataset = Dataset.from_dict(dataset_dict)

    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving dataset to: {output_path}")
    hf_dataset.save_to_disk(f"{output_path}/train")

    print(f"Dataset saved successfully!")
    print(f"Dataset info: {hf_dataset}")

    return hf_dataset

if __name__ == "__main__":
    main()
