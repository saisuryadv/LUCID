"""
Script to load datasets from HuggingFace parquet links, filter by token length,
and save as a HuggingFace dataset in arrow format.
"""

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Configuration
tokenizer_path = "/work/01318/nnp528/vista/LUCID/tinyllama_no_qknorm_directory/tinyllama-1.1b-config"  # Update this path as needed

# Dataset links
dataset_links = [
    "https://huggingface.co/datasets/Xnhyacinth/LongBench/resolve/main/multifieldqa_en/test-00000-of-00001.parquet",
    "https://huggingface.co/datasets/Xnhyacinth/LongBench/resolve/main/2wikimqa/test-00000-of-00001.parquet"
]

# Maximum token length (2^15 = 32768)
MAX_TOKEN_LENGTH = 2 ** 15


def main():
    # Load tokenizer
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Lists to store filtered samples
    filtered_inputs = []
    filtered_outputs = []

    # Process each dataset link
    for link in dataset_links:
        print(f"\nProcessing: {link}")

        # Load parquet file directly from URL using load_dataset
        ds = load_dataset('parquet', data_files=link, split='train', num_proc=32)
        print(f"Loaded {len(ds)} samples")

        # Process each sample
        for row in tqdm(ds, total=len(ds), desc="Filtering samples"):
            # Extract fields
            context = row['context']
            question = row['question']
            answers = row['answers']

            # Create input and output
            input_text = context + question
            output_text = answers[0] if isinstance(answers, list) else answers

            # Create total text for length calculation
            tot_text = input_text + output_text

            # Tokenize and check length
            tokens = tokenizer.encode(tot_text, add_special_tokens=True)
            token_length = len(tokens)

            # Filter by token length
            if token_length < MAX_TOKEN_LENGTH:
                filtered_inputs.append(input_text)
                filtered_outputs.append(output_text)

    print(f"\nTotal filtered samples: {len(filtered_inputs)}")

    # Create HuggingFace dataset
    dataset_dict = {
        "input": filtered_inputs,
        "output": filtered_outputs
    }

    hf_dataset = Dataset.from_dict(dataset_dict)

    # Save dataset in arrow format with train subdirectory only
    # (run_clm_chunk.py only loads validation if it exists, so we skip it)
    output_path = "/work/01318/nnp528/vista/LUCID/FT_dataset_split"
    print(f"Saving dataset to: {output_path}")
    hf_dataset.save_to_disk(f"{output_path}/train")

    print(f"Dataset saved successfully!")
    print(f"Dataset info: {hf_dataset}")

    return hf_dataset


if __name__ == "__main__":
    main()
