import json
import os
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Download BABILong dataset")
    parser.add_argument("--output_dir", type=str, default="./data/babilong_dataset_32k",
                        help="Output directory for the dataset")
    parser.add_argument("--context_length", type=str, default="32k",
                        help="Context length variant (e.g., 32k, 64k, 128k)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    splits = [f"qa{i}" for i in range(1, 11)]
    all_data = []

    for split in splits:
        print(f"Downloading split: {split}")
        dataset = load_dataset("RMT-team/babilong", name=args.context_length, split=split)

        for example in dataset:
            combined_input = example["input"] + " " + example["question"]
            all_data.append({
                "input": combined_input,
                "output": example["target"]
            })

        print(f"  Added {len(dataset)} examples from {split}")

    output_file = os.path.join(output_dir, f"babilong_{args.context_length}_combined.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\nTotal examples: {len(all_data)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
