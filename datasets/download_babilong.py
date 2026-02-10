import json
import os
from datasets import load_dataset

def main():
    output_dir = "/work/01318/nnp528/vista/LUCID/babilong_dataset_32k"
    os.makedirs(output_dir, exist_ok=True)

    splits = [f"qa{i}" for i in range(1, 11)]
    all_data = []

    for split in splits:
        print(f"Downloading split: {split}")
        dataset = load_dataset("RMT-team/babilong", name="32k", split=split)

        for example in dataset:
            combined_input = example["input"] + " " + example["question"]
            all_data.append({
                "input": combined_input,
                "output": example["target"]
            })

        print(f"  Added {len(dataset)} examples from {split}")

    output_file = os.path.join(output_dir, "babilong_32k_combined.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\nTotal examples: {len(all_data)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
