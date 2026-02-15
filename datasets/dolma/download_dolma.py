import argparse
import os
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Download Dolma dataset")
    parser.add_argument("--output_dir", type=str, default="./data/dolma_70_30_split",
                        help="Output directory for the dataset splits")
    parser.add_argument("--test_size", type=float, default=0.3,
                        help="Fraction of data to use for test split (default: 0.3)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for split (default: 0)")
    args = parser.parse_args()

    dataset_name = "allenai/dolma"
    dataset_config = "v1_6-sample"

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory for splits: {output_dir}")

    print(f"Downloading dataset '{dataset_name}' with config '{dataset_config}'...")
    full_dataset = load_dataset(dataset_name, dataset_config, trust_remote_code=True)['train']

    print(f"Creating {int((1-args.test_size)*100)}/{int(args.test_size*100)} split with seed={args.seed}...")
    split_dataset_dict = full_dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    print(f"Split created: {split_dataset_dict}")

    print(f"Saving splits to {output_dir}...")
    split_dataset_dict.save_to_disk(output_dir)

    print("Download complete. Splits are saved.")

if __name__ == "__main__":
    main()
