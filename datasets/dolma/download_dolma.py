from datasets import load_dataset
import os

dataset_name = "allenai/dolma"
dataset_config = "v1_6-sample"

output_dir = os.path.join(os.environ["SCRATCH"], "dolma_70_30_split")
print(f"Output directory for splits: {output_dir}")

print(f"Ensuring dataset '{dataset_name}' with config '{dataset_config}' is cached...")
full_dataset = load_dataset(dataset_name, dataset_config, trust_remote_code=True)['train']

print("Creating 70/30 split with seed=0...")
split_dataset_dict = full_dataset.train_test_split(test_size=0.3, seed=0)

print(f"Split created: {split_dataset_dict}")

print(f"Saving splits to {output_dir}...")
split_dataset_dict.save_to_disk(output_dir)

print("Preprocessing complete. Splits are saved.")
