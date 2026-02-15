import argparse
import zipfile
import os
from huggingface_hub import snapshot_download

def download_and_extract_scrolls(target_dir):
    # Download the entire repo to the target directory
    repo_dir = snapshot_download(
        repo_id="tau/scrolls",
        repo_type="dataset",
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )

    print(f"Downloaded to: {repo_dir}")

    # Zip files to extract
    zip_files = ["quality.zip", "qasper.zip", "qmsum.zip"]

    for zip_name in zip_files:
        zip_path = os.path.join(repo_dir, zip_name)
        if not os.path.exists(zip_path):
            print(f"WARNING: {zip_name} not found.")
            continue

        extract_dir = os.path.join(repo_dir, zip_name.replace(".zip", ""))
        print(f"Extracting {zip_name} -> {extract_dir}")

        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    print("Done.")
    return repo_dir

def main():
    parser = argparse.ArgumentParser(description="Download SCROLLS dataset")
    parser.add_argument("--output_dir", type=str, default="./data/scrolls",
                        help="Output directory for the dataset")
    args = parser.parse_args()

    download_and_extract_scrolls(args.output_dir)

if __name__ == "__main__":
    main()
