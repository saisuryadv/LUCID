from huggingface_hub import snapshot_download
import zipfile
import os

def download_and_extract_scrolls(target_dir):
    # 1. Download the entire repo to the target directory
    repo_dir = snapshot_download(
        repo_id="tau/scrolls",
		repo_type="dataset",
        local_dir=target_dir,
        local_dir_use_symlinks=False   # ensures real files, not symlinks
    )

    print(f"Downloaded to: {repo_dir}")

    # 2. Zip files we want to extract
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

path = "/work/01318/nnp528/vista/LUCID/scrolls"
download_and_extract_scrolls(path)
