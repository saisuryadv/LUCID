# TACC Vista Setup Guide

This guide covers LUCID installation on [TACC Vista](https://www.tacc.utexas.edu/systems/vista/) (NVIDIA GH200 / ARM aarch64). Vista has specific quirks around compiler versions, module management, and filesystem layout that differ from typical x86 GPU clusters.

## System Overview

- **Architecture:** ARM aarch64 (NOT x86_64)
- **GPU:** NVIDIA GH200 120GB, compute capability 9.0 (sm_90)
- **CUDA:** 12.8
- **Python:** 3.11.8 (via `python3_mpi` module)
- **No prebuilt wheels** for flash-attn or other CUDA extensions — everything must compile from source

## 1. Module Setup

Vista loads default modules on login. Some need to be swapped to compatible versions:

```bash
module swap ucc/1.5.1 ucc/1.4.4
module swap ucx/1.19.1 ucx/1.18.1
module swap cmake/4.1.1 cmake/3.31.5
module load tacc-apptainer/1.3.3
module save
```

Your final module list should include:

| Module | Version |
|--------|---------|
| gcc | 15.1.0 |
| cuda | 12.8 |
| openmpi | 5.0.5 |
| python3_mpi | 3.11.8 |
| nccl | 12.4 |
| nvpl | 25.5 |
| ucc | 1.4.4 |
| ucx | 1.18.1 |
| cmake | 3.31.5 |

## 2. Filesystem Layout

Vista has three main filesystems:

| Path | Alias | Use | Notes |
|------|-------|-----|-------|
| `/home1/...` | `$HOME` | Small files, dotfiles | NFS, limited quota |
| `/work/...` | `$WORK` | Project code, venvs | Persistent |
| `/scratch/...` | `$SCRATCH` | Large data, caches | **Purged after 10 days of inactivity** |

**Recommendation:** Clone LUCID and create your venv under `$WORK`. Put datasets and HuggingFace cache on `$SCRATCH`.

```bash
export HF_HOME="$SCRATCH/.cache/huggingface"
echo 'export HF_HOME="$SCRATCH/.cache/huggingface"' >> ~/.bashrc
```

Since `$HOME` is NFS, the DeepSpeed Triton cache can be slow. Optionally redirect it:

```bash
export TRITON_CACHE_DIR="$SCRATCH/.triton"
```

## 3. Create Virtual Environment

```bash
cd $WORK
git clone https://github.com/saisuryadv/LUCID.git
cd LUCID

python3 -m venv LUCID-env
source LUCID-env/bin/activate
```

Install PyTorch first (CUDA 12.8 wheels):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then install the rest:

```bash
pip install -r requirements.txt
```

## 4. Building flash-attn from Source

This is the trickiest part. CUDA 12.8 requires gcc <= 14, but Vista loads gcc 15.1 by default.

```bash
source LUCID-env/bin/activate

# Temporarily load gcc 14 for the build
module load gcc/14.2.0

CC=/opt/apps/gcc/14.2.0/bin/gcc \
CXX=/opt/apps/gcc/14.2.0/bin/g++ \
CUDAHOSTCXX=/opt/apps/gcc/14.2.0/bin/g++ \
TORCH_CUDA_ARCH_LIST="9.0" \
MAX_JOBS=4 \
pip install flash-attn --no-build-isolation
```

Build takes ~15-30 minutes. After it finishes, restore your modules:

```bash
module swap gcc/14.2.0 gcc/15.1.0
module load nvpl/25.5
```

### Gotcha: module load changes system pip

When `gcc/14.2.0` is loaded, the system `pip` path changes. Even with your venv activated, `pip` may resolve to the system pip and install packages to `~/.local/` instead of your venv.

**Always verify after module loads:**

```bash
which pip  # Should point to your venv, e.g., .../LUCID-env/bin/pip
```

If it doesn't, install the cached wheel directly:

```bash
source LUCID-env/bin/activate
pip install ~/.cache/pip/wheels/.../flash_attn-*.whl
```

## 5. Install Transformers Fork

```bash
cd $WORK
git clone https://github.com/saisuryadv/transformers.git
cd transformers
pip install -e .
cd ..

# Copy model file into the transformers source tree
cp LUCID/modeling_lucid.py transformers/src/transformers/models/llama/modeling_lucid.py
```

## 6. Install flash-linear-attention (for LUCID-Path)

```bash
cd $WORK
git clone https://github.com/sustcsonglin/flash-linear-attention.git
cd flash-linear-attention
pip install -e .
cd ..

# Symlink custom Triton kernels
ln -s $WORK/LUCID/lucid_path/triton_code \
      $WORK/flash-linear-attention/fla/ops/lucid_path
```

If you get an `ImportError` about `CacheLayerMixin`, patch `flash-linear-attention/fla/models/utils.py`:

```python
# Replace this:
if version.parse(_TF_VERSION) > version.parse(_NEED_NEW):
    from transformers.cache_utils import CacheLayerMixin
else:
    CacheLayerMixin = object

# With this:
try:
    from transformers.cache_utils import CacheLayerMixin
except ImportError:
    CacheLayerMixin = object
```

## 7. SLURM Partitions

| Partition | Type | Use | Notes |
|-----------|------|-----|-------|
| `gg` | CPU only | Data downloads, `accelerate config` | No GPU access |
| `gh-dev` | GPU | Interactive development (`idev`) | Max 8 nodes, short queue |
| `gh` | GPU | Production batch jobs | Full-scale training |

**Interactive GPU session:**

```bash
idev -p gh-dev -N 1 -n 1 -t 2:00:00 -A <ALLOC>
```

**Download data on CPU partition:**

```bash
idev -p gg -N 1 -n 1 -m 120 -A <ALLOC>
```

## 8. Verify Setup

Get an interactive GPU node and run the smoke test from the main README:

```bash
idev -p gh-dev -N 1 -n 1 -t 0:30:00 -A <ALLOC>

source LUCID-env/bin/activate

WANDB_DISABLED=true python transformers/examples/pytorch/language-modeling/run_clm.py \
  --model_type lucid \
  --config_name lucid/config \
  --tokenizer_name lucid/config \
  --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
  --do_train --output_dir ./test_output \
  --per_device_train_batch_size 1 --max_steps 5 \
  --block_size 512 --overwrite_output_dir --report_to none
```

## Known Working Package Versions

For reference, these versions are confirmed working on Vista:

| Package | Version |
|---------|---------|
| torch | 2.7.1+cu128 |
| transformers | 4.54.0.dev0 (editable) |
| flash-attn | 2.8.3 |
| deepspeed | 0.17.1 |
| triton | 3.3.1 |
| accelerate | 1.8.1 |
| datasets | 3.6.0 |
| lm_eval | 0.4.9.1 |
