# LUCID: Linear Attention with Unrestricted Context and Ideal Decay

This repository contains the code for reproducing the LUCID model and its variant LUCID-Path.

## Directory Structure

```
LUCID-CODE/
├── modeling_llama.py           # Modified LLaMA implementation with LUCID attention
├── run_clm_chunk.py           # Training script for chunked causal language modeling
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── lm_eval_ruler/             # RULER benchmark task definitions (from lm_eval/tasks/ruler)
│   ├── niah_utils.py          # Multi-needle evaluation utilities
│   ├── niah_*.yaml            # NIAH task configurations
│   ├── ruler.yaml             # Main RULER configuration
│   └── ...                    # Other RULER task files
├── datasets/                  # Dataset download scripts
│   ├── dolma/
│   │   ├── download_dolma.py
│   │   └── download_dolma.slurm
│   ├── download_scrolls.py
│   ├── download_babilong.py
│   └── download_longbench.py
├── lucid/                     # LUCID implementation
│   ├── config/                # Model configuration files
│   ├── train_lucid.slurm     # Pre-training script
│   ├── finetune/              # Finetuning scripts
│   │   ├── finetune_longbench.slurm
│   │   └── finetune_scrolls.slurm
│   └── evaluation/            # Evaluation scripts
│       ├── mniah/
│       │   └── varying_needles/
│       │       ├── mniah.slurm
│       │       └── submit.sh
│       ├── ruler/
│       │   ├── ruler.slurm
│       │   └── submit.sh
│       └── longbench/
│           ├── longbench.slurm
│           └── submit.sh
└── lucid_path/                # LUCID-Path implementation
    ├── config/                # Model configuration files
    ├── train_lucid_path.slurm # Pre-training script
    ├── triton_code/           # Triton kernels (adapted from FLA repository)
    ├── finetune/              # Finetuning scripts
    │   ├── finetune_babilong.slurm
    │   ├── finetune_longbench.slurm
    │   └── scrolls/
    │       ├── finetune_qasper.slurm
    │       └── finetune_qmsum.slurm
    └── evaluation/            # Evaluation scripts
        ├── babilong/
        │   ├── babilong.slurm
        │   └── submit.sh
        └── scrolls/
            ├── scrolls.slurm
            └── submit.sh
```

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- Transformers library (modified version included)
- Flash Attention 2
- Triton
- lm-eval-harness (for evaluations)

### Installation

1. Clone this repository and set up your environment:

```bash
# Create virtual environment
python -m venv LUCID-env
source LUCID-env/bin/activate  # On Windows: LUCID-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install torch torchvision torchaudio --index-url https://cuda.pytorch.org/whl/cu118
pip install transformers datasets accelerate einops flash-attn triton
pip install lm-eval
```

2. Update placeholder values in all SLURM scripts:
   - Replace `<PATH_TO_LUCID>` with your project directory path
   - Replace `<ALLOC>` with your SLURM allocation ID
   - Replace `<EMAIL>` with your email address
   - For evaluation scripts, replace model path placeholders:
     - `<PATH_TO_PRETRAINED_MODEL>` with path to pre-trained checkpoint
     - `<PATH_TO_FINETUNED_MODEL>` with path to fine-tuned checkpoint
     - `<PATH_TO_QASPER_FINETUNED_MODEL>` with path to QASPER fine-tuned model
     - `<PATH_TO_QMSUM_FINETUNED_MODEL>` with path to QMSUM fine-tuned model

## Datasets

### Download Datasets

We provide scripts to download the training and evaluation datasets:

#### Dolma (Pre-training)
```bash
cd datasets/dolma
sbatch download_dolma.slurm  # Or run: python download_dolma.py
```

#### SCROLLS Benchmark
```bash
python datasets/download_scrolls.py
```

#### BABILong Benchmark
```bash
python datasets/download_babilong.py
```

#### LongBench
```bash
python datasets/download_longbench.py
```

## Training

### LUCID Pre-training

Train LUCID from scratch on the Dolma dataset:

```bash
cd lucid
sbatch train_lucid.slurm
```

**Key hyperparameters:**
- Batch size: 16 per device
- Learning rate: 5e-4
- Block size: 2048
- Optimizer: AdamW (β2=0.99, ε=1e-12)
- LR scheduler: Cosine with 6% warmup
- Weight decay: 0.01

### LUCID-Path Pre-training

Train LUCID-Path from scratch:

```bash
cd lucid_path
sbatch train_lucid_path.slurm
```

**Key hyperparameters:**
- Batch size: 8 per device
- Learning rate: 5e-4
- Block size: 2048
- Training nodes: 32

## Finetuning

### LUCID Finetuning

#### LongBench
```bash
cd lucid/finetune
sbatch finetune_longbench.slurm
```

#### SCROLLS
```bash
cd lucid/finetune
sbatch finetune_scrolls.slurm
```

### LUCID-Path Finetuning

#### LongBench
```bash
cd lucid_path/finetune
sbatch finetune_longbench.slurm
```

#### BABILong
```bash
cd lucid_path/finetune
sbatch finetune_babilong.slurm
```

#### SCROLLS - QASPER
```bash
cd lucid_path/finetune/scrolls
sbatch finetune_qasper.slurm
```

#### SCROLLS - QMSUM
```bash
cd lucid_path/finetune/scrolls
sbatch finetune_qmsum.slurm
```

## Evaluation

All evaluation scripts use the `lm-eval-harness` framework. Before running evaluations, ensure you've installed `lm-eval` and updated the model paths in the scripts.

### LUCID Evaluations

#### MNIAH (Multi-Needle In A Haystack)
Evaluates multi-key retrieval across varying context lengths.

```bash
cd lucid/evaluation/mniah/varying_needles
bash submit.sh  # Submits jobs for 2, 4, 6, 8, 10 needles
```

**Configuration:**
- Sequence lengths: 2K, 4K, 6K, 8K
- Number of needles: 2, 4, 6, 8, 10
- Max length: 32K tokens

#### RULER Benchmark
Comprehensive suite of 12 synthetic tasks testing retrieval and reasoning.

```bash
cd lucid/evaluation/ruler
bash submit.sh  # Submits all 12 RULER tasks
```

**Tasks included:**
- NIAH variants (single_1/2/3, multikey_1/2/3, multiquery, multivalue)
- Variable tracking (ruler_vt)
- Common words extraction (ruler_cwe)
- Frequent words extraction (ruler_fwe)
- Question answering (ruler_qa_squad)

**Configuration:**
- Sequence lengths: 2K, 4K, 6K, 8K
- Max length: 32K tokens
- Wall time: 4 hours per task

**Note:** The RULER task definitions are provided in `lm_eval_ruler/` (copied from `lm_eval/tasks/ruler`). The `niah_utils.py` file supports multi-needle evaluation with configurable `num_needle_k`, `num_needle_v`, and `num_needle_q` parameters for testing retrieval of multiple key-value pairs across different context lengths.

#### LongBench
Question answering tasks on long documents.

```bash
cd lucid/evaluation/longbench
bash submit.sh  # Submits 4 LongBench tasks
```

**Tasks included:**
- 2WikiMQA
- HotpotQA
- MultifieldQA-EN
- QASPER

**Configuration:**
- Max length: 32K tokens
- Wall time: 30 minutes per task

### LUCID-Path Evaluations

#### BABILong
Evaluates performance on 10 reasoning tasks with 64K context.

```bash
cd lucid_path/evaluation/babilong
bash submit.sh  # Submits all 10 BABILong QA tasks
```

**Tasks included:**
- babilong_qa1 through babilong_qa10

**Configuration:**
- Context length: 64K tokens
- Max length: 150K tokens
- Evaluation limit: 50 examples per task
- Wall time: 2 hours per task

#### SCROLLS
Document understanding tasks requiring long-range reasoning.

```bash
cd lucid_path/evaluation/scrolls
bash submit.sh  # Submits QASPER and QMSUM tasks
```

**Tasks included:**
- QASPER (Question Answering on Scientific Papers)
- QMSUM (Query-based Multi-domain Meeting Summarization)

**Configuration:**
- Max length: 32K tokens
- Batch size: 8
- Uses task-specific fine-tuned models
- Wall time: 2 hours per task

## Model Architecture

### LUCID Attention

LUCID implements a linear attention mechanism with:
- **Exponential decay**: Ideal forgetting mechanism for long sequences
- **RMSNorm**: Query and key normalization for stability
- **Efficient triangular solver**: Handles sequences beyond GPU memory
- **GQA support**: Grouped Query Attention for efficiency
- **Block-wise computation**: Custom CUDA implementation for long sequences

**Key equation:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V'
where V' solves: L * V' = V (L is lower triangular exponential matrix)
```

### LUCID-Path Attention

LUCID-Path extends LUCID with path-based computation:
- **Path-based attention**: Efficient parallel computation using path formulation
- **Custom Triton kernels**: Optimized forward/backward implementations
- **Householder transformations**: Efficient orthogonalization
- **Extended context**: Supports up to 65K+ tokens efficiently
- **Memory optimized**: Chunk-wise processing with gradient checkpointing

**Triton kernels included:**
- `parallel_path_fwd.py`: Forward pass computation
- `parallel_path_bwd_*.py`: Backward pass gradients
- `cumprod_householder_*.py`: Householder cumulative products
- `triangular_solve_*.py`: Efficient triangular solvers

## Key Features

1. **Linear Complexity**: O(n) attention computation vs O(n²) for standard attention
2. **Long Context**: Efficient processing of sequences up to 65K+ tokens
3. **Memory Efficient**: Chunked processing for sequences exceeding GPU memory
4. **Gradient Checkpointing**: Reduces memory footprint during training
5. **Custom Kernels**: Optimized Triton implementations for critical operations
6. **Production Ready**: Supports caching, generation, and standard HuggingFace APIs

## Configuration

Model configurations are stored in the `config/` directories. Key configuration parameters:

### LUCID Configuration
- `attention_type`: "lucid" (set to "llama" for baseline)
- `which_rmsnorm`: "exp_k_norm" (normalization strategy)
- `use_rope`: true/false (rotary positional embeddings)
- `use_beta`: false (beta scaling for forget gates)
- `use_token_grad_clip`: true/false (gradient clipping)

### LUCID-Path Configuration
- `attention_type`: "lucid_path"
- `chunk_size`: 2048 (size of chunks for path computation)
- `use_triton`: true (enable custom Triton kernels)
- Additional path-specific parameters in config.json

## Performance Notes

### Training
- LUCID: ~3.5B parameters, trains on 16 GPUs
- LUCID-Path: ~3.5B parameters, trains on 32 GPUs
- Training time: ~13 hours for LUCID-Path on Dolma subset

### Evaluation
- MNIAH: Linear scaling with sequence length
- RULER: Consistent performance across all 12 tasks
- BABILong: Handles 64K context efficiently
- SCROLLS: Task-specific fine-tuning improves performance

## Troubleshooting

### Common Issues

1. **Out of Memory during training:**
   - Reduce `per_device_train_batch_size`
   - Enable `gradient_checkpointing`
   - Reduce `block_size` or use chunked processing

2. **SLURM job failures:**
   - Check allocation limits (`squeue -u $USER`)
   - Verify paths in SLURM scripts are correct
   - Check logs in `.o` and `.e` files

3. **Evaluation errors:**
   - Ensure `lm-eval` is installed: `pip install lm-eval`
   - Verify model paths are correct in scripts
   - Check that model was trained with compatible config

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{lucid2024,
  title={LUCID: Linear Attention with Unrestricted Context and Ideal Decay},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## License

[Specify your license here]

## Acknowledgments

This implementation is based on the Hugging Face Transformers library and incorporates ideas from:
- Flash Attention (Dao et al., 2022)
- Rotary Position Embeddings (Su et al., 2021)
- Linear Attention mechanisms
- Triton for GPU kernel optimization
- [Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention) - LUCID-Path Triton kernels are adapted from the PaTH implementation

## Contact

For questions or issues, please open an issue on GitHub or contact [contact information].
