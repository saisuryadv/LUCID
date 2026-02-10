#!/bin/bash
#SBATCH -p gh				# Or your target GPU partition
#SBATCH -N 1				# Number of nodes
#SBATCH -n 1				# Total tasks/GPUs
#SBATCH -t 2:00:00			# Wall time
#SBATCH -A <ALLOC>			# Your project allocation
#SBATCH --mail-user=<EMAIL>
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,CANCELLED

set -euo pipefail
echo "--- STARTING VIRUTAL ENVIRONMENT ---"
source <PATH_TO_LUCID>/LUCID-env/bin/activate

# --- CONFIGURATION ---
PROJ_DIR=<PATH_TO_LUCID>
MODEL_PATH="$PROJ_DIR/<PATH_TO_FINETUNED_MODEL>"
BASE_RESULTS_DIR="$PROJ_DIR/evaluation_results/MNIAH"

# --- ACCELERATE LAUNCH CONFIG ---
NUM_MACHINES=$SLURM_NNODES
NUM_PROCESSES=$SLURM_NNODES
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "============================================================"
echo "MNIAH evaluations started!"
echo "============================================================"

export LAUNCHER="accelerate launch \
    --num_machines=$NUM_MACHINES \
    --num_processes=$NUM_PROCESSES \
    --rdzv_backend c10d \
    --main_process_ip $HEAD_NODE \
    --main_process_port 29500 \
    -m \
"
export SCRIPT_AND_ARGS="lm_eval \
    --model hf \
    --model_args pretrained=$MODEL_PATH,max_length=32768 \
    --tasks niah_multikey_1,niah_multikey_1 \
    --metadata {\"max_seq_lengths\":[2048,4096,6144,8192],\"num_needle_k\":${NN}} \
    --output_path "$BASE_RESULTS_DIR/${NN}_results.json" \
    --batch_size 30 \
    --trust_remote_code \
    --seed 0 \
"
export CMD="$LAUNCHER $SCRIPT_AND_ARGS"
srun $CMD

echo "============================================================"
echo "MNIAH evaluations completed!"
echo "============================================================"

echo "Job finished successfully."
echo "Job finished on: $(date)"
