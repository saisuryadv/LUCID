#!/bin/bash
echo "Submitting SLURM jobs for SCROLLS Evaluation"

PROJ_DIR=<PATH_TO_LUCID>
BASE_RESULTS_DIR="$PROJ_DIR/evaluation_results/SCROLLS"
MODEL_PATHS=(
  "$PROJ_DIR/<PATH_TO_QASPER_FINETUNED_MODEL>"
  "$PROJ_DIR/<PATH_TO_QMSUM_FINETUNED_MODEL>"
)

TASKS=(
  "scrolls_qasper"
  "scrolls_qmsum"
)

# Loop over indices to match TASKS and MODEL_PATHS
for i in "${!TASKS[@]}"; do
  TASK="${TASKS[i]}"
  MODEL_PATH="${MODEL_PATHS[i]}"

  sbatch --export=ALL,TASK="$TASK",MODEL_PATH="$MODEL_PATH",BASE_RESULTS_DIR="$BASE_RESULTS_DIR" \
    --job-name="Evaluate-LUCID-Path-$TASK" \
    --output="eval_$TASK.o%j" \
    --error="eval_$TASK.e%j" \
    scrolls.slurm
done

echo "All jobs submitted"
