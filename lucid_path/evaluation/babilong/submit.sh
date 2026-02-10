#!/bin/bash
echo "Submitting SLURM jobs for BABILONG Evaluation"

TASKS=(
  "babilong_qa1"
  "babilong_qa2"
  "babilong_qa3"
  "babilong_qa4"
  "babilong_qa5"
  "babilong_qa6"
  "babilong_qa7"
  "babilong_qa8"
  "babilong_qa9"
  "babilong_qa10"
)

for TASK in "${TASKS[@]}"; do
  sbatch --export=ALL,TASK="$TASK" \
    --job-name="Evaluate-LUCID-Path-$TASK" \
    --output="eval_lucid_path_$TASK.o%j" \
    --error="eval_lucid_path_$TASK.e%j" \
    babilong.slurm
done

echo "All jobs submitted"
