#!/bin/bash
echo "Submitting SLURM jobs for RULER Evaluation"

TASKS=(
  "niah_single_1"
  "niah_single_2"
  "niah_single_3"
  "niah_multikey_1"
  "niah_multikey_2"
  "niah_multikey_3"
  "niah_multiquery"
  "niah_multivalue"
  "ruler_vt"
  "ruler_cwe"
  "ruler_fwe"
  "ruler_qa_squad"
)

for TASK in "${TASKS[@]}"; do
  sbatch --export=ALL,TASK="$TASK" \
    --job-name="Evaluate-LUCID-$TASK" \
    --output="eval_lucid_$TASK.o%j" \
    --error="eval_lucid_$TASK.e%j" \
    ruler.slurm
done

echo "All jobs submitted"
