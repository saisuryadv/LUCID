#!/bin/bash
echo "Submitting SLURM jobs for LONGBENCH Evaluation"

TASKS=(
  "longbench_2wikimqa"
  "longbench_hotpotqa"
  "longbench_multifieldqa_en"
  "longbench_qasper"
)

for TASK in "${TASKS[@]}"; do
  sbatch --export=ALL,TASK="$TASK" \
    --job-name="Evaluate-LUCID-$TASK" \
    --output="eval_lucid_$TASK.o%j" \
    --error="eval_lucid_$TASK.e%j" \
    longbench.slurm
done

echo "All jobs submitted"
