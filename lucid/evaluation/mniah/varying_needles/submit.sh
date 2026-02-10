#!/bin/bash
echo "Submitting SLURM jobs for MNIAH Evaluation"

NUM_NEEDLES=( 2 4 6 8 10 )

for NN in "${NUM_NEEDLES[@]}"; do
  sbatch --export=ALL,NN="$NN" \
    --job-name="Evaluate-LUCID-$NN" \
    --output="mniah_lucid_num_needle_$NN.o%j" \
    --error="mniah_lucid_num_needle_$NN.e%j" \
    mniah.slurm
done

echo "All jobs submitted"
