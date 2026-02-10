#!/bin/bash
#SBATCH -J Finetune-OgLucid	# Job name
#SBATCH -o oglucid_expknorm_rope.o%j		# stdout output file
#SBATCH -e oglucid_expknorm_rope.e%j		# stderr error file
#SBATCH -p gh			# Or your target GPU partition
#SBATCH -N 16			# Number of nodes
#SBATCH -n 16			# Total tasks/GPUs
#SBATCH -t 24:00:00		# Wall time
#SBATCH -A <ALLOC>		# Your project allocation
#SBATCH --mail-user=<EMAIL>
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,CANCELLED

set -euo pipefail
echo "--- STARTING VIRUTAL ENVIRONMENT ---"
source <PATH_TO_LUCID>/LUCID-env/bin/activate

PROJ_DIR=<PATH_TO_LUCID>
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "--- STARTING TRAINING ---"
export SCRIPT_AND_ARGS="$PROJ_DIR/transformers/examples/pytorch/language-modeling/run_clm.py \
	--model_name_or_path $PROJ_DIR/oglucid_no_qknorm_directory/oglucid_checkpoints_finetune_init \
	--dataset_name $PROJ_DIR/dolma_70_30_split \
	--tokenizer_name $PROJ_DIR/oglucid_no_qknorm_directory/oglucid_checkpoints_finetune_init \
	--validation_split_percentage 10 \
	--do_train \
	--output_dir $PROJ_DIR/oglucid_no_qknorm_directory/oglucid_rope_theta64K_finetune_$SLURM_JOB_ID \
	--per_device_train_batch_size 1 \
	--max_steps 500 \
	--logging_steps 50 \
	--logging_first_step True \
	--log_on_each_node False \
	--logging_strategy steps \
	--logging_dir $PROJ_DIR/oglucid_no_qknorm_directory/oglucid_rope_theta64K_finetune_$SLURM_JOB_ID \
	--save_steps 100 \
	--learning_rate 5e-6 \
	--lr_scheduler_type cosine \
	--warmup_ratio 0.06 \
	--block_size 65536 \
	--gradient_checkpointing \
	--dataloader_num_workers 16 \
	--preprocessing_num_workers 32 \
	--data_seed 0 \
	--save_total_limit 1 \
	--disable_tqdm True \
	--adam_beta2 0.99 \
	--adam_epsilon 1e-12 \
	--weight_decay 0.01 \
"
#	--preprocessing_num_workers 32"
export LAUNCHER="accelerate launch \
	--num_processes $SLURM_NNODES \
	--num_machines $SLURM_NNODES \
	--rdzv_backend c10d \
	--main_process_ip $HEAD_NODE \
	--main_process_port 29500 \
"
export CMD="$LAUNCHER $SCRIPT_AND_ARGS"

srun bash -c "$CMD"

echo "Job finished successfully."
echo "Job finished on: $(date)"
