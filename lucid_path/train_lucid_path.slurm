#!/bin/bash
#SBATCH -J Train-Lucid-Path		# Job name
#SBATCH -o lucid_path.o%j		# stdout output file
#SBATCH -e lucid_path.e%j		# stderr error file
#SBATCH -p <PARTITION>			# Your GPU partition
#SBATCH -N 32				# Number of nodes
#SBATCH -n 32				# Total tasks/GPUs
#SBATCH -t 13:00:00			# Wall time
#SBATCH -A <ALLOC>			# Your project allocation
#SBATCH --mail-user=<EMAIL>
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,CANCELLED

set -euo pipefail
echo "--- STARTING VIRUTAL ENVIRONMENT ---"
source <PATH_TO_LUCID>/LUCID-env/bin/activate

PROJ_DIR=<PATH_TO_LUCID>
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "--- STARTING TRAINING ---"
export SCRIPT_AND_ARGS="$PROJ_DIR/transformers/examples/pytorch/language-modeling/run_clm.py \
	--config_name $PROJ_DIR/lucid_path_directory/lucid_path-config \
	--dataset_name $PROJ_DIR/dolma_70_30_split \
	--tokenizer_name $PROJ_DIR/lucid_path_directory/lucid_path-config \
	--output_dir $PROJ_DIR/lucid_path_directory/lucid_path_checkpoints_$SLURM_JOB_ID \
	--logging_dir $PROJ_DIR/lucid_path_directory/lucid_path_$SLURM_JOB_ID \
	--validation_split_percentage 10 \
	--do_train \
	--per_device_train_batch_size 8 \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 1 \
	--logging_steps 100 \
	--logging_first_step True \
	--log_on_each_node False \
	--logging_strategy steps \
	--save_steps 2000 \
	--learning_rate 5e-4 \
	--lr_scheduler_type cosine \
	--warmup_ratio 0.06 \
	--block_size 2048 \
	--dataloader_num_workers 32 \
	--preprocessing_num_workers 64 \
	--data_seed 0 \
	--save_total_limit 1 \
	--disable_tqdm True \
	--adam_beta2 0.99 \
	--adam_epsilon 1e-12 \
	--weight_decay 0.01 \
"
export LAUNCHER="accelerate launch \
	--num_processes $SLURM_NNODES \
	--num_machines $SLURM_NNODES \
	--dynamo_backend no \
	--rdzv_backend c10d \
	--main_process_ip $HEAD_NODE \
	--main_process_port 29500 \
"
export CMD="$LAUNCHER $SCRIPT_AND_ARGS"

srun bash -c "$CMD"

echo "Job finished successfully."
echo "Job finished on: $(date)"
