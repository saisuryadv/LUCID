#!/bin/bash
#SBATCH -J QMSUM-Lucid-Path	# Job name
#SBATCH -o qmsum_lucid_path.o%j		# stdout output file
#SBATCH -e qmsum_lucid_path.e%j		# stderr error file
#SBATCH -p gh			# Or your target GPU partition
#SBATCH -N 8			# Number of nodes
#SBATCH -n 8			# Total tasks/GPUs
#SBATCH -t 8:00:00		# Wall time
#SBATCH -A <ALLOC>		# Your project allocation
#SBATCH --mail-user=<EMAIL>
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,CANCELLED

set -euo pipefail
echo "--- STARTING VIRUTAL ENVIRONMENT ---"
source <PATH_TO_LUCID>/LUCID-env/bin/activate

PROJ_DIR=<PATH_TO_LUCID>
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "--- STARTING TRAINING ---"
export SCRIPT_AND_ARGS="$PROJ_DIR/transformers/examples/pytorch/language-modeling/run_clm_chunk.py \
	--model_name_or_path $PROJ_DIR/lucid_path_directory/lucid_path_checkpoints_finetune_init \
	--train_file $PROJ_DIR/scrolls_cleaned/qmsum/train.json \
	--tokenizer_name $PROJ_DIR/Tokenizer_Qmsum \
	--output_dir $PROJ_DIR/lucid_path_directory/FT_SCROLLS/QMSUM_finetune_$SLURM_JOB_ID \
	--logging_dir $PROJ_DIR/lucid_path_directory/FT_SCROLLS/QMSUM_finetune_$SLURM_JOB_ID \
	--do_train \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 4 \
	--num_train_epochs 2 \
	--logging_steps 10 \
	--logging_first_step True \
	--log_on_each_node False \
	--logging_strategy steps \
	--save_steps 100 \
	--learning_rate 5e-6 \
	--lr_scheduler_type cosine \
	--warmup_ratio 0.06 \
	--block_size 32768 \
	--gradient_checkpointing \
	--dataloader_num_workers 32 \
	--preprocessing_num_workers 64 \
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
	--dynamo_backend no \
	--rdzv_backend c10d \
	--main_process_ip $HEAD_NODE \
	--main_process_port 29500 \
"
export CMD="$LAUNCHER $SCRIPT_AND_ARGS"

srun bash -c "$CMD"

echo "Job finished successfully."
echo "Job finished on: $(date)"
