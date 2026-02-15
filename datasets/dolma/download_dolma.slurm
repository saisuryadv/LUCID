#!/bin/bash
#SBATCH -J DolmaDownload			# Job name
#SBATCH -o download_dolma.o%j		# Name of stdout output file
#SBATCH -e download_dolma.e%j		# Name of stderr error file
#SBATCH -p <PARTITION>				# Your GPU partition
#SBATCH -N 1						# Total number of nodes needed (1 is sufficient)
#SBATCH -n 1						# Total number of mpi tasks (1 is sufficient)
#SBATCH -t 02:00:00					# Run time (hh:mm:ss) - 2 hours should be plenty
#SBATCH -A <ALLOC>					# Your project allocation name

set -e	# Exit the script if any command fails

echo "Starting dataset download job..."
echo "Job started on: $(date)"

# --- ACTIVATE ENVIRONMENT ---
source <PATH_TO_LUCID>/LUCID-env/bin/activate
echo "Virtual environment activated."

# --- DEFINE PROJECT DIRECTORY ---
export PROJ_DIR=<PATH_TO_LUCID>

# --- RUN THE DOWNLOAD SCRIPT ---
echo "Running the dataset check and download script..."
python ${PROJ_DIR}/download_dolma.py

echo "Job finished successfully."
echo "Job finished on: $(date)"
