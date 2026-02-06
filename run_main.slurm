#!/bin/bash
#SBATCH --job-name=main_notebook
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --account=def-sponsor00

set -euo pipefail

# Change to project directory (submit from project dir so SLURM_SUBMIT_DIR is set)
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/def-sponsor00/benhoangquan/ey_challenge_2026}"
mkdir -p logs

# Load modules (adjust version numbers for your cluster; check with: module avail python)
module load python/3.10
# Optional: load CUDA if needed for GPU libraries (uncomment if required)
# module load cuda/12.6

pip install --upgrade pip setuptools wheel -q
pip install -r requirements_minimal.txt -q

# Run notebook execution
python main.py
