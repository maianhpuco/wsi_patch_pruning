#!/bin/bash
#SBATCH --job-name=ig_experiments    # Job name
#SBATCH --output=slurm-%x-%j.out     # Output file (job name and job ID)
#SBATCH --error=slurm-%x-%j.err      # Error file
#SBATCH --time=12:00:00              # Max time (hh:mm:ss)
#SBATCH --partition=gpu              # Specify partition (change if needed)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=4            # Request CPU cores
#SBATCH --mem=16G                    # Memory per node
#SBATCH --ntasks=1                   # Number of tasks

# Load required modules (modify if needed)
module load cuda/11.7  # Adjust based on available versions

# Activate environment
source env/bin/activate
conda activate camil

# Set project directory
export PROJECT_DIR=$(pwd)

# Run different IG methods
for ig_name in integrated_gradient expected_gradient integrated_decision_gradient; do
    echo "Running with ig_name=${ig_name}"
    python main_ig.py --ig_name ${ig_name}
    python main_plot_ig.py --ig_name ${ig_name}
done
