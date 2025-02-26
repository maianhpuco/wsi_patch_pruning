#!/bin/bash

#SBATCH --job-name=run_06_igg
#SBATCH --output=run_06_idg.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition=<your_partition_name>  # Replace with the correct partition if needed
conda activate camil  

export PROJECT_DIR=$(pwd) 

echo "Running with ig_name=integrated_decision_gradient"
python main_ig.py --ig_name integrated_decision_gradient
echo "Done"
