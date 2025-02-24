#!/bin/bash

#SBATCH --job-name=gradients
#SBATCH --output=gradients_output.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition=<your_partition_name>  # Replace with the correct partition if needed
conda activate camil  

export PROJECT_DIR=$(pwd) 

echo "Group 2 : contrastive_gradient, square_integrated_gradient"
echo "Running with ig_name=contrastive_gradient"
python main_ig.py --ig_name contrastive_gradient

echo "Running with ig_name=square_integrated_gradient" 
python main_ig.py --ig_name square_integrated_gradient 

echo "All tasks (group 2) contrastive_gradient, square_integrated_gradient."
