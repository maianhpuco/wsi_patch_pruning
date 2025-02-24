#!/bin/bash

#SBATCH --job-name=gradients
#SBATCH --output=gradients_output.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition=<your_partition_name>  # Replace with the correct partition if needed

echo "Group 1 : integrated_gradient, expected_gradient"
echo "Running with ig_name=integrated_gradient"
python main_ig.py --ig_name integrated_gradient

echo "Running with ig_name=expected_gradient" 
python main_ig.py --ig_name expected_gradient 

echo "All tasks (group 1) integrated_gradient, expected_gradient."
