#!/bin/bash

#SBATCH --job-name=group1
#SBATCH --output=group1.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00

conda activate camil  

export PROJECT_DIR=$(pwd) 

echo "Group 1 : integrated_gradient, expected_gradient"
echo "Running with ig_name=integrated_gradient"
python main_ig.py --ig_name integrated_gradient

echo "Running with ig_name=expected_gradient" 
python main_ig.py --ig_name expected_gradient 

echo "All tasks (group 1) integrated_gradient, expected_gradient."
