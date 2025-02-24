#!/bin/bash

#SBATCH --job-name=gradients
#SBATCH --output=gradients_output.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition=<your_partition_name>  # Replace with the correct partition if needed

echo "Group 3 : vanilla_gradient, integrated_decision_gradient, optim_square_integrated_gradient"
echo "Running with ig_name=vanilla_gradient"
python main_ig.py --ig_name vanilla_gradient

@echo "Running with ig_name=integrated_decision_gradient" 
python main_ig.py --ig_name integrated_decision_gradient 

@echo "Running with ig_name=optim_square_integrated_gradient" 
python main_ig.py --ig_name optim_square_integrated_gradient  

echo "All tasks (group 3) vanilla_gradient, integrated_decision_gradient, optim_square_integrated_gradient"
