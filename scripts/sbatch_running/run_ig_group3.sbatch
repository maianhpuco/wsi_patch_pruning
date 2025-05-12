#!/bin/bash

#SBATCH --job-name=group3 
#SBATCH --output=group3.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
conda activate camil  


export PROJECT_DIR=$(pwd) 

echo "Group 3 : vanilla_gradient, integrated_decision_gradient, optim_square_integrated_gradient"
echo "Running with ig_name=vanilla_gradient"
python main_ig.py --ig_name vanilla_gradient

echo "Running with ig_name=integrated_decision_gradient" 
python main_ig.py --ig_name integrated_decision_gradient 

@echo "Running with ig_name=optim_square_integrated_gradient" 
python main_ig.py --ig_name optim_square_integrated_gradient  

echo "All tasks (group 3) vanilla_gradient, integrated_decision_gradient, optim_square_integrated_gradient"
