#!/bin/bash

#SBATCH --job-name=group1
#SBATCH --output=group1.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00

conda activate camil  

export PROJECT_DIR=$(pwd) 

echo "Running with ig_name=integrated_gradient"
python main_plot_ig_nobar.py --ig_name integrated_gradient

echo "Running with ig_name=expected_gradient"
python main_plot_ig_nobar.py --ig_name expected_gradient

echo "Running with ig_name=contrastive_gradient"
python main_plot_ig_nobar.py --ig_name contrastive_gradient

echo "Running with ig_name=square_integrated_gradient"
python main_plot_ig_nobar.py --ig_name square_integrated_gradient

echo "Running with ig_name=vanilla_gradient"
python main_plot_ig_nobar.py --ig_name vanilla_gradient 

echo "Running with ig_name=integrated_decision_gradient"
python main_plot_ig_nobar.py --ig_name integrated_decision_gradient

echo "Running with ig_name=optim_square_integrated_gradient"
python main_plot_ig_nobar.py --ig_name optim_square_integrated_gradient
