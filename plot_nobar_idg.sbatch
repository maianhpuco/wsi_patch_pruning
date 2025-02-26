#!/bin/bash

#SBATCH --job-name=plot_idg
#SBATCH --output=plot_nobar_idg.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00

conda activate camil  

export PROJECT_DIR=$(pwd) 

echo "Running with ig_name=integrated_decision_gradient"
python main_plot_ig_nobar.py --ig_name integrated_decision_gradient 

