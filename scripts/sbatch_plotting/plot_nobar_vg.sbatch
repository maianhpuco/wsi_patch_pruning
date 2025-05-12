#!/bin/bash

#SBATCH --job-name=plot_vg
#SBATCH --output=plot_nobar_vg.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00

conda activate camil  

export PROJECT_DIR=$(pwd) 

echo "Running with ig_name=vanilla_gradient"
python main_plot_ig_nobar.py --ig_name vanilla_gradient 
 

