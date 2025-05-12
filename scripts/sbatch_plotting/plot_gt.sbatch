#!/bin/bash

#SBATCH --job-name=plot_gt
#SBATCH --output=plot_gt.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
conda activate camil  

export PROJECT_DIR=$(pwd)

python main_plot_gt.py 
echo "done" 