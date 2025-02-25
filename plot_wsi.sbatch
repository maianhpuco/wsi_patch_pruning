#!/bin/bash

#SBATCH --job-name=plot_wsi
#SBATCH --output=plot_wsi.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
conda activate camil  

export PROJECT_DIR=$(pwd)

python main_plot_wsi.py 
echo "done" 