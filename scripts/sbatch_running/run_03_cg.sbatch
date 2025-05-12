#!/bin/bash

#SBATCH --job-name=run_03_cg
#SBATCH --output=run_03_cg.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
conda activate camil  

export PROJECT_DIR=$(pwd) 
echo "Running with ig_name=contrastive_gradient"
python main_ig.py --ig_name contrastive_gradient
echo "Done"
