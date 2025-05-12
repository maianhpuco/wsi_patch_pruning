#!/bin/bash

#SBATCH --job-name=run_02_eg
#SBATCH --output=run_02_eg.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
conda activate camil  

export PROJECT_DIR=$(pwd) 

echo "Running with ig_name=expected_gradient"
python main_ig.py --ig_name expected_gradient 
echo "Done"
