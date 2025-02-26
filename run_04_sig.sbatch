#!/bin/bash

#SBATCH --job-name=run_04_sig
#SBATCH --output=run_04_sig.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
conda activate camil  

export PROJECT_DIR=$(pwd) 
echo "Running with ig_name=square_integrated_gradient"
python main_ig.py --ig_name square_integrated_gradient 
echo "Done"
