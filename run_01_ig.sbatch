#!/bin/bash

#SBATCH --job-name=run_01_ig
#SBATCH --output=run_01_ig.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition=<your_partition_name>  # Replace with the correct partition if needed
conda activate camil  

export PROJECT_DIR=$(pwd) 

echo "Running with ig_name=square_integrated_gradient" 
python main_ig.py --ig_name square_integrated_gradient 
echo "Done"
