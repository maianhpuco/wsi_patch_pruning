#!/bin/bash

#SBATCH --job-name=groundtruth
#SBATCH --output=groundtruth.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
conda activate camil  

export PROJECT_DIR=$(pwd)

echo "run the ground truth folder" 
python metrics_segmentation/main_interior_mask.py 
echo "done running ground truth folder" 