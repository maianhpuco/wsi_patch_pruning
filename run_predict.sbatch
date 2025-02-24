#!/bin/bash

#SBATCH --job-name=predict
#SBATCH --output=predict.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
conda activate camil  

export PROJECT_DIR=$(pwd)

echo "run prediction on test" 
python main_predict.py 
echo "done prediction on test" 