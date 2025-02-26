#!/bin/bash

#SBATCH --job-name=check
#SBATCH --output=check.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00

conda activate camil  

export PROJECT_DIR=$(pwd) 

python main_check.py 

