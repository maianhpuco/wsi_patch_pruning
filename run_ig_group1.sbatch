#!/bin/bash

#SBATCH --job-name=group1
#SBATCH --output=group1.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00

conda activate camil  

export PROJECT_DIR=$(pwd) 

echo "Group 1 : integrated_gradient, expected_gradient"
echo "Running with ig_name=integrated_gradient"
python main_ig.py --ig_name integrated_gradient

echo "Running with ig_name=expected_gradient" 
python main_ig.py --ig_name expected_gradient 

echo "All tasks (group 1) integrated_gradient, expected_gradient."
    if args.ig_name=='integrated_gradient':
        from attr_method.integrated_gradient import IntegratedGradients as AttrMethod 
       
    elif args.ig_name=='vanilla_gradient':
        from attr_method.vanilla_gradient import VanillaGradients as AttrMethod 
    
    elif args.ig_name=='contrastive_gradient':
        from attr_method.contrastive_gradient import ContrastiveGradients as AttrMethod 

    elif args.ig_name=='expected_gradient':
       from attr_method.expected_gradient import ExpectedGradients as AttrMethod   
    
    elif args.ig_name=='integrated_decision_gradient':
       from attr_method.integrated_decision_gradient import IntegratedDecisionGradients as AttrMethod     
  
    elif args.ig_name=='optim_square_integrated_gradient':
       from attr_method.optim_square_integrated_gradient import OptimSquareIntegratedGradients as AttrMethod
    
    elif args.ig_name=='square_integrated_gradient':
       from attr_method.square_integrated_gradient import SquareIntegratedGradients as AttrMethod      