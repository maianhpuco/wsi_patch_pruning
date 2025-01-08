import os
import sys 
import torch
import glob
import pandas as pd
import numpy as np
import argparse



PROJECT_DIR = os.environ.get('PROJECT_DIR')
SLIDE_DIR = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16'

sys.path.append(os.path.join(PROJECT_DIR))
sys.path.append(os.path.join(SLIDE_DIR))


def parse_arguments():
    """
    Parses the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Training configuration for CAMIL model")
    parser.add_argument('--dry_run', type=bool, default=False, help="test running okay? ")    
    # Dataset and paths
    parser.add_argument('--train_or_test', type=str, choices=["train", "test"], default='train', help="training or inferencing")  
    parser.add_argument('--dataset_name', type=str, choices=["camelyon16"], default='camelyon16', help="dataset name") 
    parser.add_argument('--input_shape', type=int, default=512, help="Input feature dimension (default: 512)")
    parser.add_argument('--n_classes', type=int, default=2, help="Number of output classes (default: 2)")
    parser.add_argument('--subtyping', type=bool, default=False, help="Whether to use subtyping (default: False)")
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs to train (default: 10)")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate (default: 1e-3)")
    parser.add_argument('--checkpoint_filename', type=str, default=None)
    # Device (GPU/CPU)
    parser.add_argument('--device', type=str, choices=["cpu", "cuda"], default="cuda", help="Device for training (default: cuda)")

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print(f"Training for dataset {format(args.dataset_name)}")
    train_or_test_or_val = 'train'

    if args.dataset_name == 'camelyon16':     
        args.feature_folder =os.path.join(SLIDE_DIR)  
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
 
    import datetime

    
if __name__ == '__main__':
    main()
#checkpoint_path camelyon16_20241118-1932_completed.pth
# /project/hnguyen2/mvu9/camelyon16_features_data

# cp -r /home/mvu9/digital_pathology/camil_pytorch/data/camelyon16_features/* /project/hnguyen2/mvu9/camelyon16_features_data/ 
  