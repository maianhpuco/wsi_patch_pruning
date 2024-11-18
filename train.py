import os
import sys 
import torch
import glob
import pandas as pd
import numpy as np
import argparse



PROJECT_DIR = os.environ.get('PROJECT_DIR') 
sys.path.append(os.path.join(PROJECT_DIR))
sys.path.append(os.path.join(PROJECT_DIR, "utils"))
sys.path.append(os.path.join(PROJECT_DIR, "src")) 
sys.path.append(os.path.join(PROJECT_DIR, "datase_utils")) 


from src.camil import CAMIL  # Assuming CAMIL is the model class from camil.py
from data.camelyon16_dataset import CustomDataset

from utils.utils import train




def parse_arguments():
    """
    Parses the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Training configuration for CAMIL model")
    
    # Dataset and paths
    parser.add_argument('--dataset_name', type=str, choices=["camelyon16"], default='camelyon16', help="dataset name") 
    parser.add_argument('--input_shape', type=int, default=512, help="Input feature dimension (default: 512)")
    parser.add_argument('--n_classes', type=int, default=2, help="Number of output classes (default: 2)")
    parser.add_argument('--subtyping', type=bool, default=False, help="Whether to use subtyping (default: False)")
    # Training parameters
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs to train (default: 10)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    
    # Device (GPU/CPU)
    parser.add_argument('--device', type=str, choices=["cpu", "cuda"], default="cuda", help="Device for training (default: cuda)")

    return parser.parse_args()

def main():

    # Parse arguments
    args = parse_arguments()
    print(f"Training for dataset {format(args.dataset_name)}")
    
    if args.dataset_name == 'camelyon16':     
        args.label_file = os.path.join(PROJECT_DIR, "label_files/camelyon_data.csv")
        args.split_paths = os.path.join(PROJECT_DIR, "data/camelyon_csv_splits")
        args.file_paths =  glob.glob(
            os.path.join(
                PROJECT_DIR, 
                "data/camelyon16_features/h5_files/*.h5")  
        )
        
        
    # Initialize the model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
 
    model = CAMIL(args)
    model.to(device)   

    # Create dataset
    dataset = CustomDataset(file_paths=args.file_paths, label_file=args.label_file, shuffle=True)

    # # Set device (GPU or CPU)
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # # Call the train function
    train(model, dataset, epochs=10, learning_rate=1e-3, device=device)

if __name__ == '__main__':
    main()
