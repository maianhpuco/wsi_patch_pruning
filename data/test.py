import os 
import sys
import torch 
from torch.utils.data import DataLoader
import glob 

PROJECT_DIR = os.environ.get('PROJECT_DIR') 
sys.path.append(os.path.join(PROJECT_DIR))
sys.path.append(os.path.join(PROJECT_DIR, "src"))
from data.data_loader import CustomDataset 


if __name__=='__main__':
    # Sample arguments for dataset
    args = type('', (), {})()  # Create a simple object to hold args
    args.label_file = os.path.join(PROJECT_DIR, "label_files/camelyon_data.csv")
    # args.file_path = os.path.join(PROJECT_DIR, "data/camelyon16_features")
    args.file_paths =  glob.glob(
       os.path.join(PROJECT_DIR, "data/camelyon16_features/h5_files/*.h5")  
    )
    dataset = CustomDataset(filenames=filenames, label_file=args.label_file, shuffle=True)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Iterate through the DataLoader
    for features, sparse_matrix, labels in dataloader:
        # print("Features Shape: ", features.shape)
        # print("Sparse Matrix Shape: ", sparse_matrix.shape)
        # print("Labels Shape: ", labels.shape)
        break  # Stop after the first batch for testing
