import os
import sys 
from tqdm import tqdm 
import glob
import pandas as pd
import numpy as np
import argparse
import time   
import timm 
import yaml 
import h5py 
import openslide

import torch
import torch.nn as nn
import torch.optim as optim  
import random 

from src.bag_classifier.mil_classifier import MILClassifier # in the repo
from data.full_feature_dataset import FeaturesDataset  # in the repo
from utils.train_classifier.train_mlclassifier import (
    FocalLoss, 
    load_checkpoint,collate_mil_fn,  
    evaluate_mil_classifier, 
    predict_and_save
)
from torch.utils.data import DataLoader


def load_config(config_file):
    # Load configuration from the provided YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_mean_std_for_normal_dist(feature_folder, basename_list, save_path):
    paths = glob.glob(os.path.join(feature_folder, '*.h5'))  # Get all HDF5 files
    paths = [i for i in paths if os.path.basename(i).split(".h5")[0] in basename_list] 
    # Initialize accumulators
    feature_sum = None
    feature_sq_sum = None
    total_samples = 0
    start = time.time()
    
    print("Start computing the mean and std for normalization...")

    for file_path in tqdm(paths, desc="Computing the Mean and Std"):
        with h5py.File(file_path, "r") as f:
            features = f['features'][:]  # Extract features
            
        features = torch.tensor(features, dtype=torch.float32)  # Convert to tensor

        # Initialize accumulators on first iteration
        if feature_sum is None:
            feature_sum = torch.zeros_like(features.sum(dim=0))
            feature_sq_sum = torch.zeros_like(features.sum(dim=0))

        feature_sum += features.sum(dim=0)
        feature_sq_sum += (features ** 2).sum(dim=0)
        total_samples += features.shape[0]  # Number of patches

    # Compute mean and std
    mean = feature_sum / total_samples
    std = torch.sqrt((feature_sq_sum / total_samples) - (mean ** 2))

    print(f"Completed computation in {time.time() - start:.2f} seconds.")
    print(">>>>> mean and std shape", mean.shape, std.shape)
    # Save to HDF5 (convert to NumPy before saving)
    with h5py.File(save_path, "w") as f:
        f.create_dataset("mean", data=mean.numpy())
        f.create_dataset("std", data=std.numpy())

    print(f"Mean and Std saved to {save_path}")  
    return mean, std


def main(args):
    if args.recompute_mean_std:
        train_val_files = [i for i in os.listdir(args.features_h5_path) if i.split(".h5")[0].split("_")[0] != "test"]
        train_val_list = [i.split(".h5")[0] for i in train_val_files] 
        mean, std = get_mean_std_for_normal_dist(
            args.features_h5_path,
            train_val_list, 
            args.feature_mean_std_path
            )
    else:
        with h5py.File(args.feature_mean_std_path, "r") as f:
            mean = torch.tensor(f["mean"][:], dtype=torch.float32)
            std = torch.tensor(f["std"][:], dtype=torch.float32) 
         
    test_dataset = FeaturesDataset(
        feature_folder=args.features_h5_path,
        split_csv=args.split_path, 
        transform=None, 
        mean=mean,
        std=std, 
        dataset_type='test' 
    )   
    print("example test ")
    print(test_dataset[0].keys())
    print("- test dataset" , len(test_dataset)) 
    
    # Define model & optimizer
    input_dim = 768  # Adjust according to dataset
    model = MILClassifier(input_dim=input_dim, pooling='attention')
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Define checkpoint path
    checkpoint_path = os.path.join(args.checkpoint_folder, CHECK_POINT_FILE) 

    print("--- check the model ----")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    mil_model, optimizer, start_epoch, best_auc = load_checkpoint(
        model, optimizer, checkpoint_path) 
    
    criterion = FocalLoss(gamma=1, alpha=0.8)
    mil_model.to(device)
    mil_model.eval() 
     

    
    print("------Run the evaluation on test set") 
    predict_and_save(
        mil_model, 
        test_dataset, 
        criterion, 
        device, 
        output_file=args.pred_path)
    
    
    print("------>>>>>> Done the evaluation on test set") 
    
    test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, collate_fn=collate_mil_fn)  
    test_loss, test_acc, test_auc = evaluate_mil_classifier(
        mil_model, 
        test_loader, 
        criterion, 
        device) 

   

if __name__ == '__main__': 
    arg_file_name = 'ma_exp002' 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dry_run', type=bool, default=False)
    parser.add_argument('--config_file', default=arg_file_name)
    args = parser.parse_args()
    
    if os.path.exists(f'./testbest_config/{args.config_file}.yaml'):
        config = load_config(f'./testbest_config/{args.config_file}.yaml')
        args.use_features = config.get('use_features', True)
        args.slide_path = config.get('SLIDE_PATH')
        args.json_path = config.get('JSON_PATH')
        args.spixel_path = config.get('SPIXEL_PATH')
        args.patch_path = config.get('PATCH_PATH') # save all the patch (image)
        args.features_h5_path = config.get("FEATURES_H5_PATH") # save all the features
        args.batch_size = config.get('batch_size')
        args.feature_extraction_model = config.get('feature_extraction_model')
        args.checkpoint_folder = config.get('CHECKPOINT_PATH')
        args.feature_mean_std_path = config.get("FEATURE_MEAN_STD_PATH")
        args.split_path = config.get("SPLIT_PATH")
        
        args.recompute_mean_std = False
        args.pred_path = config.get("PRED_PATH") 
        
    # os.makedirs(args.checkpoint_folder, exist_ok=True)    
    # os.makedirs(os.path.dirname(args.feature_mean_std_path), exist_ok=True)   
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # CHECK_POINT_FILE = 'mil_checkpoint.pth' 
    # CHECK_POINT_FILE = 'mil_checkpoint_draft.pth'
    CHECK_POINT_FILE = 'mil_checkpoint_official.pth'  
    
    main(args)