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
from data.feature_dataset import FeaturesDataset  # in the repo
from utils.train_classifier.train_mlclassifier import (
    save_checkpoint, 
    load_checkpoint, 
    train_mil_classifier
)

from utils.plotting import (
    plot_heatmap_with_bboxes,
    get_region_original_size,
    downscaling,
    rescaling_stat_for_segmentation, 
    min_max_scale,
    replace_outliers_with_bounds)


def load_config(config_file):
    # Load configuration from the provided YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

 
def main(args):
    
    example_list = [i for i in args.features_h5_path if i.split(".h5")[0].split("_")[0] != "test"]
    
    print(len(example_list))
    print("- total files", len(example_list))
    train_num = int(len(example_list)*0.8)
    random.shuffle(example_list)  # Shuffle the list in-place
    train_files = example_list[:train_num]  # First 80% as train
    test_files = example_list[train_num:]  # Remaining 20% as test

    train_list = [i.split(".h5")[0] for i in train_files]
    test_list = [i.split(".h5")[0] for i in test_files]

    print("Prepare the dataset for training")
    
    train_dataset = FeaturesDataset(
        feature_folder = args.features_h5_path,
        basename_list = train_list,
        transform=None
    )

    test_dataset = FeaturesDataset(
        feature_folder = args.features_h5_path,
        basename_list = test_list,
        transform=None
    )

    print("train dataset", len(train_dataset))
    print("test dataset", len(test_dataset)) 


    print("Start testing")
    
    # Define model & optimizer
    input_dim = 768  # Adjust according to dataset
    mil_model = MILClassifier(input_dim=input_dim, pooling='attention')
    optimizer = optim.AdamW(mil_model.parameters(), lr=0.0005)

    # Define checkpoint path
    checkpoint_path = os.path.join(args.checkpoint_folder, 'mil_checkpoint.pth')
    model, optimizer, start_epoch, best_auc = load_checkpoint(mil_model, optimizer, checkpoint_path)

    print("--- check the model ----")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_features = torch.randn(5000, 768)
    test_features = test_features.to(device)

    model.to(device)
    model.eval()  # Set to evaluation mode

    with torch.no_grad():
        bag_output = model(test_features, [test_features.shape[0]])  # Forward pass
    print(bag_output.shape)
    
    print("- Start training")
    
    train_mil_classifier(
        model, 
        train_dataset, 
        test_dataset, 
        num_epochs=30, 
        batch_size=32, 
        checkpoint_path=checkpoint_path
        )

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
        
    os.makedirs(args.checkpoint_folder, exist_ok=True)   
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    main(args)