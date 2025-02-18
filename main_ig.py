import os
import sys 
import torch
from tqdm import tqdm 
import glob
import pandas as pd
import numpy as np
import argparse
import time   
import timm 
# import yaml 
import random 

import numpy as np
from src.bag_classifier.mil_classifier import MILClassifier 

import torch
import torch.nn as nn 
import torch.optim as optim

import saliency.core as saliency
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS

import torch 
from torch.utils.data import DataLoader 

from src.bag_classifier.mil_classifier import MILClassifier # in the repo
from data.feature_dataset import FeaturesDataset  # in the repo
from utils.utils import load_config
from utils.train_classifier.train_mlclassifier import save_checkpoint, load_checkpoint
from utils.plotting import (
    plot_heatmap_with_bboxes,
    get_region_original_size,
    downscaling,
    rescaling_stat_for_segmentation) 
from data.ig_dataset import IG_dataset 

def load_model(checkpoint_path, model):
    input_dim = 768  # Adjust according to dataset
    mil_model = MILClassifier(input_dim=input_dim, pooling='attention')
    optimizer = optim.AdamW(mil_model.parameters(), lr=0.0005)

    model, _, _, _ = load_checkpoint(mil_model, optimizer, checkpoint_path)  
    return model 

def get_data(feature_h5_path):
    pass 

def get_baseline():
    pass 

def main(args): 
    '''
    Input: h5 file
    Output: save scores into a json folder
    '''
    if args.ig_name=='ig':
        from attr_method.integrated_gradient import get_ig 
        print("Running for Integrated Gradient Attribution method")
        #adding more args relating to the ig here 
        
    scores = get_ig()
    
    dataset = IG_dataset(
        args.features_h5_path,
        args.slide_path,
        )
    print("Get the baseline")
    import torch
    import numpy as np

    # Initialize accumulators
    feature_sum = None
    feature_sq_sum = None
    total_samples = 0

    # Iterate over dataset to compute mean and std in a memory-efficient way
    start = time.time()
    for i in tqdm(range(len(dataset)), desc="Computing the Mean and Std"):
        sample = dataset[i]
        features = torch.tensor(sample['features'], dtype=torch.float32)  # Convert to tensor

        if feature_sum is None:
            feature_sum = torch.zeros_like(features.sum(dim=0))
            feature_sq_sum = torch.zeros_like(features.sum(dim=0))

        feature_sum += features.sum(dim=0)
        feature_sq_sum += (features ** 2).sum(dim=0)
        total_samples += features.shape[0]  # Number of patches

    # Compute mean and std
    mean = feature_sum / total_samples
    std = torch.sqrt((feature_sq_sum / total_samples) - (mean ** 2))

    # Print results
    print("Mean shape:", mean.shape)
    print("Std shape:", std.shape)
    print("Duration", time.time()-start)
        
 
        
    print("Total number of sample in dataset:", len(dataset))
    for data in dataset:
        basename = data['basename']
        features = data['features']  # Shape: (batch_size, num_patches, feature_dim)
        label = data['label']
        patch_indices = data['patch_indices']
        coordinates = data['coordinates']
        spixel_idx = data['spixel_idx']

        # Run Integrated Gradients on batch
        # scores = get_ig(features, label, patch_indices, coordinates, spixel_idx)

        # print("IG Scores Shape:", scores.shape)
        print(basename)
        print(features.shape)
        break 
    
if __name__=="__main__":
    # get config 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=bool, default=False)
    parser.add_argument('--config_file', default='ma_exp002')
    args = parser.parse_args()
    
    if os.path.exists(f'./testbest_config/{args.config_file}.yaml'):
        config = load_config(f'./testbest_config/{args.config_file}.yaml')
        args.use_features = config.get('use_features', True)
        
        args.slide_path = config.get('SLIDE_PATH')
        args.json_path = config.get('JSON_PATH')
        args.spixel_path = config.get('SPIXEL_PATH')
        args.patch_path = config.get('PATCH_PATH') # save all the patch (image)
        args.features_h5_path = config.get("FEATURES_H5_PATH") # save all the features
         
        print(args.features_h5_path)
        
        os.makedirs(args.features_h5_path, exist_ok=True)  
        
        args.batch_size = config.get('batch_size')
        args.feature_extraction_model = config.get('feature_extraction_model')
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.ig_name = "ig"
    main(args) 