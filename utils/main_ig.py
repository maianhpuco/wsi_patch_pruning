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
import yaml 
 
import numpy as np
from src.bag_classifier.mil_classifier import MILClassifier 

import torch
import torch.nn as nn 
import torch.optim as optim

import saliency.core as saliency
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS

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
    pass
    if args.ig_name=='vanilla_gradient':
        from attr_method.integrated_gradient import get_ig 
        print("Running for Integrated Gradient Attribution method")
    scores = get_ig()
    dataset = IG_dataset(
        args.features_h5_path,
        args.slide_path,
        )
    print("Total number of sample in dataset:", len(dataset))


    
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
        