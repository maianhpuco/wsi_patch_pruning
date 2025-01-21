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

from data.merge_dataset import SuperpixelDataset, PatchDataset, SlidePatchesDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from utils import utils  
from src.importance_scores import get_scoring_do_nothing
from src.pruning import get_pruning_do_nothing

from src.bag_classifier.clam import CLAM_MB
from utils.train_classifier.train_clam import * 
from utils.utils import setup_logger


#TODO
# clean the code of the main function -> make it cleaner

SCORING_FUNCTION_MAP = {
    "get_scoring_do_nothing": get_scoring_do_nothing,
}


PRUNING_FUNCTION_MAP = {
    "get_pruning_do_nothing": get_pruning_do_nothing,
}   

def load_config(config_file):
    # Load configuration from the provided YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config



PROJECT_DIR = os.environ.get('PROJECT_DIR')
sys.path.append(os.path.join(PROJECT_DIR))  

# SLIDE_DIR = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16'
example_list = ['normal_072', 'normal_001', 'normal_048', 'tumor_026', 'tumor_031', 'tumor_032']



def main(args):
    if args.dry_run:
        print("Running the dry run")
    else:
        print("Running on full data") 
    
    model = timm.create_model(args.feature_extraction_model, pretrained=True)  # You can choose any model
    model = model.to(args.device) 
    model.eval()   
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the patch to 224x224
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
        # You can add other transformations like RandomHorizontalFlip, RandomRotation, etc.
    ])
    dataset = FeaturesDataset(
        feature_folder=args.features_h5_path
    )
    for features, patch_indices, label  in train_dataset:
        print(features.shape)
        print(patch_indices)
        
         
if __name__ == '__main__':
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
        
        args.scoring_function = SCORING_FUNCTION_MAP.get(
            config.get("scoring_function")
        )
        args.pruning_function = PRUNING_FUNCTION_MAP.get(
            config.get('pruning_function') 
        )
        args.batch_size = config.get('batch_size')
        args.feature_extraction_model = config.get('feature_extraction_model')
        
        # args.scoring_function("")
        # args.pruning_function("")
        
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main(args) 
    