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

import openslide

from data.merge_dataset import SuperpixelDataset, PatchDataset, SlidePatchesDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from patch_merging import tome 
from utils import utils  
from testbed.importance_scores import get_scoring_do_nothing
from testbed.pruning import get_pruning_do_nothing

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

model = timm.create_model('vit_base_patch16_224', pretrained=True)  # You can choose any model
model.eval()  

def main(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the patch to 224x224
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
        # You can add other transformations like RandomHorizontalFlip, RandomRotation, etc.
    ])
    if args.dry_run:
        print("Running the dry run")
    else:
        print("Running on full data")
    start_slide = time.time()
    
    wsi_paths = glob.glob(os.path.join(args.slide_path, '*.tif'))
    wsi_paths = [path for path in wsi_paths if os.path.basename(path).split(".")[0] in example_list]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),          # Convert the image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
    ]) 
    
    for wsi_path in wsi_paths:
        start_slide = time.time()
        slide_basename = os.path.basename(wsi_path).split(".")[0]
        slide_patch_dataset = SlidePatchesDataset(
            patch_dir = os.path.join(args.patch_path, slide_basename),
            transform = transform
        )
        dataloader = DataLoader(slide_patch_dataset, batch_size=args.batch_size, shuffle=True)

        for batch in dataloader:
            batch_image = batch['image']
            batch_patch_info = batch['patch_info']
            
            parsed_batch_info = [] 
            for i in range(args.batch_size):
                parsed_info = {
                    'ymin': batch_patch_info['ymin'][i].item(),
                    'ymax': batch_patch_info['ymax'][i].item(),
                    'xmin': batch_patch_info['xmin'][i].item(),
                    'xmax': batch_patch_info['xmax'][i].item(),
                    'spixel_idx': batch_patch_info['spixel_idx'][i].item(),
                    'patch_idx': batch_patch_info['patch_idx'][i].item()
                }
                parsed_batch_info.append(parsed_info)

            # Print the parsed batch info
            print("Parsed Batch Info:", parsed_batch_info)
            print("Batch Image Shape:", batch_image.shape) 
    
    
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
        args.patch_path = config.get('PATCH_PATH')
        
        args.scoring_function = SCORING_FUNCTION_MAP.get(
            config.get("scoring_function")
        )
        args.pruning_function = PRUNING_FUNCTION_MAP.get(
            config.get('pruning_function') 
        )
        args.batch_size = config.get('batch_size')
        args.scoring_function("")
        args.pruning_function("")
    
    main(args) 
    