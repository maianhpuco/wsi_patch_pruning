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
import shutil 

import random 
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader 
from utils.utils import load_config
from utils.plotting import (
    plot_heatmap_with_bboxes,
    get_region_original_size,
    downscaling,
    rescaling_stat_for_segmentation
) 
from data.ig_dataset import IG_dataset 
import openslide


def main(args): 
    '''
    Input: h5 file
    Output: save scores into a json folder
    '''
    
    if args.ig_name=='integrated_gradients':
        score_save_path = os.path.join(args.attribution_scores_folder, 'integrated_gradient')
        print("score_save_path", score_save_path)
    else:
        print("No attribution method is valid")
        dataset = IG_dataset(
        args.features_h5_path,
        args.slide_path,
        )
    print("Total number of sample in dataset:", len(dataset))
    
    for idx, data in enumerate(dataset):
        total_file = len(dataset)
        print(f"Processing the file numner {idx+1}/{total_file}")
        basename = data['basename']
        features = data['features']  # Shape: (batch_size, num_patches, feature_dim)
        label = data['label']
        patch_indices = data['patch_indices']
        coordinates = data['coordinates']
        spixel_idx = data['spixel_idx']  
        slide = openslide.open_slide(os.path.join(args.slide_path, f'{basename}.tif'))
        (
            downsample_factor,
            new_width,
            new_height,
            original_width,
            original_height
        ) = rescaling_stat_for_segmentation(
            slide, downsampling_size=1096)

        scale_x = new_width / original_width
        scale_y = new_height / original_height
        
        # scale_x, scale_y, new_height, new_width  


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
        args.checkpoints_dir = config.get("CHECKPOINT_PATH")
        args.attribution_scores_folder = config.get("SCORE_FOLDER")    
        os.makedirs(args.features_h5_path, exist_ok=True)  
        os.makedirs(args.attribution_scores_folder, exist_ok=True) 
        args.batch_size = config.get('batch_size')
        args.feature_extraction_model = config.get('feature_extraction_model')
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.ig_name = "integrated_gradients"
        
    main(args) 
    