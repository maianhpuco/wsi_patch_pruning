import os
import sys 
import torch
from tqdm import tqdm 
import glob
import pandas as pd
import numpy as np
import argparse
import time   
import shutil 
import h5py
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
    rescaling_stat_for_segmentation, 
    min_max_scale, 
    replace_outliers_with_bounds 
) 
from data.ig_dataset import IG_dataset 
import openslide
import glob 
import matplotlib.pyplot as plt
 
def main(args): 
    '''
    Input: h5 file
    Output: save scores into a json folder
    '''
    if args.ig_name=='integrated_gradients':
        from attr_method.integrated_gradient import IntegratedGradients as AttrMethod 
        attribution_method = AttrMethod() 
        print("Running for Integrated Gradient Attribution method")

    scores_dir = os.path.join(args.attribution_scores_folder, f'{args.ig_name}') 

    
    
    # print("Total number of sample in dataset:", len(dataset))
    all_scores_paths = glob.glob(os.path.join(scores_dir, "tumor_*.npy"))
    for idx, scores_path in enumerate(all_scores_paths):
        print(f"Print the plot {idx+1}/{len(all_scores_paths)}")
        print(scores_path)
        scores_array = np.load(scores_path)
        print("scores array shape", scores_array.shape)
        basename = os.path.basename(scores_path).split(".")[0]
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
        h5_file_path = os.path.join(args.features_h5_path, f'{basename}.h5') 
        
        result = {} 
        with h5py.File(h5_file_path, "r") as f:
            coordinates= f['coordinates'][:]
        scaled_scores = min_max_scale(replace_outliers_with_bounds(scores_array.copy()))

        
        plot_dir = os.path.join(args.plot_path, f'{args.ig_name}')
        if os.path.exists(plot_dir):
            shutil.rmtree(plot_dir)  # Delete the existing directory
        os.makedirs(plot_dir)  
        plot_path = os.path.join(plot_dir, f'{basename}.png')
        plot_heatmap_with_bboxes(
            scale_x, scale_y, new_height, new_width,
            coordinates,
            scaled_scores,
            name = "expected gradient, background mean of normal",
            save_path = plot_path
        ) 
        print("-> Save the plot at: ", plot_path)
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
        args.plot_path = config.get("PLOT_PATH")
        os.makedirs(args.plot_path, exist_ok=True)  
        os.makedirs(args.features_h5_path, exist_ok=True)  
        os.makedirs(args.attribution_scores_folder, exist_ok=True) 
        args.batch_size = config.get('batch_size')
        args.feature_extraction_model = config.get('feature_extraction_model')
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.ig_name = "integrated_gradients"
        
    main(args) 
    