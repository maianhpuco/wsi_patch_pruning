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
    score_path = os.path.join(args.attribution_scores_folder, f'{args.ig_name}', f'{args.wsi_name}.npy')
    scores = np.load(score_path)
    print(scores.shape)
    
    h5_file_path = os.path.join(args.features_h5_path, f'{args.wsi_name}.h5')

    slide = openslide.open_slide(os.path.join(args.slide_path, f'{args.wsi_name}.tif'))
    (
        downsample_factor,
        new_width,
        new_height,
        original_width,
        original_height
    ) = rescaling_stat_for_segmentation(slide, downsampling_size=1096)

    scale_x = new_width / original_width
    scale_y = new_height / original_height

    with h5py.File(h5_file_path, "r") as f:
        coordinates = f['coordinates'][:]

    
    scaled_scores = min_max_scale(replace_outliers_with_bounds(scores.copy()))

    print(len(coordinates))
    print(len(scaled_scores))
    plot_heatmap_with_bboxes(
        scale_x, scale_y, new_height, new_width,
        coordinates,
        scaled_scores,
        name = "",
        color_bar=args.color_bar,
        show_plot=args.show_plot,
        save_path=os.path.join(args.plot_path, f'{args.ig_name}', f'{args.wsi_name}.png')
    )


if __name__ == '__main__':
    # get config 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config_file', default='ma_exp002')
    parser.add_argument('--color_bar', type=int, default=1)
    parser.add_argument('--show_plot', type=int, default=1)
    parser.add_argument('--wsi_name', type=str, default='tumor_026')
    parser.add_argument('--ig_name', 
                    default='integrated_gradients', 
                    choices=[
                        'integrated_gradient', 
                        'expected_gradient', 
                        'integrated_decision_gradient', 
                        'contrastive_gradient', 
                        'vanilla_gradient', 
                        'square_integrated_gradient', 
                        'optim_square_integrated_gradient'
                        ],
                    help='Choose the attribution method to use.') 
    
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
        if args.dry_run==1:
            args.attribution_scores_folder = config.get("SCORE_FOLDER_DRYRUN") 
            args.plot_path = config.get("PLOT_PATH_DRYRUN")    
        else: 
            args.attribution_scores_folder = config.get("SCORE_FOLDER")    
            args.plot_path = config.get("PLOT_PATH")
            
        print("Attribution folder path", args.attribution_scores_folder) 
        # args.attribution_scores_folder = config.get("SCORE_FOLDER")    
        # os.makedirs(args.features_h5_path, exist_ok=True)  
        os.makedirs(args.attribution_scores_folder, exist_ok=True) 
        
        args.batch_size = config.get('batch_size')
        args.feature_extraction_model = config.get('feature_extraction_model')
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        # args.ig_name = "integrated_gradients"
        
     
    main(args) 
