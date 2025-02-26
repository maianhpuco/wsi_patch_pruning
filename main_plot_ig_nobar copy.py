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
    plot_heatmap_with_bboxes_nobar,
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
    
    gt_path = os.path.join(args.plot_path, 'ground_truth') 
    print(">> Ground truth count", len(glob.glob(os.path.join(gt_path, "*.png")))) 
     
    for method in os.listdir(args.plot_path):
        print("Method: ", method)
        all_scores_paths = glob.glob(os.path.join(
            args.plot_path, method, "*.png"))
        print("Number of file in the file", len(all_scores_paths)) 
        
        


if __name__ == '__main__':
    # get config 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config_file', default='ma_exp002')
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
            args.plot_path = config.get("PLOT_NOBAR_PATH")
            
        print("Attribution folder path", args.attribution_scores_folder) 
        # args.attribution_scores_folder = config.get("SCORE_FOLDER")    
        # os.makedirs(args.features_h5_path, exist_ok=True)  
        os.makedirs(args.attribution_scores_folder, exist_ok=True) 
        
        args.batch_size = config.get('batch_size')
        args.feature_extraction_model = config.get('feature_extraction_model')
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        # args.ig_name = "integrated_gradients"
        
     
    main(args) 
    
    # python main_plot_ig.py --ig_name=integrated_decision_gradient --dry_run=1   
    # python main_plot_ig.py --ig_name=contrastive_gradient --dry_run=1  
    # python main_plot_ig.py --ig_name=integrated_gradient --dry_run=1   
    
    # get the plot
    # scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/plotting/integrated_decision_gradient/tumor_026.png . 
    # scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/plotting/vanilla_gradients/tumor_026.png .
    # scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/plotting/contrastive_gradient/tumor_026.png .   
    # scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/plotting/integrated_gradient/tumor_026.png .
    # scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16/evaluation/reference.csv . i
    
    
    # scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/plotting_dryrun  .
    # scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/plotting_dryrun/optim_square_integrated_gradient .
    # scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/ground_truth_mask . 
    
    #scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/attribution_scores/ .
    
    #scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/pred_on_testset.csv  . 
    
    #scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/ground_truth_mask_ver2 . 
    #scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/attribution_scores_ver2 .
    #scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/features_h5_files .   
        