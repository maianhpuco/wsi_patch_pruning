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
    plot_wsi, 
) 
import openslide
import glob 
import matplotlib.pyplot as plt
 
def main(args): 
    '''
    Input: h5 file
    Output: save scores into a json folder
    '''
    
    slide_paths = glob.glob(os.path.join( args.slide_path, "*.tif"))  
    print("Image will be plotted at:", args.plot_slide_dir)
    
    if os.path.exists(args.plot_slide_dir):
        shutil.rmtree(args.plot_slide_dir)  # Delete the existing directory
    os.makedirs(args.plot_slide_dir)   
    
    for idx, slide_path in enumerate(slide_paths):
        print(f"Print the plot {idx+1}/{len(slide_paths)}")
        basename = os.path.basename(slide_path).split(".")[0] 
        
        plot_wsi(basename, args.slide_path, args.plot_slide_dir)
        print(f"-> Save the plot {basename} at: ", args.plot_slide_dir)        # scale_x, scale_y, new_height, new_width  


if __name__ == '__main__':
    # get config 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config_file', default='ma_exp002')

    args = parser.parse_args()
    
    if os.path.exists(f'./testbest_config/{args.config_file}.yaml'):
        config = load_config(f'./testbest_config/{args.config_file}.yaml')
        args.use_features = config.get('use_features', True)
        args.slide_path = config.get('SLIDE_PATH')
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.gt_path = config.get("GROUND_TRUTH_PATH")
        args.plot_slide_dir = config.get("PLOT_SLIDE") 
        

     
    main(args) 
    
    #scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/plotting_slide
    #scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/plotting_annotation   