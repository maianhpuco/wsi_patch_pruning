import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from utils.utils import load_config
from utils.plotting import (
    plot_heatmap_with_bboxes,
    get_region_original_size,
    downscaling,
    rescaling_stat_for_segmentation, 
    min_max_scale, 
    replace_outliers_with_bounds 
) 





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
 
# 'integrated_gradient', 
# 'expected_gradient', 
# 'integrated_decision_gradient', 
# 'contrastive_gradient', 
# 'vanilla_gradient', 
# 'square_integrated_gradient', 
# 'optim_square_integrated_gradient'

def main(args): 
    # List of image paths and titles
    image_paths = [
        os.path.join(args.plot_path, f'{args.wsi_name}_raw.png'),
        os.path.join(args.plot_path, f'{args.wsi_name}_ground_truth.png'),
        os.path.join(args.plot_path, f'integrated_gradient/{args.wsi_name}.png'),
        os.path.join(args.plot_path, f'expected_gradient/{args.wsi_name}.png'),
        os.path.join(args.plot_path, f'integrated_decision_gradient/{args.wsi_name}.png'),
        os.path.join(args.plot_path, f'contrastive_gradient/{args.wsi_name}.png'),
        os.path.join(args.plot_path, f'vanilla_gradient/{args.wsi_name}.png'),
        os.path.join(args.plot_path, f'square_integrated_gradient/{args.wsi_name}.png'),
        os.path.join(args.plot_path, f'optim_square_integrated_gradient/{args.wsi_name}.png'),
    ]
    titles = ["Raw", "Ground Truth", "Integrated Gradient", "Expected Gradient", "Integrated Decision Gradient", "Contrastive Gradient", "Vanilla Gradient", "Square Integrated Gradient", "Optimized Square Integrated Gradient"]

    # Load the first image to set the target size for all images
    first_image = Image.open(image_paths[0])
    target_size = first_image.size  # (width, height)

    num_images = len(image_paths)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 4))

    for i, (ax, img_path, title) in enumerate(zip(axes, image_paths, titles)):
        img = Image.open(img_path).resize(target_size, Image.LANCZOS)
        img_with_border = ImageOps.expand(img, border=1, fill="black")

        # Example: if you highlight the last column
        if i == num_images - 1:
            img_with_border = ImageOps.expand(img, border=7, fill="red")
            # Convert to RGBA for blending
            img_rgba = img_with_border.convert("RGBA")
            # Create an overlay of color #e3c6c6 with 30% opacity
            overlay = Image.new("RGBA", img_rgba.size, (227, 198, 198, int(255 * 0.3)))

        ax.imshow(img_with_border)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Remove all spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    plt.savefig(os.path.join(args.plot_path, f'{args.wsi_name}_all.png'), bbox_inches='tight', dpi=100)
    if args.show_plot:
        plt.show()
    
        


if __name__ == '__main__':
    # get config 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config_file', default='ma_exp002')
    parser.add_argument('--show_plot', type=int, default=0)
    parser.add_argument('--wsi_name', type=str, default='tumor_026')
    
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
        args.ground_truth_path = config.get("GROUND_TRUTH_PATH")
        if args.dry_run==1:
            args.attribution_scores_folder = config.get("SCORE_FOLDER_DRYRUN") 
            args.plot_path = config.get("PLOT_PATH_DRYRUN")    
        else: 
            args.attribution_scores_folder = config.get("SCORE_FOLDER")    
            args.plot_path = config.get("PLOT_PATH")
            
        os.makedirs(args.features_h5_path, exist_ok=True)  
        os.makedirs(args.attribution_scores_folder, exist_ok=True) 
        
        args.batch_size = config.get('batch_size')
        args.feature_extraction_model = config.get('feature_extraction_model')
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
     
    main(args) 