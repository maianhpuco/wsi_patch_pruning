import os
import sys 
import torch
import glob
import pandas as pd
import numpy as np
import argparse
import time   
import timm 
import yaml 

import openslide
import zipfile
 
from data.merge_dataset import SuperpixelDataset, PatchDataset
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
# SLIDE_DIR = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16'
example_list = ['normal_072', 'normal_001', 'normal_048', 'tumor_026', 'tumor_031', 'tumor_032']
# example_list =['normal_048', 'normal_001', 'tumor_026', 'tumor_031'] 
# example_list=['normal_072']


 
 
sys.path.append(os.path.join(PROJECT_DIR))

model = timm.create_model('vit_base_patch16_224', pretrained=True)  # You can choose any model
model.eval()  

def save_region_as_npy(region_np, slide_basename, superpixel_name):
    # Create a directory for the slide if it doesn't exist
    save_dir = os.path.join("read_folder", slide_basename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for the numpy file
    npy_file_path = os.path.join(save_dir, f"{superpixel_name}.npy")
    
    # Save the region as a numpy file
    np.save(npy_file_path, region_np)

    # Optionally, zip the file (if needed)
    zip_file_path = npy_file_path + ".zip"
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(npy_file_path, os.path.basename(npy_file_path))
        # Optionally remove the .npy file after zipping
        os.remove(npy_file_path)
 

def main(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the patch to 256x256
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
    json_folder = args.json_path
    
    superpixel_dataset = SuperpixelDataset(
        slide_paths=wsi_paths,
        json_folder=json_folder,
        )
    print("Number of slide in dataset:", len(superpixel_dataset)) 
    
    for slide_index in range(len(superpixel_dataset)):
        superpixel_datas, wsi_path = superpixel_dataset[slide_index]
        print(wsi_path)
        slide_basename = os.path.basename(wsi_path).split(".")[0]
        print(slide_basename)
        
        slide = openslide.open_slide(wsi_path)  
        print(len(superpixel_datas))
        for each_superpixel in superpixel_datas:
            foreground_idx = each_superpixel['foreground_idx'] 
            xywh_abs_bbox = each_superpixel['xywh_abs_bbox']
            superpixel_extrapolated = each_superpixel['superpixel_extrapolated']
            
            start = time.time()
            region = utils.get_region_original_size(slide, xywh_abs_bbox)
            region_np = np.array(region)
            print(f"Slicing time: {time.time() - start} seconds")  

            # Save the region as a NumPy file and zip it
            save_region_as_npy(region_np, slide_basename, f"foreground_idx") 
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=bool, default=False)
    parser.add_argument('--config_file', default='ma_exp001')
    args = parser.parse_args()
    
    if os.path.exists(f'./testbest_config/{args.config_file}.yaml'):
        config = load_config(f'./testbest_config/{args.config_file}.yaml')
        args.use_features = config.get('use_features', True)
        args.slide_path = config.get('SLIDE_PATH')
        args.json_path = config.get('JSON_PATH')
        args.scoring_function = SCORING_FUNCTION_MAP.get(
            config.get("scoring_function")
        )
        args.pruning_function = PRUNING_FUNCTION_MAP.get(
            config.get('pruning_function') 
        )

        args.scoring_function("")
        args.pruning_function("")
    
    main(args) 
     