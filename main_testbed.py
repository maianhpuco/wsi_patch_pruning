import os
import sys 
import torch
import glob
import pandas as pd
import numpy as np
import argparse
import time   
import timm 
from data.merge_dataset import SuperpixelDataset, PatchDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from patch_merging import tome 
from utils import utils  

PROJECT_DIR = os.environ.get('PROJECT_DIR')
# SLIDE_DIR = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16'
# example_list = ['normal_072', 'normal_001', 'normal_048', 'tumor_026', 'tumor_031', 'tumor_032']
example_list =['normal_072', 'normal_048', 'tumor_026', 'tumor_031'] 

SLIDE_PATH = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16/images' #replace you path 
JSON_PATH = '/project/hnguyen2/mvu9/camelyon16/json_files' # replace your path 

 
sys.path.append(os.path.join(PROJECT_DIR))

model = timm.create_model('vit_base_patch16_224', pretrained=True)  # You can choose any model
model.eval()  

def main(): 
    wsi_paths = glob.glob(os.path.join(SLIDE_PATH, '*.tif'))
    wsi_paths = [path for path in wsi_paths if os.path.basename(path).split(".")[0] in example_list]
    json_folder = JSON_PATH  
    
    for wsi_path in wsi_paths: 
        print(wsi_path) 
        
        dataset = SuperpixelDataset(
            slide_path=wsi_path,
            json_folder=json_folder,
            )
        for (foreground_idx, xywh_abs_bbox, superpixel_extrapolated) in dataset:
            print(foreground_idx)
        break 
        
    
    
    
    
if __name__ == '__main__':
    main()
    # TODO
    # get argument: dry run only get 1 wsi, 1 superpixel, run thru from the start to the end 
    