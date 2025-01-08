import torch
from torch.utils.data import Dataset
import h5py
import os
import sys 
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import glob
import openslide 
import json 

example_list = ['normal_072', 'normal_001', 'normal_048', 'tumor_026', 'tumor_031', 'tumor_032']
SLIDE_PATH = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16/images'
JSON_PATH = '/project/hnguyen2/mvu9/camelyon16/json_files'

class SuperpixelDataset(Dataset):
    def __init__(self, slide_root, superpixel_root, basename):
        self.slide = None 
        self.basename = os.path.basename(slide_path)
    
    def __getitem__(self, index):
        
        return None  
    
class PatchDataset(Dataset):
    def __init__(self):
        pass 
    def __getitem__(self, idx):
        pass 

if __name__ == '__main__':
    wsi_paths = glob.glob(os.path.join(SLIDE_PATH, '*.tif'))
    wsi_paths = [path for path in wsi_paths if os.path.basename(path).split(".")[0] in example_list]
    
    for wsi_path in wsi_paths: 
        basename = os.path.basename(wsi_path).split(".")[0]
        print(wsi_path)
        
        slide = openslide.open_slide(wsi_path)
        print(slide.dimensions)
        
        json_path = os.path.join(JSON_PATH, f'{basename}.json')
        print(json_path)
        with open(json_path, 'r') as json_file:
            loaded_data = json.load(json_file)

            superpixel_labels = np.array(loaded_data['superpixel_labels'])
            downscaled_region_array = np.array(loaded_data['downscaled_region_array'])
            output_image_with_bboxes = np.array(loaded_data['output_image_with_bboxes'])

            # Convert float values in 'foreground_superpixels' and 'background_superpixels' back to integers
            foreground_superpixels = [int(i) for i in loaded_data['foreground_superpixels']]
            background_superpixels = [int(i) for i in loaded_data['background_superpixels']]

            # Convert the 'bounding_boxes' keys back to integers and values back to tuples
            bounding_boxes = {int(k): tuple(v) for k, v in loaded_data['bounding_boxes'].items()}

            # Scalar values remain as they are
            downsample_factor = loaded_data['downsample_factor']
            new_width = loaded_data['new_width']
            new_height = loaded_data['new_height']

            # Print the loaded data to verify
            print(f"Superpixel Labels: {superpixel_labels.shape}")
            print(f"Downscaled Region Array: {downscaled_region_array.shape}")
            print(f"Output Image with BBoxes: {output_image_with_bboxes.shape}")
            print(f"Foreground Superpixels: {foreground_superpixels[:5]}")  # First 5 for brevity
            print(f"Background Superpixels: {background_superpixels[:5]}")  # First 5 for brevity
            print(f"Bounding Boxes: {list(bounding_boxes.keys())[:2]}")  # First 2 keys for brevity
            print(f"Downsample Factor: {downsample_factor}, Width: {new_width}, Height: {new_height}") 
                # reading superpixel 
            
            
        break
            