#run after superpixel (json) >> processing patch and save patches  
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
import shutil
import openslide

PROJECT_DIR = os.environ.get('PROJECT_DIR')
print("PROJECT DIR", PROJECT_DIR)
sys.path.append(PROJECT_DIR) 
 
 
from data.merge_dataset import SuperpixelDataset, PatchDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from utils import utils  
import os
import numpy as np
from scipy import ndimage
from PIL import Image, ImageFilter, ImageStat

SCORING_FUNCTION_MAP = {
    "get_scoring_do_nothing": get_scoring_do_nothing,
}

PRUNING_FUNCTION_MAP = {
    "get_pruning_do_nothing": get_pruning_do_nothing,
}   

PROJECT_DIR = os.environ.get('PROJECT_DIR')
sys.path.append(os.path.join(PROJECT_DIR))  


# example_list = ['normal_072', 'normal_001', 'normal_048', 'tumor_026', 'tumor_031', 'tumor_032']
example_list = ['normal_031', 'tumor_024', 'normal_047', 'tumor_009', 'tumor_057', 'normal_093', 'normal_051', 'tumor_014', 'tumor_015', 'tumor_067', 'normal_003', 'tumor_084', 'tumor_101', 'normal_148', 'normal_022', 'tumor_012', 'normal_039', 'normal_084', 'normal_101', 'tumor_010', 'normal_088', 'normal_155', 'normal_087', 'normal_016', 'normal_114', 'normal_024', 'tumor_048', 'normal_078', 'tumor_049', 'tumor_086']

model = timm.create_model('vit_base_patch16_224', pretrained=True)  # You can choose any model
model.eval()   



def load_config(config_file):
    # Load configuration from the provided YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_patches_with_updated_bboxes(
    region, mask, abs_bbox, patch_size=(224, 224), 
    coverage_threshold=0.5, 
    edge_threshold=20, 
    spixel_idx=None, 
    save_dir=None):
    """
    Extracts patches from the region, updates bounding boxes according to abs_bbox, and saves them to a specified folder.
    
    Parameters:
        region (np.array): The image region (numpy array).
        mask (np.array): The mask corresponding to the region (numpy array).
        abs_bbox (tuple): The absolute bounding box of the superpixel (xmin, ymin, width, height).
        patch_size (tuple): Size of the patches (height, width).
        coverage_threshold (float): The minimum percentage of the patch area that must be covered by the mask.
        edge_threshold (float): The minimum edge detection threshold for including the patch.
        save_dir (str): Directory to save the extracted patches.
    """
    region_height, region_width = region.shape[:2]
    patch_height, patch_width = patch_size

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create the directory if it doesn't exist
   
    patch_original_idx = 0  
    patch_idx = 0  # Initialize patch index
    
    xmin, ymin, w, h = abs_bbox  # Extract the superpixel's absolute bounding box values
    
    # Extracting patches
    for top in range(0, region_height, patch_height):
        for left in range(0, region_width, patch_width):
            # Ensure the patch is within bounds
            bottom = min(top + patch_height, region_height)
            right = min(left + patch_width, region_width)

            # Extract the patch and corresponding mask region (no changes in slicing)
            patch = region[top:bottom, left:right]
            patch_mask = mask[top:bottom, left:right]

            patch_area = patch.shape[0] * patch.shape[1]
            mask_coverage = np.sum(patch_mask) / patch_area  # Proportion of the patch covered by the mask

            # Only include patches that satisfy the coverage threshold
            if mask_coverage > coverage_threshold:
                edge_mean = filter_by_edge_detection(patch, patch_area)

                if edge_mean > edge_threshold:
                    # Update the bounding box position (keep the size of the patch the same)
                    # Calculate the new position relative to the superpixel's bounding box
                    adjusted_top = top + ymin
                    adjusted_left = left + xmin
                    adjusted_bottom = min(adjusted_top + patch_height, ymin + h)
                    adjusted_right = min(adjusted_left + patch_width, xmin + w)

                    bbox = (adjusted_top, adjusted_left, adjusted_bottom, adjusted_right)
                    bbox_str = f"{adjusted_top}_{adjusted_left}_{adjusted_bottom}_{adjusted_right}" 
                    # Save the patch as an image file
                    patch_img = Image.fromarray(patch.astype(np.uint8))  # Convert numpy array to PIL image
                    patch_filename = f"{bbox_str}_{spixel_idx}_{patch_original_idx}.png"
                    patch_img.save(os.path.join(save_dir, patch_filename))

                    # Save the updated bounding box and patch index
                    patch_idx += 1
            patch_original_idx += 1
            
    return patch_idx

    # print(f"Saved {patch_idx} patches to {save_dir}.") 
    
    
def filter_by_edge_detection(patch, patch_area):
    # Convert the NumPy array (patch) to a PIL image
    patch_pil = Image.fromarray(patch.astype(np.uint8))

    # Apply edge detection using PIL's ImageFilter.FIND_EDGES
    edge = patch_pil.filter(ImageFilter.FIND_EDGES)

    # Compute the sum of the edge values using ImageStat
    edge_stat = ImageStat.Stat(edge).sum
    edge_mean = np.mean(edge_stat) / patch_area  # Normalize by patch area

    return edge_mean 




def main(args):
    transform = transforms.Compose([
        transforms.Resize((args.patch_size, args.patch_size)),  # Resize the patch to 224x224
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
   
    from tqdm import tqdm
    
    for slide_index in tqdm(range(len(superpixel_dataset))):
        
        superpixel_datas, wsi_path = superpixel_dataset[slide_index]
        print(wsi_path)
        #slide = openslide.open_slide(wsi_path)  
        print(len(superpixel_datas))
        
        slide_basename = os.path.basename(wsi_path).split(".")[0]
        print("Basename:", slide_basename)
        
        save_dir = os.path.join(args.patch_path, slide_basename) 
        
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)  # Remove the existing directory and its contents
        os.makedirs(save_dir) 
        
        start_slide = time.time()
        total = 0 
        
        for each_superpixel in tqdm(superpixel_datas):
            foreground_idx = each_superpixel['foreground_idx'] 
            # print("Processing foreground:", foreground_idx)
            
            xywh_abs_bbox = each_superpixel['xywh_abs_bbox']
            superpixel_extrapolated = each_superpixel['superpixel_extrapolated']

            
            superpixel_np = utils.read_region_from_npy(
                args.spixel_path, 
                slide_basename, 
                foreground_idx
                )
             
            num_patch = save_patches_with_updated_bboxes(
                superpixel_np, 
                superpixel_extrapolated, 
                xywh_abs_bbox, 
                patch_size=(args.patch_size, args.patch_size), 
                coverage_threshold=0.5, 
                edge_threshold=20, 
                spixel_idx=foreground_idx, 
                save_dir=save_dir)

            total += num_patch 
            
        print(">>>>>> total patch in this slide: ", total)
        print("Finish after ", time.time()-start_slide)
        # print('Complete an Slide after: ', time.time()-start_slide)


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
        args.spixel_path = config.get('SPIXEL_PATH')
        args.patch_path = config.get('PATCH_PATH')
        
        os.makedirs(args.patch_path, exist_ok=True) 
        
        args.patch_size = config.get('patch_size')
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
    