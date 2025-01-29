
import os
import sys 
from tqdm import tqdm 
import glob
import pandas as pd
import numpy as np
import argparse
import time   
import yaml 
import json 
import glob 
import h5py 
import openslide
import numpy as np
import cv2
from skimage import segmentation
from skimage import color 
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries 
PROJECT_DIR = os.environ.get('PROJECT_DIR')
print("PROJECT DIR", PROJECT_DIR)
sys.path.append(PROJECT_DIR) 

import torch
import torch.nn as nn
import torch.optim as optim 
from data.merge_dataset import SuperpixelDataset, PatchDataset, SlidePatchesDataset
from PIL import Image
from utils import utils  

from torchvision import transforms 

wsi_basenames = ['normal_031', 'tumor_024', 'normal_047', 'tumor_009', 'tumor_057', 'normal_093', 'normal_051', 'tumor_014', 'tumor_015', 'tumor_067', 'normal_003', 'tumor_084', 'tumor_101', 'normal_148', 'normal_022', 'tumor_012', 'normal_039', 'normal_084', 'normal_101', 'tumor_010', 'normal_088', 'normal_155', 'normal_087', 'normal_016', 'normal_114', 'normal_024', 'tumor_048', 'normal_078', 'tumor_049', 'tumor_086']



def load_config(config_file):
    # Load configuration from the provided YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def rescaling_stat_for_segmentation(obj, downsampling_size=1024):
    """
    Rescale the image to a new size and return the downsampling factor.
    """
    if hasattr(obj, 'shape'):
        original_width, original_height = obj.shape[:2]
    elif hasattr(obj, 'size'):  # If it's an image (PIL or similar)
        original_width, original_height = obj.size
    elif hasattr(obj, 'dimensions'):  # If it's a slide (e.g., a TIFF object)
        original_width, original_height = obj.dimensions
    else:
        raise ValueError("The object must have either 'size' (image) or 'dimensions' (slide) attribute.")

    if original_width > original_height:
        downsample_factor = int(downsampling_size * 100000 / original_width) / 100000
    else:
        downsample_factor = int(downsampling_size * 100000 / original_height) / 100000

    new_width = int(original_width * downsample_factor)
    new_height = int(original_height * downsample_factor)

    return downsample_factor, new_width, new_height, original_width, original_height 


def downscaling(obj, new_width, new_height):
    """
    Downscale the given object (image or slide) to the specified size.
    """
    if isinstance(obj, np.ndarray):  # If it's a NumPy array
        # Resize using scikit-image (resize scales and interpolates)
        image_numpy = resize(obj, (new_height, new_width), anti_aliasing=True)
        image_numpy = (image_numpy * 255).astype(np.uint8)

    elif hasattr(obj, 'size'):  # If it's an image (PIL or similar)
        obj = obj.resize((new_width, new_height))
        image_numpy = np.array(obj)

    elif hasattr(obj, 'dimensions'):  # If it's a slide (e.g., a TIFF object)
        thumbnail = obj.get_thumbnail((new_width, new_height))
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts the image to a tensor (C, H, W)
        ])
        image_tensor = transform(thumbnail)
        image_numpy = image_tensor.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C) numpy format
    else:
        raise ValueError("The object must have either 'size' (image) or 'dimensions' (slide) attribute.")
 
    return image_numpy


def superpixel_segmenting(obj, downsample_size = 1096, n_segments=2000, compactness=10.0, start_label=0):
    # start = time.time()
    downsample_factor, new_width, new_height, curr_width, curr_height = rescaling_stat_for_segmentation(
        obj, downsample_size)

    # Downscale the region and prepare for mask generation
    downscaled_region = downscaling(
        obj, new_width, new_height)
    
    downscaled_region_array = np.array(downscaled_region)
    # print("downsample_factor", downsample_factor)
    # print(downscaled_region_array.shape)
    
    lab_image = color.rgb2lab(downscaled_region_array)
    superpixel_labels = segmentation.slic(lab_image, n_segments=n_segments, compactness=compactness, start_label=start_label)

    # print((time.time()-start)/60.00)
    segmented_mask = segmentation.mark_boundaries(downscaled_region_array, superpixel_labels)

    return superpixel_labels, segmented_mask, downsample_factor, new_width, new_height, downscaled_region_array, lab_image

 
def equalize_image(downscaled_region_array):
    # Convert to YUV color space
    yuv_image = cv2.cvtColor(downscaled_region_array, cv2.COLOR_RGB2YUV)

    # Extract the Y (luminance) channel
    y_channel = yuv_image[..., 0]

    # Ensure the Y channel is 8-bit single channel (CV_8UC1)
    y_channel = np.uint8(y_channel)  # Convert to 8-bit unsigned integer

    # Apply histogram equalization to the Y channel (luminance)
    y_channel_equalized = cv2.equalizeHist(y_channel)

    # Replace the Y channel in the YUV image with the equalized Y channel
    yuv_image[..., 0] = y_channel_equalized

    # Convert back to RGB color space
    equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)

    return equalized_image 

def identify_foreground_background(equalized_image, superpixel_labels, threshold=240):
    """
    """
    # Convert to a numpy array and initialize the lists
    equalized_image = np.array(equalized_image)
    unique_superpixels = np.unique(superpixel_labels)

    # Determine the pixels that are close to white (background pixels)
    background_mask = np.all(equalized_image >= threshold, axis=-1)  # RGB close to white

    foreground_superpixels = []
    background_superpixels = []

    # Iterate through each superpixel label
    for label in unique_superpixels:
        # Create a mask for the current superpixel
        superpixel_mask = superpixel_labels == label

        # Check the percentage of background pixels within this superpixel
        superpixel_background = np.sum(background_mask[superpixel_mask]) / np.sum(superpixel_mask)

        # If more than 50% of the superpixel is background (white), consider it background
        if superpixel_background > 0.5:
            background_superpixels.append(label)
        else:
            foreground_superpixels.append(label)

    return foreground_superpixels, background_superpixels 

def get_bounding_boxes_for_foreground_segments(original_image, superpixel_labels, foreground_superpixels):
    # Ensure the image is in uint8 format (required by OpenCV)
    if original_image.dtype != np.uint8:
        original_image = np.uint8(original_image)  # Convert to uint8

    # Ensure the image is in BGR format for OpenCV (if it's in RGB)
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    # Create a copy of the original image to draw bounding boxes on
    output_image = np.copy(original_image)

    # Initialize the dictionary to store bounding boxes for each foreground superpixel
    bounding_boxes = {}

    # Loop through each foreground superpixel label
    for label in foreground_superpixels:
        # Get the coordinates of the pixels for this label
        coords = np.column_stack(np.where(superpixel_labels == label))

        # Get the bounding box for the superpixel (min and max of x and y)
        ymin, xmin = np.min(coords, axis=0)
        ymax, xmax = np.max(coords, axis=0)

        # Draw the bounding box on the image (in green)
        cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Store the bounding box in the dictionary
        bounding_boxes[label] = (xmin, ymin, xmax, ymax)

    return bounding_boxes, output_image 

def plot_foreground_boundaries_on_original_image(original_image, superpixel_labels, foreground_superpixels):
    # Create a mask for foreground superpixels
    foreground_mask = np.isin(superpixel_labels, foreground_superpixels)

    # Find the superpixel boundaries for the entire image
    boundaries = segmentation.find_boundaries(superpixel_labels, connectivity=2)

    # Highlight the boundaries for foreground superpixels only (in yellow)
    boundary_image = np.zeros_like(original_image)

    # Create a mask for only the foreground superpixels' boundaries
    foreground_boundaries = boundaries & foreground_mask

    # Set the pixels of the foreground boundaries to yellow (RGB: [1, 1, 0])
    boundary_image[foreground_boundaries] = [1, 1, 0]  # Yellow color for the boundary

    # Combine the boundary image with the original image
    # Keep the original image where there are background pixels (foreground will be transparent)
    combined_image = np.copy(original_image)
    combined_image[foreground_boundaries] = boundary_image[foreground_boundaries]  # Apply yellow boundary where foreground
    return combined_image

def processing_superpixel(slide_path, JSON_SAVE_PATH):
    
    print("Start processing: ", slide_path)
    basename = os.path.basename(slide_path).split('.')[0]
    os.makedirs(os.path.dirname(JSON_SAVE_PATH), exist_ok=True)
    # print(os.listdir(os.path.dirname(JSON_SAVE_PATH)))

    ############################segment ############################################
    start  = time.time()

    slide = openslide.open_slide(slide_path)

    (
        superpixel_labels,
        segmented_mask,
        downsample_factor,
        new_width,
        new_height,
        downscaled_region_array,
        lab_image )= superpixel_segmenting(
            slide, 
            downsample_size = 1096, 
            n_segments=500, 
            compactness=10.0, 
            start_label=0)
        
    ################################################################################
    equalized_image = equalize_image(downscaled_region_array)

    foreground_superpixels, background_superpixels = identify_foreground_background(equalized_image, superpixel_labels)
    sp_plot = plot_foreground_boundaries_on_original_image(downscaled_region_array, superpixel_labels, foreground_superpixels)

    bounding_boxes, output_image_with_bboxes = get_bounding_boxes_for_foreground_segments(
        downscaled_region_array,
        superpixel_labels,
        foreground_superpixels
        )
    ################################################################################
    # save the RESULT ----------------------------
    # Convert numpy arrays and scalars to Python-native types
    data_to_save = {
        'superpixel_labels': superpixel_labels.tolist(),  # Convert numpy array to list
        'downsample_factor': downsample_factor,  # Convert numpy scalar to Python float
        'new_width': new_width,  # Convert numpy int64 to Python int
        'new_height': new_height,  # Convert numpy int64 to Python int
        'downscaled_region_array': downscaled_region_array.tolist(),  # Convert numpy array to list
        'foreground_superpixels': [float(i) for i in foreground_superpixels],
        'background_superpixels': [float(i) for i in background_superpixels],
        'bounding_boxes': {str(k): list([float(j) for j in v]) for k, v in bounding_boxes.items()},
        'output_image_with_bboxes': output_image_with_bboxes.tolist(),   # Convert numpy array to list
        'superpixels_plot': sp_plot.tolist()
    }
    

    # Save the dictionary to a single JSON file
    with open(JSON_SAVE_PATH, 'w') as json_file:
        json.dump(data_to_save, json_file)

    # Print confirmation
    print("All results saved successfully in one JSON file!")
    
def main(args):
    total = len(wsi_basenames)
    count = 1 
    for wsi_basename in wsi_basenames:
        start = time.time()
        print("------ processing: ", wsi_basename, "------", count, "/", total) 
        print(wsi_basename)
        slide_path = glob.glob(os.path.join(args.slide_path, f'{wsi_basename}*'))[0]
        JSON_SAVE_PATH = os.path.join(args.json_path, f'{wsi_basename}.json')
        print(JSON_SAVE_PATH) 
        processing_superpixel(slide_path, JSON_SAVE_PATH) 
        count += 1 
        print("Time (mins): ", (time.time()-start)/60.00) 



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=bool, default=False)
    parser.add_argument('--config_file', default='ma_exp002')
    args = parser.parse_args()
    
    if os.path.exists(f'./testbest_config/{args.config_file}.yaml'):
        config = load_config(f'./testbest_config/{args.config_file}.yaml') 

    
    args.slide_path = config.get('SLIDE_PATH')
    args.json_path =  '/project/hnguyen2/mvu9/camelyon16/json_files_v2' # replace your path  #config.get('JSON_PATH') 

        
    if not os.path.exists(args.json_path):
        os.makedirs(args.json_path)
        print(f"Created directory: {args.json_path}")
    else:
        print(f"Directory {args.json_path} already exists")  
        
    # slide_items = ['normal_031', 'tumor_024', 'normal_047', 'tumor_009', 'tumor_057', 'normal_093', 'normal_051', 'tumor_014', 'tumor_015', 'tumor_067', 'normal_003', 'tumor_084', 'tumor_101', 'normal_148', 'normal_022', 'tumor_012', 'normal_039', 'normal_084', 'normal_101', 'tumor_010', 'normal_088', 'normal_155', 'normal_087', 'normal_016', 'normal_114', 'normal_024', 'tumor_048', 'normal_078', 'tumor_049', 'tumor_086']
    slide_items  = [i.split('.')[0] for i in os.listdir(args.slide_path) if i.endswith('tif')]
    print(">>>>>slide_items: ", len(slide_items)) 
    # slide_items = [i.split(".")[0] for i in os.listdir(args.slide_path) if i.endswith('tif')]
    json_items = [i.split('.')[0] for i in os.listdir(args.json_path) if i.endswith('json')]
    
    items_not_in_json = [item for item in slide_items if item not in json_items] 
    wsi_basenames = items_not_in_json    

    print(">>>>count", len(wsi_basenames))
    
    
     
    main(args)
