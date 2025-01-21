
import os
import sys 
from tqdm import tqdm 
import glob
import pandas as pd
import numpy as np
import argparse
import time   
import timm 
import yaml 
import json 
import glob 
import h5py 
import openslide

PROJECT_DIR = os.environ.get('PROJECT_DIR')
print("PROJECT DIR", PROJECT_DIR)
sys.path.append(PROJECT_DIR) 

import torch
import torch.nn as nn
import torch.optim as optim 
from data.merge_dataset import SuperpixelDataset, PatchDataset, SlidePatchesDataset
from PIL import Image
from utils import utils  



wsi_basenames = ['normal_031', 'tumor_024', 'normal_047', 'tumor_009', 'tumor_057', 'normal_093', 'normal_051', 'tumor_014', 'tumor_015', 'tumor_067', 'normal_003', 'tumor_084', 'tumor_101', 'normal_148', 'normal_022', 'tumor_012', 'normal_039', 'normal_084', 'normal_101', 'tumor_010', 'normal_088', 'normal_155', 'normal_087', 'normal_016', 'normal_114', 'normal_024', 'tumor_048', 'normal_078', 'tumor_049', 'tumor_086']


def load_config(config_file):
    # Load configuration from the provided YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def superpixel_segmenting(obj, downsample_size = 1096, n_segments=2000, compactness=10.0, start_label=0):
    # start = time.time()
    downsample_factor, new_width, new_height, curr_width, curr_height = rescaling_stat_for_segmentation(
        obj, downsample_size)

    # Downscale the region and prepare for mask generation
    downscaled_region = downscaling(
        obj, new_width, new_height)
    downscaled_region_array = np.array(downscaled_region)

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

def processing_superpixel(slide_path, JSON_SAVE_PATH):
    
    print("Start processing: ", slide_path)
    basename = os.path.basename(slide_path).split('.')[0]
    os.makedirs(os.path.dirname(JSON_SAVE_PATH), exist_ok=True)
    print(os.listdir(os.path.dirname(JSON_SAVE_PATH)))

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
    # sp_plot = plot_foreground_boundaries_on_original_image(downscaled_region_array, superpixel_labels, foreground_superpixels)

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
    # with open(JSON_SAVE_PATH, 'w') as json_file:
    #     json.dump(data_to_save, json_file)

    # Print confirmation
    print("All results saved successfully in one JSON file!")
    
def main(args):
    for wsi_basename in wsi_basenames:
        print(wsi_basename)
        slide_path = glob.glob(os.path.join(args.slide_path, f'{wsi_basename}*'))
        JSON_SAVE_PATH = os.path.join(args.json_path, wsi_basename)
        print(JSON_SAVE_PATH) 
        processing_superpixel(slide_path, JSON_SAVE_PATH) 



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=bool, default=False)
    parser.add_argument('--config_file', default='ma_exp002')
    args = parser.parse_args()
    
    if os.path.exists(f'./testbest_config/{args.config_file}.yaml'):
        config = load_config(f'./testbest_config/{args.config_file}.yaml') 
    
    args.slide_path = config.get('SLIDE_PATH')
    args.json_path = config.get('JSON_PATH') 
    
    main(args)
    