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
import cv2 
import time 
from PIL import Image 


class PatchDataset(Dataset):
    def __init__(
        self, 
        region, 
        mask, 
        patch_size=(256, 256), 
        coverage_threshold=0.1, 
        transform=None, 
        return_feature=True, 
        model=None
        ):
        self.region_np = region
        self.mask = mask
        self.patch_size = patch_size
        self.coverage_threshold = coverage_threshold
        self.transform = transform
        self.model = model
        
        # Get region dimensions and patch size
        region_height, region_width = region.shape[:2]
        patch_height, patch_width = patch_size

        self.patches = []
        self.bboxes = []

        # Loop through the region and extract patches
        for top in range(0, region_height, patch_height):
            for left in range(0, region_width, patch_width):  
                # Ensure the patch is within bounds
                bottom = min(top + patch_height, region_height)
                right = min(left + patch_width, region_width)
                
                # Extract the patch and corresponding mask region
                patch = region[top:bottom, left:right]
                patch_mask = mask[top:bottom, left:right]

                patch_area = patch.size
                mask_coverage = np.sum(patch_mask) / patch_area  # Proportion of the patch covered by the mask
                
                # Only include patches that satisfy the coverage threshold
                if mask_coverage >= self.coverage_threshold:
                    bbox = (top, left, bottom, right)
                    self.patches.append(patch)
                    self.bboxes.append(bbox)
                    

    def __len__(self):
        """Returns the total number of patches."""
        return len(self.patches)

    def __getitem__(self, idx):
        """Returns a patch and its corresponding bounding box."""
        patch = self.patches[idx]
        bbox = self.bboxes[idx]
        # Convert patch to PIL image for torchvision transforms
        patch_pil = Image.fromarray(patch.astype(np.uint8))  # Convert numpy array to PIL Image

        # Apply the transformations if provided
        if self.transform:
            patch_pil = self.transform(patch_pil)
            
        if self.return_feature:
            patch_tensor = patch_pil.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                features = self.model(patch_tensor)  # Pass through the model to extract features
            return features.squeeze(0), patch_pil, bbox  # Remove batch dimension and return features
        else:
            return _, patch_pil, bbox 
        

class SuperpixelDataset(Dataset):
    def __init__(self, slide_paths, json_folder):
        """
        Args:
        """
        self.slide_paths = slide_paths
        self.json_folder = json_folder 

        # Get list of WSI paths and filter by example_list

    def __len__(self):
        """Returns the total number of samples (WSI images)."""
        return len(self.wsi_paths)

    def __getitem__(self, index):
        """Returns a sample (WSI image and associated data)."""
        # Get the WSI path and basename
        wsi_path = self.slide_paths[index]
        basename = os.path.basename(wsi_path).split(".")[0]
        print(basename) 
        
        slide = openslide.open_slide(wsi_path)
        print("complete reading WSIs")
        # Load corresponding JSON data
        json_path = os.path.join(self.json_folder, f'{basename}.json')
        sample = self.read_json_superpixel(json_path)
        
        bounding_boxes = sample['bounding_boxes']
        downsample_factor = sample['downsample_factor']
        foreground_superpixels = sample['foreground_superpixels']
        superpixel_labels = sample['superpixel_labels']
          
        for foreground_idx in foreground_superpixels:
            bbox = bounding_boxes[foreground_idx]
            xywh_abs_bbox = self._get_absolute_bbox_coordinate(bbox, downsample_factor) 
             
            superpixel_downsampling = superpixel_labels == foreground_idx
            superpixel_extrapolated = self.extrapolate_superpixel_mask_segment(
                superpixel_labels, foreground_idx, bounding_boxes, downsample_factor)
            
            region_cropped = self.get_region_original_size(slide, xywh_abs_bbox)
            region_np = np.array(region_cropped) 
            
            yield (foreground_idx, region_np, superpixel_extrapolated)
            

    
    @staticmethod
    def get_region_original_size(slide, xywh_abs_bbox):
        xmin_original, ymin_original, width_original, height_original = xywh_abs_bbox
        region = slide.read_region(
            (xmin_original, ymin_original),  # Top-left corner (x, y)
            0,  # Level 0
            (width_original, height_original)  # Width and height
        )
        return region.convert('RGB')
    
    @staticmethod 
    def extrapolate_superpixel_mask_segment(
        superpixel_downsampling, 
        superpixel_idx, 
        bounding_boxes, 
        downsample_factor):
        mask = (superpixel_downsampling == superpixel_idx).astype(np.uint8)
        xmin, ymin, xmax, ymax = [int(i) for i in bounding_boxes[superpixel_idx]]
        cropped_mask = mask[ymin:ymax, xmin:xmax]  # Corrected cropping

        upscaled_mask = cv2.resize(
            cropped_mask,
            (int(cropped_mask.shape[1] / downsample_factor), int(cropped_mask.shape[0] / downsample_factor)),
            interpolation=cv2.INTER_NEAREST  # Nearest-neighbor interpolation to keep binary mask intact
        )

        upscaled_mask_bool = (upscaled_mask > 0).astype(bool)  # Convert to boolean (True/False)

        return upscaled_mask_bool 
        
    @staticmethod
    def _get_absolute_bbox_coordinate(bbox, downsample_factor):
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin

        xmin_original = int(xmin / downsample_factor)
        ymin_original = int(ymin / downsample_factor)
        width_original = int(width / downsample_factor)
        height_original = int(height / downsample_factor)

        relative_bbox = [xmin_original, ymin_original, width_original, height_original]

        return relative_bbox  

    @staticmethod 
    def read_json_superpixel(json_path):
        with open(json_path, 'r') as json_file:
            loaded_data = json.load(json_file)
        
        # Process JSON data
        superpixel_labels = np.array(loaded_data['superpixel_labels'])
        downscaled_region_array = np.array(loaded_data['downscaled_region_array'])
        output_image_with_bboxes = np.array(loaded_data['output_image_with_bboxes'])

        foreground_superpixels = [int(i) for i in loaded_data['foreground_superpixels']]
        background_superpixels = [int(i) for i in loaded_data['background_superpixels']]

        bounding_boxes = {int(k): tuple(v) for k, v in loaded_data['bounding_boxes'].items()}

        downsample_factor = loaded_data['downsample_factor']
        new_width = loaded_data['new_width']
        new_height = loaded_data['new_height']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        
        # Return the sample (image, features, bounding boxes)
        sample = {
            'superpixel_labels': superpixel_labels,
            'downscaled_region_array': downscaled_region_array,
            'output_image_with_bboxes': output_image_with_bboxes,
            'foreground_superpixels': foreground_superpixels,
            'background_superpixels': background_superpixels,
            'bounding_boxes': bounding_boxes,
            'downsample_factor': downsample_factor,
            'new_width': new_width,
            'new_height': new_height
        }
            
        return sample


        
    
# if __name__ == '__main__':

            
             
    