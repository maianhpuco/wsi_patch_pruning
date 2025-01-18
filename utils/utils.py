import os
import sys 
import numpy as np

def get_region_original_size(slide, xywh_abs_bbox):
    xmin_original, ymin_original, width_original, height_original = xywh_abs_bbox
    region = slide.read_region(
        (xmin_original, ymin_original),  # Top-left corner (x, y)
        0,  # Level 0
        (width_original, height_original)  # Width and height
    )
    return region.convert('RGB')



def read_region_from_npy(dir_folder, slide_basename, foreground_idx):
    # Construct the file path
    npy_file_path = os.path.join(dir_folder, slide_basename, f"{foreground_idx}.npy")

    # Check if the file exists
    if not os.path.exists(npy_file_path):
        raise FileNotFoundError(f"The file {npy_file_path} does not exist.")

    # Load and return the NumPy array from the file
    region_np = np.load(npy_file_path)

    return region_np 