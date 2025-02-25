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
    plot_anno_with_mask, 
) 
import openslide
import glob 
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import glob
import pandas as pd
 
def parse_xml(file_path):
    try:
        tree = ET.parse(file_path)  # Load XML file
        root = tree.getroot()  # Get root element
        return root
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return None

# Function to extract coordinate data
def extract_coordinates(file_path):
    coordinates = [] 
    root = parse_xml(file_path)
    if root is None:
        return  # Skip if parsing failed

    print(f"Processing XML: {file_path}")

    # Loop through all 'Coordinate' elements
    for coordinate in root.findall(".//Coordinate"):
        order = coordinate.attrib.get("Order")
        x = coordinate.attrib.get("X")
        y = coordinate.attrib.get("Y")

        # Append extracted data
        if order and x and y:
            coordinates.append({
                "File": file_path.split("/")[-1],  # Extract only the filename
                "Order": int(order),
                "X": float(x),
                "Y": float(y)
            }) 
    return coordinates 

            
def main(args): 
    '''
    Input: h5 file
    Output: save scores into a json folder
    '''
    # Get all XML file paths
    anno_path = args.anno_path 
    file_paths = glob.glob(f"{anno_path}/*.xml")
    print(f"Found {len(file_paths)} XML files.")

    # List to store extracted data
    all_coordinates = [] 
    # Process all XML files
    for xml_file in file_paths:
        # print(f"Processing XML: {xml_file}")
        all_coordinates.extend(extract_coordinates(xml_file))

    # Convert to DataFrame
    df = pd.DataFrame(all_coordinates)  
    anno_paths = glob.glob(os.path.join(args.anno_path, "*.xml"))  
    
    print("Image will be plotted at:", args.plot_anno_dir)
    
    if os.path.exists(args.plot_anno_dir):
        shutil.rmtree(args.plot_anno_dir)  # Delete the existing directory
    os.makedirs(args.plot_anno_dir)   
    
    # scale_x, scale_y, new_height, new_width  
        
    for idx, slide_path in enumerate(anno_paths):
        print(f"Print the plot {idx+1}/{len(anno_paths)}")
        basename = os.path.basename(slide_path).split(".")[0] 
        plot_anno_with_mask(basename, args.slide_path, df, save_dir=args.plot_anno_dir)
    print(">>>> done")

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
        args.plot_anno_dir = config.get("PLOT_ANNO") 
        args.anno_path = config.get("ANNOTATION_PATH") 
        
     
    main(args) 
