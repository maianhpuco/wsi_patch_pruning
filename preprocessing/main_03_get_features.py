# OUPUT: FEATURE VECTORS
# take image and put thru VIT backbone 
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
import h5py 
import openslide
from tqdm import tqdm 
PROJECT_DIR = os.environ.get('PROJECT_DIR')
print("PROJECT DIR", PROJECT_DIR)
sys.path.append(PROJECT_DIR) 


import torch
import torch.nn as nn
import torch.optim as optim 

from data.merge_dataset import SuperpixelDataset, PatchDataset, SlidePatchesDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from utils import utils  
from src.important_scores import get_scoring_do_nothing
from src.pruning import get_pruning_do_nothing

from src.bag_classifier.clam import CLAM_MB
from utils.train_classifier.train_clam import * 
from utils.utils import setup_logger


#TODO
# clean the code of the main function -> make it cleaner

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
sys.path.append(os.path.join(PROJECT_DIR))  


# SLIDE_DIR = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16'
# example_list = ['normal_072', 'normal_001', 'normal_048', 'tumor_026', 'tumor_031', 'tumor_032']
# example_list = ['tumor_026'] #a, 'tumor_032']
# example_list = ['normal_031', 'tumor_024', 'normal_047', 'tumor_009', 'tumor_057', 'normal_093', 'normal_051', 'tumor_014', 'tumor_015', 'tumor_067', 'normal_003', 'tumor_084', 'tumor_101', 'normal_148', 'normal_022', 'tumor_012', 'normal_039', 'normal_084', 'normal_101', 'tumor_010', 'normal_088', 'normal_155', 'normal_087', 'normal_016', 'normal_114', 'normal_024', 'tumor_048', 'normal_078', 'tumor_049', 'tumor_086']
 
def main(args):
    
    if args.dry_run:
        print("Running the dry run")
    else:
        print("Running on full data") 
    
    model = timm.create_model(args.feature_extraction_model, pretrained=True)  # You can choose any model
    model = model.to(args.device) 
    model.eval()   
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the patch to 224x224
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
        # You can add other transformations like RandomHorizontalFlip, RandomRotation, etc.
    ])

    start_slide = time.time()    
    
    wsi_paths = glob.glob(os.path.join(args.slide_path, '*.tif'))
    
    wsi_paths = [path for path in wsi_paths if os.path.basename(path).split(".")[0] in example_list]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),          # Convert the image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
    ]) 
    count = 0
    for wsi_path in wsi_paths:
        print(">------ processing", count+1, "over", len(wsi_paths))
        start_slide = time.time()
        slide_basename = os.path.basename(wsi_path).split(".")[0]
        
        print("Processinging: ", slide_basename)
 
        slide_patch_dataset = SlidePatchesDataset(
            patch_dir = os.path.join(args.patch_path, slide_basename),
            transform = transform
        )
        dataloader = DataLoader(slide_patch_dataset, batch_size=args.batch_size, shuffle=True)
        
        _slide_features = []
        _patch_idxes = []
        _coordinates = []  # To store coordinates as a combined list of [xmin, xmax, ymin, ymax]
        _spixel_idx = []  # To store the superpixel indices  
        
        for batch in tqdm(dataloader):
            batch_image = batch['image'].to(args.device) 
            batch_patch_info = batch['patch_info']
            
            parsed_batch_info = [] 
            for i in range(len(batch_patch_info['ymin'])):
                parsed_info = {
                    'ymin': batch_patch_info['ymin'][i].item(),
                    'ymax': batch_patch_info['ymax'][i].item(),
                    'xmin': batch_patch_info['xmin'][i].item(),
                    'xmax': batch_patch_info['xmax'][i].item(),
                    'spixel_idx': batch_patch_info['spixel_idx'][i].item(),
                    'patch_idx': batch_patch_info['patch_idx'][i].item()
                }
                parsed_batch_info.append(parsed_info)
                
            batch_idxes = [info['patch_idx'] for info in parsed_batch_info] 
            _coordinates.extend([
                [info['xmin'], info['xmax'], info['ymin'], info['ymax']] for info in parsed_batch_info
            ])
            _spixel_idx.extend([
                info['spixel_idx'] for info in parsed_batch_info
            ])
 
            # Print the parsed batch info
            # print("Parsed Batch Info of a sample:", parsed_batch_info[0])
            # print("Batch Image Shape:", batch_image.shape) 
            
            with torch.no_grad():  # Disable gradient calculation for inference
                _batch_features = model.forward_features(batch_image)
                class_token_features = _batch_features[:, 0, :]  
                
            # 0. apply feature extraction here on batch_image
                # input: batch_image
                # output: slide_features (remember to cat them into a slide's features)
         
            _slide_features.append(class_token_features)
            _patch_idxes.append(batch_idxes)
            
        count += 1 
            
        slide_features = torch.cat(_slide_features, dim=0)   # Concatenate all features for the slide on CPU
        slide_patch_idxes = torch.cat(
            [torch.tensor(idxes) for idxes in _patch_idxes], dim=0
            ).to(args.device)
        
        _coordinates = np.array(_coordinates)  # Shape will be [num_patches, 4] for [xmin, xmax, ymin, ymax]
        _spixel_idx = np.array(_spixel_idx) 
        
        
         # Label for the slide (assuming binary classification, 0 for normal, 1 for tumor)
        label = 0 if slide_basename.split("_")[0] == "normal" else 1
        # label = label.unsqueeze(0)
        # Create or open the HDF5 file for saving data
        output_file = os.path.join(args.features_h5_path, f"{slide_basename}.h5")
        
        with h5py.File(output_file, 'w') as f:
            # Save the features, patch indices, and label into the HDF5 file
            f.create_dataset('features', data=slide_features.cpu().numpy())  # Save features as dataset
            f.create_dataset('patch_indices', data=slide_patch_idxes.cpu().numpy())  # Save patch indices
            f.create_dataset('label', data=np.array([label]))  # Save label as dataset
             # Save the coordinates (xmin, xmax, ymin, ymax) as a single dataset
            f.create_dataset('coordinates', data=_coordinates)  # Save coordinates as dataset (shape: [num_patches, 4])
            # Save the superpixel indices
            f.create_dataset('spixel_idx', data=_spixel_idx)  # Save superpixel indices (shape: [num_patches]) 
        
        print(f"Saved features for {slide_basename} to {output_file}")
        print(f"Finish a slide after: {(time.time()-start_slide)/60.0000} mins")
        print(f"Slide feature shape {slide_features.shape}")
    
   
   
    
# 1. adding scoring + pruning here; 
    # input: slide_features 
    # output: reduced_slide_features
# 2. SSL apply on reduced_slide_features 
# 3. Bag Classifier (DSMIL, DTFD-MIL, Snuffy, Camil)
# 4. Compute metric: GLOP, AUC, Acc, etc. 
   
   
   
         
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=bool, default=False)
    parser.add_argument('--config_file', default='ma_exp002')
    args = parser.parse_args()
    
    if os.path.exists(f'./testbest_config/{args.config_file}.yaml'):
        config = load_config(f'./testbest_config/{args.config_file}.yaml')
        args.use_features = config.get('use_features', True)
        
        args.slide_path = config.get('SLIDE_PATH')
        args.json_path = config.get('JSON_PATH')
        args.spixel_path = config.get('SPIXEL_PATH')
        args.patch_path = config.get('PATCH_PATH') # save all the patch (image)
        args.features_h5_path = config.get("FEATURES_H5_PATH") # save all the features 
        print(args.features_h5_path)
        
        os.makedirs(args.features_h5_path, exist_ok=True)  
        
        args.scoring_function = SCORING_FUNCTION_MAP.get(
            config.get("scoring_function")
        )
        args.pruning_function = PRUNING_FUNCTION_MAP.get(
            config.get('pruning_function') 
        )
        args.batch_size = config.get('batch_size')
        args.feature_extraction_model = config.get('feature_extraction_model')
        
        # args.scoring_function("")
        # args.pruning_function("")
        
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # example_list = ['normal_031', 'tumor_024', 'normal_047', 'tumor_009', 'tumor_057', 'normal_093', 'normal_051', 'tumor_014', 'tumor_015', 'tumor_067', 'normal_003', 'tumor_084', 'tumor_101', 'normal_148', 'normal_022', 'tumor_012', 'normal_039', 'normal_084', 'normal_101', 'tumor_010', 'normal_088', 'normal_155', 'normal_087', 'normal_016', 'normal_114', 'normal_024', 'tumor_048', 'normal_078', 'tumor_049', 'tumor_086']
    
    example_list = [i.split('.')[0] for i in os.listdir(args.patch_path)]
    avai_items = [i.split('.')[0] for i in os.listdir(args.features_h5_path) if i.endswith("h5")]
    print("number of available h5", len(avai_items)) 
    items_to_process = [item for item in example_list if item not in avai_items] 
    
    remove_item = ['normal_114', 'tumor_026', 'tumor_009', 'tumor_024', 'tumor_015', 'normal_076','normal_070', 'normal_066', 'normal_053', 'normal_104','normal_112']  
    items_to_process = [item for item in items_to_process if item not in remove_item] 
  
    example_list = items_to_process    
    print("len(example_list)", len(example_list)) 
    
    main(args) 
    