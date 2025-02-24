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
import shutil 

PROJECT_DIR = os.environ.get('PROJECT_DIR')
print("PROJECT DIR", PROJECT_DIR)
sys.path.append(PROJECT_DIR)   

# import yaml 
import random 
import numpy as np
from src.bag_classifier.mil_classifier import MILClassifier 
import torch
import torch.nn as nn 
import torch.optim as optim
import saliency.core as saliency
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
from torch.utils.data import DataLoader 
from src.bag_classifier.mil_classifier import MILClassifier # in the repo
from data.feature_dataset import FeaturesDataset  # in the repo
from utils.utils import load_config
from utils.train_classifier.train_mlclassifier import (
    save_checkpoint, 
    load_checkpoint
)
import h5py 
import pickle 

from utils.plotting import (
    plot_heatmap_with_bboxes,
    get_region_original_size,
    downscaling,
    rescaling_stat_for_segmentation
) 
from data.ig_dataset import IG_dataset 
import numpy as np
np.random.seed(0)

from metrics_segmentation.method import (
    calculate_dice_score,
    calculate_iou_score,
    calculate_fp,
    calculate_fn,
    calculate_tn,
    calculate_tp,
)
from metrics_segmentation.utils_metrics import (
    extract_coordinates, 
    check_xy_in_coordinates_fast, 
) 

import openslide 

def read_h5_data(file_path, dataset_name=None):
    data = None
    with h5py.File(file_path, "r") as file:
        if dataset_name is not None:
            if dataset_name in file:
                dataset = file[dataset_name]
                data = dataset[()]
            else:
                raise KeyError(f"Dataset '{dataset_name}' not found in the file.")
        else:
            datasets = {}

            def visitor(name, node):
                if isinstance(node, h5py.Dataset):
                    datasets[name] = node[()]

            file.visititems(visitor)

            if len(datasets) == 1:
                data = list(datasets.values())[0]
            else:
                data = datasets
    return data 
 
def main(args):
    # Assume that have path of h5 file
    _annotation_list = os.listdir(args.annotation_path)
    _excluded_list = os.listdir(args.ground_truth_corr_path)
    _h5_files = os.listdir(args.features_h5_path)
    
    annotation_list = [] 
    for anno_filename in _annotation_list:
        name = anno_filename.split(".")[0]
        if f"{name}.csv" not in _excluded_list and  f"{name}.h5" in _h5_files: 
            annotation_list.append(anno_filename)
    
    # reset_directory(args.ground_truth_corr_path)
    # reset_directory(args.ground_truth_path)  
    
    total_file = len(annotation_list)
    print("total file to process:", total_file)
    for idx, basename in enumerate(annotation_list):
        print(basename)
        print(f">>> Processing the annotation file number {idx+1}/{total_file}")
        name = basename.split(".")[0]
        h5_path = os.path.join(args.features_h5_path, f"{name}.h5")
        xml_path = os.path.join(args.annotation_path, f"{name}.xml")
        
        df_xml_save_path = os.path.join(args.ground_truth_corr_path, f'{name}.csv')

        df_xml = extract_coordinates(
            xml_path,
            df_xml_save_path)
        print("Save the df fill contour into:", df_xml_save_path)  
        
        print("df_xml.shape: ", df_xml.shape)
        
        h5_data = read_h5_data(h5_path)
    
        mask = check_xy_in_coordinates_fast(
            df_xml, h5_data["coordinates"])
        mask_save_path = os.path.join(args.ground_truth_path, f"{name}.npy")
        print("Save the mask into:", mask_save_path)
        np.save(mask_save_path, mask) 
         
        # print("shape of mask", mask.shape)
        # print("sum of mask", np.sum(mask))
        # print(">>> MASK: ", mask[:10]) 

    print("- Check total number of file")
    
    import glob 
    df_file = glob.glob(os.path.join(args.ground_truth_corr_path, "*.csv"))
    mask_file = glob.glob(os.path.join(args.ground_truth_path, "*.npy")) 
    print("+ Number of annotation: ", len(annotation_list))
    print("+ Number of df file: ", len(df_file))
    print("+ Numner mask file: ", len(mask_file))
    
def reset_directory(path):
    if os.path.exists(path):
        print("Current path exist, will delete and recreate: ", path)
        shutil.rmtree(path)  
    os.makedirs(path)  
 
if __name__=="__main__":
    # get config 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config_file', default='ma_exp002')
    parser.add_argument('--ig_name', 
                    default='integrated_gradients', 
                    choices=[
                        'integrated_gradient', 
                        'expected_gradient', 
                        'guided_gradient', 
                        'contrastive_gradient', 
                        'vanilla_gradient', 
                        'squareintegrated_gradient'],
                    help='Choose the attribution method to use.') 
    
    args = parser.parse_args()
    
    if os.path.exists(f'./testbest_config/{args.config_file}.yaml'):
        config = load_config(f'./testbest_config/{args.config_file}.yaml')
        args.use_features = config.get('use_features', True)
        args.slide_path = config.get('SLIDE_PATH')
        args.json_path = config.get('JSON_PATH')
        args.spixel_path = config.get('SPIXEL_PATH')
        args.patch_path = config.get('PATCH_PATH') # save all the patch (image)
        args.features_h5_path = config.get("FEATURES_H5_PATH") # save all the features
        args.checkpoints_dir = config.get("CHECKPOINT_PATH")
        args.attribution_scores_folder = config.get("SCORE_FOLDER")    
        os.makedirs(args.features_h5_path, exist_ok=True)  
        os.makedirs(args.attribution_scores_folder, exist_ok=True) 
        args.batch_size = config.get('batch_size')
        args.feature_extraction_model = config.get('feature_extraction_model')
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.feature_mean_std_path=config.get("FEATURE_MEAN_STD_PATH")
        args.annotation_path = config.get("ANNOTATION_PATH")
        
        # args.ig_name = "integrated_gradients"
        args.sanity_check_path = config.get("SANITY_CHECK_PATH")   
        args.ground_truth_corr_path = config.get("GROUND_TRUTH_CORR_PATH") 
        args.ground_truth_path = config.get("GROUND_TRUTH_PATH") 
        
        print("args.ground_truth_corr_path", args.ground_truth_corr_path)
        print("args.ground_truth_path", args.ground_truth_path)        
        
        os.makedirs(args.ground_truth_corr_path, exist_ok=True)   
        os.makedirs(args.ground_truth_path, exist_ok=True)    
        os.makedirs(args.sanity_check_path, exist_ok=True) 

        args.do_normalizing = True

    main(args)
    
#  scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/sanity_check/tumor_026.png 