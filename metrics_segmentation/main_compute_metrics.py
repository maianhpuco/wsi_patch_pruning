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
from metrics_segmentation.utils_metrics_ver2 import (
    # read_all_xml_file_base_tumor,
    check_xy_in_coordinates,
    # read_h5_data,
    extract_coordinates, 
    check_xy_in_coordinates_fast 
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
    for basename in os.listdir(args.annotation_path): 
        h5_path = os.path.join(args.features_h5_path, "tumor_026.h5")
        xml_path = os.path.join(args.annotation_path, "tumor_026.xml")  
        
        # path = "/Users/nam.le/Desktop/research/camil_pytorch/data/camelyon16_feature/h5_files/tumor_048.h5"
        # h5_name = path.split("/")[-1].replace("h5", "xml")
        
        mask = pd.read_csv(args.ground_truth_path, "tumor_026")
        
        # df_xml = pd.read_csv()
        
        # print(df_xml, type(df_xml))
        
        h5_data = read_h5_data(h5_path)
        
        print("---- run the fast version")
        # mask = check_xy_in_coordinates_fast(df_xml, h5_data["coordinates"])
        
        print("shape of mask", mask.shape)
        print("sum of mask", np.sum(mask))
        print(">>> MASK: ", mask[:10])
    # 0 is back ground, 1 is tumor
    predict = np.random.randint(0, 2, size=(h5_data["coordinates"].shape[0], 1))
    print("predict.shape", predict.shape)
    
    
    tp = calculate_tp(mask, predict)
    fp = calculate_fp(mask, predict)
    tn = calculate_tn(mask, predict)
    fn = calculate_fn(mask, predict)
    dice = calculate_dice_score(mask, predict)
    iou = calculate_iou_score(mask, predict)

    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    print(f"Dice Score: {dice:.4f}")
    print(f"IoU Score: {iou:.4f}")    
    
    basename = 'tumor_026' 
    # basename = os.path.basename(scores_path).split(".")[0]
    slide = openslide.open_slide(os.path.join(args.slide_path, f'{basename}.tif'))
    (
        downsample_factor,
        new_width,
        new_height,
        original_width,
        original_height
    ) = rescaling_stat_for_segmentation(
        slide, downsampling_size=1096)

    scale_x = new_width / original_width
    scale_y = new_height / original_height
    h5_file_path = os.path.join(args.features_h5_path, f'{basename}.h5') 

    result = {} 
    with h5py.File(h5_file_path, "r") as f:
        coordinates= f['coordinates'][:]
    
    scaled_scores = mask
    print(">>>>mask", mask[:5])
    plot_dir = args.sanity_check_path 
    # plot_dir = os.path.join(args.plot_path, f'{args.ig_name}')
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)  # Delete the existing directory
    os.makedirs(plot_dir)  
    plot_path = os.path.join(plot_dir, f'{basename}.png')
    plot_heatmap_with_bboxes(
        scale_x, scale_y, new_height, new_width,
        coordinates,
        scaled_scores,
        name = "",
        save_path = plot_path
    ) 
    print("-> Save the plot at: ", plot_path)
    # scale_x, scale_y, new_height, new_width       

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
        os.makedirs(args.ground_truth_corr_path, exist_ok=True)   
        os.makedirs(args.ground_truth_path, exist_ok=True)    
        os.makedirs(args.sanity_check_path, exist_ok=True)   
        args.do_normalizing = True

    main(args) 
    
#  scp -r mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/sanity_check/tumor_026.png 