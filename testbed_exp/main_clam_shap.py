import os
import sys

project_dir = os.environ.get("PROJECT_DIR")
print("project_dir", project_dir) 
sys.path.append(os.path.join(project_dir))
sys.path.append(os.path.join(project_dir, "includes", "shap"))

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
PROJECT_DIR = os.environ.get('PROJECT_DIR')
print("PROJECT DIR", PROJECT_DIR)
sys.path.append(PROJECT_DIR) 
 


import torch
import torch.nn as nn
import torch.optim as optim 

from data.merge_dataset import SuperpixelDataset, PatchDataset, SlidePatchesDataset
from data.classification_dataset import FeaturesDataset 

from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from utils import utils  
from src.important_scores import get_scoring_do_nothing
from src.pruning import get_pruning_do_nothing

from src.bag_classifier.clam import CLAM_MB
from utils.train_classifier.train_clam import * 
from utils.utils import setup_logger


def load_config(config_file):
    # Load configuration from the provided YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config
 
def train_eval_clam(train_dataset, test_dataset):
    model_clam = CLAM_MB(
        gate=False, 
        size_arg="small", 
        dropout=0.15, 
        k_sample=10, 
        n_classes=2, 
        subtyping=False, 
        embed_dim=768
        )
    
    loss_fn = nn.CrossEntropyLoss()  # Common loss function for classification
    optimizer = optim.Adam(model_clam.parameters(), lr=0.0001) 
     
    model_clam = model_clam.to(args.device) 
    n_classes = 2 
    bag_weight = 0.7
    epoch_num = 50
    file_name = os.path.basename(__file__)
    logger = setup_logger(f'./logs/{file_name}.txt')
    
    print('>>> Ready to test 1 epoch') 
    
    train_losses = [] 
    for epoch in range(epoch_num):
        train_loss = train_epoch(
            epoch, 
            model_clam, 
            train_dataset,
            optimizer, 
            n_classes, 
            bag_weight, 
            logger, 
            loss_fn
            )
        train_losses.append(train_loss)

    print("Train loss:", [f"{loss:.2f}" for loss in train_losses])
    
    eval(
        epoch, 
        model_clam, 
        test_dataset,
        n_classes, 
        bag_weight, 
        logger, 
        loss_fn
        )
    
def main(args):
    if args.dry_run:
        print("Running the dry run")
    else:
        print("Running on full data") 

    
    train_dataset = FeaturesDataset(
        feature_folder=args.features_h5_path, 
        basename_list = args.train_list, 
        transform=None
    )
    
    test_dataset = FeaturesDataset(
        feature_folder=args.features_h5_path, 
        basename_list = args.test_list, 
        transform=None
    )
    from src.pruning import Uniform
     
    
    
    print("Processing training dataset with length: ", len(train_dataset)) 
    print("Processing test dataset with length: ", len(test_dataset)) 
    train_eval_clam(train_dataset, test_dataset)
    
         
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
        
        args.batch_size = config.get('batch_size')
        args.feature_extraction_model = config.get('feature_extraction_model')

        
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # _list = ['normal_031', 'tumor_024', 'normal_047', 'tumor_009', 'tumor_057', 'normal_093', 'normal_051', 'tumor_014', 'tumor_015', 'tumor_067', 'normal_003', 'tumor_084', 'tumor_101', 'normal_148', 'normal_022', 'tumor_012', 'normal_039', 'normal_084', 'normal_101', 'tumor_010', 'normal_088', 'normal_155', 'normal_087', 'normal_016', 'normal_114', 'normal_024', 'tumor_048', 'normal_078', 'tumor_049', 'tumor_086'] 
    avai_items = [i.split('.')[0] for i in os.listdir(args.features_h5_path)]
    
    example_list = avai_items   
     
    print(example_list)
    max_len = len(example_list)
    train_num = int(0.7*max_len)
    test_num = max_len - train_num 
    import random
    random.seed(123)
    random.shuffle(example_list)
    
    train_list = random.sample(example_list, train_num)
    test_list = [item for item in example_list if item not in train_list] 
    
    print("Total example {}, train {} test {}")
    print("\n--Train list")
    tr_n = [i for i in train_list if i.split("_")[0]=='normal']
    tr_r = [i for i in train_list if i.split("_")[0]=='tumor'] 
    print(" + Normal: ", len(tr_n))
    print(" + Tumor: ", len(tr_r))

    print("\n--Test list", test_list)
    t_n = [i for i in test_list if i.split("_")[0]=='normal']
    t_r = [i for i in test_list if i.split("_")[0]=='tumor'] 
    print(" + Normal: ",len(t_n))
    print(" + Tumor: ",len(t_r))
    
    # [i for i in os.listdir(args.features_h5_path)]
        
    args.train_list = train_list
    args.test_list = test_list 
    
    main(args) 
    