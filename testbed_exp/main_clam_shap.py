import os
import sys

project_dir = os.environ.get("PROJECT_DIR")
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

from src.bag_classifier.bag_classifier import Bag_Classifier
from utils.train_classifier.train_bag_classifer import * 
from utils.utils import setup_logger



from shap import datasets as shap_datasets  
from shap import GradientExplainer 


def load_config(config_file):
    # Load configuration from the provided YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config
 
def train_eval_bagcls(train_dataset, test_dataset):
    pass 

        
def main(args):
    if args.dry_run:
        print("Running the dry run")
    else:
        print("Running on full data") 

    
    train_dataset = FeaturesDataset(
        feature_folder=args.features_h5_path, 
        basename_list=args.train_list + args.test_list,
        # basename_list = args.train_list, 
        transform=None
    )
    
    test_dataset = FeaturesDataset(
        feature_folder=args.features_h5_path, 
        basename_list = args.test_list, 
        transform=None
    )
    print("Processing Pruning training dataset with length: ", len(train_dataset)) 
    print("Processing Pruning test dataset with length: ", len(test_dataset))  
    train_losses = [] 

    pruning_model = Bag_Classifier(
        gate=False, 
        size_arg="small", 
        dropout=0.15, 
        n_classes=2, 
        subtyping=False, 
        embed_dim=768
        )
    
    loss_fn = nn.CrossEntropyLoss()  # Common loss function for classification
    optimizer = optim.Adam(pruning_model.parameters(), lr=0.0001) 
     
    pruning_model = pruning_model.to(args.device) 
    n_classes = 2 
    epoch_num = 50
    file_name = os.path.basename(__file__)
    logger = setup_logger(f'./logs/pruning_{file_name}.txt')    
    

    
    # for epoch in range(epoch_num):
    #     train_loss = train_epoch(
    #         epoch, 
    #         pruning_model, 
    #         train_dataset,
    #         optimizer, 
    #         n_classes, 
    #         logger, 
    #         loss_fn, 
    #         checkpoint_filename=checkpoint_file, 
    #         save_last_epoch_checkpoint=True 
    #         )
    #     train_losses.append(train_loss)

    
    print("------Train loss:", [f"{loss:.4f}" for loss in train_losses])
    
    eval(
        pruning_model, 
        optimizer,
        test_dataset,
        n_classes, 
        logger, 
        loss_fn, 
        checkpoint_filename=checkpoint_file
        )  
    
    print("------Start Pruning") 

    # max_features_size = 0 
    # for features, _, _, _, _ in train_dataset:  # If using DataLoader
    #     batch_size = features.shape[0]
    #     if batch_size > max_features_size:
    #         max_features_size = batch_size

    # print("Max features size: ", max_features_size) 
    # bg_1 = torch.randn(max_features_size, 768) 
    # black_bg_1 = torch.zeros_like(bg_1).float() 
    # black_bg_1 = black_bg_1.unsqueeze(0) 
    # print("black_bg_1.shape", black_bg_1.shape)
    
    count=1
    total = len(train_dataset)
    
    for data in train_dataset: 
        start = time.time()
        features, label, patch_indices, coordinates, spixels_indices, file_basename = data   
        features, label = features.to(device), label[0].to(device) 
        # print(coordinates.shape)
        # print(spixels_indices.shape)
        # print(spixels_indices.shape)
        
        bg_1 = torch.randn(features.shape[0], 768) 
        black_bg_1 = torch.zeros_like(bg_1).float() 
        black_bg_1 = black_bg_1.unsqueeze(0)
        black_bg_1 = black_bg_1.to(device) 
         
        # print("black_bg_1.shape", black_bg_1.shape)
        
        to_explain = features.unsqueeze(0) 
        
        explainer = GradientExplainer(
            pruning_model, black_bg_1, local_smoothing=100, batch_size=1)  
        shap_values, indexes = explainer.shap_values(
            to_explain, nsamples=10, ranked_outputs=2, rseed=123)  
        
        
        shap_values = shap_values # Convert shap_values to NumPy array if needed
        indexes = indexes  # Convert indexes to NumPy array if needed
        if isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.cpu().numpy()  # Move to CPU (if needed) and convert to numpy
        if isinstance(indexes, torch.Tensor):
            indexes = indexes.cpu().numpy()  # Move to CPU (if needed) and convert to numpy
        shap_values = shap_values[0]
        # print(f"Shap values shape: {shap_values.shape}")  
        shap_values_sliced = shap_values[:, :, 0]  
        # print("shap_values_sliced.shape", shap_values_sliced.shape) 
        
        shap_values_avg = shap_values_sliced.mean(axis=1).squeeze() 
        # Specify the output HDF5 file path
        output_file = os.path.join(args.important_scores_path, f"{file_basename[0]}.h5")
        
        # Save shap_values and indexes to an HDF5 file
        with h5py.File(output_file, 'w') as f:
            # Create datasets for shap_values and indexes
            f.create_dataset('shap_values_avg', data=shap_values_avg)
            f.create_dataset('indexes', data=indexes)
            f.create_dataset("coordinates", data=coordinates) 
            
        print(f"Saved {count}/{total} shap_values and indexes to {output_file}")  
        count+=1 
        
        
        # shap_values = shap_values[0]
        # print(f"Shap values shape: {shap_values.shape}")  
        # shap_values_sliced = shap_values[:, :, 0]  
        # print("shap_values_sliced.shape", shap_values_sliced.shape) 
        
        # shap_values_avg = shap_values_sliced.mean(axis=1).squeeze()
        
         
        # print("shap_values_avg.shape", shap_values_avg.shape) 
        # print(shap_values_avg[:10]) 
        # min_val = np.min(shap_values_avg)
        # max_val = np.max(shap_values_avg)
        # median_val = np.median(shap_values_avg)
        # average_val = np.mean(shap_values_avg)
        # variance_val = np.var(shap_values_avg) 
         
        # print(f"Min: {min_val}, Max: {max_val}, Median: {median_val}, Average: {average_val}, Variance: {variance_val}")
        # print(f"---Complete the first features after {time.time()-start}" )
        # print("------") 
        
        

     
        # shap_values, indexes = explainer.shap_values(to_explain_padded, nsamples=3, ranked_outputs=2, rseed=123)   
        # unique_patch_indices = torch.unique(patch_indices)
        # if len(unique_patch_indices) < len(patch_indices):
    #         print("Duplicate patch indices found!")
    #     else:
    #         print("No duplicates in patch indices.") 
         
    # train_eval_bagcls(train_dataset, test_dataset)
    
# TODO:
# - GET THE RESULT OF CLASSIFICATION 
# - GET VISUALIZAITON OF THE FEATURES THAT HAS BEEN REMOVE AND KEEP (HEATMAP)
      
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
        args.patch_path = config.get('PATCH_PATH') # save all the patch (image)
        args.features_h5_path = config.get("FEATURES_H5_PATH") # save all the features
        args.checkpoint_path = config.get('CHECKPOINT_PATH') 
        args.important_scores_path = config.get('IMPORTANT_SCORES_PATH2') 
       
        #------------------  
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
            print(f"Created directory: {args.checkpoint_path}")
        else:
            print(f"Directory {args.checkpoint_path} already exists")
        checkpoint_file = os.path.join(args.checkpoint_path, "bag_classifer.pth") 
        #------------------
        if not os.path.exists(args.important_scores_path):
            os.makedirs(args.important_scores_path)
            print(f"Created directory: {args.important_scores_path}")
        else:
            print(f"Directory {args.important_scores_path} already exists")
        
        
        
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
    print("------------") 
    tr_n = [i for i in train_list if i.split("_")[0]=='normal']
    tr_r = [i for i in train_list if i.split("_")[0]=='tumor'] 
    print(" + Normal: ", len(tr_n))
    print(" + Tumor: ", len(tr_r))
    print("---")
    print(tr_n)
    print(tr_r)

    print("\n--Test list")
    t_n = [i for i in test_list if i.split("_")[0]=='normal']
    t_r = [i for i in test_list if i.split("_")[0]=='tumor'] 
    print(" + Normal: ",len(t_n))
    print(" + Tumor: ",len(t_r))
    print("------------")
    # [i for i in os.listdir(args.features_h5_path)]
        
    args.train_list = train_list
    args.test_list = test_list 
    
    main(args) 
    