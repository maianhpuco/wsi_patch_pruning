# import os
# import sys 
# from tqdm import tqdm 
# import glob
# import pandas as pd
# import numpy as np
# import argparse
# import time   
# import timm 
# import yaml 
# import h5py 
# import openslide

# import torch
# import torch.nn as nn
# import torch.optim as optim 

# from data.merge_dataset import SuperpixelDataset, PatchDataset, SlidePatchesDataset
# from data.classification_dataset import FeaturesDataset 

# from torchvision import transforms
# from torch.utils.data import DataLoader
# from PIL import Image
# from utils import utils  
# from src.importance_scores import get_scoring_do_nothing
# from src.pruning import get_pruning_do_nothing

# from src.bag_classifier.clam import CLAM_MB
# from utils.train_classifier.train_clam import * 
# from utils.utils import setup_logger


# #TODO
# # clean the code of the main function -> make it cleaner

# SCORING_FUNCTION_MAP = {
#     "get_scoring_do_nothing": get_scoring_do_nothing,
# }


# PRUNING_FUNCTION_MAP = {
#     "get_pruning_do_nothing": get_pruning_do_nothing,
# }   

# def load_config(config_file):
#     # Load configuration from the provided YAML file
#     with open(config_file, 'r') as f:
#         config = yaml.safe_load(f)
#     return config

# # def train_clam(features_dataset):
# #     loss_fn = nn.CrossEntropyLoss()  # Common loss function for classification
# #     optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    
# #     n_classes = 2 
# #     bag_weight = 0.7
# #     epoch_num = 30 
# #     model_clam = CLAM_MB(
# #         gate=False, 
# #         size_arg="small", 
# #         dropout=0.15, 
# #         k_sample=20, 
# #         n_classes=2, 
# #         subtyping=False, 
# #         embed_dim=768
# #         )
# #     model_clam = model_clam.to(args.device) 
    
# #     logger = setup_logger('./logs/test_clam.txt')
    
# #     # print('>>> Ready to test 1 epoch') 

# #     train_losses = [] 
# #     for epoch in range(epoch_num):
# #         train_loss = train_epoch(
# #             epoch, 
# #             model_clam, 
# #             features_dataset,
# #             optimizer, 
# #             n_classes, 
# #             bag_weight, 
# #             logger, 
# #             loss_fn
# #             )
# #         train_losses.append(train_loss)

# #     print("Train loss:", [f"{loss:.2f}" for loss in train_losses]) 
    
# def train_dsmil(features_dataset):
#     pass 


# PROJECT_DIR = os.environ.get('PROJECT_DIR')
# sys.path.append(os.path.join(PROJECT_DIR))  

# # SLIDE_DIR = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16'
# # example_list = ['normal_072', 'normal_001', 'normal_048', 'tumor_026', 'tumor_031', 'tumor_032']
# example_list = ['normal_072', 'normal_001', 'normal_048', 'tumor_031', 'tumor_032']  

# def main(args):
#     if args.dry_run:
#         print("Running the dry run")
#     else:
#         print("Running on full data") 
    
#     model = timm.create_model(args.feature_extraction_model, pretrained=True)  # You can choose any model
#     model = model.to(args.device) 
#     model.eval()   
    
#     features_dataset = FeaturesDataset(
#         feature_folder=args.features_h5_path, 
#         transform=None
#     )
    
#     train_dsmil(features_dataset)


#     # adding testing set here!  
        
         
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dry_run', type=bool, default=False)
#     parser.add_argument('--config_file', default='ma_exp002')
#     args = parser.parse_args()
    
#     if os.path.exists(f'./testbest_config/{args.config_file}.yaml'):
#         config = load_config(f'./testbest_config/{args.config_file}.yaml')
#         args.use_features = config.get('use_features', True)
        
#         args.slide_path = config.get('SLIDE_PATH')
#         args.json_path = config.get('JSON_PATH')
#         args.spixel_path = config.get('SPIXEL_PATH')
#         args.patch_path = config.get('PATCH_PATH') # save all the patch (image)
#         args.features_h5_path = config.get("FEATURES_H5_PATH") # save all the features
         
#         print(args.features_h5_path)
        
#         os.makedirs(args.features_h5_path, exist_ok=True)  
        
#         args.scoring_function = SCORING_FUNCTION_MAP.get(
#             config.get("scoring_function")
#         )
#         args.pruning_function = PRUNING_FUNCTION_MAP.get(
#             config.get('pruning_function') 
#         )
#         args.batch_size = config.get('batch_size')
#         args.feature_extraction_model = config.get('feature_extraction_model')
        
#         # args.scoring_function("")
#         # args.pruning_function("")
        
#     args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     example_list = ['normal_031', 'tumor_024', 'normal_047', 'tumor_009', 'tumor_057', 'normal_093', 'normal_051', 'tumor_014', 'tumor_015', 'tumor_067', 'normal_003', 'tumor_084', 'tumor_101', 'normal_148', 'normal_022', 'tumor_012', 'normal_039', 'normal_084', 'normal_101', 'tumor_010', 'normal_088', 'normal_155', 'normal_087', 'normal_016', 'normal_114', 'normal_024', 'tumor_048', 'normal_078', 'tumor_049', 'tumor_086'] 

#     avai_items = [i.split('.')[0] for i in os.listdir(args.patch_path)]
#     items_not_in_json = [item for item in example_list if item not in avai_items] 
#     example_list = items_not_in_json    
#     print(example_list)
    
#     main(args) 
    