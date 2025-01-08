import os
import sys 
import torch
import glob
import pandas as pd
import numpy as np
import argparse
from data.merge_dataset import SuperpixelDataset, PatchDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import timm 
import patch_merging

 # Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the patch to 256x256
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
    # You can add other transformations like RandomHorizontalFlip, RandomRotation, etc.
])
 

PROJECT_DIR = os.environ.get('PROJECT_DIR')
# SLIDE_DIR = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16'
example_list = ['normal_072', 'normal_001', 'normal_048', 'tumor_026', 'tumor_031', 'tumor_032']

SLIDE_PATH = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16/images'
JSON_PATH = '/project/hnguyen2/mvu9/camelyon16/json_files'
 
 
sys.path.append(os.path.join(PROJECT_DIR))
# Load a pre-trained ViT model from timm

model = timm.create_model('vit_base_patch16_224', pretrained=True)  # You can choose any model
model.eval()  # Set the model to evaluation mode 


def main(): 
    model_merge = timm.create_model("vit_base_patch16_224", pretrained=True) 
    patch_merging.patch.timm(model_merge) 
    tokens = torch.randn(1, 3030, 768)  
    model_merge.eval()
    with torch.no_grad():
        output = model_merge(tokens)
 
        
    wsi_paths = glob.glob(os.path.join(SLIDE_PATH, '*.tif'))
    wsi_paths = [path for path in wsi_paths if os.path.basename(path).split(".")[0] in example_list]
    json_folder = JSON_PATH  
    
    dataset = SuperpixelDataset(
        slide_paths=wsi_paths,
        json_folder=json_folder,
        )
    
    import time 
    
    # llop though each WSI image 
    for wsi_data in dataset:
        #loop though each superpixel
        for foreground_idx, region_np, superpixel_extrapolated in wsi_data:
            start = time.time()
            print("foreground_id", foreground_idx)
            print("region shape", region_np.shape)
            print("superpixel shape", superpixel_extrapolated.shape) 
            patch_dataset = PatchDataset(
                region_np,
                superpixel_extrapolated,
                patch_size = (224, 224),
                transform = transform, 
                return_feature=True,  # Enable feature extraction
                model=model 
            )
            patch_dataloader = DataLoader(patch_dataset, batch_size=32, shuffle=True)
            all_features = [] 
            for features, patches, bboxes in patch_dataloader: 
                flattened_features = features.view(-1, features.shape[-1]) 
                all_features.append(flattened_features)
                 
            ts_all_features_of_superpixel = torch.cat(all_features, dim=0) 
            ts_all_features_of_superpixel=ts_all_features_of_superpixel[None, ...] 

            print(">> feature output size", ts_all_features_of_superpixel.shape)
            model_merge.r = 8 
            out = model_merge(ts_all_features_of_superpixel) 
            print(f"r = {model_merge.r} first 5 layers 's most likely class",out.topk(5).indices[0].tolist()) 
            print(out.shape) 
            
            print("Time to finish a superpixel", time.time() - start, "second")
        break
    
if __name__ == '__main__':
    main()
#checkpoint_path camelyon16_20241118-1932_completed.pth
# /project/hnguyen2/mvu9/camelyon16_features_data

# cp -r /home/mvu9/digital_pathology/camil_pytorch/data/camelyon16_features/* /project/hnguyen2/mvu9/camelyon16_features_data/ 
  