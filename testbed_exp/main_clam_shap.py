import os
import sys

project_dir = os.environ.get("PROJECT_DIR")
print("project_dir", project_dir) 
sys.path.append(os.path.join(project_dir))
sys.path.append(os.path.join(project_dir, "includes", "shap"))

import torch 
# import torchvision
# import torchvision.transforms as transforms
import numpy as np
import json

from torch.utils.data import DataLoader, Dataset
from PIL import Image

import torch
import torch.nn as nn 

# from shap import GradientExplainer  
# from shap import datasets as shap_datasets 

# import matplotlib.pyplot as plt
# import numpy as np
# from skimage.transform import resize  
 

# # Load pre-trained VGG16 model from torchvision
# model = torchvision.models.vgg16(pretrained=True)
# model.eval()  # Set the model to evaluation mode a


# # Load a small dataset (Imagenet50) for testing
# X, y = shap_datasets.imagenet50()
# to_explain = X[[39, 41]]  # Choose two images to explain
# to_explain = X[[41]]
 
# print(X.shape)
# print(to_explain.shape)
# X_tensor = torch.from_numpy(X).float() 
# X_tensor = X_tensor.permute(0, 3, 1, 2) # Change the shape of the input tensor 

# to_explain_tensor = torch.from_numpy(to_explain).float() 
# to_explain_tensor = to_explain_tensor.permute(0, 3, 1, 2) # Change the shape of the input tensor     


# layer = model.features[7] # Choose the first layer of the model  
# # explainer = GradientExplainer((model, layer), X_tensor) 
# explainer = GradientExplainer(model, X_tensor, local_smoothing=100)  
# to_explain_tensor_out = model.features[:7](to_explain_tensor) 
# print("to_explain_tensor_out.shape", to_explain_tensor_out.shape) 
# print("---------------shap values-----------------") 


# if __name__=='__main__':
#     print("true") 

