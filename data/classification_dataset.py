import torch
from torch.utils.data import Dataset
import h5py
import os
import sys 
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import glob 
class CustomDataset(Dataset):
    def __init__(
        self, 
        feature_folder, 
        feature_file_end ='h5',  
        shuffle=True, 
        ):
        """
        Args:
            label_file (str): Path to the CSV file containing labels for each sample.
            shuffle (bool): Whether to shuffle the data or not. Default is False.
        """
        self.feature_folder = feature_folder
        self.shuffle = shuffle
        self.paths = glob.glob()
        self.indices = np.arange(len(self.names))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
        
    def get_feature_path(self, basename):
        return os.path.join(self.feature_folder, 
                            f'{basename}.h5')
    
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.indices)

    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample.
        
        Returns:
            tuple: (feature, adjacency matrix, label)
        """
        # Get the index of the file
        file_idx = self.indices[index]
        file_basename = self.names[file_idx]
        file_path = self.get_feature_path(file_basename)
        # Load the HDF5 file
        with h5py.File(file_path,  "r") as f:
            features = f['features'][:]
            neighbor_indices = f['indices'][:] # how to get these indice 
            values = f['similarities'][:]
            values = np.nan_to_num(values)
            label = self.name_label_dict[file_basename]
            
        label_tensor = torch.tensor([label], dtype=torch.float32).view(1, 1)
         
        Idx = neighbor_indices[:, :8]
        rows = np.asarray([[enum] * len(item) for enum, item in enumerate(Idx)]).ravel()
        columns = Idx.ravel()
        
        neighbor_matrix = values[:, 1:]
        normalized_matrix = normalize(neighbor_matrix, norm="l2")

        similarities = np.exp(-normalized_matrix)
        values = np.concatenate((np.max(similarities, axis=1).reshape(-1, 1), similarities), axis=1)
        values = values[:, :8]
        values = values.ravel().tolist()
        sparse_coords = list(zip(rows, columns))

        # Create sparse adjacency matrix
        sparse_matrix = torch.sparse_coo_tensor(
            indices=torch.tensor(sparse_coords).t(),
            values=torch.tensor(values),
            size=(features.shape[0], features.shape[0])
        )
        
        features_tensor = torch.tensor(features, dtype=torch.float3

        return features_tensor, sparse_matrix, label_tensor  

