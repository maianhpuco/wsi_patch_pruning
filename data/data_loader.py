import torch
from torch.utils.data import Dataset
import h5py
# import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

class CustomDataset(Dataset):
    def __init__(self, file_paths, label_file, shuffle=False):
        """
        Args:
            filenames (list): List of HDF5 file paths containing the data.
            label_file (str): Path to the CSV file containing labels for each sample.
            shuffle (bool): Whether to shuffle the data or not. Default is False.
        """
        self.file_paths = file_paths
        self.label_file = label_file
        self.shuffle = shuffle
        
        # Load labels from CSV file
        self.labels_df = pd.read_csv(label_file)
        
        # Optionally shuffle filenames if required
        self.indices = np.arange(len(filenames))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.filenames)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample.
        
        Returns:
            tuple: (feature, adjacency matrix, label)
        """
        # Get the index of the file
        file_idx = self.indices[index]
        filename = self.filenames[file_idx]
        
        # Load the HDF5 file
        with h5py.File(file_paths, "r") as f:
            features = f['features'][:]
            neighbor_indices = f['indices'][:]
            values = f['similarities'][:]
        print("--- item")
        print(features.shape)
        
        #     values = np.nan_to_num(values)
            
        #     # Get the label
        #     base_name = os.path.splitext(os.path.basename(filename))[0]
        #     label = self.labels_df.loc[self.labels_df["slide_id"] == base_name, "slide_label"].values[0]
        
        # print("- label", label.shape)
        
        # # Process adjacency matrix
        # Idx = neighbor_indices[:, :8]
        # rows = np.asarray([[enum] * len(item) for enum, item in enumerate(Idx)]).ravel()
        # columns = Idx.ravel()

        # neighbor_matrix = values[:, 1:]
        # normalized_matrix = normalize(neighbor_matrix, norm="l2")

        # similarities = np.exp(-normalized_matrix)
        # values = np.concatenate((np.max(similarities, axis=1).reshape(-1, 1), similarities), axis=1)
        # values = values[:, :8]
        # values = values.ravel().tolist()
        # sparse_coords = list(zip(rows, columns))

        # # Create sparse adjacency matrix
        # sparse_matrix = torch.sparse_coo_tensor(
        #     indices=torch.tensor(sparse_coords).t(),
        #     values=torch.tensor(values),
        #     size=(features.shape[0], features.shape[0])
        # )
        # print("- sparse_matrix", sparse_matrix.shape)

        # # Convert features to tensor
        # features_tensor = torch.tensor(features, dtype=torch.float32)

        # return features_tensor, sparse_matrix, torch.tensor(label, dtype=torch.float32)

