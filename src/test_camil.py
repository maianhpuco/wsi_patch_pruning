import os
import sys  
import torch
import torch.optim as optim
import torch.nn as nn
import time
from collections import deque


PROJECT_DIR = os.environ.get('PROJECT_DIR') 
sys.path.append(os.path.join(PROJECT_DIR))
sys.path.append(os.path.join(PROJECT_DIR, "src"))
from src.nystromformer import NystromAttention     
from src.camil import CAMIL 
import numpy as np
from src.nystromformer import NystromAttention     
from src.camil import Encoder 
def create_sparse_adj_matrix(grid_size=16):
    """
    Create a sparse adjacency matrix for a grid-based graph, where each node is connected to its 8 neighbors (left, right, up, down, and 4 diagonals).
    The adjacency matrix is weighted using the formula exp(-sqrt(i)), where i is the index of the node.

    Args:
        grid_size (int): The size of the grid (grid_size x grid_size). Default is 16.

    Returns:
        torch.sparse.Tensor: The sparse adjacency matrix in COO format.
    """
    
    num_nodes = grid_size * grid_size  # Total number of nodes (e.g., 256 for 16x16 grid)
    
    # Initialize the adjacency matrix with zeros
    mask = np.zeros((num_nodes, num_nodes))
    
    # Define the 8 neighbors for each node in the grid
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # Create the adjacency matrix with weights exp(-sqrt(i)) for neighbors
    weights = np.zeros_like(mask)

    for i in range(grid_size):
        for j in range(grid_size):
            # Get the index for the current node (flattened 256x256 grid)
            node_idx = i * grid_size + j

            # Iterate through the neighbors and set weights using the formula
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                # Check if the neighbor is within bounds
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    neighbor_idx = ni * grid_size + nj
                    # Calculate the weight using the formula exp(-sqrt(i))
                    weight = np.exp(-np.sqrt(node_idx))
                    mask[node_idx, neighbor_idx] = weight
                    weights[node_idx, neighbor_idx] = weight  # Store the weight

    # Convert the mask to a sparse tensor in COO format
    mask_tensor = torch.tensor(mask, dtype=torch.float32)  # Dense tensor with the weights
    sparse_adj = torch.sparse_coo_tensor(mask_tensor.nonzero().t(), mask_tensor[mask_tensor != 0], size=mask_tensor.size())

    return sparse_adj


class Args:
    def __init__(self):
        self.input_shape = (5, 512)  # (seq_len, feature_dim)
        self.n_classes = 2  # Number of classes for classification
        self.subtyping = False  # Boolean flag for subtyping 

if __name__ == "__main__": 
    # Example instantiation of the model
    args = Args()
    
    model = CAMIL(args) 
    grid_size = 40 
    batch_size = 1 # Two graphs in the batch
    seq_len =  grid_size * grid_size  # nodes per graph
    feature_dim = 512  # Each node has 512 features
    bag = torch.randn(seq_len, feature_dim)  # Random tensor for node features
    print("+ bag:", bag.shape)

    # Create the sparse tensor in COO format 
    # # if the image is 16x16 then then attention matrix is 256*256
    sparse_adj = create_sparse_adj_matrix(grid_size=grid_size)  # Create the sparse adjacency matrix for a 16x16 grid
    print("+ sparse_adj: ", sparse_adj.shape)
    out, alpha, k_alpha  = model(bag, sparse_adj)

    print("out", out.shape)
    print("alpha", alpha.shape)
    print("k_alpha", k_alpha.shape)
