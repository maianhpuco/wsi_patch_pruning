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
from src.camil import Encoder

if __name__ == "__main__":
    # Define the input dimensions
    batch_size = 8  # Number of samples in the batch
    seq_length = 128  # Length of the sequence (e.g., number of tokens or features)
    dim = 512  # Dimensionality of the feature vectors
    num_landmarks = 256  # Number of landmarks for Nystrom Attention
    heads = 8  # Number of attention heads
    dim_head = 64  # Dimensionality of each attention head
    depth = 4  # Number of transformer layers in Nystromformer
    pinv_iterations = 6  # Number of iterations for the Moore-Penrose pseudo-inverse calculation
    attn_dropout = 0.1  # Dropout rate for attention
    ff_dropout = 0.1  # Dropout rate for feed-forward layers

    # Create a dummy input tensor
    # Simulate a batch of input data with random values
    # x = torch.randn(batch_size, seq_length, dim) 
    # x = torch.randn(1, 300, 512)
 
    # print("input shape", x.shape)
    # mask = None

    # # Initialize the NystromAttention model separately for testing
    # nystrom_attention = NystromAttention(
    #     dim=dim,
    #     dim_head=dim_head,
    #     heads=heads,
    #     num_landmarks=num_landmarks,
    #     pinv_iterations=pinv_iterations,
    #     dropout=attn_dropout
    # )

    # # Pass the dummy input tensor through the NystromAttention model
    # nystrom_attention_output = nystrom_attention(x, mask=mask, return_attn=False)

    # # Print the shape of the output from NystromAttention
    # print("Output shape from NystromAttention:", nystrom_attention_output.shape)
    
    # # Initialize the Encoder
    # encoder = Encoder()

    # dense_input = torch.randn(12345, 512)  
    # sparse_adj = torch.randn(12345, 12345) 

    # encoded_output, attention_scores = encoder([dense_input, sparse_adj])

    # print("Encoded Output Shape: ", encoded_output.shape)
    # print("Attention Scores Shape: ", attention_scores.shape)
  

    # Generate a dense tensor for the input
      # Tensor of shape (12345, 512)

    # Create a sparse adjacency matrix (random sparse matrix for the example)
    num_nodes = 1234
    density = 0.01  # 1% non-zero entries
    
    indices = torch.nonzero(torch.rand(num_nodes, num_nodes) < density)
    print("indices", indices.shape)
    dense_input = torch.randn(num_nodes, 512) 
    # Generate random values for the non-zero entries
    values = torch.randn(indices.size(0))

    # Create the sparse tensor in COO format
    sparse_adj = torch.sparse_coo_tensor(indices.t(), values, size=(num_nodes, num_nodes))
    print("-- input shape ")
    print("dense input", dense_input.shape)
    print("sparse_adj: ", sparse_adj.shape)
    
    # # Initialize the Encoder
    encoder = Encoder()

    # Pass the dense input and sparse adjacency matrix to the encoder
    encoded_output, attention_scores = encoder([dense_input, sparse_adj])
    