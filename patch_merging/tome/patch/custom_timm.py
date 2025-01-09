# from typing import Tuple

# import torch
# from timm.models.vision_transformer import Attention, Block, VisionTransformer

# from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
# from tome.utils import parse_r


# import os
# import sys  
# import torch
# import torch.optim as optim
# import torch.nn as nn
# import time
# from collections import deque


# PROJECT_DIR = os.environ.get('PROJECT_DIR') 
# sys.path.append(os.path.join(PROJECT_DIR))
# sys.path.append(os.path.join(PROJECT_DIR, "src"))


# from src.custom_layers import (
#     NeighborAggregator, LastSigmoid, MILAttentionLayer, CustomAttention) 
# from src.nystromformer import NystromAttention  

# class CAMIL(nn.Module):
#     def __init__(self, args):
#         """
#         Build the architecture of the Graph Attention Network (Graph Att Net)
#         Parameters:
#         ----------
#         args: Argument class containing all the necessary model parameters
#         """
#         super(CAMIL, self).__init__()

#         self.input_shape = args.input_shape
#         self.args = args
#         self.n_classes = args.n_classes
#         self.subtyping = args.subtyping

#         # Initialize layers and components of the model
#         self.attcls = MILAttentionLayer(
#             input_dim=512, 
#             weight_params_dim=128, 
#             use_gated=False, 
#             )
        
#         self.neigh = NeighborAggregator(output_dim=1)
#         self.nyst_att = NystromAttention(
#             dim=512, 
#             dim_head=64, 
#             heads=8, 
#             num_landmarks=256, 
#             pinv_iterations=6
#             )
#         self.encoder = Encoder()

#         # Define the final fully connected layer for classification
#         if self.subtyping:
#             self.class_fc = LastSigmoid(
#                 input_dim=512,output_dim=self.n_classes, subtyping=self.subtyping, pooling_mode='sum')
#         else:
#             self.class_fc = LastSigmoid(
#                 input_dim=512, output_dim=1, subtyping=self.subtyping, pooling_mode='sum')


#     def forward(self, bag, adjacency_matrix):
#         """
#         Define the forward pass of the model
#         """
#         # Pass through the encoder
#         xo, alpha = self.encoder([bag, adjacency_matrix])
#         k_alpha = self.attcls(xo)
        
#         attn_output = torch.mul(k_alpha, xo)
#         out = self.class_fc(attn_output)s
#         return out, alpha, k_alpha
    
# class Encoder(nn.Module):
#     def __init__(
#         self, 
#         embedding_dim=512, weight_params_dim=256):
#         super(Encoder, self).__init__()
        
#         self.embedding_dim = embedding_dim 
#         self.weight_params_dim = weight_params_dim 
#         # Initialize components like custom attention, neighbor aggregator, and Nystrom Attention
#         self.custom_att = CustomAttention(
#             input_dim=self.embedding_dim, 
#             weight_params_dim=self.weight_params_dim
#             ) 
#         self.wv = nn.Linear(self.embedding_dim, self.embedding_dim)  # Linear layer for value transformation
        
#         self.neigh = NeighborAggregator(output_dim=1)
        
#         self.nyst_att = NystromAttention(
#             dim=self.embedding_dim, dim_head=64, heads=8, num_landmarks=16, pinv_iterations=6) #256 

#     def forward(self, inputs):
#         dense = inputs[0]  # Dense input data (e.g., feature vectors)
#         sparse_adj = inputs[1]  # Sparse adjacency matrix for graph structure
#         encoder_output = self.nyst_att(dense.unsqueeze(0))  # Add an extra dimension to the input (batch dimension) 
#         xg = encoder_output.squeeze(0)  # Remove the extra dimension (batch dimension)
#         # Combine the encoder output with the original dense input (residual connection)
#         encoder_output = xg + dense
#         attention_matrix = self.custom_att(encoder_output) 
#         norm_alpha, alpha = self.neigh([attention_matrix, sparse_adj]
        
#         # Transform the dense input using the value transformation layer (wv)
#         value = self.wv(dense)
#         # Compute element-wise multiplication of attention scores and transformed values
#         norm_alpha = norm_alpha.unsqueeze(1)  # Reshape to [300, 1]
#         norm_alpha = norm_alpha.expand(-1, value.size(1))  # Expand to [300, 512] 
#         xl = norm_alpha * value
#         wei = torch.sigmoid(-xl)  # Sigmoid activation for weighting
#         squared_wei = wei ** 2
#         xo = (xl * 2 * squared_wei) + 2 * encoder_output * (1 - squared_wei)
#         return xo, alpha
     