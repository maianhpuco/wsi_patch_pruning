import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MILAttentionLayer(nn.Module):
    def __init__(
        self, 
        input_dim=512, 
        weight_params_dim=128, 
        kernel_initializer='xavier_uniform', 
        use_gated=False
        ):
        """
        Initialize the MILAttentionLayer class.
        
        Args:
            weight_params_dim (int): Dimension of the weight matrices.
            kernel_initializer (str): The initialization method for the weight matrices. Default is 'xavier_uniform'.
            use_gated (bool): If True, uses a gated mechanism.
        """
        super(MILAttentionLayer, self).__init__()
        self.input_dim = input_dim
        # Initialize the parameters
        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated
        self.kernel_initializer = kernel_initializer

        # Initialize weight matrices for Q (queries) and K (keys)
        self.v_weight_params = nn.Parameter(torch.empty(self.input_dim, self.weight_params_dim))  # v weight matrix
        self.w_weight_params = nn.Parameter(torch.empty(self.weight_params_dim, 1))  # w weight matrix

        # If the gated mechanism is used, initialize u_weight_params
        if self.use_gated:
            self.u_weight_params = nn.Parameter(torch.empty(self.input_dim, self.weight_params_dim))
        else:
            self.u_weight_params = None

        # Initialize weights using the specified initializer
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization (equivalent to glorot_uniform in TensorFlow)."""
        if self.kernel_initializer == "xavier_uniform":
            nn.init.xavier_uniform_(self.v_weight_params)  # Initialize v weight matrix
            nn.init.xavier_uniform_(self.w_weight_params)  # Initialize w weight matrix
            if self.use_gated:
                nn.init.xavier_uniform_(self.u_weight_params)  # Initialize u weight matrix if gated

    def forward(self, inputs):
        """
        Forward pass for the MILAttentionLayer.
        
        Args:
            inputs (torch.Tensor): Input tensor representing instances for the attention mechanism.
        
        Returns:
            torch.Tensor: The attention scores (alpha) after applying softmax.
        """
        # Compute the attention scores for the input instances
        instances = self.compute_attention_scores(inputs)

        # Apply softmax over the computed attention scores to normalize them
        alpha = F.softmax(instances, dim=0)

        return alpha

    def compute_attention_scores(self, instance):
        """
        Compute the attention scores using the formula tanh(v * h_k^T).
        
        Args:
            instance (torch.Tensor): Input tensor of instances.

        Returns:
            torch.Tensor: Computed attention scores.
        """
        original_instance = instance        
        # Compute tanh(v * h_k^T)
        # instance.shape = (batch_size, seq_len, 512), v_weight_params.shape = (512, weight_params_dim)
        instance = torch.tanh(torch.matmul(instance, self.v_weight_params))  # Attention computation
        # print("- tanh result: ", instance.shape)
        # If gated mechanism is used, apply gating mechanism
        if self.use_gated:
            # print("-sigmoid_result", sigmoid_result.shape)
            instance = instance * torch.sigmoid(torch.matmul(original_instance, self.u_weight_params)) 

        # Compute w^T * (tanh(v * h_k^T)) to obtain attention scores
        attention_scores = torch.matmul(instance, self.w_weight_params)

        return attention_scores

class NeighborAggregator(nn.Module):
    def __init__(self, output_dim):
        """
        Initializes the NeighborAggregator module. This module performs neighborhood aggregation in a graph-based model.

        Args:
            output_dim (int): The output dimension for the aggregation, though it's not directly used in this implementation.
        """
        super(NeighborAggregator, self).__init__()
        self.output_dim = output_dim  # The dimension of the output (not used here but could be useful in extended models)

    def forward(self, inputs):
        """
        The forward pass that aggregates neighborhood information using the adjacency matrix.

        Args:
            inputs (tuple): A tuple of two tensors:
                - data_input (torch.Tensor): A tensor of shape (batch_size, seq_len, feature_dim), representing node feature vectors.
                - adj_matrix (torch.Tensor): A sparse adjacency matrix of shape (batch_size, seq_len, seq_len), 
                  indicating the graph structure (1 for connection, 0 for no connection).
        
        Returns:
            tuple: 
                - alpha (torch.Tensor): A tensor of attention scores computed using softmax, representing the importance of each node's neighbors.
                - reduced_sum (torch.Tensor): The raw aggregated sum of features of each node's neighbors.
        """
        
        data_input = inputs[0]  # Extract the dense input data (node feature vectors)
        adj_matrix = inputs[1]  # Extract the adjacency matrix (graph structure)
        # # Ensure the adjacency matrix is in COO format to access the indices and values
        # indices = adj_matrix._indices()  # Get indices of non-zero entries in the adjacency matrix
        # values = adj_matrix._values()  # Get the corresponding values (the actual weights or connections)
        # # Indices give us the rows and columns of the sparse adjacency matrix
        # row_indices = indices[0]  # The row indices of the non-zero elements
        # col_indices = indices[1]  # The column indices of the non-zero elements
        adj_matrix_dense = adj_matrix.to_dense() 
        # selected_rows = data_input[row_indices]  # This selects the rows from `data_input` corresponding to the row indices
        sparse_data_input =  adj_matrix_dense * data_input # values should be reshaped to match the feature dimension
    
        # print(">> neighbor:", sparse_data_input.shape)
 
        # Perform element-wise multiplication between the adjacency matrix and the node features (data_input)
        # This operation sets the features of non-neighboring nodes to 0, keeping only the features of neighboring nodes.
        # sparse_data_input = adj_matrix * data_input
        
        # Aggregate the information by summing over the neighbors (rows of the adjacency matrix)
        # This operation results in a tensor where each element represents the summed features of the neighboring nodes.
        reduced_sum = sparse_data_input.sum(dim=1)  # Sum along the second dimension (seq_len), aggregating features of neighbors

        # Apply softmax to the aggregated sum to get attention scores for each node's neighborhood.
        # The softmax operation normalizes the values so that they sum to 1 and act as importance weights.
        alpha = F.softmax(reduced_sum, dim=0)  # Apply softmax over the aggregated sum to normalize the scores
        
        return alpha, reduced_sum  # Return the attention scores and raw aggregated sum

class LastSigmoid(nn.Module):
    def __init__(self,
                 input_dim, 
                 output_dim, subtyping, kernel_initializer='glorot_uniform', bias_initializer='zeros', pooling_mode="sum", use_bias=True):
        super(LastSigmoid, self).__init__()
        
        # Initialize key parameters such as the output dimension, subtyping flag, and pooling mode
        self.output_dim = output_dim
        self.subtyping = subtyping
        self.pooling_mode = pooling_mode
        self.use_bias = use_bias
        
        # Define the weight kernel (output transformation matrix)
        self.kernel = nn.Parameter(torch.randn(512, output_dim)) 
        
        # Optionally add a bias term
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.bias = None

    def forward(self, x):
        # Pooling: Apply either max pooling or sum pooling based on the pooling_mode
        if self.pooling_mode == 'max':
            x = x.max(dim=0, keepdim=True)[0]
        elif self.pooling_mode == 'sum':
            x = x.sum(dim=0, keepdim=True)

        # Perform the final transformation
        if self.subtyping:
            x = torch.matmul(x, self.kernel)
            if self.use_bias:
                x += self.bias
            return F.softmax(x, dim=-1)  # For subtyping, use softmax
        else:
            x = torch.matmul(x, self.kernel)
            if self.use_bias:
                x += self.bias
            return torch.sigmoid(x)  # For binary classification, use sigmoid

class CustomAttention(nn.Module):
    """
    Implements the Q(ti) * K(ti) operation in the attention mechanism, preparing the input to be multiplied 
    with the attention weights (s_ij) later
    Args:
        weight_params_dim (int): The dimension to which the queries and keys will be projected.
        kernel_initializer (str, optional): The initializer for the weight matrices. Default is "xavier_uniform".

    Example usage:
        # Initialize CustomAttention layer
        attention_layer = CustomAttention(weight_params_dim=256)
        # Input tensor of shape (batch_size, seq_len, input_dim)
        inputs = torch.randn(batch_size, seq_len, weight_params_dim)
        # Forward pass through the attention layer
        attention_weights = attention_layer(inputs)
    """
    def __init__(self, input_dim=512, weight_params_dim=256, kernel_initializer="xavier_uniform", **kwargs):
        super(CustomAttention, self).__init__()
        self.input_dim = input_dim
        self.weight_params_dim = weight_params_dim  # The dimension of queries and keys after projection
        self.kernel_initializer = kernel_initializer

        # Initialize weight matrices for Q (queries) and K (keys)
        self.wq_weight_params = nn.Parameter(torch.empty(self.input_dim, self.weight_params_dim))
        self.wk_weight_params = nn.Parameter(torch.empty(self.input_dim, self.weight_params_dim))

        # Initialize the weights using the specified initializer
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization (equivalent to glorot_uniform in TensorFlow)."""
        if self.kernel_initializer == "xavier_uniform":
            nn.init.xavier_uniform_(self.wq_weight_params)
            nn.init.xavier_uniform_(self.wk_weight_params)

    def forward(self, inputs):
        """
        Forward pass for computing attention logits Q(ti) * K(ti).

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        
        Returns:
            torch.Tensor: Scaled attention logits, shape (batch_size, seq_len_q, seq_len_k).
        """
        # Assuming `inputs` has shape (batch_size, seq_len, input_dim)
        q = torch.matmul(inputs, self.wq_weight_params)  # (batch_size, seq_len, weight_params_dim)
        k = torch.matmul(inputs, self.wk_weight_params)  # (batch_size, seq_len, weight_params_dim)

        # Transpose k for Q * K^T operation
        k_t = k.transpose(-2, -1)  # (batch_size, seq_len, weight_params_dim) -> (batch_size, weight_params_dim, seq_len)

        # Compute Q * K^T (scaled dot-product)
        matmul_qk = torch.matmul(q, k_t)  # (batch_size, seq_len_q, seq_len_k)

        # Scale by sqrt of the last dimension of K (or Q, since they are the same size)
        dk = torch.tensor(self.weight_params_dim, dtype=torch.float32)  # Scaling factor
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        return scaled_attention_logits 