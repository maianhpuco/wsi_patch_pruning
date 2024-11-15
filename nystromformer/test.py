import tensorflow as tf
import numpy as np
import os 
import sys
PROJECT_DIR = os.environ['PROJECT_DIR']
sys.path.append(os.path.join(PROJECT_DIR))

def create_test_data(batch_size, seq_len, dim):
    """
    Create random data for testing. 
    """
    return tf.random.normal((batch_size, seq_len, dim))

# Test NystromAttention
print("Testing NystromAttention...")


if __name__ == "__main__":
    print
    # Define testing parameters
    batch_size = 16
    seq_len = 128  # Length of the sequence
    dim = 512  # Dimensionality of input
    dim_head = 64  # Dimensionality of the head in attention
    heads = 8  # Number of attention heads
    num_landmarks = 256  # Number of landmarks for Nyström method
    pinv_iterations = 6  # Number of iterations for Moore-Penrose Pseudoinverse


     # Create test data
    test_data = create_test_data(batch_size, seq_len, dim)
    
    # Call the Nyström Attention layer
    attn_output = nystrom_attention_layer(test_data)
    print(f"NystromAttention output shape: {attn_output.shape}")
    
    # Test Nystromformer
    print("Testing Nystromformer...")

    nystrom_attention_layer = NystromAttention(
       dim=dim,
       dim_head=dim_head,
       heads=heads,
       num_landmarks=num_landmarks,
       pinv_iterations=pinv_iterations,
       residual=True,
       dropout=0.1
    )
    nystromformer_model = Nystromformer(
        dim=dim,
        depth=4,
        dim_head=dim_head,
        heads=heads,
        num_landmarks=num_landmarks,
        pinv_iterations=pinv_iterations,
        attn_values_residual=True,
        attn_values_residual_conv_kernel=33,
        attn_dropout=0.1,
        ff_dropout=0.1
    )
    
   
    nystromformer_output = nystromformer_model(test_data)
    print(f"Nystromformer output shape: {nystromformer_output.shape}")
