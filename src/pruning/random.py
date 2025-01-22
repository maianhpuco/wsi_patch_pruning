import numpy as np

# def random_feature_selection(features, patch_indices, pruning_rate=0.1):
#     total_sample = features.shape[0]
#     n_to_select = int(total_sample * (1-pruning_rate))
#     selected_indices = np.random.choice(
#             range(total_sample), size=n_to_select, replace=False)
#     _features = features[selected_indices, :] 
#     _patch_indices = patch_indices[selected_indices,: ]  
#     return _features, _patch_indices
def random_feature_selection(features, patch_indices, pruning_rate=0.1):
    total_sample = features.shape[0]
    n_to_select = int(total_sample * (1 - pruning_rate))
    
    # For PyTorch tensors, use torch.randperm instead of np.random.choice
    selected_indices = torch.randperm(total_sample)[:n_to_select]

    # Slice the features and patch_indices using the selected indices
    _features = features[selected_indices]
    _patch_indices = patch_indices[selected_indices]
    
    return _features, _patch_indices 
