import numpy as np

def random_feature_selection(features, patch_indices, pruning_rate=0.1):
    total_sample = features.shape[0]
    n_to_select = int(total_sample * (1-pruning_rate))
    selected_indices = np.random.choice(
        range(total_sample), size=n_to_select, replace=False)
    return features[selected_indices, :], patch_indices[selected_indices,: ] 
