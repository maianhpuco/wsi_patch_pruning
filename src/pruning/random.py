import numpy as np

def random_feature_selection(features_tensor, patch_indices, pruning_rate=0.1):
    total_sample = features.shape[0]
    # Step 1: Calculate the number of features to select based on the fraction
    n_to_select = int(total_sample * (1-pruning_rate)

    # Step 2: Randomly select `n_features_to_select` feature indices
    selected_indices = np.random.choice(
        range(total_sample), size=n_to_select, replace=False)
    
    print(selected_indices)
    
    # Step 3: Extract the selected features
    selected_sample = features[selected_indices, :]
    patch_indices = selected_sample[selected_feature_indices, ]
    # Step 4: Return the selected feature indices and the corresponding features
    return selected_sample, patch_indices 
