import numpy as np


def sample_features_indices(features, selection_ratio=0.5, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    num_samples = features.shape[0]
    num_to_select = int(num_samples * selection_ratio)
    selected_indices = np.random.choice(
        np.arange(num_samples), size=num_to_select, replace=False
    )
    return selected_indices
