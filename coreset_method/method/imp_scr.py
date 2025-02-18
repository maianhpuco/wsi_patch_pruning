import numpy as np


def sample_important_indices(features, selection_ratio=0.5, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    num_samples = features.shape[0]
    num_to_select = int(num_samples * selection_ratio)

    # Compute importance scores for each sample, here we use squared L2 norm.
    importance_scores = np.sum(features**2, axis=1)

    total_score = np.sum(importance_scores)
    if total_score == 0:
        probabilities = np.full(num_samples, 1 / num_samples)
    else:
        probabilities = importance_scores / total_score

    selected_indices = np.random.choice(
        np.arange(num_samples), size=num_to_select, replace=False, p=probabilities
    )

    return selected_indices
