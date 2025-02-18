import numpy as np


def k_center_greedy(features, ratio=0.5, random_seed=None):
    k = int(ratio * features.shape[0])
    if random_seed is not None:
        np.random.seed(random_seed)

    num_samples = features.shape[0]
    centers = []
    distances = np.full(num_samples, np.inf)
    first_center = np.random.choice(num_samples)
    centers.append(first_center)
    # Update distances: each point's distance to the first center.
    distances = np.minimum(
        distances, np.linalg.norm(features - features[first_center], axis=1)
    )
    for _ in range(1, k):
        next_center = int(np.argmax(distances))
        centers.append(next_center)
        # Update distances: each point gets the minimum distance to a chosen center.
        distances = np.minimum(
            distances, np.linalg.norm(features - features[next_center], axis=1)
        )

    return centers
