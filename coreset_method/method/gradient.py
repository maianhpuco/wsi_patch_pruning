import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def gradient_coreset_selection(X, model, loss_fn, ratio, batch_size=32, device="cpu"):

    model.to(device)
    model.eval()
    n_samples = X.shape[0]
    m = int(n_samples * ratio)  # Number of samples to select

    # Validate ratio
    if m <= 0 or m > n_samples:
        raise ValueError("Invalid ratio: results in zero or too many samples.")
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Collect model parameters that require gradients
    params = [param for param in model.parameters() if param.requires_grad]

    # Prepare storage for per-sample gradients
    per_sample_grads = []
    sample_indices = []

    # Global sample index
    global_sample_idx = 0
    for batch in dataloader:
        inputs = batch[0].to(device)

        # Forward pass
        outputs = model(inputs)
        # Compute reconstruction loss
        losses = loss_fn(outputs, inputs)  # Shape: (batch_size, input_size)
        losses = losses.mean(dim=1)  # Per-sample loss

        # Compute per-sample gradients
        for i in range(len(losses)):
            # Zero gradients
            model.zero_grad()

            # Backward pass for a single sample
            losses[i].backward(retain_graph=True)

            # Collect gradients
            grads = []
            for param in params:
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
                else:
                    grads.append(torch.zeros_like(param).view(-1))

            grad_vector = torch.cat(grads)  # Shape: (num_params,)
            per_sample_grads.append(grad_vector.cpu())
            sample_indices.append(global_sample_idx)
            global_sample_idx += 1

    per_sample_grads = torch.stack(per_sample_grads)  # Shape: (n_samples, num_params)

    # Compute the full gradient
    full_grad = per_sample_grads.sum(dim=0)

    # Gradient coreset selection
    selected_indices = []
    current_grad_sum = torch.zeros_like(full_grad)
    remaining_indices = set(range(n_samples))

    for _ in range(m):
        # Residual gradient
        residual = full_grad - current_grad_sum

        max_gain = None
        best_idx = None

        # Find the sample with the maximum gain
        for idx in remaining_indices:
            grad_i = per_sample_grads[idx]
            gain = torch.dot(grad_i, residual).item()

            if (max_gain is None) or (gain > max_gain):
                max_gain = gain
                best_idx = idx

        # Update selected indices and current gradient sum
        selected_indices.append(best_idx)
        current_grad_sum += per_sample_grads[best_idx]
        remaining_indices.remove(best_idx)

    return selected_indices


def gradient_complete_function(X, ratio=0.5):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    loss_fn = nn.MSELoss(reduction="none")
    input_size = 768
    encoding_dim = 128

    autoencoder = Autoencoder(input_size, encoding_dim)

    selected_indices = gradient_coreset_selection(
        X=X_tensor, model=autoencoder, loss_fn=loss_fn, ratio=ratio
    )
    return selected_indices
