import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

def train_one_epoch(model, dataset, optimizer, loss_fn, device):
    """
    Trains the model for one epoch on individual samples (without batching).
    
    Args:
        model: The model to be trained.
        dataset: The dataset to load training data.
        optimizer: The optimizer to update model parameters.
        loss_fn: The loss function to compute the error.
        device: The device to train on ('cuda' or 'cpu').
        
    Returns:
        avg_loss: The average loss for this epoch.
        accuracy: The accuracy for this epoch.
    """
    
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for features, sparse_matrix, labels in dataset:
        features = features.to(device)
        sparse_matrix = sparse_matrix.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass: Get model outputs
        outputs, _, _ = model(features, sparse_matrix)
        predicted_prob = torch.sigmoid(outputs) 

        # Calculate loss (assuming outputs are logits)
        loss = loss_fn(outputs, labels)  # Unsqueeze labels if they are scalar

        # Backward pass: Compute gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        # Track running loss
        running_loss += loss.item()

        # Calculate accuracy (for binary classification)
        predicted = torch.sigmoid(outputs) > 0.5  # Convert logits to binary class predictions
        correct += (predicted == labels.unsqueeze(1)).sum().item()
        total += labels.size(0)

    # Calculate average loss and accuracy for this epoch
    avg_loss = running_loss / len(dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train(model, dataset, epochs=10, learning_rate=1e-3, device="cuda"):
    """
    Train the model for multiple epochs.
    
    Args:
        model: The model to be trained.
        dataset: The dataset to load training data from.
        epochs: Number of epochs to train.
        learning_rate: Learning rate for optimizer.
        device: The device to train on ('cuda' or 'cpu').
    """
    
    # Move model to the specified device (GPU/CPU)
    model.to(device)
    
    # Set up the optimizer (using Adam here)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function (using binary cross-entropy for binary classification)
    loss_fn = torch.nn.BCELoss()  # Use this if your model outputs raw logits

    # Training loop over multiple epochs
    for epoch in range(epochs):
        avg_loss, accuracy = train_one_epoch(model, dataset, optimizer, loss_fn, device)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    print("Training complete")
