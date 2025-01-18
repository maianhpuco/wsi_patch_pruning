import os
import sys  
import torch
import torch.optim as optim
import torch.nn as nn
import time
from collections import deque
import json
import numpy as np 

from sklearn.metrics import roc_auc_score 








# Helper function to save checkpoints
def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    print(f"Saving checkpoint at epoch {epoch}...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)

# Helper function to load checkpoints and resume training
def load_checkpoint(model, optimizer, filename=None):
    if filename is None:
        print("No checkpoint found, starting fresh.")
        return model, optimizer, 0, None
    else: 
        print(f"Loading checkpoint from {filename}...") 
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"Resuming from epoch {epoch}, last loss: {loss}")
            return model, optimizer, epoch, loss
        

def train_one_epoch(model, train_dataset, val_dataset, optimizer, loss_fn, device):
    """
    Trains the model for one epoch on individual samples (without batching) and validates it on the validation set.
    
    Args:
        model: The model to be trained.
        train_dataset: The dataset to load training data.
        val_dataset: The dataset to load validation data.
        optimizer: The optimizer to update model parameters.
        loss_fn: The loss function to compute the error.
        device: The device to train on ('cuda' or 'cpu').
        
    Returns:
        avg_train_loss: The average loss for the training epoch.
        train_accuracy: The accuracy for the training epoch.
        avg_val_loss: The average loss for the validation epoch.
        val_accuracy: The accuracy for the validation epoch.
        train_auc: The AUC for the training epoch.
        val_auc: The AUC for the validation epoch.
    """
    
    # Training Phase
    model.train()  # Set the model to training mode
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0
    all_train_preds = []
    all_train_labels = []

    for features, sparse_matrix, labels in train_dataset:
        
        features = features.to(device)
        sparse_matrix = sparse_matrix.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass: Get model outputs
        predicted_prob, _, _ = model(features, sparse_matrix)
        # Calculate loss (assuming outputs are logits)
        loss = loss_fn(predicted_prob, labels)

        # Backward pass: Compute gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        # Track running loss
        running_train_loss += loss.item()
        # Calculate accuracy (for binary classification)
        predicted = predicted_prob > 0.5  # Convert logits to binary class predictions
        correct_train += (predicted == labels.unsqueeze(1)).sum().item()
        total_train += labels.size(0)
        
        # Collect predictions and labels for AUC calculation
        all_train_preds.extend(predicted_prob.detach().cpu().numpy().flatten())  # Flatten to 1D
        all_train_labels.extend(labels.detach().cpu().numpy().flatten())  # Flatten to 1D 
        
        # Convert the predictions and labels to numpy arrays of float64
    all_train_preds_np = np.array(all_train_preds, dtype=np.float64)
    all_train_labels_np = np.array(all_train_labels, dtype=np.float64)

    # If you want them as Python lists, you can convert them using tolist():
    all_train_preds_list = all_train_preds_np.tolist()
    all_train_labels_list = all_train_labels_np.tolist() 
    
    # print("- train pred", all_train_preds_list)
    # print("- train label", all_train_labels_list)
      
    # print("all_train_preds", all_train_preds)
    # print("all_train_labels", all_train_labels)  
    # Calculate average loss and accuracy for training
    avg_train_loss = running_train_loss / len(train_dataset)
    train_accuracy = correct_train / total_train

    # Calculate AUC for training
    train_auc = roc_auc_score(all_train_labels, all_train_preds)

    # Validation Phase
    model.eval()  # Set the model to evaluation mode
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():  # Disable gradient computation during validation
        for features, sparse_matrix, labels in val_dataset:
            features = features.to(device)
            sparse_matrix = sparse_matrix.to(device)
            labels = labels.to(device)
            print("features.shape", features.shape)
            print("label", labels.shape)
            print(labels)
            # Forward pass: Get model outputs
            predicted_prob, _, _ = model(features, sparse_matrix)

            # Calculate loss
            loss = loss_fn(predicted_prob, labels)

            # Track running loss
            running_val_loss += loss.item()

            # Calculate accuracy (for binary classification)
            predicted = predicted_prob > 0.5  # Convert logits to binary class predictions
            correct_val += (predicted == labels.unsqueeze(1)).sum().item()
            total_val += labels.size(0)

            # Collect predictions and labels for AUC calculation
            all_val_preds.extend(predicted_prob.detach().cpu().numpy().flatten())  # Flatten to 1D
            all_val_labels.extend(labels.detach().cpu().numpy().flatten())  # Flatten to 1D 
            
    # Convert the predictions and labels to numpy arrays of float64
    all_val_preds = np.array(all_val_preds, dtype=np.float64)
    all_val_labels = np.array(all_val_labels, dtype=np.float64)

    # If you want them as Python lists, you can convert them using tolist():
    all_val_preds_list = all_val_preds.tolist()
    all_val_labels_list = all_val_labels.tolist() 
    
    # print("val pred", all_val_preds_list)
    # print("val label", all_val_labels_list)
    
    # Calculate average loss and accuracy for validation
    avg_val_loss = running_val_loss / len(val_dataset)
    val_accuracy = correct_val / total_val

    # Calculate AUC for validation
    val_auc = roc_auc_score(all_val_labels, all_val_preds)
        # Convert to a standard numpy array (float64)
    # Print the results
    # print("-----")
    # print(f"Validation Loss: {avg_val_loss:.4f}")
    # print(f"Validation Accuracy: {val_accuracy:.4f}")
    # print(f"Validation AUC: {val_auc:.4f}") 
    return avg_train_loss, train_accuracy, avg_val_loss, val_accuracy, train_auc, val_auc

def train(
    model, train_dataset, val_dataset, epochs=10, learning_rate=1e-3, 
    device="cuda", save_path="best_model.pth", log_file=None, checkpoint_path=None):
    """
    Train the model for multiple epochs, saving the model with the best validation performance and logging the results.
    
    Args:
        model: The model to be trained.
        train_dataset: The dataset to load training data from.
        val_dataset: The dataset to load validation data from.
        epochs: Number of epochs to train.
        learning_rate: Learning rate for optimizer.
        device: The device to train on ('cuda' or 'cpu').
        save_path: Path where to save the model weights of the best epoch.
        log_file: Path to save the log file containing epoch results.
        checkpoint_path: Path to load/save the checkpoint file.
    """
    
    # Move model to the specified device (GPU/CPU)
    model.to(device)
    
    # Set up the optimizer (using Adam here)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function (using binary cross-entropy for binary classification)
    loss_fn = torch.nn.BCELoss()  # Use this if your model outputs raw logits
    
    best_val_accuracy = 0.0  # Track the best validation accuracy
    best_epoch = -1  # To store the epoch of the best model
    best_model_weights = None  # To store the best model's weights

    # List to store logs for each epoch
    log_data = []

    # Load checkpoint if it exists
    model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
    
    # Training loop over multiple epochs
    for epoch in range(start_epoch, epochs):
        # Train and validate for the current epoch
        epoch_start_time = time.time() 
        avg_train_loss, train_accuracy, avg_val_loss, val_accuracy, train_auc, val_auc = train_one_epoch(
            model, train_dataset, val_dataset, optimizer, loss_fn, device)
        
        # End the timer for the epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Print results for the current epoch, including AUC for both train and validation
        print(f"Epoch [{epoch+1}/{epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train AUC: {train_auc:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}, "
            f"Time: {epoch_duration:.2f} seconds") 
        
        # Log results for the current epoch, including AUC
        if log_file is not None:
            log_data.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "train_auc": train_auc,  # Add train AUC
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_auc": val_auc   # Add validation AUC
            }) 
        
        # If the current epoch has better validation accuracy, save the model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_model_weights = model.state_dict()  # Save the model's state dict

            # # Save the best model weights if there is an improvement
            # torch.save(best_model_weights, save_path)
            # print(f"Best model saved at epoch {best_epoch} with validation accuracy: {best_val_accuracy:.4f}")
            # temp_cheeck_point_basename = checkpoint_path.split(".pth")[0]
            # check_point_file_path= os.path.join(
            #     os.path.dirname(checkpoint_path), 
            #     checkpoint_path.split(".pth")[0], 
            #     f"epoch_{best_epoch}.pth"
            #     )
             
            # print(f"save to file {check_point_file_path}")
        # Save checkpoint every epoch
            # save_checkpoint(model, optimizer, epoch + 1, avg_val_loss, checkpoint_path)

# After all epochs, save the best model's weights
    if best_model_weights is not None:
        torch.save(best_model_weights, save_path)
        print(f"Best model saved at epoch {best_epoch} with validation accuracy: {best_val_accuracy:.4f}")
    else:
        print("No model improvement detected.")
    
    # Save the log data to a JSON file
    if log_file is not None:
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=4)
        print(f"Training log saved to {log_file}")
    
    print("Training complete") 
    # Save the log data to a JSON file
    if log_file is not None:
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=4)
        print(f"Training log saved to {log_file}")
    
    print("Training complete")








 
# def train(
#     model, train_dataset, val_dataset, epochs=10, learning_rate=1e-3, 
#     device="cuda", save_path="best_model.pth", log_file=None, checkpoint_path=None):
#     """
#     Train the model for multiple epochs, saving the model with the best validation performance and logging the results.
    
#     Args:
#         model: The model to be trained.
#         train_dataset: The dataset to load training data from.
#         val_dataset: The dataset to load validation data from.
#         epochs: Number of epochs to train.
#         learning_rate: Learning rate for optimizer.
#         device: The device to train on ('cuda' or 'cpu').
#         save_path: Path where to save the model weights of the best epoch.
#         log_file: Path to save the log file containing epoch results.
#         checkpoint_path: Path to load/save the checkpoint file.
#     """
    
#     # Move model to the specified device (GPU/CPU)
#     model.to(device)
    
#     # Set up the optimizer (using Adam here)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
#     # Loss function (using binary cross-entropy for binary classification)
#     loss_fn = torch.nn.BCELoss()  # Use this if your model outputs raw logits
    
#     best_val_accuracy = 0.0  # Track the best validation accuracy
#     best_epoch = -1  # To store the epoch of the best model
#     best_model_weights = None  # To store the best model's weights

#     # List to store logs for each epoch
#     log_data = []

#     # Load checkpoint if it exists
#     model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
    
#     # Training loop over multiple epochs
#     for epoch in range(start_epoch, epochs):
#         # Train and validate for the current epoch
#         epoch_start_time = time.time() 
#         avg_train_loss, train_accuracy, avg_val_loss, val_accuracy, train_auc, val_auc = train_one_epoch(
#             model, train_dataset, val_dataset, optimizer, loss_fn, device)
        
#         # End the timer for the epoch
#         epoch_end_time = time.time()
#         epoch_duration = epoch_end_time - epoch_start_time

#         # Print results for the current epoch, including AUC for both train and validation
#         print(f"Epoch [{epoch+1}/{epochs}], "
#             f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train AUC: {train_auc:.4f}, "
#             f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}, "
#             f"Time: {epoch_duration:.2f} seconds") 
        
#         # Log results for the current epoch, including AUC
#         if log_file is not None:
#             log_data.append({
#                 "epoch": epoch + 1,
#                 "train_loss": avg_train_loss,
#                 "train_accuracy": train_accuracy,
#                 "train_auc": train_auc,  # Add train AUC
#                 "val_loss": avg_val_loss,
#                 "val_accuracy": val_accuracy,
#                 "val_auc": val_auc   # Add validation AUC
#             }) 
        
#         # If the current epoch has better validation accuracy, save the model
#         if val_accuracy > best_val_accuracy:
#             best_val_accuracy = val_accuracy
#             best_epoch = epoch + 1
#             best_model_weights = model.state_dict()  # Save the model's state dict

#         # Save checkpoint every epoch
#         # save_checkpoint(model, optimizer, epoch + 1, avg_val_loss, checkpoint_path)

#     # After all epochs, save the best model's weights
#     if best_model_weights is not None:
#         torch.save(best_model_weights, save_path)
#         print(f"Best model saved at epoch {best_epoch} with validation accuracy: {best_val_accuracy:.4f}")
#     else:
#         print("No model improvement detected.")
    
#     # Save the log data to a JSON file
#     if log_file is not None:
#         with open(log_file, "w") as f:
#             json.dump(log_data, f, indent=4)
#         print(f"Training log saved to {log_file}")
    
#     print("Training complete")
 