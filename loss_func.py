import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F


def compute_accuracy(y_true, y_pred_binary):
    """
    Computes Accuracy (ACC).
    """
    return accuracy_score(y_true, y_pred_binary)


def compute_auc(y_true, y_pred_prob):
    """
    Computes the Area Under the ROC Curve (AUC).
    """
    return roc_auc_score(y_true, y_pred_prob)


def compute_ece(y_true, y_pred_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    """
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    # Create bin edges from 0.0 to 1.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    # Determine the bin index for each probability
    bin_indices = (
        np.digitize(y_pred_prob, bin_edges) - 1
    )  # subtract 1 to make 0-indexed

    ece = 0.0
    for i in range(n_bins):
        in_bin = bin_indices == i
        if not np.any(in_bin):
            continue
        avg_pred_prob = np.mean(y_pred_prob[in_bin])
        frac_positives = np.mean(y_true[in_bin])
        prop_in_bin = np.sum(in_bin) / len(y_true)
        ece += np.abs(avg_pred_prob - frac_positives) * prop_in_bin
    return ece


def ece_loss(predictions, true_labels, num_bins=15):
    """
    Calculate a calibration-inspired loss based on the Expected Calibration Error (ECE).

    Args:
        predictions (torch.Tensor): Model's predicted logits, shape (n_samples, n_classes).
        true_labels (torch.Tensor): Ground truth labels, shape (n_samples,).
        num_bins (int): Number of bins for computing ECE.

    Returns:
        torch.Tensor: The calibration-inspired loss.
    """

    # Convert logits to probabilities using softmax
    probs = F.softmax(predictions, dim=1)

    # Get the predicted class probabilities and ground truth as probabilities
    max_probs, predicted_classes = torch.max(probs, dim=1)
    true_probs = torch.zeros_like(probs).scatter_(1, true_labels.unsqueeze(1), 1)

    # Establish bin limits
    bin_edges = torch.linspace(0, 1, num_bins + 1, device=probs.device)

    # Initialize ECE loss
    total_ece_loss = torch.zeros(1, device=probs.device)

    # Compute ECE for each bin
    for lower_edge, upper_edge in zip(bin_edges[:-1], bin_edges[1:]):
        # Samples whose predicted probabilities fall within the current bin
        within_bin = (max_probs > lower_edge) & (max_probs <= upper_edge)
        prop_in_bin = within_bin.float().mean()

        if prop_in_bin > 0:
            # Average accuracy of true labels in the current bin
            accuracy_in_bin = torch.mean(
                (predicted_classes[within_bin] == true_labels[within_bin]).float()
            )

            # Average confidence of probabilities in the current bin
            confidence_in_bin = torch.mean(max_probs[within_bin])

            # Weighted difference between confidence and accuracy
            bin_ece_loss = torch.abs(confidence_in_bin - accuracy_in_bin) * prop_in_bin
            total_ece_loss += bin_ece_loss

    return total_ece_loss
