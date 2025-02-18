import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import matplotlib.pyplot as plt
 


def get_ig():
    pass 


class IntegratedGradients(CoreSaliency):
    """Efficient Integrated Gradients with Counterfactual Attribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, x_value, call_model_function,
                model, call_model_args=None, baseline_features=None,
                x_steps=25, batch_size=1, eta=1.0,
                memmap_file="ig_memmap.npy", gradients_memmap_file="gradients_avg_memmap.npy"):
        baseline_num = 1
        expected_gradient =  np.zeros_like(x_value, dtype=np.float32)
        total_gradients = np.zeros_like(x_value, dtype=np.float32)
        new_term = np.zeros_like(x_value, dtype=np.float32)
        # counterfactual_gradients = np.zeros_like(x_value, dtype=np.float32)
        # **Remove old files if they exist**
        for file in [memmap_file, gradients_memmap_file]:
            if os.path.exists(file):
                os.remove(file)

        list_of_counterfactual_grad = np.memmap(memmap_file, dtype=np.float32, mode="w+", shape=(x_steps, x_value.shape[0]))
        list_of_gradients = np.memmap(gradients_memmap_file, dtype=np.float32, mode="w+", shape=(x_steps, x_value.shape[0]))
        # logits_list = []
        # baseline_mean = baseline_normal_features_batch.mean(axis=0)
        # baseline_std = baseline_normal_features_batch.std(axis=0)

        alphas = np.linspace(0, 1, x_steps)
        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing IG²", ncols=100), start=1):
            sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
            x_baseline_batch = baseline_features[sampled_indices]

            # sampled_matrix = np.random.normal(loc=baseline_mean, scale=baseline_std, size=x_value.shape)
            # x_baseline_batch = sampled_matrix
            x_diff = x_value - x_baseline_batch

            x_step_batch = x_baseline_batch + alpha * x_diff
            # x_step_batch = x_step_batch.reshape(-1, x_value.shape[1])
            # print("x_step_batch", x_step_batch.shape)
            x_step_batch_tensor = torch.tensor(x_step_batch, dtype=torch.float32, requires_grad=True)

            call_model_output, logit = call_model_function(
                x_step_batch_tensor,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )
            self.format_and_check_call_model_output(call_model_output, x_step_batch_tensor.shape, self.expected_keys)

            gradients_batch = call_model_output[INPUT_OUTPUT_GRADIENTS].reshape(baseline_num, x_value.shape[0], x_value.shape[1])
            # gradients_batch = normalize_by_2norm(gradients_batch)
            gradients_avg = np.mean(gradients_batch, axis=0)
            # print("gradients_avg.shape", gradients_avg.shape)
            total_gradients += gradients_avg
            # logits_list.append(logit.cpu().detach().numpy().item())
            # -------- COUNTER FACTUAL GRADIENT ------------------------------------------------------------------------
            x_baseline_torch = torch.tensor(x_baseline_batch.copy(), dtype=torch.float32, requires_grad=False)
            logits_x_r = model(x_baseline_torch, [x_baseline_torch.shape[0]])

            # Compute counterfactual gradients using logits difference
            x_step_batch_torch = torch.tensor(x_step_batch, dtype=torch.float32, requires_grad=True)
            logits_x_step = model(x_step_batch_torch, [x_step_batch_torch.shape[0]])
            logits_difference = torch.norm(logits_x_step - logits_x_r, p=2) ** 2
            logits_difference.backward()

            if x_step_batch_torch.grad is None:
                raise RuntimeError("Gradients are not being computed! Ensure tensors require gradients.")

            grad_logits_diff = x_step_batch_torch.grad.numpy()
            # grad_logits_diff = grad_logits_diff.reshape(baseline_num, x_value.shape[0], x_value.shape[1])
            # grad_logits_diff = normalize_by_2norm(grad_logits_diff)

            counterfactual_gradients = grad_logits_diff.mean(axis=0)
            # print("counterfactual_gradients.shape", counterfactual_gradients.shape)
            W_j = np.linalg.norm(gradients_avg) + 1e-8  # Avoid division by zero
            ig_mask = (gradients_avg * counterfactual_gradients) * (eta / W_j)
            list_of_gradients[step_idx - 1, :] = gradients_avg.mean(axis=1)
            list_of_gradients.flush()
            list_of_counterfactual_grad[step_idx - 1, :] = counterfactual_gradients.mean(axis=1)
            list_of_counterfactual_grad.flush()

            del gradients_batch, gradients_avg, x_step_batch_tensor, x_step_batch_torch
            torch.cuda.empty_cache()
            x_diff = x_diff.reshape(-1, x_value.shape[1])
            new_term += (counterfactual_gradients * x_diff)
            expected_gradient += ig_mask

        return new_term, expected_gradient, total_gradients, list_of_counterfactual_grad, list_of_gradients, alphas, x_diff 