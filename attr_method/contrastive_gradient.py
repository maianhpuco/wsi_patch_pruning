import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import matplotlib.pyplot as plt
from attr_method.common import PreprocessInputs, call_model_function 

class ContrastiveGradients(CoreSaliency):
    """Efficient Integrated Gradients with Counterfactual Attribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value")
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features", None)
        x_steps = kwargs.get("x_steps", 25) 
        
        attribution_values =  np.zeros_like(x_value, dtype=np.float32)
        # total_grad =  np.zeros_like(x_value, dtype=np.float32) 
        alphas = np.linspace(0, 1, x_steps)
        
        sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
        x_baseline_batch = baseline_features[sampled_indices]
        x_diff = x_value - x_baseline_batch 
        
        sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
        x_baseline_batch = baseline_features[sampled_indices] 
        
        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing:", ncols=100), start=1):
            # ------------ Counter Factual Gradient ------------ 
            x_diff = x_value - x_baseline_batch  
            x_step_batch = x_baseline_batch + alpha * x_diff
            
            # x_step_batch_tensor = torch.tensor(x_step_batch, dtype=torch.float32, requires_grad=True)

            # call_model_output = call_model_function(
            #     x_step_batch_tensor,
            #     model,
            #     call_model_args=call_model_args,
            #     expected_keys=self.expected_keys
            # )
            # self.format_and_check_call_model_output(call_model_output, x_step_batch_tensor.shape, self.expected_keys)
            
            # baseline_num = 1 
            # gradients_batch = call_model_output[INPUT_OUTPUT_GRADIENTS].reshape(baseline_num, x_value.shape[0], x_value.shape[1])
            
            # # reshape after computing 
            # gradients_avg = gradients_batch.reshape(-1, x_value.shape[-1])      
            
            # ------------ Counter Factual Gradient ------------
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
            
            # ------------ Conbine Gradient and X_diff ------------ 
            counterfactual_gradients = grad_logits_diff.mean(axis=0) 
            x_diff = x_diff.mean(axis=0)
            print("check shape")
            print(x_diff.shape, counterfactual_gradients.shape)
            attribution_values += (counterfactual_gradients * x_diff) 
            
        return attribution_values / x_steps