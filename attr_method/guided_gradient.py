import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import matplotlib.pyplot as plt
from attr_method.common import PreprocessInputs, call_model_function 
import math 

EPSILON = 1E-9


def l1_distance(x1, x2):
  """Returns L1 distance between two points."""
  return np.abs(x1 - x2).sum()


def translate_x_to_alpha(x, x_input, x_baseline):
  """Translates a point on straight-line path to its corresponding alpha value.

  Args:
    x: the point on the straight-line path.
    x_input: the end point of the straight-line path.
    x_baseline: the start point of the straight-line path.

  Returns:
    The alpha value in range [0, 1] that shows the relative location of the
    point between x_baseline and x_input.
  """
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.where(x_input - x_baseline != 0,
                    (x - x_baseline) / (x_input - x_baseline), np.nan)


def translate_alpha_to_x(alpha, x_input, x_baseline):
  """Translates alpha to the point coordinates within straight-line interval.

   Args:
    alpha: the relative location of the point between x_baseline and x_input.
    x_input: the end point of the straight-line path.
    x_baseline: the start point of the straight-line path.

  Returns:
    The coordinates of the point within [x_baseline, x_input] interval
    that correspond to the given value of alpha.
  """
  assert 0 <= alpha <= 1.0
  return x_baseline + (x_input - x_baseline) * alpha

 
class GuidedGradients(CoreSaliency):
    """Efficient Integrated Gradients with Counterfactual Attribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value")
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features", None)
        x_steps = kwargs.get("x_steps", 25) 
        fraction = kwargs.get("fraction", 0.10)
        max_dist = kwargs.get("max_dist", 0.02)
        
        attribution_values =  np.zeros_like(x_value, dtype=np.float32)
        # total_grad =  np.zeros_like(x_value, dtype=np.float32) 
        sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
        x_baseline_batch = baseline_features[sampled_indices]
        x = x_baseline_batch.copy()
        x_baseline_tensor = torch.tensor(x_baseline_batch, dtype=torch.float32, requires_grad=True)
         
        for step in range(x_steps):
            print(f"----Step {step}/{x_steps}")
            call_model_output = call_model_function(
                x_baseline_tensor,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )
            self.format_and_check_call_model_output(call_model_output, x_baseline_tensor.shape, self.expected_keys) 
            baseline_num = 1
            
            grad_actual = call_model_output[INPUT_OUTPUT_GRADIENTS].reshape(-1, x_value.shape[1])  
            # grad 
            print("grad shape", grad_actual.shape)  
            grad = grad_actual.copy() 
            
            #---- 
                        
            x_baseline_batch = x.reshape(-1, x_value.shape[-1])  
            x = x.reshape(-1, x_value.shape[-1])
            x_diff = x_diff.reshape(-1, x_value.shape[-1]) 
            x_max = x_max.reshape(-1, x_max.shape[-1]) 
            x_min = x_min.reshape(-1, x_max.shape[-1])    
            #----
            
            
            l1_total = l1_distance(x_value, x_baseline_batch) 
            x_diff = x_value - x_baseline_batch   
            alpha = (step + 1.0) / x_steps 
       
            alpha_min = max(alpha - max_dist, 0.0)
            alpha_max = min(alpha + max_dist, 1.0)
            
            # for the purpose of cliping input x_min, x_max and adjust is just for clipping the gradient etc ... 
            x_min =  x_baseline_batch + alpha_min * x_diff 
            x_max =  x_baseline_batch + alpha_max * x_diff 
            
            l1_target = l1_total * (1 - (step + 1) / x_steps) 
            #  l1 target is the total L1 distance between the baseline and input: | X_value - X_baseline | 
            #  represents how much the feature values should have changed at a given step.
 
 

            
            gamma = np.inf 
            count = 0 
            while gamma > 1.0: 
                x_old = x_value.copy()
                x_old = x_old.reshape(-1, x_value.shape[-1]) 
                x_alpha = x_baseline_batch + alpha * x_diff             
                x_alpha[np.isnan(x_alpha)] = alpha_max 
              
                x_value[x_alpha < alpha_min] = x_min[x_alpha < alpha_min] 
                print("smaller than alpha_min", x_value[x_alpha < alpha_min].shape)
                
                l1_current = l1_distance(x_value, x) # measures how far the current feature values are from the previous state.
                
                if math.isclose(l1_target, l1_current, rel_tol=EPSILON, abs_tol=EPSILON):
                    attr += (x - x_old) * grad_actual
                    break 
                grad[x==x_max] = np.inf 
                
                # <---Select features with the lowest absolute gradient. 
                threshold = np.quantile(np.abs(grad), fraction, interpolation='lower')
                s = np.logical_and(np.abs(grad) <= threshold, grad != np.inf) 
                # --->
                 
                # Find by how much the L1 distance can be reduced by changing only the
                # selected features. # Compute L1 distance reduction possible by modifying `s` 
                l1_s = (np.abs(x - x_max) * s).sum() 
               
 
                # Calculate ratio `gamma` that show how much the selected features should
                # be changed toward `x_max` to close the gap between current L1 and target
                # L1.
                if l1_s > 0:
                    gamma = (l1_current - l1_target) / l1_s
                else:
                    gamma = np.inf 
                print("- count")
                print(f"-> l1_current: {l1_current}, l1_target: {l1_target}, l1_s: {l1_s}, gamma: {gamma}") 
                
                # Gamma higher than 1.0 means that changing selected features is not
                # enough to close the gap. Therefore change them as much as possible to
                # stay in the valid range.  
                              
                if gamma > 1.0:
                    x[s] = x_max[s]
                else:
                    assert gamma > 0, gamma
                    x[s] = translate_alpha_to_x(gamma, x_max, x)[s]
                
                # Update attribution to reflect changes in `x`.
                attribution_values += (x - x_old) * grad_actual
                count+=1 
                
            return attribution_values

        
                    
