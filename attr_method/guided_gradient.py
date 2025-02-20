import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import matplotlib.pyplot as plt
from attr_method.common import PreprocessInputs, call_model_function 
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
        fraction = kwargs.get("fraction", 0.25)
        max_dist = kwargs.get("max_dist", 0.02)
        
        attribution_values =  np.zeros_like(x_value, dtype=np.float32)
        # total_grad =  np.zeros_like(x_value, dtype=np.float32) 
        for step in range(x_steps):
            sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
            x_baseline_batch = baseline_features[sampled_indices]
            x = x_baseline_batch.copy()
            x_baseline_tensor = torch.tensor(x_baseline_batch, dtype=torch.float32, requires_grad=True)

            call_model_output = call_model_function(
                x_baseline_tensor,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            ) 
            l1_total = l1_distance(x_value, x_baseline_batch) 
            x_diff = x_value - x_baseline_batch   
            alpha = (step + 1.0) / x_steps 
       
            alpha_min = max(alpha - max_dist, 0.0)
            alpha_max = min(alpha + max_dist, 1.0)
            x_min =  x_baseline_batch + alpha_min * x_diff 
            x_max =  x_baseline_batch + alpha_max * x_diff 
            
            l1_target = l1_total * (1 - (step + 1) / x_steps) 
            gamma = np.inf 
            while gamma > 1.0: 
                x_old = x_value.copy()
                x_alpha = x_baseline_batch + alpha * x_diff             
                x_alpha[np.isnan(x_alpha)] = alpha_max 
                x_value[x_alpha < alpha_min] = x_min[x_alpha < alpha_min] 
                
                
                l1_current = l1_distance(x_value, x) 
                if math.isclose(l1_target, l1_current, rel_tol=EPSILON, abs_tol=EPSILON):
                    attr += (x - x_old) * grad_actual
                    break 
                        
            

    @staticmethod
    def guided_ig_impl(
        x_input, 
        x_baseline, 
        grad_func, 
        steps=200, 
        fraction=0.25,
        max_dist=0.02
        ):
        """Calculates and returns Guided IG attribution.

        Args:
            x_input: model input that should be explained.
            x_baseline: chosen baseline for the input explanation.
            grad_func: gradient function that accepts a model input and returns
            the corresponding output gradients. In case of many class model, it is
            responsibility of the implementer of the function to return gradients
            for the specific class of interest.
            steps: the number of Riemann sum steps for path integral approximation.
            fraction: the fraction of features [0, 1] that should be selected and
            changed at every approximation step. E.g., value `0.25` means that 25% of
            the input features with smallest gradients are selected and changed at
            every step.
            max_dist: the relative maximum L1 distance [0, 1] that any feature can
            deviate from the straight line path. Value `0` allows no deviation and,
            therefore, corresponds to the Integrated Gradients method that is
            calculated on the straight-line path. Value `1` corresponds to the
            unbounded Guided IG method, where the path can go through any point within
            the baseline-input hyper-rectangular.
        """
        import math 
        x_input = np.asarray(x_input, dtype=np.float64)
        x_baseline = np.asarray(x_baseline, dtype=np.float64)
        x = x_baseline.copy()
        l1_total = l1_distance(x_input, x_baseline)
        attr = np.zeros_like(x_input, dtype=np.float64)

        # If the input is equal to the baseline then the attribution is zero.
        total_diff = x_input - x_baseline
        if np.abs(total_diff).sum() == 0:
            return attr

        # Iterate through every step.
        for step in range(steps):
            # Calculate gradients and make a copy.
            grad_actual = grad_func(x)
            grad = grad_actual.copy()
            # Calculate current step alpha and the ranges of allowed values for this
            # step.
            alpha = (step + 1.0) / steps
            alpha_min = max(alpha - max_dist, 0.0)
            alpha_max = min(alpha + max_dist, 1.0)
            x_min = translate_alpha_to_x(alpha_min, x_input, x_baseline)
            x_max = translate_alpha_to_x(alpha_max, x_input, x_baseline)
            # The goal of every step is to reduce L1 distance to the input.
            # `l1_target` is the desired L1 distance after completion of this step.
            l1_target = l1_total * (1 - (step + 1) / steps)

            # Iterate until the desired L1 distance has been reached.
            gamma = np.inf
            while gamma > 1.0:
                x_old = x.copy()
                x_alpha = translate_x_to_alpha(x, x_input, x_baseline)
                x_alpha[np.isnan(x_alpha)] = alpha_max
                # All features that fell behind the [alpha_min, alpha_max] interval in
                # terms of alpha, should be assigned the x_min values.
                x[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]

                # Calculate current L1 distance from the input.
                l1_current = l1_distance(x, x_input)
                # If the current L1 distance is close enough to the desired one then
                # update the attribution and proceed to the next step.
                if math.isclose(l1_target, l1_current, rel_tol=EPSILON, abs_tol=EPSILON):
                    attr += (x - x_old) * grad_actual
                    break

            # Features that reached `x_max` should not be included in the selection.
            # Assign very high gradients to them so they are excluded.
            grad[x == x_max] = np.inf

            # Select features with the lowest absolute gradient.
            threshold = np.quantile(np.abs(grad), fraction, interpolation='lower')
            s = np.logical_and(np.abs(grad) <= threshold, grad != np.inf)

            # Find by how much the L1 distance can be reduced by changing only the
            # selected features.
            l1_s = (np.abs(x - x_max) * s).sum()

            # Calculate ratio `gamma` that show how much the selected features should
            # be changed toward `x_max` to close the gap between current L1 and target
            # L1.
            if l1_s > 0:
                gamma = (l1_current - l1_target) / l1_s
            else:
                gamma = np.inf

            if gamma > 1.0:
                # Gamma higher than 1.0 means that changing selected features is not
                # enough to close the gap. Therefore change them as much as possible to
                # stay in the valid range.
                x[s] = x_max[s]
            else:
                assert gamma > 0, gamma
                x[s] = translate_alpha_to_x(gamma, x_max, x)[s]
            # Update attribution to reflect changes in `x`.
            attr += (x - x_old) * grad_actual
        return attr