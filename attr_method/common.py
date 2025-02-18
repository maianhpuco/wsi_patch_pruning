import torch
import saliency.core as saliency 

def PreprocessInputs(inputs):
    """ Convert inputsa to a PyTorch tensor and enable gradient tracking """
    inputs = torch.tensor(inputs, dtype=torch.float32).clone().detach()
    return inputs.requires_grad_(True)
 
def call_model_function(images, model, call_model_args=None, expected_keys=None):
    """ Compute model logits and gradients """
    images = PreprocessInputs(images)
    model.eval()
    logits = model(images, [images.shape[0]])
    output = -logits
    grads = torch.autograd.grad(output, images, grad_outputs=torch.ones_like(output), create_graph=False)
    gradients = grads[0].detach().cpu().numpy()
    return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
 