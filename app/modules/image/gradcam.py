# app/gradcam.py
import torch
import torch.nn as nn
import numpy as np


class GradCAM:
    """
    Stable Grad-CAM implementation (EfficientNetV2 compatible).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        self.model.eval()

        self.activations = None
        self.gradients = None

        # Auto-detect last Conv2d if not provided
        if target_layer is None:
            self.target_layer = self._find_last_conv_module(self.model)
        else:
            self.target_layer = (
                target_layer
                if isinstance(target_layer, nn.Conv2d)
                else self._find_last_conv_module(target_layer)
            )

        if self.target_layer is None:
            raise RuntimeError("GradCAM: No Conv2d layer found.")

    def _find_last_conv_module(self, module: nn.Module):
        last_conv = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Generate Grad-CAM heatmap.

        Returns:
            cam (np.ndarray): HxW normalized [0,1]
        """

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Register hooks (FULL backward hook = PyTorch safe)
        fh = self.target_layer.register_forward_hook(self._forward_hook)
        bh = self.target_layer.register_full_backward_hook(self._backward_hook)

        # Forward
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx].sum()

        # Backward
        score.backward()

        # Remove hooks
        fh.remove()
        bh.remove()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM failed: missing activations or gradients")

        # GAP on gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum
        cam = torch.relu((weights * self.activations).sum(dim=1))

        # Normalize
        cam = cam[0]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        # Cleanup
        self.activations = None
        self.gradients = None

        return cam.detach().cpu().numpy().astype(np.float32)
