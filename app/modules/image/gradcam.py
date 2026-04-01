# app/gradcam.py

import torch
import torch.nn as nn
import numpy as np
import cv2


class GradCAM:
    """
    Stable Grad-CAM implementation (EfficientNetV2 compatible).
    Also includes Patch-Level Frequency Attention visualization.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        self.model.eval()

        self.activations = None
        self.gradients = None

        # Auto detect last conv layer
        if target_layer is None:
            self.target_layer = self._find_last_conv_module(self.model)
        else:
            self.target_layer = target_layer

        if self.target_layer is None:
            raise RuntimeError("GradCAM: No Conv2d layer found.")

    def _find_last_conv_module(self, module: nn.Module):
        last_conv = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        fh = self.target_layer.register_forward_hook(self._forward_hook)
        bh = self.target_layer.register_full_backward_hook(self._backward_hook)

        self.model.zero_grad()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]

        score.backward()

        fh.remove()
        bh.remove()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM failed")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * self.activations, dim=1)

        cam = torch.relu(cam)

        cam = cam[0]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        self.activations = None
        self.gradients = None

        return cam.cpu().numpy().astype(np.float32)


# ---------------------------------------------
# PATCH LEVEL FREQUENCY ATTENTION (NEW)
# ---------------------------------------------

def frequency_attention_map(face_img):
    """
    Detect frequency artifacts commonly produced by GAN deepfakes.

    Returns:
        freq_map (np.ndarray): heatmap highlighting suspicious areas
    """

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Fourier Transform
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.log(np.abs(fshift) + 1)

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        magnitude,
        cv2.COLORMAP_JET
    )

    return heatmap


# ---------------------------------------------
# COMBINED VISUALIZATION
# ---------------------------------------------

def overlay_gradcam(face_img, cam):

    cam = cv2.resize(cam, (face_img.shape[1], face_img.shape[0]))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        face_img,
        0.6,
        heatmap,
        0.4,
        0
    )

    return overlay