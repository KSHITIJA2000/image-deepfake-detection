import torch
import torch.nn as nn
import numpy as np
import cv2


class GradCAM:

    def __init__(self, model: nn.Module, target_layer: nn.Module = None):

        self.model = model
        self.model.eval()

        self.activations = None
        self.gradients = None

        # safer layer selection
        self.target_layer = target_layer or self._find_layer()

        if self.target_layer is None:
            raise RuntimeError("GradCAM: No valid Conv layer found")

    # =====================================================
    # SAFE LAYER SELECTION
    # =====================================================
    def _find_layer(self):

        convs = [
            m for m in self.model.modules()
            if isinstance(m, nn.Conv2d)
        ]

        if len(convs) == 0:
            return None

        # use deeper conv (more semantic features)
        return convs[-1]

    # =====================================================
    # HOOKS
    # =====================================================
    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    # =====================================================
    # GRADCAM GENERATION
    # =====================================================
    def generate(self, x, class_idx=None):

        device = next(self.model.parameters()).device
        x = x.to(device)

        self.activations = None
        self.gradients = None

        h1 = self.target_layer.register_forward_hook(self._forward_hook)
        h2 = self.target_layer.register_backward_hook(self._backward_hook)

        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)

        probs = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = torch.argmax(probs, dim=1).item()

        score = logits[:, class_idx]

        score.backward()

        h1.remove()
        h2.remove()

        # =================================================
        # CAM COMPUTATION (STABLE)
        # =================================================
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)

        cam = torch.relu(cam)

        cam = cam[0].detach().cpu().numpy()

        # normalize safely
        cam = self._normalize(cam)

        return cam

    # =====================================================
    # NORMALIZATION (SAFE)
    # =====================================================
    def _normalize(self, cam):

        cam = cv2.GaussianBlur(cam, (7, 7), 0)

        mn, mx = cam.min(), cam.max()

        if abs(mx - mn) < 1e-8:
            return np.zeros_like(cam, dtype=np.float32)

        cam = (cam - mn) / (mx - mn)

        return cam.astype(np.float32)


# =========================================================
# FREQUENCY MAP (UNCHANGED BUT SAFE)
# =========================================================
def frequency_attention_map(face_img):

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    mag = np.log(np.abs(fshift) + 1)

    mag = cv2.normalize(
        mag, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return cv2.applyColorMap(mag, cv2.COLORMAP_JET)


# =========================================================
# OVERLAY (SAFE)
# =========================================================
def overlay_gradcam(face_img, cam, alpha=0.55):

    cam = cv2.resize(cam, (face_img.shape[1], face_img.shape[0]))
    cam = np.clip(cam, 0, 1)

    heatmap = cv2.applyColorMap(
        (cam * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    return cv2.addWeighted(face_img, 1 - alpha, heatmap, alpha, 0)


# =========================================================
# FUSION VISUALIZATION
# =========================================================
def build_explainability(face_img, cam):

    gradcam_img = overlay_gradcam(face_img, cam)
    freq_map = frequency_attention_map(face_img)

    fused = cv2.addWeighted(gradcam_img, 0.7, freq_map, 0.3, 0)

    return gradcam_img, freq_map, fused