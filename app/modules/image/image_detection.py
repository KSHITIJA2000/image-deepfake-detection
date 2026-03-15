import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
from collections import OrderedDict

from app.gradcam import GradCAM
from app.face_utils import extract_face
from app.config import IMAGE_MODEL_PATH
from app.hybrid_model import HybridDeepfakeDetector

class ImageDeepfakeDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = HybridDeepfakeDetector()
        
        # --- SMART WEIGHT LOADER (Fixes RuntimeError) ---
        state_dict = torch.load(IMAGE_MODEL_PATH, map_location=self.device)
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            # If your weights were saved as 'features.X' instead of 'effnet.features.X'
            if k.startswith('features.'):
                new_state_dict[f"effnet.{k}"] = v
            # If Swin weights need prefixing
            elif k.startswith('norm.'):
                new_state_dict[f"swin.{k}"] = v
            else:
                new_state_dict[k] = v
        
        # Load flexibly (strict=False ignores minor non-weight keys)
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device).eval()
        # --- END SMART LOADER ---

        # Initialize Grad-CAM for Explainable AI
        self.gradcam = GradCAM(
            self.model,
            target_layer=self.model.effnet.features[-1] 
        )

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        os.makedirs("static/gradcam", exist_ok=True)

    def predict(self, img_path):
        image = Image.open(img_path).convert("RGB")
        face = extract_face(image)

        input_tensor = self.transform(face).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        real_prob = probs[0].item()
        fake_prob = probs[1].item()

        # --- CORRECT BINARY DECISION LOGIC ---
        # Picking the higher probability ensures a "Correct" classification
        if fake_prob > real_prob:
            label = "FAKE"
            confidence = fake_prob
        else:
            label = "REAL"
            confidence = real_prob

        # Generate Grad-CAM Heatmap for verification
        cam = self.gradcam.generate(input_tensor, class_idx=1)
        cam = cv2.resize(cam, face.size)

        face_np = np.array(face)
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(face_np, 0.6, heatmap, 0.4, 0)

        # Save the result
        cam_filename = os.path.basename(img_path).rsplit(".", 1)[0] + "_cam.jpg"
        cam_path = os.path.join("static/gradcam", cam_filename)

        cv2.imwrite(
            cam_path,
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

        return label, confidence, fake_prob, real_prob, cam_filename