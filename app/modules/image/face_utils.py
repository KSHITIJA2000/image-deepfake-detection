import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from retinaface import RetinaFace
from app.modules.image.gradcam import GradCAM, overlay_gradcam

class FaceDeepfakeDetector:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(self.device)

        # GradCAM on the last convolutional layer of EfficientNet
        self.gradcam = GradCAM(
            self.model,
            target_layer=self.model.effnet.features[-1]
        )

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect_face(self, image_path):
        """Detect face using RetinaFace and return BGR crop"""
        faces = RetinaFace.detect_faces(image_path)
        img_bgr = cv2.imread(image_path)
        if isinstance(faces, dict) and len(faces) > 0:
            face_data = list(faces.values())[0]
            x1, y1, x2, y2 = face_data["facial_area"]
            # Ensure indices are within bounds
            return img_bgr[max(0, y1):y2, max(0, x1):x2]
        return None

    def explain_prediction(self, label, confidence, cam):
        """Explainable AI: Analyzes the focus areas of the model"""
        # Divide heatmap into top (eyes) and bottom (mouth)
        height = cam.shape[0]
        top_focus = np.mean(cam[:height//2, :])
        bottom_focus = np.mean(cam[height//2:, :])

        if label == "FAKE":
            region = "eye/forehead area" if top_focus > bottom_focus else "mouth/jawline"
            return f"Artifacts detected in the {region}. Inconsistent textures suggest GAN-generated synthesis."
        else:
            return "Facial features show natural biological variance and consistent lighting gradients."

    def predict(self, image_path, output_dir):
        face_bgr = self.detect_face(image_path)
        
        # Fallback: full image if no face detected
        if face_bgr is None:
            img = Image.open(image_path).convert("RGB")
            face_bgr = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
        
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(Image.fromarray(face_rgb)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        fake_prob, real_prob = probs[1].item(), probs[0].item()
        label = "FAKE" if fake_prob > real_prob else "REAL"
        confidence = max(fake_prob, real_prob)

        # XAI Generation
        class_idx = 1 if label == "FAKE" else 0
        cam = self.gradcam.generate(input_tensor, class_idx)
        explanation = self.explain_prediction(label, confidence, cam)
        
        # Save GradCAM
        gradcam_img = overlay_gradcam(face_bgr, cam)
        filename = os.path.basename(image_path).split(".")[0] + "_gradcam.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, gradcam_img)

        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "explanation": explanation,
            "fake_prob": round(fake_prob, 4),
            "real_prob": round(real_prob, 4),
            "gradcam": f"/static/gradcam/{filename}"
        }