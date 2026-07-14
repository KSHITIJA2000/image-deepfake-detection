import os
import cv2
import torch
import numpy as np
import uuid
from collections import OrderedDict

from app.modules.image.hybrid_model import DeepfakeDetector
from app.modules.image.face_helper import FaceHelper
from app.modules.image.gradcam import GradCAM
from app.config import IMAGE_MODEL_PATH, IMAGE_GRADCAM_DIR


class ImageDeepfakeDetector:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("\n[INFO] Loading Image Deepfake Model...\n")

        # ==========================================
        # MODEL
        # ==========================================

        self.model = DeepfakeDetector()

        checkpoint = torch.load(
            IMAGE_MODEL_PATH,
            map_location=self.device,
            weights_only=False
        )

        state_dict = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )

        clean_state = OrderedDict()

        for k, v in state_dict.items():
            clean_state[k.replace("module.", "")] = v

        self.model.load_state_dict(
            clean_state,
            strict=True
        )

        self.model.to(self.device)
        self.model.eval()

        print("[OK] Model loaded successfully")


        # ==========================================
        # HELPERS
        # ==========================================

        self.face_helper = FaceHelper()

        self.gradcam = GradCAM(
            self.model
        )

        # Separate GradCAM folder
        os.makedirs(
            IMAGE_GRADCAM_DIR,
            exist_ok=True
        )


    # =====================================================
    # PREDICTION
    # =====================================================

    def predict(self, img_path):

        try:

            # ==========================================
            # FACE EXTRACTION
            # ==========================================

            img_tensor = self.face_helper.extract_face(
                img_path
            )

            if img_tensor is None:
                raise ValueError(
                    "Face extraction failed"
                )


            img_tensor = (
                img_tensor
                .unsqueeze(0)
                .to(self.device)
            )


            # ==========================================
            # INFERENCE
            # ==========================================

            with torch.no_grad():

                logits = self.model(
                    img_tensor
                )

                probs = torch.softmax(
                    logits,
                    dim=1
                )[0]


                pred_class = torch.argmax(
                    probs
                ).item()


                real_prob = float(
                    probs[0].item()
                )

                fake_prob = float(
                    probs[1].item()
                )


                prediction = (
                    "fake"
                    if pred_class == 1
                    else "real"
                )


                confidence = float(
                    probs[pred_class].item()
                )


            confidence = float(
                np.clip(
                    confidence,
                    0.0,
                    1.0
                )
            )


            # ==========================================
            # GRADCAM GENERATION
            # ==========================================

            cam_url = None


            image = cv2.imread(
                img_path
            )


            if image is not None:


                # GradCAM requires gradients
                cam = self.gradcam.generate(
                    img_tensor
                )


                h, w = image.shape[:2]


                cam = cv2.resize(
                    cam,
                    (w, h)
                )


                heatmap = cv2.applyColorMap(
                    np.uint8(cam * 255),
                    cv2.COLORMAP_JET
                )


                overlay = cv2.addWeighted(
                    image,
                    0.6,
                    heatmap,
                    0.4,
                    0
                )


                # Unique filename
                cam_filename = (
                    f"cam_{uuid.uuid4().hex}.jpg"
                )


                cam_path = os.path.join(
                    IMAGE_GRADCAM_DIR,
                    cam_filename
                )


                cv2.imwrite(
                    cam_path,
                    overlay
                )


                # Served through FastAPI mount
                cam_url = (
                    f"/gradcam/{cam_filename}"
                    f"?t={uuid.uuid4().hex}"
                )


            # ==========================================
            # EXPLANATION
            # ==========================================

            explanation = (
                f"Prediction: {prediction.upper()} | "
                f"Fake Probability: {fake_prob:.4f} | "
                f"Real Probability: {real_prob:.4f}"
            )


            return (
                prediction,
                confidence,
                fake_prob,
                real_prob,
                cam_url,
                explanation
            )


        except Exception as e:


            print(
                f"[IMAGE ERROR] {e}"
            )


            return (
                "error",
                0.0,
                0.0,
                0.0,
                None,
                str(e)
            )