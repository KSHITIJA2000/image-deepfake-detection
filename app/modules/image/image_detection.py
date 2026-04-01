import torch
import os
from collections import OrderedDict
from app.modules.image.hybrid_model import HybridDeepfakeDetector
from app.modules.image.face_utils import FaceDeepfakeDetector
from app.config import IMAGE_MODEL_PATH, GRADCAM_DIR

class ImageDeepfakeDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HybridDeepfakeDetector()
        
        # Load weights
        state_dict = torch.load(IMAGE_MODEL_PATH, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("features."): new_state_dict[f"effnet.{k}"] = v
            elif k.startswith("norm."): new_state_dict[f"swin.{k}"] = v
            elif k.startswith("module."): new_state_dict[k.replace("module.", "")] = v
            else: new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.face_helper = FaceDeepfakeDetector(self.model, self.device)
        os.makedirs(GRADCAM_DIR, exist_ok=True)

    def predict(self, img_path):
        res = self.face_helper.predict(img_path, GRADCAM_DIR)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Returns: label, confidence, fake_prob, real_prob, gradcam_url, explanation
        return (
            res["prediction"], 
            res["confidence"], 
            res["fake_prob"], 
            res["real_prob"], 
            res["gradcam"], 
            res["explanation"]
        )