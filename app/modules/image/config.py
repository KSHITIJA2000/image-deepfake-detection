import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "image_model",
    "Hybrid_Swin_EffNet_best1.pth"
)
