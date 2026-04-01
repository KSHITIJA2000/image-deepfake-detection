import os

# Root project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Image model path
IMAGE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "image_model",
    "Hybrid_Swin_EffNet_best1.pth"
)

audio_model_path = os.path.join(
    BASE_DIR,
    "models",
    "audio_model",
    "audio_model.pth"
)
GRADCAM_DIR = "static/gradcam"

# GradCAM output folder
GRADCAM_DIR = os.path.join(BASE_DIR, "static", "gradcam")

# Ensure GradCAM folder exists
os.makedirs(GRADCAM_DIR, exist_ok=True)