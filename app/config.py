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
# Image GradCAM output
IMAGE_GRADCAM_DIR = "gradcam_outputs"


# Video GradCAM output
VIDEO_GRADCAM_DIR = "static/gradcam"

# GradCAM output folder
GRADCAM_DIR = os.path.join(BASE_DIR, "static", "gradcam")

# Ensure GradCAM folder exists
os.makedirs(GRADCAM_DIR, exist_ok=True)

import os

# =========================================================
# ROOT DIRECTORY
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =========================================================
# MODEL PATHS
# =========================================================

IMAGE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "image_model",
    "Hybrid_Swin_EffNet_best1.pth"
)

AUDIO_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "audio_model",
    "audio_model.pth"
)

VIDEO_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "video_model",
    "video_model.pth"
)



LIPSYNC_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lip_sync_model",
    "sync_model_final_best.pth"
)
FUSION_MODEL_PATH = (
    "models/fusion_model/"
    "best_fusion_model.pth"
)


# =========================================================
# OUTPUT DIRECTORIES
# =========================================================

GRADCAM_DIR = os.path.join(BASE_DIR, "static", "gradcam")

VIDEO_FRAME_DIR = os.path.join(BASE_DIR, "static", "video_frames")

UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")


# =========================================================
# DATASET PATHS
# =========================================================

IMAGE_DATASET_PATH = os.path.join(BASE_DIR, "data", "image")

AUDIO_DATASET_PATH = os.path.join(BASE_DIR, "data", "audio")

VIDEO_DATASET_PATH = os.path.join(BASE_DIR, "data", "video")


# =========================================================
# TRAINING PARAMETERS (GLOBAL)
# =========================================================

BATCH_SIZE = 8
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


# =========================================================
# MODEL PARAMETERS
# =========================================================

FRAME_SIZE = 224
SEQUENCE_LENGTH = 20
FEATURE_DIM = 512
NUM_CLASSES = 2


# =========================================================
# CREATE DIRECTORIES IF NOT EXISTS
# =========================================================

os.makedirs(GRADCAM_DIR, exist_ok=True)
os.makedirs(VIDEO_FRAME_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)