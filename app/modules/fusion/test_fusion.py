import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import warnings

warnings.filterwarnings("ignore")

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)

import torch
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


from app.modules.fusion.FusionDatasetCached import FusionDatasetCached
from app.modules.fusion.fusion_model import FusionModel



####################################################
# DEVICE
####################################################

DEVICE = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)

print(
    "Device:",
    DEVICE
)



####################################################
# PATHS
####################################################

IMAGE_MODEL_PATH = (
    "models/image_model/"
    "Hybrid_Swin_EffNet_best1.pth"
)


VIDEO_MODEL_PATH = (
    "models/video_model/"
    "video_model.pth"
)


AUDIO_MODEL_PATH = (
    "models/audio_model/"
    "audio_model.pth"
)


LIP_MODEL_PATH = (
    "models/lip_sync_model/"
    "sync_model_final_best.pth"
)



FUSION_MODEL_PATH = (
    "models/fusion_model/"
    "best_fusion_model.pth"
)



CACHE_DIR = "fusion_cache"
####################################################
# TEST DATASET
####################################################

test_dataset = FusionDatasetCached(

    CACHE_DIR,

    split="test"

)



test_loader = DataLoader(

    test_dataset,

    batch_size=4,

    shuffle=False,

    num_workers=0,

    pin_memory=True

)



####################################################
# LOAD FUSION MODEL
####################################################

model = FusionModel(

    image_model_path=IMAGE_MODEL_PATH,

    video_model_path=VIDEO_MODEL_PATH,

    audio_model_path=AUDIO_MODEL_PATH,

    lip_model_path=LIP_MODEL_PATH

).to(DEVICE)



model.load_state_dict(

    torch.load(

        FUSION_MODEL_PATH,

        map_location=DEVICE

    )

)


model.eval()



####################################################
# INFERENCE
####################################################

all_labels = []

all_predictions = []

all_probabilities = []



with torch.no_grad():


    for batch in tqdm(

        test_loader,

        desc="Testing Fusion Model"

    ):


        image = batch["image"].to(
            DEVICE
        )


        video = batch["video"].to(
            DEVICE
        )


        audio = batch["audio"].to(
            DEVICE
        )


        lip = batch["lip"].to(
            DEVICE
        )


        labels = batch["label"].to(
            DEVICE
        )



        outputs = model(

            image,

            video,

            audio,

            lip

        )



        probabilities = F.softmax(

            outputs,

            dim=1

        )



        predictions = torch.argmax(

            probabilities,

            dim=1

        )



        all_labels.extend(

            labels.cpu().numpy()

        )


        all_predictions.extend(

            predictions.cpu().numpy()

        )


        # Fake probability

        all_probabilities.extend(

            probabilities[:,1]
            .cpu()
            .numpy()

        )



print(
    "Testing completed"
)
####################################################
# METRICS
####################################################

accuracy = accuracy_score(

    all_labels,

    all_predictions

)


precision = precision_score(

    all_labels,

    all_predictions,

    zero_division=0

)


recall = recall_score(

    all_labels,

    all_predictions,

    zero_division=0

)


f1 = f1_score(

    all_labels,

    all_predictions,

    zero_division=0

)


auc = roc_auc_score(

    all_labels,

    all_probabilities

)



####################################################
# PRINT RESULTS
####################################################

print("\n")
print("=" * 60)
print("FUSION MODEL RESULTS")
print("=" * 60)


print(
    f"Accuracy  : {accuracy:.4f}"
)


print(
    f"Precision : {precision:.4f}"
)


print(
    f"Recall    : {recall:.4f}"
)


print(
    f"F1 Score  : {f1:.4f}"
)


print(
    f"ROC-AUC   : {auc:.4f}"
)



####################################################
# CLASSIFICATION REPORT
####################################################

print(

    classification_report(

        all_labels,

        all_predictions,

        target_names=[

            "Real",

            "Fake"

        ]

    )

)



####################################################
# CONFUSION MATRIX
####################################################

cm = confusion_matrix(

    all_labels,

    all_predictions

)


print(

    "Confusion Matrix:"

)


print(cm)



####################################################
# SAVE RESULTS
####################################################

os.makedirs(

    "results/fusion",

    exist_ok=True

)



np.save(

    "results/fusion/confusion_matrix.npy",

    cm

)


np.save(

    "results/fusion/test_labels.npy",

    np.array(all_labels)

)


np.save(

    "results/fusion/test_predictions.npy",

    np.array(all_predictions)

)


np.save(

    "results/fusion/fake_probabilities.npy",

    np.array(all_probabilities)

)



print(

    "\nResults saved in results/fusion/"

)