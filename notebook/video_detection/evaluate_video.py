import os
import torch
import numpy as np

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from app.modules.video.video_dataset import VideoDataset
from app.modules.video.video_model import VideoDeepfakeModel


# ==========================
# CONFIG
# ==========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

TEST_DIR = os.path.join(
    "data",
    "cache_faces",
    "test"
)

MODEL_PATH = "video_model.pth"

BATCH_SIZE = 4



# ==========================
# LOAD MODEL
# ==========================

def load_model():

    model = VideoDeepfakeModel()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    return model



# ==========================
# EVALUATION
# ==========================

def evaluate():

    print("Loading dataset...")

    dataset = VideoDataset(
        TEST_DIR,
        train=False
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )


    model = load_model()


    y_true = []
    y_prob = []


    print("\nEvaluating...")


    with torch.no_grad():

        for frames, labels in loader:


            frames = frames.to(
                DEVICE,
                non_blocking=True
            )


            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            logits, _ = model(
                frames
            )


            probs = torch.softmax(
                logits,
                dim=1
            )[:, 1]


            y_true.extend(
                labels.cpu().numpy()
            )


            y_prob.extend(
                probs.cpu().numpy()
            )



    # Convert numpy

    y_true = np.array(
        y_true
    )

    y_prob = np.array(
        y_prob
    )



    # ==========================
    # AUC
    # ==========================

    auc = roc_auc_score(
        y_true,
        y_prob
    )



    # ==========================
    # FIND BEST THRESHOLD
    # ==========================

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_prob
    )


    best_index = np.argmax(
        tpr - fpr
    )


    best_threshold = thresholds[
        best_index
    ]



    # ==========================
    # APPLY THRESHOLD
    # ==========================

    y_pred = (
        y_prob >= best_threshold
    ).astype(int)



    # ==========================
    # METRICS
    # ==========================

    acc = accuracy_score(
        y_true,
        y_pred
    )


    f1 = f1_score(
        y_true,
        y_pred
    )



    # ==========================
    # RESULTS
    # ==========================

    print("\n==============================")
    print("VIDEO MODEL RESULTS")
    print("==============================")

    print(
        f"Best Threshold : {best_threshold:.4f}"
    )

    print(
        f"Accuracy       : {acc:.4f}"
    )

    print(
        f"F1 Score       : {f1:.4f}"
    )

    print(
        f"AUC            : {auc:.4f}"
    )


    print("\nConfusion Matrix")

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )


    print("\nClassification Report")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Real",
                "Fake"
            ]
        )
    )



# ==========================
# ENTRY POINT
# ==========================

if __name__ == "__main__":

    evaluate()