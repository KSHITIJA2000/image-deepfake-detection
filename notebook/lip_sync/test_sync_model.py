import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from torch.utils.data import DataLoader

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.append(PROJECT_ROOT)

# ==========================================================
# IMPORTS
# ==========================================================

from app.modules.lip_sync.mouth_dataset import MouthDataset
from app.modules.lip_sync.lip_sync_model import LipSyncModel

# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Lip Sync Deepfake Detection - Testing")
print("=" * 60)
print(f"Using Device : {DEVICE}")

# ==========================================================
# PATHS
# ==========================================================

DATASET_PATH = "data/lip_sync/mouth_roi"

TEST_PATH = os.path.join(DATASET_PATH, "test")

SAVE_DIR = "models/lip_sync_model"

os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    SAVE_DIR,
    "sync_model_final_best.pth"
)

# ==========================================================
# DATASET
# ==========================================================

test_dataset = MouthDataset(
    TEST_PATH,
    augment=False
)

print(f"Test Samples : {len(test_dataset)}")

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ==========================================================
# MODEL
# ==========================================================

model = LipSyncModel().to(DEVICE)

print("\nLoading Best Model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model Loaded Successfully!")

# ==========================================================
# TESTING
# ==========================================================

test_true = []
test_pred = []
test_prob = []

print("\nRunning Test Set...\n")

with torch.no_grad():

    for x, y in tqdm(test_loader):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        outputs = model(x)

        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(probs, dim=1)

        test_true.extend(y.cpu().numpy())
        test_pred.extend(preds.cpu().numpy())
        test_prob.extend(probs[:, 1].cpu().numpy())

# ==========================================================
# METRICS
# ==========================================================

accuracy = accuracy_score(
    test_true,
    test_pred
)

precision = precision_score(
    test_true,
    test_pred,
    zero_division=0
)

recall = recall_score(
    test_true,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    test_true,
    test_pred,
    zero_division=0
)

try:
    auc = roc_auc_score(
        test_true,
        test_prob
    )
except:
    auc = 0.0

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {auc:.4f}")

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    test_true,
    test_pred
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Real", "Fake"],
    yticklabels=["Real", "Fake"]
)

plt.title("Lip Sync Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()
# ==========================================================
# ROC CURVE
# ==========================================================

fpr, tpr, _ = roc_curve(
    test_true,
    test_prob
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "roc_curve.png"
    ),
    dpi=300
)

plt.close()

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report = classification_report(
    test_true,
    test_pred,
    target_names=[
        "Real",
        "Fake"
    ]
)

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(report)

with open(
    os.path.join(
        SAVE_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)

# ==========================================================
# SAVE FINAL METRICS
# ==========================================================

with open(
    os.path.join(
        SAVE_DIR,
        "final_metrics.txt"
    ),
    "w"
) as f:

    f.write("=" * 40 + "\n")
    f.write("Lip Sync Test Results\n")
    f.write("=" * 40 + "\n\n")

    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1 Score  : {f1:.4f}\n")
    f.write(f"ROC AUC   : {auc:.4f}\n")

# ==========================================================
# SAVE TEST PREDICTIONS
# ==========================================================

import csv

csv_path = os.path.join(
    SAVE_DIR,
    "test_predictions.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "Actual",
        "Predicted",
        "Probability_Fake"
    ])

    for gt, pred, prob in zip(
        test_true,
        test_pred,
        test_prob
    ):

        writer.writerow([
            gt,
            pred,
            float(prob)
        ])

# ==========================================================
# SUMMARY
# ==========================================================

print("\n")
print("=" * 60)
print("FILES SAVED")
print("=" * 60)

print(f"Model                 : {MODEL_PATH}")
print(f"Confusion Matrix      : {os.path.join(SAVE_DIR,'confusion_matrix.png')}")
print(f"ROC Curve             : {os.path.join(SAVE_DIR,'roc_curve.png')}")
print(f"Classification Report : {os.path.join(SAVE_DIR,'classification_report.txt')}")
print(f"Metrics               : {os.path.join(SAVE_DIR,'final_metrics.txt')}")
print(f"Predictions CSV       : {os.path.join(SAVE_DIR,'test_predictions.csv')}")

print("\n")
print("=" * 60)
print("LIP SYNC MODEL EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)