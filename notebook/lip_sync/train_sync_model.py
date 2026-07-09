import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
)


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
from torch.amp import autocast, GradScaler

from app.modules.lip_sync.mouth_dataset import MouthDataset
from app.modules.lip_sync.lip_sync_model import LipSyncModel

# ==========================================================
# SEED
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device : {DEVICE}")

# ==========================================================
# PATHS
# ==========================================================

DATASET_PATH = "data/lip_sync/mouth_roi"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
TEST_PATH = os.path.join(DATASET_PATH, "test")

SAVE_DIR = "models/lip_sync_model"

os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    SAVE_DIR,
    "sync_model_final_best.pth"
)

# ==========================================================
# HYPERPARAMETERS
# ==========================================================

BATCH_SIZE = 8

EPOCHS = 50

LR = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 10

NUM_WORKERS = 0      # Windows safe

PIN_MEMORY = True

# ==========================================================
# DATASETS
# ==========================================================

train_dataset = MouthDataset(
    TRAIN_PATH,
    augment=True
)

val_dataset = MouthDataset(
    VAL_PATH,
    augment=False
)

test_dataset = MouthDataset(
    TEST_PATH,
    augment=False
)

print(f"\nTrain Samples      : {len(train_dataset)}")
print(f"Validation Samples : {len(val_dataset)}")
print(f"Test Samples       : {len(test_dataset)}")

# ==========================================================
# DATALOADERS
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

# ==========================================================
# MODEL
# ==========================================================

model = LipSyncModel().to(DEVICE)

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

scaler = GradScaler()

print(model)

# ==========================================================
# HISTORY
# ==========================================================

history = {

    "train_loss": [],
    "val_loss": [],

    "train_acc": [],
    "val_acc": [],

    "precision": [],
    "recall": [],

    "f1": [],
    "auc": []

}

best_loss = float("inf")

patience_counter = 0

csv_file = os.path.join(
    SAVE_DIR,
    "training_log.csv"
)

with open(csv_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Epoch",
        "Train Loss",
        "Validation Loss",
        "Train Accuracy",
        "Validation Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ])

print("\n==============================")
print("🚀 TRAINING STARTED")
print("==============================\n")
# ==========================================================
# TRAINING LOOP
# ==========================================================

for epoch in range(EPOCHS):

    # --------------------------
    # TRAIN
    # --------------------------
    model.train()

    train_loss = 0
    train_true = []
    train_pred = []

    for x, y in tqdm(train_loader):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=torch.float16):

            outputs = model(x)
            loss = criterion(outputs, y)

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        train_true.extend(y.cpu().numpy())
        train_pred.extend(preds.cpu().numpy())

    train_acc = accuracy_score(train_true, train_pred)
    train_loss = train_loss / len(train_loader)

    # --------------------------
    # VALIDATION
    # --------------------------
    model.eval()

    val_loss = 0
    val_true = []
    val_pred = []
    val_prob = []

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            outputs = model(x)
            loss = criterion(outputs, y)

            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            val_true.extend(y.cpu().numpy())
            val_pred.extend(preds.cpu().numpy())
            val_prob.extend(probs[:, 1].cpu().numpy())

    val_loss = val_loss / len(val_loader)

    # --------------------------
    # METRICS
    # --------------------------
    acc = accuracy_score(val_true, val_pred)
    prec = precision_score(val_true, val_pred, zero_division=0)
    rec = recall_score(val_true, val_pred, zero_division=0)
    f1 = f1_score(val_true, val_pred, zero_division=0)

    try:
        auc = roc_auc_score(val_true, val_prob)
    except:
        auc = 0.0

    scheduler.step()

    # --------------------------
    # LOG
    # --------------------------
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(acc)
    history["precision"].append(prec)
    history["recall"].append(rec)
    history["f1"].append(f1)
    history["auc"].append(auc)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Train Acc  : {train_acc:.4f}")
    print(f"Val Acc    : {acc:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print(f"ROC-AUC    : {auc:.4f}")

    # --------------------------
    # SAVE BEST MODEL
    # --------------------------
    if val_loss < best_loss:

        best_loss = val_loss
        patience_counter = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "f1": f1,
            "auc": auc
        }, MODEL_PATH)

        print("✅ BEST MODEL SAVED")

    else:

        patience_counter += 1
        print(f"EarlyStopping: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("🛑 EARLY STOPPING TRIGGERED")
            break
        # ==========================================================
        # ==========================================================
# PART 4D - TESTING + FINAL EVALUATION
# ==========================================================

print("\n================================")
print("📊 LOADING BEST MODEL")
print("================================\n")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_true, test_pred, test_prob = [], [], []

with torch.no_grad():
    for x, y in tqdm(test_loader):

        x, y = x.to(DEVICE), y.to(DEVICE)

        out = model(x)
        probs = torch.softmax(out, dim=1)

        preds = torch.argmax(probs, dim=1)

        test_true.extend(y.cpu().numpy())
        test_pred.extend(preds.cpu().numpy())
        test_prob.extend(probs[:, 1].cpu().numpy())


# ==========================================================
# METRICS
# ==========================================================

acc = accuracy_score(test_true, test_pred)
precision = precision_score(test_true, test_pred, zero_division=0)
recall = recall_score(test_true, test_pred, zero_division=0)
f1 = f1_score(test_true, test_pred, zero_division=0)

try:
    auc = roc_auc_score(test_true, test_prob)
except:
    auc = 0.0

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {auc:.4f}")


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(test_true, test_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()


# ==========================================================
# ROC CURVE
# ==========================================================

fpr, tpr, _ = roc_curve(test_true, test_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()


# ==========================================================
# LOSS / ACCURACY CURVES
# ==========================================================

plt.figure()
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig("loss_curve.png")
plt.close()

plt.figure()
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig("accuracy_curve.png")
plt.close()

plt.figure()
plt.plot(history["f1"], label="F1 Score")
plt.legend()
plt.title("F1 Curve")
plt.savefig("f1_curve.png")
plt.close()


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\nCLASSIFICATION REPORT:\n")
print(classification_report(test_true, test_pred, target_names=["Real", "Fake"]))


# SAVE FINAL METRICS
with open("final_metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1: {f1}\n")
    f.write(f"AUC: {auc}\n")

print("\n✅ ALL EVALUATIONS COMPLETED")