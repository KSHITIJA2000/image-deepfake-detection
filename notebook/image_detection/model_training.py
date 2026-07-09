import os
import sys
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# =========================================================
# PROJECT ROOT
# =========================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(PROJECT_ROOT)

from notebook.image_detection.image_detection_dataset import ImageDataset
from app.modules.image.hybrid_model import DeepfakeDetector

# =========================================================
# DATASET (CACHED CROPPED FACES ONLY)
# =========================================================
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache_faces")

TRAIN_DIR = os.path.join(CACHE_DIR, "train")
VAL_DIR   = os.path.join(CACHE_DIR, "val")
TEST_DIR  = os.path.join(CACHE_DIR, "test")

# =========================================================
# MODEL SAVE PATH
# =========================================================
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "image_model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "Hybrid_Swin_EffNet_bestceleb.pth")

# =========================================================
# TRAINING
# =========================================================
def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # =====================================================
    # DATASET (NO FACE DETECTION)
    # =====================================================
    train_dataset = ImageDataset(TRAIN_DIR)
    val_dataset   = ImageDataset(VAL_DIR)
    test_dataset  = ImageDataset(TEST_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # =====================================================
    # MODEL
    # =====================================================
    model = DeepfakeDetector().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-6
    )

    scaler = GradScaler("cuda")

    epochs = 40
    accumulation_steps = 4

    best_acc = 0.0
    best_weights = None
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []

    # =====================================================
    # EVALUATION
    # =====================================================
    def evaluate(loader):

        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(device_type="cuda", enabled=device.type == "cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)

                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    # =====================================================
    # TRAIN LOOP
    # =====================================================
    for epoch in range(epochs):

        print("\n" + "=" * 60)
        print(f"Epoch {epoch+1}/{epochs}")
        print("=" * 60)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0
        running_correct = 0
        running_total = 0

        for step, (images, labels) in enumerate(train_loader):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)

                scheduler.step(epoch + step / len(train_loader))

            running_loss += loss.item() * accumulation_steps * images.size(0)

            preds = outputs.argmax(1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc = evaluate(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.4f}")

        # =================================================
        # SAVE BEST MODEL
        # =================================================
        if val_acc > best_acc:

            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0

            torch.save(best_weights, MODEL_PATH)
            print("✅ Best model saved")

        else:

            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print("🛑 Early stopping")
                break

    # =====================================================
    # LOAD BEST MODEL
    # =====================================================
    print("\nLoading best model...")

    model.load_state_dict(
        best_weights if best_weights else torch.load(MODEL_PATH)
    )
    model.eval()

    # =====================================================
    # TEST EVALUATION
    # =====================================================
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]

            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # =====================================================
    # METRICS
    # =====================================================
    acc = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    cm = confusion_matrix(all_labels, all_preds)

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(cm)

    # =====================================================
    # GRAPHS
    # =====================================================

    # LOSS CURVE
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(PROJECT_ROOT, "loss_curve.png"))
    plt.close()

    # CONFUSION MATRIX
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.savefig(os.path.join(PROJECT_ROOT, "confusion_matrix.png"))
    plt.close()

    # ROC CURVE
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("ROC Curve")

    plt.savefig(os.path.join(PROJECT_ROOT, "roc_curve.png"))
    plt.close()

    print("\nTraining Completed Successfully")
    print("Best Val Acc:", best_acc)


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    train()