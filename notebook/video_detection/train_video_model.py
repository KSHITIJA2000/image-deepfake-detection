import os

from torchgen import model
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import warnings

warnings.filterwarnings("ignore")

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))    
import multiprocessing
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
from tqdm import tqdm

from app.modules.video.video_dataset import VideoDataset
from app.modules.video.video_model import VideoDeepfakeModel


############################################################
# CONFIG
############################################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CACHE_DIR = os.path.join("data", "cache_faces")
PLOTS_DIR = "plots"  # Directory to save the graphs

BATCH_SIZE = 4
EPOCHS = 30

LR = 1e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 5
MIN_DELTA = 0.001
GRAD_ACCUM = 2


############################################################
# METRICS
############################################################

def compute_metrics(y_true, y_pred, y_prob):

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5

    return acc, f1, auc


############################################################
# MAIN FUNCTION
############################################################

def main():

    ########################################################
    # DATA
    ########################################################

    train_dataset = VideoDataset(os.path.join(CACHE_DIR, "train"), train=True)
    val_dataset = VideoDataset(os.path.join(CACHE_DIR, "val"), train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    ########################################################
    # MODEL
    ########################################################

    model = VideoDeepfakeModel().to(DEVICE)

    print("Freezing EfficientNet backbone...")
    model.freeze_backbone()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    ########################################################
    # STORAGE
    ########################################################

    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    train_aucs, val_aucs = [], []
    train_accs, val_accs = [], []

    ########################################################
    # EARLY STOPPING
    ########################################################

    best_score = 0
    best_f1 = 0
    patience_counter = 0

    ########################################################
    # TRAIN LOOP
    ########################################################

    for epoch in range(EPOCHS):
        if epoch == 5:
            print("\n===== Unfreezing EfficientNet Last Blocks =====")
            for p in model.backbone.features[-2:].parameters():
                p.requires_grad = True

            # Add the newly unfrozen parameters to the existing optimizer
            optimizer.add_param_group({
                "params": model.backbone.features[-2:].parameters(),
                "lr": 1e-6
            })
            scheduler.min_lrs.append(0.0)
        if epoch == 8:
            print("\n===== Unfreezing Entire EfficientNet =====")
            model.unfreeze_backbone()

            # Filter out parameters that are already in the optimizer to avoid duplicates
            existing_params = set(p for group in optimizer.param_groups for p in group['params'])
            new_params = [p for p in model.backbone.parameters() if p not in existing_params and p.requires_grad]

            optimizer.add_param_group({
                "params": new_params,
                "lr": 5e-7
            })
            scheduler.min_lrs.append(0.0)

        ####################################################
        # TRAIN
        ####################################################

        model.train()
        # ---> ADD THESE 3 LINES RIGHT HERE <---
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.eval()

        train_loss = 0
        y_true, y_pred, y_prob = [], [], []

        optimizer.zero_grad()

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]")

        for step, (frames, labels) in enumerate(train_bar):

            frames = frames.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                logits, _ = model(frames)
                loss = criterion(logits, labels)

            train_loss += loss.item()

            loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

            train_bar.set_postfix(loss=loss.item())

        ####################################################
        # VALIDATION
        ####################################################

        model.eval()

        val_loss = 0
        y_true_v, y_pred_v, y_prob_v = [], [], []

        with torch.no_grad():

            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [VAL]")

            for frames, labels in val_bar:

                frames = frames.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                    logits, _ = model(frames)
                    loss = criterion(logits, labels)

                val_loss += loss.item()

                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                y_true_v.extend(labels.cpu().numpy())
                y_pred_v.extend(preds)
                y_prob_v.extend(probs)

        ####################################################
        # METRICS
        ####################################################

        train_acc, train_f1, train_auc = compute_metrics(y_true, y_pred, y_prob)
        val_acc, val_f1, val_auc = compute_metrics(y_true_v, y_pred_v, y_prob_v)

        ####################################################
        # STORE
        ####################################################

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        ####################################################
        # PRINT
        ####################################################

        print("\n====================================================")
        print(f"Epoch {epoch+1}")
        print("----------------------------------------------------")
        print(f"Train Loss : {train_loss / len(train_loader):.4f}")
        print(f"Val Loss   : {val_loss / len(val_loader):.4f}")
        print("----------------------------------------------------")
        print(f"Train F1  : {train_f1:.4f} | Val F1  : {val_f1:.4f}")
        print(f"Train AUC : {train_auc:.4f} | Val AUC : {val_auc:.4f}")
        print("====================================================")

        ####################################################
        # SCHEDULER
        ####################################################

        scheduler.step(val_auc)

        ####################################################
        # SAVE BEST
        ####################################################

        score = val_auc

        if score > best_score + MIN_DELTA:
            best_score = score
            best_f1 = val_f1
            patience_counter = 0

            torch.save(model.state_dict(), "video_model.pth")
            print(">>> BEST MODEL SAVED")
        else:
            patience_counter += 1

        ####################################################
        # EARLY STOPPING
        ####################################################

        if patience_counter >= PATIENCE:
            print("\n>>> EARLY STOPPING TRIGGERED")
            break

    ############################################################
    # CONFUSION MATRIX
    ############################################################
    
    os.makedirs(PLOTS_DIR, exist_ok=True)

    cm = confusion_matrix(y_true_v, y_pred_v)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"])

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.show()

    ############################################################
    # PLOTS
    ############################################################

    epochs_range = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"))
    plt.show()

    plt.figure()
    plt.plot(epochs_range, train_f1s, label="Train F1")
    plt.plot(epochs_range, val_f1s, label="Val F1")
    plt.legend()
    plt.title("F1 Curve")
    plt.savefig(os.path.join(PLOTS_DIR, "f1_curve.png"))
    plt.show()

    plt.figure()
    plt.plot(epochs_range, train_aucs, label="Train AUC")
    plt.plot(epochs_range, val_aucs, label="Val AUC")
    plt.legend()
    plt.title("AUC Curve")
    plt.savefig(os.path.join(PLOTS_DIR, "auc_curve.png"))
    plt.show()

    plt.figure()
    plt.plot(epochs_range, train_accs, label="Train Acc")
    plt.plot(epochs_range, val_accs, label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy_curve.png"))
    plt.show()

    ############################################################
    # DONE
    ############################################################

    print("\nTRAINING COMPLETE")
    print("BEST F1:", best_f1)


############################################################
# ENTRY POINT (WINDOWS FIX)
############################################################

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()