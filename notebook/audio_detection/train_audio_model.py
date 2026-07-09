import os
import sys
import argparse
from typing import Tuple, List, Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from app.modules.audio.dataset_loader import AudioDeepfakeDataset, collate_skip_corrupted
from app.modules.audio.model import AudioDeepfakeCNNLSTM


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> Tuple[float, float, float, float, Any, List[int], List[float]]:
    
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    y_true, y_pred, y_probs = [], [], []

    for x, y in loader:
        if x.numel() == 0:
            continue

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == y).sum().item()
        total_samples += batch_size
        
        # Track for metrics
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.detach().cpu().numpy())

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, None, [], []

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    # Compute Advanced Metrics safely
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.5  # Fallback if batch/epoch contains only one class
    cm = confusion_matrix(y_true, y_pred)

    return avg_loss, accuracy, f1, auc, cm, y_true, y_probs


def main():
    parser = argparse.ArgumentParser(description="Train audio deepfake detection model.")
    parser.add_argument("--train_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "audio", "train"))
    parser.add_argument("--val_dir", type=str,   default=os.path.join(PROJECT_ROOT, "data", "audio", "val"))
    
    # Base dir for saving outputs (models, plots, history)
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "models", "audio_model"))
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets and DataLoaders
    train_dataset = AudioDeepfakeDataset(root_dir=args.train_dir, augment=True)
    val_dataset   = AudioDeepfakeDataset(root_dir=args.val_dir, augment=False)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory,
                              collate_fn=collate_skip_corrupted)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_memory,
                              collate_fn=collate_skip_corrupted)

    # Model, loss, optimizer
    model = AudioDeepfakeCNNLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Track F1 for the scheduler instead of accuracy
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, "audio_model.pth")

    # Tracking Variables
    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_cm, best_y_true, best_y_probs = None, [], []

    history = {
        "train_loss": [], "train_acc": [], "train_f1": [], "train_auc": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "val_auc": []
    }

    print("\nStarting Training...")
    
    # Training loop
    for epoch in range(args.epochs):
        tr_loss, tr_acc, tr_f1, tr_auc, _, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        
        val_loss, val_acc, val_f1, val_auc, val_cm, val_y_true, val_y_probs = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )

        # Record metrics
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["train_auc"].append(tr_auc)
        
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] | "
            f"Train -> Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} F1: {tr_f1:.4f} AUC: {tr_auc:.4f} | "
            f"Val -> Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} AUC: {val_auc:.4f}"
        )

        # Scheduler step based on val F1
        scheduler.step(val_f1)

        # Save best model based on F1 Score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_cm = val_cm
            best_y_true = val_y_true
            best_y_probs = val_y_probs
            epochs_no_improve = 0
            
            torch.save(model.state_dict(), model_save_path)
            print(f"  >> Best model saved (Val F1: {best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  >> No improvement ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # -----------------------------
    # POST-TRAINING VISUALIZATIONS
    # -----------------------------
    print("\nGenerating evaluation plots...")
    epochs_range = range(1, len(history["train_loss"]) + 1)
    
    # 1. Plot & Save Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", marker="s")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # 2. Plot & Save Best Confusion Matrix
    if best_cm is not None:
        plt.figure(figsize=(6, 5))
        sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
        plt.title(f"Best Validation Confusion Matrix (F1: {best_val_f1:.4f})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=300)
        plt.close()

    # 3. Plot & Save ROC Curve
    if len(best_y_true) > 0 and len(best_y_probs) > 0:
        fpr, tpr, _ = roc_curve(best_y_true, best_y_probs)
        best_auc = roc_auc_score(best_y_true, best_y_probs)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {best_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Best Epoch)")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "roc_curve.png"), dpi=300)
        plt.close()

    # 4. Save Raw History Dictionary
    history_path = os.path.join(args.output_dir, "training_history.pth")
    torch.save(history, history_path)

    print(f"\nTraining complete. Artifacts saved in: {args.output_dir}")
    print(f"Best validation F1 Score: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()

