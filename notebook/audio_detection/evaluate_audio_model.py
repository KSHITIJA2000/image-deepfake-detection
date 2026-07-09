import os
import argparse
import torch
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from torch.utils.data import DataLoader

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
)

from app.modules.audio.dataset_loader import (
    AudioDeepfakeDataset,
    collate_skip_corrupted
)

from app.modules.audio.model import AudioDeepfakeCNNLSTM


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--val_dir", default="data/audio/val")
    parser.add_argument("--model_path", default="models/audio_model/audio_model.pth")
    parser.add_argument("--history_path", default="models/audio_model/training_history.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Dataset
    # -----------------------------
    val_dataset = AudioDeepfakeDataset(
        root_dir=args.val_dir,
        augment=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_skip_corrupted
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = AudioDeepfakeCNNLSTM().to(device)

    model.load_state_dict(
        torch.load(args.model_path, map_location=device)
    )

    model.eval()

    y_true, y_pred, y_probs = [], [], []

    print("Running evaluation...")

    # -----------------------------
    # Evaluation loop
    # -----------------------------
    with torch.no_grad():
        for x, y in val_loader:

            if x.numel() == 0:
                continue

            x = x.to(device)

            outputs = model(x)

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_probs.extend(probs)
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    # -----------------------------
    # Metrics
    # -----------------------------
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0
    )

    print("\n===== Evaluation Results =====")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"]
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig("audio_confusion_matrix.png", dpi=300)
    plt.close()

    print("Saved: audio_confusion_matrix.png")

    # -----------------------------
    # ROC Curve
    # -----------------------------
    roc_auc = roc_auc_score(y_true, y_probs)
    fpr, tpr, _ = roc_curve(y_true, y_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig("audio_roc_curve.png", dpi=300)
    plt.close()

    print(f"Saved: audio_roc_curve.png (AUC = {roc_auc:.4f})")

    # -----------------------------
    # Loss Curve (FIXED)
    # -----------------------------
    if os.path.exists(args.history_path):

        history = torch.load(
        args.history_path,
        map_location="cpu",
        weights_only=False
        )

        print("History keys:", history.keys())

        train_losses = history.get("train_loss") or history.get("train_losses")
        val_losses = history.get("val_loss") or history.get("val_losses")

        if train_losses is None or val_losses is None:
            print("Loss data not found in history file.")
        else:
            epochs = range(1, len(train_losses) + 1)

            plt.figure(figsize=(8, 5))
            plt.plot(epochs, train_losses, marker="o", label="Training Loss")
            plt.plot(epochs, val_losses, marker="s", label="Validation Loss")

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training vs Validation Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("audio_loss_curve.png", dpi=300)
            plt.close()

            print("Saved: audio_loss_curve.png")

    else:
        print(f"Warning: History file not found -> {args.history_path}")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()