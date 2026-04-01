import os
import sys
import argparse
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

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
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

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

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train audio deepfake detection model.")
    parser.add_argument("--train_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "audio", "train"))
    parser.add_argument("--val_dir", type=str,   default=os.path.join(PROJECT_ROOT, "data", "audio", "val"))
    parser.add_argument("--save_path", type=str, default=os.path.join(PROJECT_ROOT, "models", "audio_model", "audio_model.pth"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check datasets
    for dir_path, name in [(args.train_dir, "Train"), (args.val_dir, "Validation")]:
        print(f"\n{name} directory: {dir_path}")
        if not os.path.isdir(dir_path):
            raise RuntimeError(f"{name} directory does not exist!")
        for class_name in ["real", "fake"]:
            class_dir = os.path.join(dir_path, class_name)
            if not os.path.isdir(class_dir):
                raise RuntimeError(f"Class folder {class_dir} does not exist!")
            files = [f for f in os.listdir(class_dir) if f.lower().endswith((".wav", ".mp3"))]
            print(f"  {class_name}: {len(files)} audio files found")
            if len(files) == 0:
                raise RuntimeError(f"No audio files found in {class_dir}")

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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    best_val_acc = 0.0
    epochs_no_improve = 0

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)

        print(
            f"\nEpoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Scheduler step based on val accuracy
        scheduler.step(val_acc)

        # Save best model + early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"  >> Best model saved (Val Acc: {best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  >> No improvement ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()