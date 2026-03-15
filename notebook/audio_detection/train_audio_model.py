import argparse
import os
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

try:
    from .dataset_loader import AudioDeepfakeDataset, collate_skip_corrupted
    from .model import AudioDeepfakeCNNLSTM
except ImportError:
    from dataset_loader import AudioDeepfakeDataset, collate_skip_corrupted
    from model import AudioDeepfakeCNNLSTM


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
                # Gradient clipping — prevents exploding gradients with 2-layer LSTM
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train audio deepfake detection model.")
    parser.add_argument("--data_dir",    type=str,   default="data/audio/train")
    parser.add_argument("--save_path",   type=str,   default="saved_models/audio_model.pth")
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--val_split",   type=float, default=0.2)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--patience",    type=int,   default=5,
                        help="Early stopping patience (epochs without improvement)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training dataset uses augmentation; val dataset does not.
    full_dataset = AudioDeepfakeDataset(root_dir=args.data_dir, augment=False)

    val_size   = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # reproducible split
    )

    # Wrap train subset in an augmenting dataset view
    # by enabling augment flag directly on the underlying dataset for train indices.
    # Simpler approach: create two separate dataset instances.
    train_dataset = AudioDeepfakeDataset(root_dir=args.data_dir, augment=True)
    val_dataset   = AudioDeepfakeDataset(root_dir=args.data_dir, augment=False)

    # Use same seed so train/val split is identical between both dataset instances.
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(train_dataset, [train_size, val_size], generator=generator)
    generator = torch.Generator().manual_seed(42)
    _, val_dataset   = random_split(val_dataset,   [train_size, val_size], generator=generator)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_skip_corrupted,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_skip_corrupted,
    )

    model     = AudioDeepfakeCNNLSTM().to(device)
    criterion = nn.CrossEntropyLoss()

    # weight_decay adds L2 regularization — reduces overfitting
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Reduce LR when val accuracy plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    best_val_acc     = 0.0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(
            model=model, loader=train_loader,
            criterion=criterion, device=device, optimizer=optimizer,
        )
        val_loss, val_acc = run_epoch(
            model=model, loader=val_loader,
            criterion=criterion, device=device, optimizer=None,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1:02d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Step scheduler based on val accuracy
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
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()