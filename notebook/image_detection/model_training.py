import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

from image_detection_dataset import ImageDataset
from hybrid_model import HybridDeepfakeDetector 

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "dataset"))
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

if not (os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR) and os.path.exists(TEST_DIR)):
    raise FileNotFoundError(f"Dataset folders not found!\nExpected:\n{TRAIN_DIR}\n{VAL_DIR}\n{TEST_DIR}")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    train_ds = ImageDataset(TRAIN_DIR, train_transform)
    val_ds = ImageDataset(VAL_DIR, val_transform)
    test_ds = ImageDataset(TEST_DIR, val_transform)

    print("Train samples:", len(train_ds))
    print("Validation samples:", len(val_ds))
    print("Test samples:", len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    train_losses, val_losses = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HybridDeepfakeDetector().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4, weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )

    scaler = GradScaler(enabled=device.type == "cuda")

    best_val_loss = float("inf")
    patience = 5
    early_stop_counter = 0
    accumulation_steps = 1  # Simulates batch size 16

    def evaluate(loader):
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with autocast(device_type="cuda", enabled=device.type == "cuda"):
                    outputs = model(x)
                    loss = criterion(outputs, y)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss_sum += loss.item() * x.size(0)
                correct += (outputs.argmax(1) == y).sum().item()
                total += y.size(0)
        if total == 0: return 0.0, 0.0
        return loss_sum / total, correct / total

    total_epochs = 15
    for epoch in range(total_epochs):

        if epoch == 5:
            print("Unfreezing deep layers of EfficientNet...")
            for p in model.effnet.features[-3:].parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-5, weight_decay=1e-4
            )
        if epoch == 10:
            print("Unfreezing full Hybrid backbone...")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=5e-6, weight_decay=1e-4
            )

        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                outputs = model(x)
                loss = criterion(outputs, y)
                loss = loss / accumulation_steps
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running_loss += loss.item() * accumulation_steps * x.size(0)

        train_loss = running_loss / len(train_ds)
        val_loss, val_acc = evaluate(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{total_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            save_path = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models", "image_model"))
            os.makedirs(save_path, exist_ok=True)
            model_file = os.path.join(save_path, "Hybrid_Swin_EffNet_best1.pth")
            torch.save(model.state_dict(), model_file)
            print("Best hybrid model saved")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered")
                break

    # Final Evaluation
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Confusion Matrix and F1
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print("\nConfusion Matrix:\n", cm)
    print(f"F1 Score: {f1:.4f}")

    # ROC-AUC
    roc_auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # Plots
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()

    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    classes = ["Real", "Fake"]
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()