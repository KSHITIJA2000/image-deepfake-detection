import argparse
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

try:
    from app.modules.audio.dataset_loader import AudioDeepfakeDataset, collate_skip_corrupted
    from app.modules.audio.model import AudioDeepfakeCNNLSTM
except ImportError:
    from dataset_loader import AudioDeepfakeDataset, collate_skip_corrupted
    from model import AudioDeepfakeCNNLSTM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", default="data/audio/val")
    parser.add_argument("--model_path", default="models/audio_model/audio_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = AudioDeepfakeDataset(root_dir=args.val_dir, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_skip_corrupted)

    model = AudioDeepfakeCNNLSTM().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in val_loader:
            if x.numel() == 0:
                continue
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()