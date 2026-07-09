import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageDataset(Dataset):

    def __init__(self, dataset_dir):
        self.samples = []

        if "train" in dataset_dir:
            self.transform = T.Compose([
                T.Resize((224,224)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
                T.RandomRotation(5),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225]
                )
            ])
        else:
            self.transform = T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225]
                )
            ])

        real_dir = os.path.join(dataset_dir, "real")
        fake_dir = os.path.join(dataset_dir, "fake")

        for root, _, files in os.walk(real_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.samples.append((os.path.join(root, file), 0))

        for root, _, files in os.walk(fake_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.samples.append((os.path.join(root, file), 1))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {dataset_dir}")

        print(f"[DATASET] Loaded {len(self.samples)} images from {dataset_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)