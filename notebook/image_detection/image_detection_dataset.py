import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from face_utils import extract_face

class ImageDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.transform = transform
        self.samples = []

        for label_dir, label in [
            (os.path.join(dataset_dir, "real"), 0),
            (os.path.join(dataset_dir, "fake"), 1)
        ]:
            for root, _, files in os.walk(label_dir):
                for file in files:
                    if file.lower().endswith((".jpg", ".png", ".jpeg")):
                        self.samples.append((os.path.join(root, file), label))

        if len(self.samples) == 0:
            raise RuntimeError("No images found in dataset!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        try:
            image = Image.open(img_path).convert("RGB")
            face = extract_face(image)
            if self.transform:
                face = self.transform(face)
            return face, torch.tensor(label, dtype=torch.long)
        except Exception:
            return self.__getitem__((index + 1) % len(self.samples))