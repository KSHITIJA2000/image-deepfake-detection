import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageDataset(Dataset):

    def __init__(self, root_dir, train=True):

        self.root_dir = root_dir
        self.train = train

        self.data = []
        self.labels = []

        self.class_map = {
            "real": 0,
            "fake": 1
        }

        for cls in ["real", "fake"]:
            cls_path = os.path.join(root_dir, cls)

            for img_name in os.listdir(cls_path):
                self.data.append(os.path.join(cls_path, img_name))
                self.labels.append(self.class_map[cls])

        # =========================
        # AUGMENTATION PIPELINE
        # =========================
        if self.train:

            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.GaussNoise(p=0.2),
                A.Rotate(limit=10, p=0.3),
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

        else:

            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = self.data[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=img)
        img = augmented["image"]

        return img, torch.tensor(label, dtype=torch.long)