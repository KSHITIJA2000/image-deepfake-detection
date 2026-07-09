import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class MouthDataset(Dataset):

    def __init__(self, root_dir, augment=False):

        self.samples = []
        self.augment = augment

        for label_name, label in [("real", 0), ("fake", 1)]:

            folder = os.path.join(root_dir, label_name)

            if not os.path.exists(folder):
                continue

            for file in os.listdir(folder):

                if file.endswith(".npy"):

                    self.samples.append(
                        (
                            os.path.join(folder, file),
                            label
                        )
                    )

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        # -------------------------------------------------
        # Load
        # -------------------------------------------------
        x = np.load(path).astype(np.float32)

        # Expected shape = (20,96,96)
        if x.shape != (20,96,96):

            fixed = np.zeros((20,96,96), dtype=np.float32)

            t = min(20, x.shape[0])
            h = min(96, x.shape[1])
            w = min(96, x.shape[2])

            fixed[:t,:h,:w] = x[:t,:h,:w]

            x = fixed

        # -------------------------------------------------
        # Normalize
        # -------------------------------------------------
        x /= 255.0

        # -------------------------------------------------
        # Train Augmentation
        # -------------------------------------------------
        if self.augment:

            # Horizontal flip
            if random.random() < 0.5:
                x = np.flip(x, axis=2).copy()

            # Brightness
            if random.random() < 0.5:

                factor = random.uniform(0.85,1.15)

                x *= factor

            # Gaussian Noise
            if random.random() < 0.5:

                noise = np.random.normal(
                    0,
                    0.02,
                    x.shape
                )

                x += noise

            # Random Frame Drop
            if random.random() < 0.3:

                idx_remove = random.randint(0,19)

                if idx_remove > 0:
                    x[idx_remove] = x[idx_remove-1]

            x = np.clip(x,0,1)

        # -------------------------------------------------
        # Add Channel
        # Final Shape:
        # (20,1,96,96)
        # -------------------------------------------------
        x = np.expand_dims(x,1)

        return (
            torch.tensor(x,dtype=torch.float32),
            torch.tensor(label,dtype=torch.long)
        )