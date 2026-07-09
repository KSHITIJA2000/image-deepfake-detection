import random
import torch

from pathlib import Path
from torch.utils.data import Dataset

import torchvision.transforms as T
import torchvision.transforms.functional as TF


############################################################
# GAUSSIAN NOISE
############################################################

class GaussianNoise:

    def __init__(
        self,
        std=0.015,
        p=0.30,
    ):

        self.std = std
        self.p = p

    def __call__(
        self,
        image,
    ):

        if random.random() > self.p:

            return image

        noise = torch.randn_like(image) * self.std

        image = image + noise

        return image.clamp(
            0.0,
            1.0,
        )


############################################################
# DATASET
############################################################

class VideoDataset(Dataset):

    def __init__(
        self,
        cache_dir,
        train=True,
    ):

        self.cache_dir = Path(cache_dir)

        self.train = train

        self.files = sorted(
            self.cache_dir.glob("*.pt")
        )

        ####################################################
        # CACHE METADATA
        ####################################################

        self.labels = []

        self.video_names = []

        for file in self.files:

            sample = torch.load(
                file,
                map_location="cpu",
                weights_only=False,
            )

            self.labels.append(
                sample["label"]
            )

            self.video_names.append(
                sample.get(
                    "video_name",
                    file.stem,
                )
            )

        ####################################################
        # COLOR AUGMENTATIONS
        ####################################################

        self.color_transform = T.ColorJitter(

            brightness=0.20,

            contrast=0.20,

            saturation=0.15,

            hue=0.05,

        )

        ####################################################
        # RANDOM ERASING
        ####################################################

        self.random_erasing = T.RandomErasing(

            p=0.25,

            scale=(0.02, 0.10),

            ratio=(0.30, 3.30),

            value="random",

        )

        ####################################################
        # GAUSSIAN NOISE
        ####################################################

        self.noise = GaussianNoise(

            std=0.015,

            p=0.30,

        )

        ####################################################
        # NORMALIZATION
        ####################################################

        self.normalize = T.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225],

        )
        
    def __len__(
        self,
    ):

         return len(
            self.files
        )

    def _apply_train_transforms(
        self,
        frames,
    ):

        ####################################################
        # SAME GEOMETRIC AUGMENTATION FOR ENTIRE CLIP
        ####################################################

        flip = random.random() < 0.5

        angle = random.uniform(
            -8.0,
            8.0,
        )

        translate = (
            random.uniform(-0.05, 0.05),
            random.uniform(-0.05, 0.05),
        )

        scale = random.uniform(
            0.95,
            1.05,
        )

        output = []

        for frame in frames:

            if flip:

                frame = TF.hflip(frame)

            h = frame.shape[1]
            w = frame.shape[2]

            frame = TF.affine(

                frame,

                angle=angle,

                translate=(
                    int(translate[0] * w),
                    int(translate[1] * h),
                ),

                scale=scale,

                shear=0,

            )

            ################################################
            # FRAME-LEVEL AUGMENTATIONS
            ################################################

            frame = self.color_transform(
                frame
            )

            frame = self.noise(
                frame
            )

            frame = self.random_erasing(
                frame
            )

            frame = self.normalize(
                frame
            )

            output.append(
                frame
            )

        return output

    def _apply_eval_transforms(
        self,
        frames,
    ):

        return [

            self.normalize(
                frame
            )

            for frame in frames

        ]

    def __getitem__(
        self,
        index,
    ):

        sample = torch.load(

            self.files[index],

            map_location="cpu",

            weights_only=False,

        )

        frames = sample["frames"].float()

        label = sample["label"]

        frames = [

            frame

            for frame in frames

        ]

        if self.train:

            frames = self._apply_train_transforms(
                frames
            )

        else:

            frames = self._apply_eval_transforms(
                frames
            )

        frames = torch.stack(

            frames,

            dim=0,

        )

        return (

            frames,

            torch.tensor(
                label,
                dtype=torch.long,
            ),

        )
    ############################################################
# METADATA
############################################################

    def get_labels(
        self,
    ):

        return self.labels

    def get_video_name(
        self,
        index,
    ):

        return self.video_names[index]


############################################################
# DATASET STATISTICS
############################################################

    def num_real(
        self,
    ):

        return self.labels.count(0)

    def num_fake(
        self,
    ):

        return self.labels.count(1)

    def summary(
        self,
    ):

        print("\n========================================")
        print("VIDEO DATASET SUMMARY")
        print("========================================")
        print(f"Total Samples : {len(self.files)}")
        print(f"Real Videos   : {self.num_real()}")
        print(f"Fake Videos   : {self.num_fake()}")
        print("========================================")