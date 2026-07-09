import os
import torch

from torch.utils.data import Dataset


class FusionDatasetCached(Dataset):

    def __init__(
        self,
        cache_dir,
        split="train"
    ):

        super().__init__()

        self.cache_dir = os.path.join(
            cache_dir,
            split
        )

        self.image_dir = os.path.join(
            self.cache_dir,
            "image"
        )

        self.video_dir = os.path.join(
            self.cache_dir,
            "video"
        )

        self.audio_dir = os.path.join(
            self.cache_dir,
            "audio"
        )

        self.lip_dir = os.path.join(
            self.cache_dir,
            "lip"
        )

        self.samples = []

        self.build_dataset()
            ########################################################
    # BUILD DATASET
    ########################################################

    def build_dataset(self):

        for label_name in [

            "real",

            "fake"

        ]:

            label = 0 if label_name == "real" else 1

            image_folder = os.path.join(

                self.image_dir,

                label_name

            )

            if not os.path.exists(image_folder):

                continue

            files = sorted(

                os.listdir(image_folder)

            )

            for file in files:

                if not file.endswith(".pt"):

                    continue

                name = os.path.splitext(file)[0]

                image_path = os.path.join(

                    self.image_dir,

                    label_name,

                    file

                )

                video_path = os.path.join(

                    self.video_dir,

                    label_name,

                    file

                )

                audio_path = os.path.join(

                    self.audio_dir,

                    label_name,

                    file

                )

                lip_path = os.path.join(

                    self.lip_dir,

                    label_name,

                    file

                )

                if not os.path.exists(video_path):

                    continue

                if not os.path.exists(audio_path):

                    continue

                if not os.path.exists(lip_path):

                    continue

                self.samples.append(

                    {

                        "image": image_path,

                        "video": video_path,

                        "audio": audio_path,

                        "lip": lip_path,

                        "label": label,

                        "name": name

                    }

                )

        print()

        print("=" * 60)

        print("Fusion Dataset")

        print("Split :", os.path.basename(self.cache_dir))

        real_count = sum(
        s["label"] == 0
        for s in self.samples
        )

        fake_count = sum(
        s["label"] == 1
        for s in self.samples
        )

        print("Real :", real_count)
        print("Fake :", fake_count)
        print("Samples :", len(self.samples))

        print("=" * 60)

        print()
    ########################################################
    # LENGTH
    ########################################################

    def __len__(self):

        return len(self.samples)


    ########################################################
    # GET ITEM
    ########################################################

    def __getitem__(self, index):

        sample = self.samples[index]

        image = torch.load(
            sample["image"],
            map_location="cpu",
            weights_only=True
        ).float()

        video = torch.load(
            sample["video"],
            map_location="cpu",
            weights_only=True
        ).float()

        audio = torch.load(
            sample["audio"],
            map_location="cpu",
            weights_only=True
        ).float()

        lip = torch.load(
            sample["lip"],
            map_location="cpu",
            weights_only=True
        ).float()

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )
        assert image.shape == (3, 224, 224)

        assert video.shape == (16, 3, 224, 224)

        assert audio.shape[0] == 1
        assert audio.shape[1] == 128

        assert lip.shape == (20, 1, 96, 96)

        return {

            "image": image,

            "video": video,

            "audio": audio,

            "lip": lip,

            "label": label,

            "name": sample["name"]

        }