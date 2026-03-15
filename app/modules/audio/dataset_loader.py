import os
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

try:
    from .audio_preprocessing import extract_mel_spectrogram
except ImportError:
    from audio_preprocessing import extract_mel_spectrogram


SUPPORTED_EXTENSIONS = (".wav", ".mp3")
CLASS_TO_LABEL = {"real": 0, "fake": 1}


class AudioDeepfakeDataset(Dataset):
    """PyTorch dataset for audio deepfake classification."""

    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000,
        max_duration: float = 3.0,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        augment: bool = False,
    ) -> None:
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.augment = augment          # True for train, False for val/test
        self.samples: List[Tuple[str, int]] = []

        self._load_samples()

    def _load_samples(self) -> None:
        for class_name, label in CLASS_TO_LABEL.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(SUPPORTED_EXTENSIONS):
                    file_path = os.path.join(class_dir, file_name)
                    self.samples.append((file_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No audio files found in {self.root_dir}. "
                f"Expected classes: {list(CLASS_TO_LABEL.keys())} "
                f"with extensions: {SUPPORTED_EXTENSIONS}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int]]:
        file_path, label = self.samples[idx]
        mel_tensor = extract_mel_spectrogram(
            file_path=file_path,
            sample_rate=self.sample_rate,
            max_duration=self.max_duration,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            augment=self.augment,       # passed through from dataset config
        )

        # Return None for corrupted/unreadable audio; collate_fn will filter it out.
        if mel_tensor is None:
            return None

        return mel_tensor, label


def collate_skip_corrupted(batch: List[Optional[Tuple[torch.Tensor, int]]]):
    """Collate function that removes corrupted samples returned as None."""
    valid = [item for item in batch if item is not None]
    if len(valid) == 0:
        return (
            torch.empty(0, 1, 128, 1, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
        )

    features, labels = zip(*valid)
    x = torch.stack(features, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y