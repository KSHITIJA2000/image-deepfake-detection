import math
import logging
from typing import Optional, List

import numpy as np
import torch
import librosa

logger = logging.getLogger(__name__)


def get_expected_time_steps(
    sample_rate: int = 16000,
    max_duration: float = 3.0,
    hop_length: int = 512,
) -> int:
    """Return expected mel time-steps after fixed-length padding/trimming."""
    max_samples = int(sample_rate * max_duration)
    return math.ceil(max_samples / hop_length) + 1


def _pad_or_trim_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with zeros or trim to an exact target length in samples."""
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
    else:
        audio = audio[:target_length]
    return audio


def _pad_or_trim_mel(mel: np.ndarray, target_time_steps: int) -> np.ndarray:
    """Pad/trim mel spectrogram time axis to fixed length."""
    current_steps = mel.shape[1]
    if current_steps < target_time_steps:
        pad_width = target_time_steps - current_steps
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel = mel[:, :target_time_steps]
    return mel


def augment_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Apply random augmentations to improve generalization.
    Only used during training, never during evaluation/inference.
    """
    # Random gain (+/- 20%)
    gain = np.random.uniform(0.8, 1.2)
    audio = audio * gain

    # Small background noise
    noise_level = np.random.uniform(0.0, 0.005)
    noise = np.random.normal(0, noise_level, len(audio))
    audio = audio + noise

    # Random time shift (up to +/- 0.1 seconds)
    max_shift = int(0.1 * sample_rate)
    shift = np.random.randint(-max_shift, max_shift)
    audio = np.roll(audio, shift)

    # Randomly zero out the shifted region to avoid wrap-around artifacts
    if shift > 0:
        audio[:shift] = 0.0
    elif shift < 0:
        audio[shift:] = 0.0

    return audio.astype(np.float32)


# Fixed dB floor — avoids instability with silent/padded regions
DB_FLOOR = -80.0
DB_CEIL  =   0.0


def extract_mel_spectrogram(
    file_path: str,
    sample_rate: int = 16000,
    max_duration: float = 3.0,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    augment: bool = False,
) -> Optional[torch.Tensor]:
    """
    Load audio and return normalized log-mel tensor shaped (1, n_mels, time_steps).
    Returns None if the file is corrupted or unreadable.

    Args:
        augment: If True, apply random augmentations (use only during training).
    """
    try:
        target_length  = int(sample_rate * max_duration)
        expected_steps = get_expected_time_steps(sample_rate, max_duration, hop_length)

        # Load and resample to 16 kHz mono.
        audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
        if audio is None or len(audio) == 0:
            logger.warning("Empty audio: %s", file_path)
            return None

        # Pad or trim waveform to fixed duration.
        audio = _pad_or_trim_audio(audio, target_length)

        # Apply augmentation during training only.
        if augment:
            audio = augment_audio(audio, sample_rate)

        # Mel spectrogram -> log scale with fixed reference.
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        # ref=1.0 gives stable dB values regardless of signal loudness.
        # Clipping to [-80, 0] dB removes extreme values from padded silence.
        log_mel = librosa.power_to_db(mel_spec, ref=1.0)
        log_mel = np.clip(log_mel, DB_FLOOR, DB_CEIL)

        # Pad/trim time axis to fixed length.
        log_mel = _pad_or_trim_mel(log_mel, expected_steps)

        # Per-sample normalization to zero mean, unit variance.
        mean = np.mean(log_mel)
        std  = np.std(log_mel) + 1e-6
        log_mel = (log_mel - mean) / std

        # Build tensor: (1, n_mels, time_steps).
        tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)

        assert tensor.shape == (1, n_mels, expected_steps), (
            f"Shape mismatch for {file_path}: expected "
            f"(1, {n_mels}, {expected_steps}), got {tuple(tensor.shape)}"
        )

        return tensor

    except AssertionError:
        raise  # let shape mismatches surface loudly
    except Exception as e:
        logger.warning("Failed to process %s: %s", file_path, e)
        return None


def extract_mel_chunks(
    file_path: str,
    sample_rate: int = 16000,
    chunk_duration: float = 3.0,
    overlap: float = 0.5,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> Optional[List[torch.Tensor]]:
    """
    Load a long audio file and split it into overlapping chunks.
    Each chunk is processed into a mel tensor exactly like extract_mel_spectrogram.

    Args:
        file_path:       Path to audio file (any length).
        chunk_duration:  Length of each chunk in seconds (default 3.0).
        overlap:         Fraction of chunk_duration to overlap between chunks (0.0–0.9).

    Returns:
        List of tensors shaped (1, n_mels, time_steps), one per chunk.
        Returns None if the file cannot be loaded.
    """
    try:
        audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
        if audio is None or len(audio) == 0:
            logger.warning("Empty audio: %s", file_path)
            return None

        chunk_samples = int(sample_rate * chunk_duration)
        step_samples  = int(chunk_samples * (1.0 - overlap))
        expected_steps = get_expected_time_steps(sample_rate, chunk_duration, hop_length)

        # If audio is shorter than one chunk, just pad and return single chunk.
        if len(audio) <= chunk_samples:
            audio_padded = _pad_or_trim_audio(audio, chunk_samples)
            tensor = _audio_to_mel_tensor(
                audio_padded, sample_rate, n_fft, hop_length, n_mels, expected_steps
            )
            return [tensor] if tensor is not None else None

        # Slide window across full audio.
        chunks = []
        start = 0
        while start < len(audio):
            end   = start + chunk_samples
            chunk = audio[start:end]

            # Pad last chunk if it's shorter than chunk_duration.
            if len(chunk) < chunk_samples:
                chunk = _pad_or_trim_audio(chunk, chunk_samples)

            tensor = _audio_to_mel_tensor(
                chunk, sample_rate, n_fft, hop_length, n_mels, expected_steps
            )
            if tensor is not None:
                chunks.append(tensor)

            start += step_samples

            # Stop if we've passed the end of the audio.
            if start >= len(audio):
                break

        if not chunks:
            logger.warning("No valid chunks extracted from %s", file_path)
            return None

        logger.info("Extracted %d chunks from %s (%.1fs audio)",
                    len(chunks), file_path, len(audio) / sample_rate)
        return chunks

    except Exception as e:
        logger.warning("Failed to extract chunks from %s: %s", file_path, e)
        return None


def _audio_to_mel_tensor(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    expected_steps: int,
) -> Optional[torch.Tensor]:
    """Convert a fixed-length audio array to a normalized mel tensor."""
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel_spec, ref=1.0)
        log_mel = np.clip(log_mel, DB_FLOOR, DB_CEIL)
        log_mel = _pad_or_trim_mel(log_mel, expected_steps)

        mean = np.mean(log_mel)
        std  = np.std(log_mel) + 1e-6
        log_mel = (log_mel - mean) / std

        return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        logger.warning("Mel conversion failed: %s", e)
        return None