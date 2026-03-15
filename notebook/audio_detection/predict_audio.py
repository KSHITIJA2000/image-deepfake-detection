import argparse

import torch

try:
    from .audio_preprocessing import extract_mel_spectrogram
    from .model import AudioDeepfakeCNNLSTM
except ImportError:
    from audio_preprocessing import extract_mel_spectrogram
    from model import AudioDeepfakeCNNLSTM


LABELS = {0: "REAL", 1: "FAKE"}


def predict_audio_file(audio_path: str, model_path: str) -> tuple[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AudioDeepfakeCNNLSTM().to(device)
    state_dict = torch.load(model_path, map_location=device,weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    mel_tensor = extract_mel_spectrogram(audio_path)
    if mel_tensor is None:
        raise ValueError(f"Could not process audio file: {audio_path}")

    x = mel_tensor.unsqueeze(0).to(device)  # (1, 1, 128, time_steps)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    label = LABELS[int(pred_idx.item())]
    confidence = float(conf.item())
    return label, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict REAL/FAKE for a single audio file.")
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="saved_models/audio_model.pth")
    args = parser.parse_args()

    try:
        label, confidence = predict_audio_file(args.audio_path, args.model_path)
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.4f}")
    except Exception as exc:
        print(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()