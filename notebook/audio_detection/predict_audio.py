import argparse
import torch

try:
    from app.modules.audio.audio_preprocessing import extract_mel_spectrogram
    from app.modules.audio.model import AudioDeepfakeCNNLSTM
except ImportError:
    from audio_preprocessing import extract_mel_spectrogram
    from model import AudioDeepfakeCNNLSTM

LABELS = {0: "REAL", 1: "FAKE"}


def predict_audio_file(audio_path: str, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AudioDeepfakeCNNLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mel_tensor = extract_mel_spectrogram(audio_path)
    if mel_tensor is None:
        raise ValueError(f"Could not process audio file: {audio_path}")

    x = mel_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    return LABELS[int(pred_idx.item())], float(conf.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--model_path", default="models/audio_model/audio_model.pth")
    args = parser.parse_args()

    label, confidence = predict_audio_file(args.audio_path, args.model_path)
    print(f"Prediction: {label}, Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()