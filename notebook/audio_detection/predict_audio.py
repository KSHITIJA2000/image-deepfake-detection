import os
import sys
import time
import argparse

import torch

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

sys.path.insert(0, PROJECT_ROOT)

from app.modules.audio.audio_preprocessing import (
    extract_mel_spectrogram
)

from app.modules.audio.model import (
    AudioDeepfakeCNNLSTM
)


LABELS = {
    0: "REAL",
    1: "FAKE"
}


def load_model(model_path, device):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    print("\nLoading model...")

    model = AudioDeepfakeCNNLSTM().to(device)

    state = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state)

    model.eval()

    print("✓ Model loaded successfully")

    return model


def predict_audio_file(

    audio_path,
    model_path,
    threshold=0.50

):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio not found:\n{audio_path}"
        )

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )

    model = load_model(
        model_path,
        device
    )

    print("\nExtracting Mel Spectrogram...")

    mel = extract_mel_spectrogram(audio_path)

    if mel is None:

        raise RuntimeError(
            "Mel Spectrogram extraction failed."
        )

    print("✓ Spectrogram extracted")

    x = mel.unsqueeze(0).to(device)

    start = time.time()

    with torch.no_grad():

        logits = model(x)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    elapsed = time.time() - start

    real_prob = probabilities[0, 0].item()
    fake_prob = probabilities[0, 1].item()

    prediction = (
        "FAKE"
        if fake_prob >= threshold
        else "REAL"
    )

    confidence = max(
        real_prob,
        fake_prob
    )

    return {

        "prediction": prediction,

        "confidence": confidence,

        "real_probability": real_prob,

        "fake_probability": fake_prob,

        "logits": logits.cpu().numpy()[0],

        "device": device,

        "time": elapsed

    }
# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Audio Deepfake Detection"
    )

    parser.add_argument(
        "--audio_path",
        required=True,
        help="Path to audio file"
    )

    parser.add_argument(
        "--model_path",
        default="models/audio_model/audio_model.pth",
        help="Path to trained model"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Fake probability threshold (default=0.50)"
    )

    args = parser.parse_args()

    try:

        result = predict_audio_file(
            audio_path=args.audio_path,
            model_path=args.model_path,
            threshold=args.threshold
        )

        print("\n" + "=" * 60)
        print("        AUDIO DEEPFAKE DETECTION")
        print("=" * 60)

        print(f"Device          : {result['device']}")
        print(f"Audio File      : {args.audio_path}")

        print("\nPrediction")
        print("-" * 60)
        print(f"Class           : {result['prediction']}")
        print(f"Confidence      : {result['confidence'] * 100:.2f}%")

        print("\nProbabilities")
        print("-" * 60)
        print(f"REAL            : {result['real_probability'] * 100:.2f}%")
        print(f"FAKE            : {result['fake_probability'] * 100:.2f}%")

        print("\nRaw Logits")
        print("-" * 60)
        print(result["logits"])

        print("\nInference Time")
        print("-" * 60)
        print(f"{result['time']:.4f} sec")

        print("=" * 60)

    except Exception as e:

        print("\n" + "=" * 60)
        print("ERROR")
        print("=" * 60)
        print(e)


if __name__ == "__main__":
    main()