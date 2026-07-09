import os
import cv2
import numpy as np
import librosa
import subprocess
from tqdm import tqdm
import face_alignment
import torch
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# PATH RESOLUTION
# =====================================================
FILE_PATH = os.path.abspath(__file__)
ROOT_DIR = os.path.abspath(os.path.join(FILE_PATH, "../../.."))

RAW_PATH = os.path.join(ROOT_DIR, "data", "lip_sync", "raw_videos")
OUT_PATH = os.path.join(ROOT_DIR, "data", "lip_sync", "features")

print(f"🚀 Root Directory: {ROOT_DIR}")
print(f"📁 Source Videos: {RAW_PATH}")
print(f"💾 Saving Features to: {OUT_PATH}")

os.makedirs(OUT_PATH, exist_ok=True)

# =====================================================
# DEVICE + MODEL INIT
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device=device,
    flip_input=False
)

# =====================================================
# LIP EXTRACTION (SAFE VERSION)
# OUTPUT: (T, 40)
# =====================================================
def extract_lips(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    lips_seq = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # skip alternate frames
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 2 != 0:
            continue

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preds = fa.get_landmarks(rgb)

            if preds is None:
                lips = np.zeros((20, 2), dtype=np.float32)
            else:
                lips = preds[0][48:68].astype(np.float32)

            lips_seq.append(lips.reshape(-1))  # ALWAYS (40,)

        except Exception:
            lips_seq.append(np.zeros(40, dtype=np.float32))

    cap.release()

    if len(lips_seq) == 0:
        return None

    return np.array(lips_seq, dtype=np.float32)

# =====================================================
# AUDIO EXTRACTION (FIXED FFmpeg PIPE ISSUE)
# =====================================================
def extract_audio(video_path):
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ac", "1", "-ar", "16000",
            "-f", "wav", "pipe:1"
        ]

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        audio_bytes = p.stdout.read()

        if not audio_bytes:
            return None

        # FIX: correct librosa usage
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)

        y, sr = librosa.load("temp_audio.wav", sr=16000)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T

        os.remove("temp_audio.wav")

        return mfcc.astype(np.float32)

    except Exception:
        return None

# =====================================================
# DATASET PROCESSING
# =====================================================
def process_dataset():

    for label in ["real", "fake"]:

        in_dir = os.path.join(RAW_PATH, label)
        out_dir = os.path.join(OUT_PATH, label)

        if not os.path.exists(in_dir):
            print(f"⚠️ Missing folder: {in_dir}")
            continue

        os.makedirs(out_dir, exist_ok=True)

        videos = [
            f for f in os.listdir(in_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        ]

        print(f"\nProcessing {label}: {len(videos)} videos")

        for v in tqdm(videos):

            video_path = os.path.join(in_dir, v)

            lips = extract_lips(video_path)
            audio = extract_audio(video_path)

            if lips is None:
                tqdm.write(f"❌ No face detected: {v}")
                continue

            # FIX: align audio length safely
            if audio is None:
                audio = np.zeros((lips.shape[0], 40), dtype=np.float32)
            else:
                min_len = min(len(lips), len(audio))
                lips = lips[:min_len]
                audio = audio[:min_len]

            sample = {
                "lip": lips,
                "audio": audio,
                "label": 0 if label == "real" else 1,
                "meta": {
                    "name": v,
                    "audio_status": "ok" if audio is not None else "missing"
                }
            }

            save_path = os.path.join(
                out_dir,
                f"{os.path.splitext(v)[0]}.npy"
            )

            np.save(save_path, sample)

    print("\n✅ Dataset extraction complete")


if __name__ == "__main__":
    process_dataset()