import os
import cv2
import numpy as np
import torch
import face_alignment
from tqdm import tqdm

# ==========================================================
# SETTINGS
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RAW_DIR = "data/lip_sync/split_videos"
SAVE_DIR = "data/lip_sync/mouth_roi"

TARGET_FRAMES = 20
ROI_SIZE = 96

print(f"Using Device : {DEVICE}")

# ==========================================================
# FACE ALIGNMENT
# ==========================================================
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device=DEVICE,
    flip_input=False
)

# ==========================================================
# MOUTH EXTRACTION
# ==========================================================
def extract_roi(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (256, 256))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        preds = fa.get_landmarks(rgb)

        if preds is None:
            continue

        pts = preds[0][48:68]

        x1 = int(np.min(pts[:, 0])) - 15
        y1 = int(np.min(pts[:, 1])) - 15
        x2 = int(np.max(pts[:, 0])) + 15
        y2 = int(np.max(pts[:, 1])) + 15

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        mouth = frame[y1:y2, x1:x2]

        if mouth.size == 0:
            continue

        mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
        mouth = cv2.resize(mouth, (ROI_SIZE, ROI_SIZE))

        frames.append(mouth)

    cap.release()

    if len(frames) == 0:
        return None

    # Uniformly sample TARGET_FRAMES
    if len(frames) >= TARGET_FRAMES:
        idx = np.linspace(
            0,
            len(frames) - 1,
            TARGET_FRAMES,
            dtype=int
        )
        frames = [frames[i] for i in idx]

    else:
        while len(frames) < TARGET_FRAMES:
            frames.append(frames[-1])

    return np.stack(frames).astype(np.uint8)

# ==========================================================
# PROCESS DATASET
# ==========================================================
def process_dataset():

    for split in ["train", "val", "test"]:

        for label in ["real", "fake"]:

            in_dir = os.path.join(RAW_DIR, split, label)
            out_dir = os.path.join(SAVE_DIR, split, label)

            os.makedirs(out_dir, exist_ok=True)

            if not os.path.exists(in_dir):
                continue

            videos = [
                f for f in os.listdir(in_dir)
                if f.lower().endswith(
                    (".mp4", ".avi", ".mov", ".mkv")
                )
            ]

            print(f"\n{split.upper()} {label.upper()} : {len(videos)} videos")

            for video in tqdm(videos):

                save_name = os.path.splitext(video)[0] + ".npy"
                save_path = os.path.join(out_dir, save_name)

                if os.path.exists(save_path):
                    continue

                roi = extract_roi(
                    os.path.join(in_dir, video)
                )

                if roi is not None:
                    np.save(save_path, roi)

    print("\nExtraction Completed Successfully.")

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    process_dataset()