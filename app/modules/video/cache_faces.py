import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import warnings

warnings.filterwarnings("ignore")

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import cv2
import torch
import random
import numpy as np


from pathlib import Path
from tqdm import tqdm

############################################################
# PROJECT ROOT
############################################################

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))

from app.modules.video.face_helper import FaceHelper

############################################################
# CONFIG
############################################################

VIDEO_ROOT = Path("data") / "video"
CACHE_ROOT = Path("data") / "cache_faces"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SEQ_LEN = 16

IMG_SIZE = 224

CACHE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

print(f"\nUsing device : {DEVICE}")

face_helper = FaceHelper(
    model_selection=0,
    min_detection_confidence=0.5,
    image_size=IMG_SIZE,
    margin=0.20,
    device=DEVICE,
)
############################################################
# ADAPTIVE TEMPORAL SAMPLING
############################################################

def sample_frame_indices(
    total_frames: int,
    train: bool,
):

    if total_frames <= SEQ_LEN:

        return list(range(total_frames))

    boundaries = np.linspace(
        0,
        total_frames,
        SEQ_LEN + 1,
        dtype=int,
    )

    indices = []

    for i in range(SEQ_LEN):

        start = boundaries[i]

        end = max(
            start + 1,
            boundaries[i + 1],
        )

        if train:

            idx = random.randint(
                start,
                end - 1,
            )

        else:

            idx = (start + end) // 2

        indices.append(idx)

    return sorted(indices)

############################################################
# EXTRACT FRAMES
############################################################

def extract_frames(
    video_path: Path,
    train: bool,
):

    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:

        cap.release()

        return torch.zeros(
            SEQ_LEN,
            3,
            IMG_SIZE,
            IMG_SIZE,
        )

    sample_indices = sample_frame_indices(
        total_frames,
        train,
    )

    frames = []

    previous_face = None

    for idx in sample_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            idx,
        )

        success, frame = cap.read()

        if not success:
            continue

        try:

            face = face_helper.crop_face(frame)

            if face is None:

                if previous_face is None:
                    continue

                face = previous_face.copy()

            else:

                previous_face = face.copy()

            face = face_helper.resize_face(face)

            tensor = face_helper.to_tensor(face)

            frames.append(
                tensor.cpu()
            )

        except Exception:

            if previous_face is not None:

                face = face_helper.resize_face(
                    previous_face
                )

                tensor = face_helper.to_tensor(face)

                frames.append(
                    tensor.cpu()
                )

    cap.release()

    if len(frames) == 0:

        blank = torch.zeros(
            3,
            IMG_SIZE,
            IMG_SIZE,
        )

        frames.append(blank)

    while len(frames) < SEQ_LEN:

        frames.append(
            frames[-1].clone()
        )

    if len(frames) > SEQ_LEN:

        frames = frames[:SEQ_LEN]

    return torch.stack(frames)
############################################################
# LOAD VIDEOS
############################################################

def load_videos(
    folder: Path,
):

    if not folder.exists():

        return []

    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.suffix.lower() in
            (
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
                ".webm",
            )
        ]
    )
############################################################
# CACHE PIPELINE
############################################################

def process_split(split):

    split_dir = VIDEO_ROOT / split

    if not split_dir.exists():

        print(f"\n[SKIP] {split_dir} not found.")

        return

    train = split == "train"

    real_dir = split_dir / "real"
    fake_dir = split_dir / "fake"

    real_videos = load_videos(real_dir)
    fake_videos = load_videos(fake_dir)

    print("\n====================================================")
    print(f"{split.upper()} SET")
    print("====================================================")
    print(f"REAL : {len(real_videos)}")
    print(f"FAKE : {len(fake_videos)}")

    cache_dir = CACHE_ROOT / split

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed = 0
    skipped = 0

    ####################################################
    # REAL
    ####################################################

    print("\nProcessing REAL Videos...\n")

    for video in tqdm(
        real_videos,
        desc=f"{split} REAL",
    ):

        try:

            frames = extract_frames(
                video,
                train=train,
            )

            torch.save(
                {
                    "frames": frames,
                    "label": 0,
                    "video_name": video.stem,
                },
                cache_dir / f"{video.stem}_real.pt",
            )

            processed += 1

        except Exception as e:

            skipped += 1

            print(
                f"[REAL ERROR] {video.name}\n{e}"
            )

    ####################################################
    # FAKE
    ####################################################

    print("\nProcessing FAKE Videos...\n")

    for video in tqdm(
        fake_videos,
        desc=f"{split} FAKE",
    ):

        try:

            frames = extract_frames(
                video,
                train=train,
            )

            torch.save(
                {
                    "frames": frames,
                    "label": 1,
                    "video_name": video.stem,
                },
                cache_dir / f"{video.stem}_fake.pt",
            )

            processed += 1

        except Exception as e:

            skipped += 1

            print(
                f"[FAKE ERROR] {video.name}\n{e}"
            )

    print("\n----------------------------------------------------")
    print(f"{split.upper()} SUMMARY")
    print("----------------------------------------------------")
    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}")
    ############################################################
# MAIN
############################################################

def main():

    print("\n====================================================")
    print("VIDEO FACE CACHE EXTRACTION")
    print("====================================================")

    print(f"Device       : {DEVICE}")
    print(f"Sequence Len : {SEQ_LEN}")
    print(f"Image Size   : {IMG_SIZE}")

    print(f"Video Root   : {VIDEO_ROOT}")
    print(f"Cache Root   : {CACHE_ROOT}")

    process_split("train")

    process_split("val")

    process_split("test")

    face_helper.close()

    print("\n====================================================")
    print("FACE CACHE EXTRACTION COMPLETED")
    print("====================================================")

    print(f"\nCache Location : {CACHE_ROOT}")

    print("""
cache_faces/

    train/
        xxxx_real.pt
        xxxx_fake.pt

    val/
        xxxx_real.pt
        xxxx_fake.pt

    test/
        xxxx_real.pt
        xxxx_fake.pt
""")


############################################################
# ENTRY POINT
############################################################

if __name__ == "__main__":

    if DEVICE.type == "cuda":

        torch.backends.cudnn.benchmark = True

        torch.set_float32_matmul_precision("high")

    with torch.inference_mode():

        main()