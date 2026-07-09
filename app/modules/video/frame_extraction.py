import os
import cv2
import uuid
import numpy as np  

def extract_frames(video_path, max_frames=16):
    print(f"max_frames = {max_frames}")

    if not os.path.exists(video_path):
        raise Exception(f"Video file not found: {video_path}")

    output_dir = os.path.join(
        "data",
        "temp_frames",
        str(uuid.uuid4())
    )

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Could not open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise Exception("Invalid video")

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        max_frames,
        dtype=int
    )

    saved = 0

    for idx in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

        ret, frame = cap.read()

        if not ret:
            continue

        frame_path = os.path.join(
            output_dir,
            f"frame_{saved:03d}.jpg"
        )

        cv2.imwrite(frame_path, frame)

        saved += 1

    cap.release()

    if saved == 0:
        raise Exception("No frames extracted from video")

    print(f"{saved} uniformly sampled frames extracted")

    return output_dir