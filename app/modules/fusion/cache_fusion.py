import os
import shutil
import tempfile

import cv2
import torch

from tqdm import tqdm

from app.modules.video.frame_extraction import extract_frames
from app.modules.video.extract_audio import extract_audio_from_video

from app.modules.image.face_helper import FaceHelper as ImageFaceHelper
from app.modules.video.face_helper import FaceHelper as VideoFaceHelper

from app.modules.audio.audio_preprocessing import (
    extract_mel_spectrogram,
    get_expected_time_steps
)
from app.modules.video.lip_sync_model import extract_mouth_sequence
############################################################
# AUDIO SETTINGS
############################################################

EXPECTED_STEPS = get_expected_time_steps()


############################################################
# DEVICE
############################################################

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


############################################################
# FACE HELPERS
############################################################

image_helper = ImageFaceHelper(device=DEVICE)
video_helper = VideoFaceHelper(device=DEVICE)


############################################################
# CACHE ONE VIDEO
############################################################

def cache_video(
    video_path,
    output_dir,
    num_frames=16,
    mouth_frames=20
):

    label = os.path.basename(
        os.path.dirname(video_path)
    )

    ########################################################
    # OUTPUT DIRECTORIES
    ########################################################

    image_dir = os.path.join(
        output_dir,
        "image",
        label
    )

    video_dir = os.path.join(
        output_dir,
        "video",
        label
    )

    audio_dir = os.path.join(
        output_dir,
        "audio",
        label
    )

    lip_dir = os.path.join(
        output_dir,
        "lip",
        label
    )

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(lip_dir, exist_ok=True)

    ########################################################
    # EXTRACT FRAMES
    ########################################################

    frames_dir = extract_frames(
    video_path,
    max_frames=num_frames
)

    frame_files = sorted(
        os.listdir(frames_dir)
    )

    # Only sample if extract_frames() returned more than required
    if len(frame_files) > num_frames:

        indices = torch.linspace(
            0,
            len(frame_files) - 1,
            num_frames
        ).long()

        frame_files = [
            frame_files[i]
            for i in indices
        ]

    ########################################################
    # IMAGE + VIDEO BRANCH
    ########################################################

    image_tensor = None

    video_faces = []

    for file in frame_files:

        frame_path = os.path.join(
            frames_dir,
            file
        )

        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        ####################################################
        # IMAGE BRANCH
        ####################################################

        if image_tensor is None:

            face = image_helper.process_frame(
                frame
            )

            if face is not None:
                image_tensor = face

        ####################################################
        # VIDEO BRANCH
        ####################################################

        face = video_helper.process_frame(
            frame
        )

        if face is not None:
            video_faces.append(face)

    ########################################################
    # HANDLE MISSING IMAGE
    ########################################################

    if image_tensor is None:
        print("[✓] Image face  NOT extracted")

        image_tensor = torch.zeros(
            3,
            224,
            224
        )
    else:
        print("[✓] Image face extracted")

    ########################################################
    # HANDLE MISSING VIDEO
    ########################################################

    if len(video_faces) == 0:

        video_faces = [

            torch.zeros(
                3,
                224,
                224
            )

            for _ in range(num_frames)

        ]

    while len(video_faces) < num_frames:

        video_faces.append(
            video_faces[-1].clone()
        )

    video_faces = video_faces[:num_frames]

    ########################################################
    # CONVERT TO CPU
    ########################################################

    image_tensor = image_tensor.cpu()

    video_tensor = torch.stack(
        video_faces
    ).cpu()
    print(f"[✓] Video faces extracted ({len(video_faces)} frames)")

    ########################################################
    # SAVE NAME
    ########################################################

    name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    ########################################################
    # SAVE IMAGE
    ########################################################

    torch.save(

        image_tensor,

        os.path.join(

            image_dir,

            name + ".pt"

        )

    )

    ########################################################
    # SAVE VIDEO
    ########################################################

    torch.save(

        video_tensor,

        os.path.join(

            video_dir,

            name + ".pt"

        )

    )
        ########################################################
    # AUDIO
    ########################################################

    fd, audio_file = tempfile.mkstemp(
        suffix=".wav"
    )
    os.close(fd)

    try:

        wav_path = extract_audio_from_video(
            video_path,
            audio_file
        )

        if (
            wav_path is not None
            and os.path.exists(wav_path)
        ):

            mel = extract_mel_spectrogram(
                wav_path,
                augment=False
            )

            if mel is None:

                mel = torch.zeros(
                    1,
                    128,
                    EXPECTED_STEPS
                )

        else:

            mel = torch.zeros(
                1,
                128,
                EXPECTED_STEPS
            )

    except Exception:

        mel = torch.zeros(
            1,
            128,
            EXPECTED_STEPS
        )

    torch.save(

        mel,

        os.path.join(

            audio_dir,

            name + ".pt"

        )
        

    )
    print(f"[✓] Audio extracted {tuple(mel.shape)}")

    ########################################################
    # LIP SYNC
    ########################################################

    mouth = extract_mouth_sequence(

        frames_dir,

        target_frames=mouth_frames

    )
    if mouth is not None:
      if mouth.dim() == 5:
        mouth = mouth.squeeze(0)

    if mouth is None:

        mouth = torch.zeros(

            1,

            mouth_frames,

            1,

            96,

            96

        )
        print("Saving lip shape:", mouth.shape)

    torch.save(

        mouth,

        os.path.join(

            lip_dir,

            name + ".pt"

        )

    )
    print(f"[✓] Lip sequence extracted {tuple(mouth.shape)}")

    ########################################################
    # CLEANUP
    ########################################################

    shutil.rmtree(

        frames_dir,

        ignore_errors=True

    )

    if os.path.exists(audio_file):

        os.remove(audio_file)
        
        print(f"[✓] Finished: {name}")
        print("-" * 60)
        ############################################################
# CACHE DATASET
############################################################

def cache_dataset(
    video_paths,
    output_dir
):

    for video in tqdm(
        video_paths,
        desc="Caching Fusion Dataset"
    ):

        try:

            cache_video(
                video_path=video,
                output_dir=output_dir
            )

        except Exception as e:

            print("\n" + "=" * 60)
            print(f"Error processing: {video}")
            print(e)
            print("=" * 60)


############################################################
# MAIN
############################################################

if __name__ == "__main__":

    DATASET_DIR = "data/video"

    OUTPUT_DIR = "fusion_cache"

    splits = [

        "train",

        "val",

        "test"

    ]


    for split in splits:

        print("\n")
        print("=" * 60)
        print(f"Processing {split.upper()}")
        print("=" * 60)

        split_output = os.path.join(
            OUTPUT_DIR,
            split
        )

        for label in [

            "real",

            "fake"

        ]:

            input_dir = os.path.join(

                DATASET_DIR,

                split,

                label

            )

            if not os.path.exists(input_dir):

                print(f"Skipping: {input_dir}")

                continue

            video_paths = [

                os.path.join(
                    input_dir,
                    file
                )

                for file in sorted(
                    os.listdir(input_dir)
                )

                if file.lower().endswith(

                    (

                        ".mp4",

                        ".avi",

                        ".mov",

                        ".mkv",

                        ".webm"

                    )

                )

            ]

            print(
                f"{label.capitalize()}: {len(video_paths)} videos"
            )

            cache_dataset(

                video_paths,

                split_output

            )
                ########################################################
    # FINISHED
    ########################################################

    print("\n")
    print("=" * 60)
    print("Fusion cache generation completed.")
    print("=" * 60)

    ########################################################
    # OPTIONAL SANITY CHECK
    ########################################################

    for split in ["train", "val", "test"]:

        print(f"\nChecking {split} cache...")

        for modality in [

            "image",

            "video",

            "audio",

            "lip"

        ]:

            real_dir = os.path.join(

                OUTPUT_DIR,

                split,

                modality,

                "real"

            )

            fake_dir = os.path.join(

                OUTPUT_DIR,

                split,

                modality,

                "fake"

            )

            real_count = (
                len(os.listdir(real_dir))
                if os.path.exists(real_dir)
                else 0
            )

            fake_count = (
                len(os.listdir(fake_dir))
                if os.path.exists(fake_dir)
                else 0
            )

            print(
                f"{modality:<8}  "
                f"Real: {real_count:<5} "
                f"Fake: {fake_count:<5}"
            )

    print("\nCache verification completed.")