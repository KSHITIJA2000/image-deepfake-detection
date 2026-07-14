import os
import cv2
import torch
import tempfile
import shutil


from app.modules.video.frame_extraction import extract_frames
from app.modules.video.extract_audio import extract_audio_from_video

from app.modules.video.face_helper import FaceHelper

from app.modules.audio.audio_preprocessing import extract_mel_spectrogram

from app.modules.video.lip_sync_model import extract_mouth_sequence



DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



def check_shapes(video_path):


    print("\nProcessing:")
    print(video_path)



    frames_dir = extract_frames(
        video_path
    )


    audio_file = tempfile.mktemp(
        suffix=".wav"
    )



    try:


        ################################################
        # IMAGE + VIDEO FRAMES
        ################################################


        face_helper = FaceHelper(
            device=DEVICE
        )


        frame_tensors = []


        files = sorted(
            os.listdir(frames_dir)
        )


        for f in files:


            path = os.path.join(
                frames_dir,
                f
            )


            frame = cv2.imread(
                path
            )


            if frame is None:
                continue



            face = face_helper.process_frame(
                frame
            )


            if face is not None:

                frame_tensors.append(
                    face.cpu()
                )



        if len(frame_tensors) == 0:

            raise Exception(
                "No face detected"
            )



        while len(frame_tensors) < 16:

            frame_tensors.append(
                frame_tensors[-1].clone()
            )


        frame_tensors = frame_tensors[:16]



        ################################################
        # IMAGE INPUT
        ################################################


        image_input = torch.stack(
            frame_tensors
        ).mean(
            dim=0
        ).unsqueeze(0)



        ################################################
        # VIDEO INPUT
        ################################################


        video_input = torch.stack(
            frame_tensors
        ).unsqueeze(0)



        ################################################
        # AUDIO INPUT
        ################################################


        audio_path = extract_audio_from_video(
            video_path,
            audio_file
        )


        if audio_path:


            mel = extract_mel_spectrogram(
                audio_path,
                augment=False
            )


            if mel.dim() == 2:

                mel = mel.unsqueeze(0)



            audio_input = mel.unsqueeze(0)


        else:


            audio_input = torch.zeros(
                1,
                1,
                128,
                95
            )



        ################################################
        # LIP INPUT
        ################################################


        lip_input = extract_mouth_sequence(
            frames_dir,
            target_frames=20
        )



        ################################################
        # PRINT SHAPES
        ################################################


        print("\n")
        print("="*60)
        print("INFERENCE INPUT SHAPES")
        print("="*60)


        print(
            "Image:",
            image_input.shape
        )


        print(
            "Video:",
            video_input.shape
        )


        print(
            "Audio:",
            audio_input.shape
        )


        print(
            "Lip:",
            lip_input.shape
        )


        print("="*60)



        ################################################
        # EXPECTED CHECK
        ################################################


        expected = {

            "Image":
            (1,3,224,224),

            "Video":
            (1,16,3,224,224),

            "Audio":
            (1,1,128,95),

            "Lip":
            (1,20,1,96,96)

        }


        print("\nCHECK RESULT")


        print(
            "Image OK:",
            tuple(image_input.shape)==expected["Image"]
        )


        print(
            "Video OK:",
            tuple(video_input.shape)==expected["Video"]
        )


        print(
            "Audio OK:",
            tuple(audio_input.shape)==expected["Audio"]
        )


        print(
            "Lip OK:",
            tuple(lip_input.shape)==expected["Lip"]
        )



    finally:


        shutil.rmtree(
            frames_dir,
            ignore_errors=True
        )


        if os.path.exists(audio_file):

            os.remove(
                audio_file
            )




if __name__ == "__main__":


    video_path = r"data/uploads/00109_2_id00701_wavtolip.mp4"


    check_shapes(
        video_path
    )