import os
import cv2
import shutil
import tempfile
import torch
import numpy as np


from app.modules.video.frame_extraction import extract_frames
from app.modules.video.extract_audio import extract_audio_from_video
from app.modules.video.gradcam import VideoGradCAM

from app.modules.image.face_helper import FaceHelper

from app.modules.audio.audio_preprocessing import extract_mel_spectrogram

from app.modules.video.lip_sync_model import extract_mouth_sequence

from app.modules.fusion.fusion_model import FusionModel


from app.config import (
    IMAGE_MODEL_PATH,
    VIDEO_MODEL_PATH,
    AUDIO_MODEL_PATH,
    LIPSYNC_MODEL_PATH,
    FUSION_MODEL_PATH
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



class VideoDeepfakeSystem:


    def __init__(self):

        print("\nLoading Fusion Detection System...\n")


        self.model = FusionModel(
            IMAGE_MODEL_PATH,
            VIDEO_MODEL_PATH,
            AUDIO_MODEL_PATH,
            LIPSYNC_MODEL_PATH,
        )
        print("\nLoading trained Fusion checkpoint...")

        print("Fusion model path:", FUSION_MODEL_PATH)
        print("Exists:", os.path.exists(FUSION_MODEL_PATH))

        checkpoint = torch.load(
        FUSION_MODEL_PATH,
        map_location=DEVICE
         )

        self.model.load_state_dict(checkpoint)

        print("Fusion checkpoint loaded successfully.")


        self.model.to(DEVICE)

        self.model.eval()
        self.gradcam = VideoGradCAM(
        self.model.video_model
        )


        self.face_helper = FaceHelper(
            device=DEVICE
        )


        print("\nFusion Model Loaded Successfully\n")



    def get_probability(self, logits):

        probs = torch.softmax(
            logits,
            dim=1
        )

        return float(
            probs[:,1].item()
        )



    def predict(self, video_path):


        print(
            "\nProcessing:",
            video_path
        )


        frames_dir = extract_frames(
            video_path
        )


        audio_file = tempfile.mktemp(
            suffix=".wav"
        )


        try:


            ################################################
            # IMAGE INPUT
            ################################################

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



                face = self.face_helper.process_frame(
                    frame
                )


                if face is not None:

                    frame_tensors.append(
                        face.cpu()
                    )



            if len(frame_tensors)==0:

                raise Exception(
                    "No face detected"
                )



            while len(frame_tensors)<16:

                frame_tensors.append(
                    frame_tensors[-1].clone()
                )


            frame_tensors = frame_tensors[:16]


            image_input = torch.stack(
                frame_tensors
            )[0].unsqueeze(0)



            image_input = image_input.to(
                DEVICE
            )



            ################################################
            # VIDEO INPUT
            ################################################

            video_input = torch.stack(
                frame_tensors
            ).unsqueeze(0).to(
                DEVICE
            )



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

                audio_input = mel.unsqueeze(0).to(
                    DEVICE
                )

            else:

                audio_input = torch.zeros(
                    1,128,128
                ).to(DEVICE)




          ################################################
# LIP INPUT
################################################

            try:
                lip_input = extract_mouth_sequence(
                frames_dir,
                target_frames=20
                 )

                if lip_input is None:
                 raise Exception("Lip extraction returned None")

                lip_input = lip_input.to(DEVICE)

            except Exception as e:
             print("\nLip extraction failed:")
             print(e)
             raise

            print("Lip tensor created:", lip_input is not None)
            print("\nINPUT SHAPES")

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



            ################################################
            # FUSION PREDICTION
            ################################################


            with torch.no_grad():

                logits = self.model(
                    image_input,
                    video_input,
                    audio_input,
                    lip_input
                )


                prob = self.get_probability(
                    logits
                )



            prediction = (
                "FAKE"
                if prob>=0.5
                else
                "REAL"
            )


            confidence = (
                prob
                if prediction=="FAKE"
                else
                1-prob
            )



            return {


                "prediction":
                prediction,


                "confidence":
                round(
                    confidence*100,
                    2
                ),


                "metrics":{


                    "fusion":
                    round(
                        prob*100,
                        2
                    )

                }

            }




        finally:


            shutil.rmtree(
                frames_dir,
                ignore_errors=True
            )


            if os.path.exists(audio_file):

                os.remove(
                    audio_file
                )