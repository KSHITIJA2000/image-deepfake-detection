import os
import cv2
import torch
import numpy as np

from torchvision import transforms


from app.modules.video.video_model import VideoDeepfakeModel



DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


MODEL_PATH = "video_model.pth"


NUM_FRAMES = 16

IMG_SIZE = 224



transform = transforms.Compose(
[
    transforms.ToPILImage(),

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])



# ==========================
# LOAD MODEL
# ==========================


def load_model():

    model = VideoDeepfakeModel()


    weights = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


    model.load_state_dict(
        weights
    )


    model.to(
        DEVICE
    )


    model.eval()


    return model



# ==========================
# EXTRACT VIDEO FRAMES
# ==========================


def extract_frames(
        video_path
):

    cap = cv2.VideoCapture(
        video_path
    )


    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    indexes = np.linspace(
        0,
        total-1,
        NUM_FRAMES
    ).astype(int)



    frames=[]


    idx=0


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        if idx in indexes:

            frame=cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            frame=transform(
                frame
            )


            frames.append(
                frame
            )


        idx+=1



    cap.release()



    while len(frames)<NUM_FRAMES:

        frames.append(
            frames[-1]
        )


    video=torch.stack(
        frames
    )


    # T,C,H,W
    return video.unsqueeze(0)



# ==========================
# PREDICT
# ==========================


def predict(video_path):


    model=load_model()


    video = extract_frames(
        video_path
    )


    video = video.to(
        DEVICE
    )



    with torch.no_grad():

        logits,_ = model(
            video
        )


        prob = torch.softmax(
            logits,
            dim=1
        )[0]


        fake_prob = prob[1].item()



    print("\n================")
    print("RESULT")
    print("================")


    if fake_prob >=0.5:

        print(
            "Prediction : FAKE"
        )

    else:

        print(
            "Prediction : REAL"
        )


    print(
        f"Fake Probability : {fake_prob*100:.2f}%"
    )

    print(
        f"Real Probability : {(1-fake_prob)*100:.2f}%"
    )




if __name__=="__main__":


    import sys


    video_path=sys.argv[1]


    predict(
        video_path
    )