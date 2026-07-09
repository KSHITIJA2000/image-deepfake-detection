import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import face_alignment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# ATTENTION
# ==========================================================
class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):

        weights = torch.softmax(
            self.attention(x),
            dim=1
        )

        context = torch.sum(weights * x, dim=1)

        return context


# ==========================================================
# LIPSYNC MODEL
# (IDENTICAL TO TRAINING)
# ==========================================================
class LipSyncModel(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        backbone.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.cnn = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.lstm = nn.LSTM(

            input_size=512,

            hidden_size=256,

            num_layers=2,

            batch_first=True,

            bidirectional=True,

            dropout=0.3

        )

        self.attention = Attention(256)

        self.classifier = nn.Sequential(

            nn.Linear(512,256),

            nn.BatchNorm1d(256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(256,128),

            nn.BatchNorm1d(128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Linear(128,64),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(64,2)

        )

    def forward(self, x, return_features=False):

     B, T, C, H, W = x.shape

     x = x.view(B * T, C, H, W)

     features = self.cnn(x)

     features = features.view(B, T, -1)

     lstm_out, _ = self.lstm(features)

     context = self.attention(lstm_out)

     logits = self.classifier(context)

     if return_features:
        return context, logits

     return logits

# ==========================================================
# MOUTH ROI EXTRACTION
# ==========================================================
def extract_mouth_sequence(frames_dir, target_frames=20):

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=str(DEVICE)
    )

    files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])

    sequence = []

    for file in files:

        frame = cv2.imread(os.path.join(frames_dir, file))

        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        preds = fa.get_landmarks(rgb)

        if preds is None:
            continue

        pts = preds[0][48:68]

        x1 = max(int(np.min(pts[:,0])) - 10, 0)
        y1 = max(int(np.min(pts[:,1])) - 10, 0)

        x2 = min(int(np.max(pts[:,0])) + 10, frame.shape[1])
        y2 = min(int(np.max(pts[:,1])) + 10, frame.shape[0])

        mouth = frame[y1:y2, x1:x2]

        if mouth.size == 0:
            continue

        mouth = cv2.cvtColor(
            mouth,
            cv2.COLOR_BGR2GRAY
        )

        mouth = cv2.resize(
            mouth,
            (96,96)
        )

        mouth = mouth.astype(np.float32) / 255.0

        sequence.append(mouth)

    if len(sequence) == 0:
        return None

    if len(sequence) < target_frames:

        pad = np.zeros(
            (96,96),
            dtype=np.float32
        )

        while len(sequence) < target_frames:
            sequence.append(pad)

    sequence = sequence[:target_frames]

    sequence = np.array(
        sequence,
        dtype=np.float32
    )

    sequence = np.expand_dims(
        sequence,
        axis=1
    )

    return torch.tensor(sequence).unsqueeze(0)