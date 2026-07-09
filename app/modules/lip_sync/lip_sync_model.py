import torch
import torch.nn as nn
import torchvision.models as models


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
# MODEL
# ==========================================================
class LipSyncModel(nn.Module):

    def __init__(self):

        super().__init__()

        # --------------------------------------------------
        # CNN Backbone
        # --------------------------------------------------
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

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

        cnn_features = 512

        # --------------------------------------------------
        # BiLSTM
        # --------------------------------------------------
        self.lstm = nn.LSTM(

            input_size=cnn_features,

            hidden_size=256,

            num_layers=2,

            bidirectional=True,

            batch_first=True,

            dropout=0.3

        )

        # --------------------------------------------------
        # Attention
        # --------------------------------------------------
        self.attention = Attention(256)

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------
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

    # ======================================================
    # Forward
    # ======================================================
    def forward(self, x):

        # x
        # (B,T,1,96,96)

        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        features = self.cnn(x)

        features = features.view(B, T, -1)

        lstm_out, _ = self.lstm(features)

        context = self.attention(lstm_out)

        output = self.classifier(context)

        return output