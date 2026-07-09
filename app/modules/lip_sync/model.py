import torch
import torch.nn as nn


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
        # x -> (batch, seq_len, hidden*2)

        weights = torch.softmax(self.attention(x), dim=1)

        context = torch.sum(weights * x, dim=1)

        return context


# ==========================================================
# LIP SYNC MODEL
# ==========================================================
class LipSyncModel(nn.Module):

    def __init__(
        self,
        input_dim=40,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.3,
        num_classes=2
    ):
        super().__init__()

        self.input_dropout = nn.Dropout(0.2)

        # ---------------------------------
        # BiLSTM
        # ---------------------------------
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # ---------------------------------
        # Transformer Encoder
        # ---------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # ---------------------------------
        # Attention
        # ---------------------------------
        self.attention = Attention(hidden_dim)

        # ---------------------------------
        # Classifier
        # ---------------------------------
        self.classifier = nn.Sequential(

            nn.Linear(hidden_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.input_dropout(x)

        # BiLSTM
        x, _ = self.lstm(x)

        # Transformer
        x = self.transformer(x)

        # Attention
        x = self.attention(x)

        # Classification
        x = self.classifier(x)

        return x