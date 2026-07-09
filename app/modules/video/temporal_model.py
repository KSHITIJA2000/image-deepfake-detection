import torch
import torch.nn as nn


############################################################
# TEMPORAL ATTENTION (STABLE VERSION)
############################################################

class TemporalAttention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.att = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: (B, T, H)

        scores = self.att(x)                  # (B, T, 1)
        scores = scores.squeeze(-1)          # (B, T)

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)

        context = torch.sum(weights * x, dim=1)

        return context, weights


############################################################
# TEMPORAL MODEL (FIXED FOR STABILITY)
############################################################

class TemporalModel(nn.Module):

    def __init__(
        self,
        input_size=1280,
        hidden_size=512,
        num_layers=1,          # IMPORTANT: reduce for stability
        dropout=0.3
    ):
        super().__init__()

        self.norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # IMPORTANT: no dropout in single-layer LSTM
        )

        self.attention = TemporalAttention(hidden_size)

    def forward(self, x):

        x = self.norm(x)

        lstm_out, _ = self.lstm(x)

        context, attn = self.attention(lstm_out)

        return context, attn