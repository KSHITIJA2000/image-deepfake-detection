import torch
from torch import nn


class AudioDeepfakeCNNLSTM(nn.Module):
    """CNN + LSTM model for audio deepfake detection."""

    def __init__(self, n_mels: int = 128, lstm_hidden_size: int = 128) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2 — added Dropout2d for spatial regularization
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # After two 2x2 pools: freq bins = n_mels / 4.
        reduced_freq = n_mels // 4
        lstm_input_size = 32 * reduced_freq

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,               # increased from 1 to 2 for more capacity
            batch_first=True,
            bidirectional=True,
            dropout=0.3,                # dropout between LSTM layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),          # increased from 0.3 to 0.5
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 128, T)
        x = self.feature_extractor(x)

        # x: (B, C, F, T2) -> (B, T2, C * F) for temporal LSTM.
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        lstm_out, _ = self.lstm(x)

        # Mean pooling over all time steps — better than last timestep only,
        # captures the full temporal context of the audio.
        pooled = lstm_out.mean(dim=1)

        logits = self.classifier(pooled)
        return logits