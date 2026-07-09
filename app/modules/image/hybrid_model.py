import torch
import torch.nn as nn
import torch.fft
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # EfficientNetV2-S pretrained on ImageNet
        self.backbone = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.DEFAULT
        )

        # Remove ImageNet classifier
        self.backbone.classifier = nn.Identity()

        # Feature size:
        # EfficientNet = 1280
        # FFT = 6
        self.classifier = nn.Sequential(
            nn.Linear(1286, 512),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(512, 2)
        )

    def fft_features(self, x):
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft = torch.fft.fftshift(fft)

        mag = torch.log1p(torch.abs(fft))

        h, w = mag.shape[2], mag.shape[3]

        low = mag[:, :, :h//2, :w//2].mean((2, 3))
        high = mag[:, :, h//2:, w//2:].mean((2, 3))

        return torch.cat([low, high], dim=1)

    def extract_features(self, x):
        spatial = self.backbone(x)
        fft = self.fft_features(x)

        features = torch.cat([spatial, fft], dim=1)

        return features

    def forward(self, x, return_features=False):
     features = self.extract_features(x)

     logits = self.classifier(features)

     if return_features:
        return features, logits

     return logits