import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights
)

from app.modules.video.temporal_model import TemporalModel


############################################################
# VIDEO DEEPFAKE MODEL (FINAL STABLE VERSION)
############################################################

class VideoDeepfakeModel(nn.Module):

    def __init__(
        self,
        hidden_size=512,
        num_layers=1,
        dropout=0.3
    ):
        super().__init__()

        ####################################################
        # BACKBONE (EfficientNetV2-S)
        ####################################################

        self.backbone = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.DEFAULT
        )

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        ####################################################
        # FEATURE NORMALIZATION (IMPORTANT FIX)
        ####################################################

        self.feature_norm = nn.LayerNorm(in_features)

        ####################################################
        # TEMPORAL MODEL
        ####################################################

        self.temporal_model = TemporalModel(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        ####################################################
        # CLASSIFIER (SIMPLIFIED FOR STABILITY)
        ####################################################

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    ########################################################
    # FREEZE / UNFREEZE
    ########################################################

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    ########################################################
    # FEATURE EXTRACTION
    ########################################################

    def extract_features(self, x):
        x = self.backbone(x)

        # SAFE FIX (NaN prevention)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        x = self.feature_norm(x)

        return x

    ########################################################
    # FORWARD
    ########################################################

    ########################################################
# FORWARD
########################################################

    def forward(self, x, return_features=False):
    

     B, T, C, H, W = x.shape

     x = x.view(B * T, C, H, W)

     features = self.extract_features(x)

     features = features.view(B, T, -1)

     context, attn = self.temporal_model(features)

     logits = self.classifier(context)

     if return_features:
        return context, logits, attn

     return logits, attn