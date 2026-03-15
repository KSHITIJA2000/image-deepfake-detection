import torch
import torch.nn as nn
from torchvision import models

class HybridDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. EfficientNetV2-S (Backbone)
        self.effnet = models.efficientnet_v2_s(weights=None)
        self.effnet.classifier = nn.Identity() 
        
        # 2. Swin Transformer Tiny (Backbone)
        self.swin = models.swin_t(weights=None)
        self.swin.head = nn.Identity() 

        # 3. Fused Classifier (Head)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),   # State_dict: classifier.0
            nn.GELU(),              # State_dict: classifier.1
            nn.Dropout(0.5),        # State_dict: classifier.2
            nn.Linear(512, 2)       # State_dict: classifier.3
        )

    def forward(self, x):
        f_effnet = self.effnet(x)
        f_swin = self.swin(x)
        fused = torch.cat((f_effnet, f_swin), dim=1)
        return self.classifier(fused)