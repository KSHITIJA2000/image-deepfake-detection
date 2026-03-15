import torch
import torch.nn as nn
from torchvision import models

class HybridDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. EfficientNetV2-S (Local features)
        weights_eff = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self.effnet = models.efficientnet_v2_s(weights=weights_eff)
        self.effnet.classifier = nn.Identity() # Outputs 1280
        
        # Freeze EfficientNet backbone initially
        for param in self.effnet.parameters():
            param.requires_grad = False
            
        # 2. Swin-Tiny (Global features)
        weights_swin = models.Swin_T_Weights.IMAGENET1K_V1
        self.swin = models.swin_t(weights=weights_swin)
        self.swin.head = nn.Identity() # Outputs 768
        
        # 3. Fused Classification Head (1280 + 768 = 2048)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2) # 2 classes (Real/Fake) for CrossEntropyLoss
        )

    def forward(self, x):
        f_effnet = self.effnet(x)
        f_swin = self.swin(x)
        fused = torch.cat((f_effnet, f_swin), dim=1)
        return self.classifier(fused)