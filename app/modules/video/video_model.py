import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights
)

from app.modules.video.temporal_model import TemporalModel



############################################################
# VIDEO DEEPFAKE MODEL WITH TEMPORAL EXPLAINABILITY
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


        in_features = (
            self.backbone
            .classifier[1]
            .in_features
        )


        self.backbone.classifier = nn.Identity()



        ####################################################
        # FEATURE NORMALIZATION
        ####################################################

        self.feature_norm = nn.LayerNorm(
            in_features
        )



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
        # STORE TEMPORAL ATTENTION
        ####################################################

        self.temporal_attention_weights = None



        ####################################################
        # CLASSIFIER
        ####################################################

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size * 2,
                256
            ),

            nn.GELU(),

            nn.Dropout(0.4),


            nn.Linear(
                256,
                2
            )

        )



    ########################################################
    # FREEZE BACKBONE
    ########################################################

    def freeze_backbone(self):

        for p in self.backbone.parameters():

            p.requires_grad = False



    ########################################################
    # UNFREEZE BACKBONE
    ########################################################

    def unfreeze_backbone(self):

        for p in self.backbone.parameters():

            p.requires_grad = True



    ########################################################
    # FEATURE EXTRACTION
    ########################################################

    def extract_features(self,x):


        x = self.backbone(x)



        # NaN protection

        x = torch.nan_to_num(

            x,

            nan=0.0,

            posinf=1.0,

            neginf=0.0

        )



        x = self.feature_norm(x)


        return x




    ########################################################
    # GET TEMPORAL ATTENTION
    ########################################################

    def get_temporal_attention(self):


        if self.temporal_attention_weights is not None:


            return (

                self.temporal_attention_weights

                .detach()

                .cpu()

                .numpy()

            )


        return None




    ########################################################
    # FORWARD
    ########################################################

    def forward(

        self,

        x,

        return_features=False

    ):


        """
        Input:

        x:
        [B,T,C,H,W]


        Example:

        [1,16,3,224,224]

        """



        B,T,C,H,W = x.shape



        ################################################
        # FRAME FEATURE EXTRACTION
        ################################################

        x = x.view(

            B*T,

            C,

            H,

            W

        )



        features = self.extract_features(

            x

        )



        features = features.view(

            B,

            T,

            -1

        )



        ################################################
        # TEMPORAL ATTENTION
        ################################################

        context, attn = self.temporal_model(

            features

        )



        # SAVE FRAME IMPORTANCE

        self.temporal_attention_weights = attn



        ################################################
        # CLASSIFICATION
        ################################################

        logits = self.classifier(

            context

        )




        ################################################
        # RETURN FEATURES
        ################################################

        if return_features:


            return (

                context,

                logits,

                attn

            )



        return logits, attn