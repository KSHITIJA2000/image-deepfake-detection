import torch
import torch.nn as nn


from app.modules.image.hybrid_model import DeepfakeDetector
from app.modules.video.video_model import VideoDeepfakeModel
from app.modules.audio.model import AudioDeepfakeCNNLSTM
from app.modules.video.lip_sync_model import LipSyncModel



class FusionModel(nn.Module):

    def __init__(
        self,
        image_model_path,
        video_model_path,
        audio_model_path,
        lip_model_path,
        embedding_dim=512,
        num_heads=8,
        num_layers=2,
        num_classes=2
    ):

        super().__init__()


        ################################################
        # LOAD IMAGE MODEL
        ################################################

        self.image_model = DeepfakeDetector()

        self.image_model.load_state_dict(
            torch.load(
                image_model_path,
                map_location="cpu"
            )
        )



        ################################################
        # LOAD VIDEO MODEL
        ################################################

        self.video_model = VideoDeepfakeModel()

        self.video_model.load_state_dict(
            torch.load(
                video_model_path,
                map_location="cpu"
            )
        )



        ################################################
        # LOAD AUDIO MODEL
        ################################################

        self.audio_model = AudioDeepfakeCNNLSTM()

        self.audio_model.load_state_dict(
            torch.load(
                audio_model_path,
                map_location="cpu"
            )
        )



        ################################################
        # LOAD LIP MODEL
        ################################################

        self.lip_model = LipSyncModel()

        lip_checkpoint = torch.load(
           lip_model_path,
           map_location="cpu"
)

        self.lip_model.load_state_dict(
        lip_checkpoint["model_state_dict"]
)


        ################################################
        # FEATURE DIMENSIONS
        ################################################

        self.image_dim = 1286
        self.video_dim = 1024
        self.audio_dim = 256
        self.lip_dim = 512



        ################################################
        # MODALITY PROJECTION
        ################################################

        self.image_proj = nn.Sequential(

            nn.Linear(
                self.image_dim,
                embedding_dim
            ),

            nn.LayerNorm(
                embedding_dim
            ),

            nn.GELU()

        )


        self.video_proj = nn.Sequential(

            nn.Linear(
                self.video_dim,
                embedding_dim
            ),

            nn.LayerNorm(
                embedding_dim
            ),

            nn.GELU()

        )


        self.audio_proj = nn.Sequential(

            nn.Linear(
                self.audio_dim,
                embedding_dim
            ),

            nn.LayerNorm(
                embedding_dim
            ),

            nn.GELU()

        )


        self.lip_proj = nn.Sequential(

            nn.Linear(
                self.lip_dim,
                embedding_dim
            ),

            nn.LayerNorm(
                embedding_dim
            ),

            nn.GELU()

        )
                ################################################
        # MODALITY TOKENS
        ################################################

        self.modality_embedding = nn.Parameter(

            torch.randn(
                1,
                4,
                embedding_dim
            )

        )


        ################################################
        # TRANSFORMER CROSS-MODAL FUSION
        ################################################

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=embedding_dim,

            nhead=num_heads,

            dim_feedforward=embedding_dim * 4,

            dropout=0.2,

            activation="gelu",

            batch_first=True

        )


        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers

        )


        ################################################
        # FUSION NORMALIZATION
        ################################################

        self.fusion_norm = nn.LayerNorm(

            embedding_dim

        )


        ################################################
        # CLASSIFIER
        ################################################

        self.classifier = nn.Sequential(

            nn.Linear(

                embedding_dim,

                256

            ),

            nn.GELU(),

            nn.Dropout(0.4),


            nn.Linear(

                256,

                128

            ),

            nn.GELU(),

            nn.Dropout(0.3),


            nn.Linear(

                128,

                num_classes

            )

        )
            ################################################
    # FREEZE PRETRAINED MODELS
    ################################################

    def freeze_backbones(self):

        for model in [

            self.image_model,

            self.video_model,

            self.audio_model,

            self.lip_model

        ]:

            for param in model.parameters():

                param.requires_grad = False



    ################################################
    # UNFREEZE PRETRAINED MODELS
    ################################################

    def unfreeze_backbones(self):

        for model in [

            self.image_model,

            self.video_model,

            self.audio_model,

            self.lip_model

        ]:

            for param in model.parameters():

                param.requires_grad = True



    ################################################
    # FREEZE ONLY BACKBONES
    # KEEP FUSION TRAINABLE
    ################################################

    def freeze_feature_extractors(self):

        self.freeze_backbones()


        trainable_modules = [

            self.image_proj,

            self.video_proj,

            self.audio_proj,

            self.lip_proj,

            self.transformer,

            self.fusion_norm,

            self.classifier

        ]


        for module in trainable_modules:

            for param in module.parameters():

                param.requires_grad = True



    ################################################
    # COUNT TRAINABLE PARAMETERS
    ################################################

    def count_trainable_parameters(self):

        total = 0

        for param in self.parameters():

            if param.requires_grad:

                total += param.numel()


        return total
        ################################################
    # FORWARD
    ################################################

    def forward(
        self,
        image,
        video,
        audio,
        lip,
        return_features=False
    ):


        ############################################
        # IMAGE FEATURE EXTRACTION
        ############################################

        image_features, _ = self.image_model(
            image,
            return_features=True
        )

        image_features = self.image_proj(
            image_features
        )


        ############################################
        # VIDEO FEATURE EXTRACTION
        ############################################

        video_features, _, _ = self.video_model(
            video,
            return_features=True
        )

        video_features = self.video_proj(
            video_features
        )


        ############################################
        # AUDIO FEATURE EXTRACTION
        ############################################

        audio_features, _ = self.audio_model(
            audio,
            return_features=True
        )

        audio_features = self.audio_proj(
            audio_features
        )


        ############################################
        # LIP FEATURE EXTRACTION
        ############################################

        lip_features, _ = self.lip_model(
            lip,
            return_features=True
        )

        lip_features = self.lip_proj(
            lip_features
        )


        ############################################
        # CREATE MODALITY TOKENS
        ############################################

        tokens = torch.stack(

            [

                image_features,

                video_features,

                audio_features,

                lip_features

            ],

            dim=1

        )
        # Shape:
        # (Batch, 4, 512)



        ############################################
        # ADD MODALITY EMBEDDING
        ############################################

        tokens = tokens + self.modality_embedding



        ############################################
        # CROSS-MODAL TRANSFORMER
        ############################################

        fused_tokens = self.transformer(
            tokens
        )
        
       # print("\n========== TOKEN CONTRIBUTION ==========")

    


        ############################################
        # POOL TOKENS
        ############################################

        fused = fused_tokens.mean(
            dim=1
        )


        fused = self.fusion_norm(
            fused
        )


        ############################################
        # CLASSIFIER
        ############################################
        
        logits = self.classifier(
            fused
        )
        

        ############################################
        # OPTIONAL FEATURES
        ############################################

        if return_features:

            return {

                "image_features": image_features,

                "video_features": video_features,

                "audio_features": audio_features,

                "lip_features": lip_features,

                "fusion_features": fused,

                "tokens": fused_tokens

            }, logits


        return logits