import torch
from torch import nn


class AudioDeepfakeCNNLSTM(nn.Module):
    def __init__(
        self,
        n_mels=128,
        lstm_hidden_size=128
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2)
        )

        reduced_freq = n_mels // 4
        lstm_input_size = 32 * reduced_freq

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                lstm_hidden_size * 2,
                64
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    ##################################
    # FEATURE EXTRACTION FOR FUSION
    ##################################
    def extract_features(self, x):
        x = self.feature_extractor(x)

        b, c, f, t = x.size()

        x = x.permute(
            0,
            3,
            1,
            2
        ).contiguous().view(
            b,
            t,
            c * f
        )

        lstm_out, _ = self.lstm(x)

        pooled = lstm_out.mean(dim=1)

        return pooled

    ##################################
# FORWARD
##################################
    def forward(self, x, return_features=False):

     features = self.extract_features(x)

     logits = self.classifier(features)

     if return_features:
        return features, logits

     return logits


##################################
# WRAPPER
##################################
class AudioDeepfakeDetector:
    def __init__(
        self,
        model_path="models/audio_model/audio_model.pth"
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.model = AudioDeepfakeCNNLSTM().to(
            self.device
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model.eval()

    def predict(self, features):
        with torch.no_grad():
            features = torch.tensor(
                features
            ).unsqueeze(0).unsqueeze(0).to(
                self.device
            )

            output = self.model(features)

            prob = torch.softmax(
                output,
                dim=1
            )

            fake_prob = prob[0][1].item()
            real_prob = prob[0][0].item()

            if fake_prob > real_prob:
                return "Fake", fake_prob
            else:
                return "Real", real_prob