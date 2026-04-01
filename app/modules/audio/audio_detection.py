import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import cv2
from .model import AudioDeepfakeCNNLSTM
from app.modules.image.gradcam import GradCAM

class AudioDeepfakeDetector:
    def __init__(self, model_path="models/audio_model/audio_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AudioDeepfakeCNNLSTM()
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()

        # Dynamic Layer Finder for CNN block (prevents AttributeError)
        target_layer = None
        for attr in ['features', 'cnn', 'conv', 'conv_layers', 'encoder']:
            if hasattr(self.model, attr):
                target_block = getattr(self.model, attr)
                target_layer = target_block[-1] if isinstance(target_block, torch.nn.Sequential) else target_block
                break
        
        if not target_layer: 
            for module in reversed(list(self.model.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break

        self.gradcam = GradCAM(self.model, target_layer=target_layer)
        os.makedirs("static/gradcam", exist_ok=True)

    def preprocess_audio(self, audio_path):
        """Convert audio to 128x128 Mel-Spectrogram"""
        y, sr = librosa.load(audio_path, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Ensure fixed size for CNN+LSTM input
        if mel_spec.shape[1] != 128:
            mel_spec = cv2.resize(mel_spec, (128, 128))
        return mel_spec

    def explain_audio(self, label, cam):
        """XAI Reasoning: Analyzes temporal/spectral focus"""
        width = cam.shape[1]
        start_heat = np.mean(cam[:, :width//3])
        end_heat = np.mean(cam[:, 2*width//3:])
        
        if label == "FAKE":
            timing = "initial segments" if start_heat > end_heat else "concluding segments"
            return f"Anomalies detected in the {timing}. Synthetic spectral artifacts found in Mel-bands, suggesting non-human vocal synthesis."
        return "Audio harmonics and noise floor appear consistent with natural speech recordings."

    def predict(self, audio_path):
        mel = self.preprocess_audio(audio_path)
        mel_tensor = torch.tensor(mel).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # 1. Standard Inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(mel_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        label = "FAKE" if predicted.item() == 1 else "REAL"

        # 2. XAI Mode (Fixes 'cudnn RNN backward' training mode error)
        mel_tensor.requires_grad = True
        with torch.enable_grad():
            # Force RNN modules to train mode so cuDNN allows the backward pass for Grad-CAM
            for m in self.model.modules():
                if isinstance(m, (torch.nn.LSTM, torch.nn.RNN, torch.nn.GRU)): 
                    m.train()
            
            cam = self.gradcam.generate(mel_tensor, predicted.item())
            
            # Switch back to eval mode immediately
            for m in self.model.modules():
                if isinstance(m, (torch.nn.LSTM, torch.nn.RNN, torch.nn.GRU)): 
                    m.eval()

        # 3. Visual Processing (Fixes OpenCV 'Sizes do not match' error)
        filename = os.path.basename(audio_path).split(".")[0]
        spec_name = f"{filename}_audio_xai.jpg"
        spec_path = os.path.join("static/gradcam", spec_name)
        
        # Target size for visualization
        target_vis_size = (400, 400) 
        
        # Prepare Mel visual background
        mel_visual = ((mel - mel.min()) / (mel.max() - mel.min()) * 255).astype(np.uint8)
        mel_visual = cv2.applyColorMap(mel_visual, cv2.COLORMAP_VIRIDIS)
        mel_visual = cv2.resize(mel_visual, target_vis_size)
        
        # Prepare Heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, target_vis_size) # Sync sizes perfectly
        
        # Overlay and save
        overlay = cv2.addWeighted(mel_visual, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(spec_path, overlay)

        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()

        return (
            label, 
            round(confidence.item(), 4), 
            round(probs[0][1].item(), 4), # fake_prob
            round(probs[0][0].item(), 4), # real_prob
            f"/static/gradcam/{spec_name}", 
            self.explain_audio(label, cam)
        )