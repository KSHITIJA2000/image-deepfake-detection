import torch
from app.modules.image.image_detection import predict_image  # Adjust based on your actual function name
from app.modules.audio.predict_audio import predict_audio

def get_multimodal_prediction(image_path, audio_path):
    # 1. Get individual probabilities (assuming values between 0 and 1)
    # 0 = Real, 1 = Fake
    image_prob = predict_image(image_path)
    audio_prob = predict_audio(audio_path)
    
    # 2. Weighted Average Fusion
    # If your audio model is more accurate, give it a higher weight (e.g., 0.6)
    w_image = 0.5
    w_audio = 0.5
    
    final_score = (w_image * image_prob) + (w_audio * audio_prob)
    
    # 3. Determine Final Label
    label = "FAKE" if final_score > 0.5 else "REAL"
    
    return {
        "final_score": round(final_score, 4),
        "label": label,
        "details": {
            "image_score": round(image_prob, 4),
            "audio_score": round(audio_prob, 4)
        }
    }