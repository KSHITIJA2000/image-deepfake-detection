from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os, shutil
from app.image_detection import ImageDeepfakeDetector

app = FastAPI(title="Hybrid Deepfake Detection")
app.mount("/static", StaticFiles(directory="static"), name="static")

detector = ImageDeepfakeDetector()
UPLOAD_DIR = "static/uploads" # Changed to static/uploads for easy UI access
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/upload.html", encoding="utf-8") as f:
        return f.read()

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    label, confidence, fake_p, real_p, cam_file = detector.predict(file_path)

    return {
        "prediction": label,
        "confidence": f"{confidence * 100:.2f}%",
        "fake_probability": f"{fake_p * 100:.2f}%",
        "real_probability": f"{real_p * 100:.2f}%",
        "gradcam_image": f"/static/gradcam/{cam_file}"
    }