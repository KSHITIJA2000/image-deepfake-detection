import os
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.modules.image.image_detection import ImageDeepfakeDetector
from app.modules.audio.audio_detection import AudioDeepfakeDetector

app = FastAPI(title="Deepfake Detection")

# Mount static folder for GradCAM images
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading models...")

# Initialize models
image_detector = ImageDeepfakeDetector()
audio_detector = AudioDeepfakeDetector()

print("Models loaded successfully")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/predict")
async def predict(
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    try:
        if image is None and audio is None:
            return {"error": "Upload image or audio"}

        if image:
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            # ✅ Now correctly unpacking 6 values
            label, conf, fake, real, cam_url, explanation = image_detector.predict(image_path)

            return JSONResponse({
                "mode": "image",
                "prediction": label,
                "confidence": conf,
                "explanation": explanation, # Show this in your UI
                "gradcam_image": cam_url
            })
        # ---------- AUDIO PREDICTION ----------
        if audio:
            audio_path = os.path.join(UPLOAD_FOLDER, audio.filename)
            with open(audio_path, "wb") as buffer:
                shutil.copyfileobj(audio.file, buffer)

            label, conf, fake, real, cam_url, explanation = audio_detector.predict(audio_path)

            return JSONResponse({
                "mode": "audio",
                "prediction": label,
                "confidence": conf,
                "explanation": explanation, # Show this in your UI
                "gradcam_image": cam_url
            })
    except Exception as e:
        print("SERVER ERROR:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})