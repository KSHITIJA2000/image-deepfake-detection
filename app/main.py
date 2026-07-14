import os
import shutil
import traceback

from typing import Optional


from fastapi import (
    FastAPI,
    UploadFile,
    File
)

from fastapi.responses import JSONResponse

from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware



from app.modules.image.image_detection import ImageDeepfakeDetector
from app.modules.audio.audio_detection import AudioDeepfakeDetector
from app.modules.video.video_detection import VideoDeepfakeSystem





# =====================================================
# APP
# =====================================================

app = FastAPI(
    title="DeepGuardX Multimodal Deepfake Detection"
)





# =====================================================
# CORS
# =====================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)





# =====================================================
# STATIC FILES
# =====================================================

# =====================================================
# STATIC FILES
# =====================================================

# Video GradCAM folder
os.makedirs(
    "static/gradcam",
    exist_ok=True
)


# Image GradCAM folder
os.makedirs(
    "gradcam_outputs",
    exist_ok=True
)


# Existing static files
app.mount(
    "/static",
    StaticFiles(
        directory="static"
    ),
    name="static"
)


# Image GradCAM files
app.mount(
    "/gradcam",
    StaticFiles(
        directory="gradcam_outputs"
    ),
    name="gradcam"
)





# =====================================================
# UPLOAD FOLDER
# =====================================================


UPLOAD_FOLDER = "data/uploads"


os.makedirs(

    UPLOAD_FOLDER,

    exist_ok=True

)





# =====================================================
# LOAD MODELS
# =====================================================


print("\n==============================")
print("Loading Detection Models")
print("==============================\n")



image_detector = ImageDeepfakeDetector()


audio_detector = AudioDeepfakeDetector()


video_detector = VideoDeepfakeSystem()



print("\nAll Models Loaded Successfully\n")







# =====================================================
# HELPERS
# =====================================================


def safe_confidence(value):

    try:

        value = float(value)


        if value <= 1:

            value *= 100


        return round(

            max(

                0,

                min(value,100)

            ),

            2

        )


    except:

        return 0.0





def format_gradcam(value):


    if not value:

        return []


    if isinstance(value,list):

        return value


    return [value]





def safe_remove(path):


    try:


        if os.path.exists(path):

            os.remove(path)


    except Exception as e:


        print(
            "Cleanup error:",
            e
        )








# =====================================================
# HEALTH
# =====================================================


@app.get("/health")

def health():


    return {

        "status":

        "DeepGuardX API Running"

    }








# =====================================================
# PREDICT
# =====================================================


@app.post("/predict")

async def predict(

    image: Optional[UploadFile] = File(None),

    audio: Optional[UploadFile] = File(None),

    video: Optional[UploadFile] = File(None)

):


    try:



        if not image and not audio and not video:


            return JSONResponse(

                status_code=400,

                content={

                    "error":
                    "No file uploaded"

                }

            )







        # =================================================
        # IMAGE
        # =================================================


        if image:


            path = os.path.join(

                UPLOAD_FOLDER,

                image.filename

            )


            with open(path,"wb") as f:


                shutil.copyfileobj(

                    image.file,

                    f

                )


            await image.close()



            result = image_detector.predict(path)



            safe_remove(path)



            return {


                "mode":

                "image",



                "prediction":

                result[0].upper(),



                "confidence":

                safe_confidence(result[1]),



                "explanation":

                result[5],



                "gradcam_images":

                format_gradcam(

                    result[4]

                ),



                "metrics":{


                    "Image Analysis":

                    safe_confidence(result[1]),


                    "Audio Analysis":

                    0,


                    "Lip Sync Analysis":

                    0,


                    "Fusion Decision":

                    0

                }

            }








        # =================================================
        # AUDIO
        # =================================================


        if audio:


            path=os.path.join(

                UPLOAD_FOLDER,

                audio.filename

            )



            with open(path,"wb") as f:


                shutil.copyfileobj(

                    audio.file,

                    f

                )



            await audio.close()



            result = audio_detector.predict(path)



            safe_remove(path)



            return {


                "mode":

                "audio",



                "prediction":

                result[0],



                "confidence":

                safe_confidence(result[1]),



                "explanation":

                result[5],



                "gradcam_images":

                format_gradcam(

                    result[4]

                ),



                "metrics":{


                    "Image Analysis":

                    0,


                    "Audio Analysis":

                    safe_confidence(result[1]),



                    "Lip Sync Analysis":

                    0,



                    "Fusion Decision":

                    0

                }

            }








        # =================================================
        # VIDEO
        # =================================================


        if video:


            path=os.path.join(

                UPLOAD_FOLDER,

                video.filename

            )



            with open(path,"wb") as f:


                shutil.copyfileobj(

                    video.file,

                    f

                )



            await video.close()



            result = video_detector.predict(path)



            safe_remove(path)



            return {


                "mode":

                "video",



                "prediction":

                result.get(

                    "prediction",

                    "UNKNOWN"

                ),




                "confidence":

                safe_confidence(

                    result.get(

                        "confidence",

                        0

                    )

                ),




                "explanation":

                result.get(

                    "explanation",

                    ""

                ),




                # ONLY TOP 5 GRADCAM

                "gradcam_images":

                format_gradcam(

                    result.get(

                        "gradcam_images",

                        []

                    )

                ),




                # FORENSIC SCORES

                "metrics":

                result.get(

                    "metrics",

                    {}

                )

            }







    except Exception as e:


        traceback.print_exc()



        return JSONResponse(

            status_code=500,

            content={

                "error":

                str(e)

            }

        )