import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"


import warnings
warnings.filterwarnings("ignore")


import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)


import shutil
import pathlib
import traceback

from typing import Optional


from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Request
)


from fastapi.responses import (
    HTMLResponse,
    JSONResponse
)


from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates



from app.modules.image.image_detection import ImageDeepfakeDetector
from app.modules.audio.audio_detection import AudioDeepfakeDetector
from app.modules.video.video_detection import VideoDeepfakeSystem



# ==========================================================
# APP INITIALIZATION
# ==========================================================

app = FastAPI(
    title="DeepGuardX Multimodal Deepfake Detection"
)



# ==========================================================
# STATIC + TEMPLATE
# ==========================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)



TEMPLATE_DIR = pathlib.Path(
    "templates"
).resolve()



templates = Jinja2Templates(
    directory=str(TEMPLATE_DIR)
)



try:
    templates.env.cache = {}
except:
    pass




# ==========================================================
# UPLOAD DIRECTORY
# ==========================================================

UPLOAD_FOLDER = "data/uploads"


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)




# ==========================================================
# LOAD MODELS
# ==========================================================

print("\n==============================")
print("Loading Detection Models")
print("==============================\n")



image_detector = ImageDeepfakeDetector()



audio_detector = AudioDeepfakeDetector()



# Loads FusionModel internally
video_detector = VideoDeepfakeSystem()



print("\nAll Models Loaded Successfully\n")





# ==========================================================
# HOME
# ==========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request:Request):


    return templates.TemplateResponse(

        "upload.html",

        {
            "request":request
        }

    )






# ==========================================================
# SAFE CONFIDENCE
# ==========================================================

def safe_confidence(value):

    try:

        value=float(value)


        if value<=1:

            value*=100


        return round(

            max(
                0,
                min(
                    value,
                    100
                )
            ),

            2

        )


    except:

        return 0.0







# ==========================================================
# PREDICT API
# ==========================================================

@app.post(
    "/predict"
)
async def predict(

    image:Optional[UploadFile]=File(None),

    audio:Optional[UploadFile]=File(None),

    video:Optional[UploadFile]=File(None)

):


    try:



        if not image and not audio and not video:


            return JSONResponse(

                status_code=400,

                content={

                    "error":
                    "Upload image, audio or video"

                }

            )





        # ==================================================
        # IMAGE MODE
        # ==================================================

        if image:


            path=os.path.join(

                UPLOAD_FOLDER,

                image.filename

            )



            with open(path,"wb") as f:


                shutil.copyfileobj(

                    image.file,

                    f

                )



            result=image_detector.predict(

                path

            )



            if os.path.exists(path):

                os.remove(path)



            return {


                "mode":

                "image",



                "prediction":

                result[0],



                "confidence":

                safe_confidence(

                    result[1]

                ),



                "explanation":

                result[5],



                "gradcam_images":

                [

                    result[4]

                ]

                if result[4]

                else [],



                "metrics":{


                    "visual":

                    safe_confidence(result[1]),


                    "audio":0,


                    "lipsync":0,


                    "fusion":0

                }

            }








        # ==================================================
        # AUDIO MODE
        # ==================================================

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




            result=audio_detector.predict(

                path

            )



            if os.path.exists(path):

                os.remove(path)




            return {



                "mode":

                "audio",



                "prediction":

                result[0],



                "confidence":

                safe_confidence(

                    result[1]

                ),



                "explanation":

                result[5],



                "gradcam_images":

                [

                    result[4]

                ]

                if result[4]

                else [],



                "metrics":{


                    "visual":0,


                    "audio":

                    safe_confidence(result[1]),


                    "lipsync":0,


                    "fusion":0

                }

            }








        # ==================================================
        # VIDEO FUSION MODE
        # ==================================================

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



            try:



                result=video_detector.predict(

                    path

                )



                print("\n==============================")
                print("FUSION RESULT")
                print(result)
                print("==============================\n")




            except Exception as e:


                traceback.print_exc()


                result={}




            finally:



                if os.path.exists(path):

                    os.remove(path)






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

                    "Video analysis completed."

                ),



                "gradcam_images":

                result.get(

                    "gradcam_images",

                    []

                ),



                "suspicious_frames":

                result.get(

                    "suspicious_frames",

                    []

                ),



                "metrics":

                result.get(

                    "metrics",

                    {


                        "visual":0,


                        "audio":0,


                        "lipsync":0,


                        "fusion":0

                    }

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