import Metrics from "./Metrics";
import GradCamGallery from "./GradCamGallery";


function ResultCard({
    result,
    file
}) {


    const fake =
        result.prediction?.toUpperCase() === "FAKE";



    return (

        <section className="result-card">


            <h2>
                Detection Result
            </h2>




            <div className={fake ? "fake badge" : "real badge"}>


                {
                    fake
                    ?
                    "DEEPFAKE DETECTED"
                    :
                    "AUTHENTIC MEDIA"
                }


            </div>





            <h1>
                {result.confidence}%
            </h1>





            <p>
                {result.explanation}
            </p>







            {
                file &&
                file.type.startsWith("image")
                &&

                <img

                    className="uploaded-preview"

                    src={
                        URL.createObjectURL(file)
                    }

                    alt="Uploaded media"

                />

            }








            {/* ============================
                AI FORENSIC METRICS
            ============================= */}



            {
                result.metrics &&

                <Metrics

                    metrics={result.metrics}

                />

            }









            {/* ============================
                GRADCAM EXPLANATION
            ============================= */}



            {
                result.gradcam_images &&
                result.gradcam_images.length > 0

                &&


                <div className="gradcam-container">


                    <h2>
                        GradCAM Explanation
                    </h2>



                    <GradCamGallery

                        key={
                            result.gradcam_images.join("_")
                        }

                        images={
                            result.gradcam_images
                        }

                    />


                </div>

            }





        </section>

    )

}



export default ResultCard;