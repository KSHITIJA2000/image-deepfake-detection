import { useState } from "react";


function GradCamGallery({
    images
}) {


    const [cacheKey] = useState(
        () => crypto.randomUUID()
    );



    if(!images || images.length === 0) {

        return (

            <div className="gradcam-empty">

                No GradCAM evidence generated

            </div>

        );

    }





    return (

        <div className="gradcam-section">


            <h3>
                Explainable AI Evidence
            </h3>




            <div className="gallery">


                {
                    images.slice(0,5).map(

                        (img,index)=>{


                            let imageURL = img;



                            if(
                                !img.startsWith("http")
                            ) {

                                imageURL =
                                "http://localhost:8000/" +
                                img.replace(/^\/+/,"");

                            }





                            return (

                                <div

                                className="cam-card"

                                key={index}

                                >



                                    <img

                                    src={
                                        imageURL +
                                        "?cache=" +
                                        cacheKey +
                                        "_" +
                                        index
                                    }

                                    alt={
                                        `GradCAM Evidence ${index+1}`
                                    }

                                    />





                                    <p>

                                        Evidence #{index+1}

                                    </p>




                                </div>

                            );


                        }

                    )

                }



            </div>



        </div>

    );


}



export default GradCamGallery;