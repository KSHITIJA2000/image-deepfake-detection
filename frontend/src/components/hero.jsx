import { motion } from "framer-motion";


function Hero(){


    return (

        <section className="hero">


            <motion.h1

                initial={{
                    opacity:0,
                    y:30
                }}

                animate={{
                    opacity:1,
                    y:0
                }}

                transition={{
                    duration:0.8
                }}

            >

                AI-Powered
                <br/>

                Multimodal Deepfake Detection


            </motion.h1>



            <motion.p

                initial={{
                    opacity:0
                }}

                animate={{
                    opacity:1
                }}

                transition={{
                    delay:0.3
                }}

            >

            Detect manipulated images, audio and videos
            using advanced AI forensic analysis,
            transformer fusion and explainable AI.


            </motion.p>


        </section>

    );

}


export default Hero;