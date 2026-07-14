function Metrics({
    metrics
}){


    if(!metrics || Object.keys(metrics).length === 0)
        return null;



    return(

        <div className="metrics">


            <h3>
                AI Forensic Metrics
            </h3>





            {
                Object.entries(metrics).map(

                    ([key,value])=>{


                        const percent = Math.min(

                            100,

                            Number(value) || 0

                        );



                        return(

                            <div

                            key={key}

                            className="metric-item"

                            >



                                <p>

                                    {key}

                                    :

                                    {" "}

                                    <strong>

                                    {percent.toFixed(2)}%

                                    </strong>


                                </p>





                                <div className="bar">


                                    <span

                                    style={{

                                        width:
                                        percent + "%"

                                    }}

                                    >

                                    </span>


                                </div>




                            </div>

                        )


                    }

                )
            }




        </div>

    )

}



export default Metrics;