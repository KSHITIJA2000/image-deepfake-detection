import axios from "axios";


const API_URL = "http://localhost:8000";


export async function detectMedia(mode,file){

    const formData = new FormData();

    formData.append(
        mode,
        file
    );


    const response = await axios.post(

        `${API_URL}/predict`,

        formData,

        {
            headers:{
                "Content-Type":
                "multipart/form-data"
            }
        }

    );


    return response.data;

}