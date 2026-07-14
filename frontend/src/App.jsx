import {useState} from "react";

import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import UploadBox from "./components/UploadBox";
import ResultCard from "./components/ResultCard";

import "./styles/app.css";


function App(){

const [result,setResult]=useState(null);

const [file,setFile]=useState(null);



return(

<>


<Navbar/>


<main>


<Hero/>


<UploadBox

setResult={setResult}

setFile={setFile}

/>



{
result &&

<ResultCard

result={result}

file={file}

/>

}



</main>



</>

)

}


export default App;