import {useState} from "react";

import {
Image,
Mic,
Video,
Upload
}
from "lucide-react";


import {
detectMedia
}
from "../api/detection";



function UploadBox({
setResult,
setFile
}){


const [mode,setMode]=useState("image");

const [file,setLocalFile]=useState(null);

const [loading,setLoading]=useState(false);



async function analyze(){


if(!file){

alert("Select file");

return;

}


try{


setLoading(true);


const result=

await detectMedia(

mode,

file

);



setResult(result);

setFile(file);



}

catch(error){

console.log(error);

alert("Backend connection failed");

}


finally{

setLoading(false);

}


}




return(

<section className="upload-card">


<div className="mode-selector">


<button

className={mode==="image"?"active":""}

onClick={()=>setMode("image")}

>

<Image/>

Image

</button>



<button

className={mode==="audio"?"active":""}

onClick={()=>setMode("audio")}

>

<Mic/>

Audio

</button>



<button

className={mode==="video"?"active":""}

onClick={()=>setMode("video")}

>

<Video/>

Video

</button>


</div>



<label className="drop-zone">


<Upload size={45}/>


<h3>
Upload Media
</h3>


<p>
Image, Audio, Video supported
</p>



<input

type="file"

accept={
mode==="image"
?
"image/*"
:
mode==="audio"
?
"audio/*"
:
"video/*"
}

onChange={(e)=>{

setLocalFile(
e.target.files[0]
)

}}


/>



{

file &&

<span>

{file.name}

</span>

}



</label>



<button

className="analyze-btn"

onClick={analyze}

>

{

loading

?

"Running Neural Analysis..."

:

"START FORENSIC ANALYSIS"

}


</button>



</section>

)


}


export default UploadBox;