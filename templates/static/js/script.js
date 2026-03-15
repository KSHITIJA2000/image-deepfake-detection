async function send() {
  const fileInput = document.getElementById("file");
  if (!fileInput.files.length) return alert("Select an image");

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const res = await fetch("/predict-image/", {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  document.getElementById("result").innerText =
    `Result: ${data.prediction}, Confidence: ${data.confidence}`;
}
