import { useState } from 'react'
import './App.css'

function App() {
    const [file, setFile] = useState(null)
    const [status, setStatus] = useState("idle")
    const [imageUrl, setImageUrl] = useState(null)

    const handleFileChange = (event) => {
    // Access the selected file from the input event
        const selectedFile = event.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setStatus('idle'); // Reset status on a new file pick
        }
    };

    const handleUpload = async (event) => {
        event.preventDefault()
        if(!file){
            alert("pleas select a file first")
            return
        }

        const formdata = new FormData()
        formdata.append("file", file)
        setStatus("uploading")

        try{
            const response = await fetch(
                "http://localhost:8000/predict",
                {
                    method: "POST",
                    body: formdata
                }
            )
            if(response.ok){
                setStatus("success")
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                setImageUrl(url);
                setFile(null)
            }else{
                setStatus("error")
                console.log("response received from the backend is not good")
            }
        }catch(err){
            console.log(err)
            setStatus("error")
        }

    }

    return (
    <main className="app-shell">
      <section className="upload-card" aria-labelledby="upload-title">
      <p className="eyebrow">Image conversion</p>
      <h1 id="upload-title">Upload your file</h1>
      <p className="intro">Choose an image to begin. Your converted result will appear below.</p>
      <form onSubmit={handleUpload}>
        <label className="file-picker">
          <input
            type="file"
            onChange={handleFileChange}
            accept=".jpg,.jpeg,.png,.pdf"
          />
          <span className="upload-icon" aria-hidden="true">↑</span>
          <span className="picker-title">Click to choose a file</span>
          <span className="picker-hint">JPG, PNG, or PDF</span>
        </label>

        {file && (
          <div className="file-details">
            <span className="file-mark">FILE</span>
            <span className="file-name">{file.name}</span>
            <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
          </div>
        )}

        <button
          type="submit"
          disabled={!file || status === 'uploading'}
          className="upload-button"
        >
          {status === 'uploading' ? 'Uploading...' : 'Upload'}
        </button>
      </form>

      {status === 'success' && <p className="status-message">Your file is ready.</p>}
      {status === 'success' && <img className="result-image" src={imageUrl} alt="Converted result" />}
      {status === 'error' && <p className="status-message error">Upload failed. Please try again.</p>}
      </section>
    </main>
  );
}

export default App
