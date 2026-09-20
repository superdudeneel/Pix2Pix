import { useState } from 'react'
import heroImg from './assets/hero.png'
import reactLogo from './assets/react.svg'
import viteLogo from './assets/vite.svg'
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
    <div style={{ padding: '20px', maxWidth: '400px' }}>
      <h3>Upload a File</h3>
      <form onSubmit={handleUpload}>
        <input
          type="file"
          onChange={handleFileChange}
          accept=".jpg,.jpeg,.png,.pdf" // Optional restriction hint
        />

        {file && (
          <div style={{ marginTop: '10px', fontSize: '14px' }}>
            <strong>Selected:</strong> {file.name} ({(file.size / 1024).toFixed(2)} KB)
          </div>
        )}

        <button
          type="submit"
          disabled={!file || status === 'uploading'}
          style={{ marginTop: '15px', display: 'block' }}
        >
          {status === 'uploading' ? 'Uploading...' : 'Upload'}
        </button>
      </form>

      {status === 'success' && <p style={{ color: 'green' }}>File uploaded! : {imageUrl}</p>}
        {status === 'success' && <img src = {imageUrl}/>}
      {status === 'error' && <p style={{ color: 'red' }}>Upload failed. Please try again.</p>}
    </div>
  );
}

export default App
