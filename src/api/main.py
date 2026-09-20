"""
FastAPI app exposing the Pix2Pix generator as a REST API.

Run locally with:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Docs available at http://localhost:8000/docs
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io

from src.api.inference import model_service
from src.api.schemas import HealthResponse

app = FastAPI(title="Pix2Pix Satellite-to-Map API")

# Allow the web frontend (running on a different origin/port) to call this API.
# Tighten allow_origins to your actual frontend domain before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model() -> None:
    """Load the production model once before accepting requests."""
    model_service.load()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=model_service.is_ready())


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()

    try:
        output_bytes, inference_ms = model_service.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return StreamingResponse(
        io.BytesIO(output_bytes),
        media_type="image/png",
        headers={"X-Inference-Time-Ms": str(round(inference_ms, 2))},
    )
