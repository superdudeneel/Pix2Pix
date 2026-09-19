"""
Pydantic models for API request/response bodies.
Only used for JSON metadata responses -- the actual image
data goes back as raw bytes (see main.py's StreamingResponse).
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictMeta(BaseModel):
    """Optional metadata you could return alongside the image,
    e.g. if you switch /predict to return JSON + base64 instead
    of raw image bytes."""
    width: int
    height: int
    inference_ms: float