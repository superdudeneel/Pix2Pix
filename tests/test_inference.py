import io

import torch
from PIL import Image

from src.api.inference import ModelService


def test_predict_returns_a_normalized_png_from_uploaded_image_bytes():
    service = ModelService(
        device=torch.device("cpu"), model_factory=torch.nn.Identity, auto_load=False
    )
    service.model = torch.nn.Identity()
    source = Image.new("RGB", (32, 20), color=(120, 80, 40))
    source_bytes = io.BytesIO()
    source.save(source_bytes, format="JPEG")

    result, elapsed_ms = service.predict(source_bytes.getvalue())
    output = Image.open(io.BytesIO(result))

    assert output.format == "PNG"
    assert output.size == (256, 256)
    assert elapsed_ms >= 0
