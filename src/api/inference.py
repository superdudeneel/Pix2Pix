"""
Model loading + inference logic, kept separate from the API layer
so it can be unit-tested / reused without spinning up FastAPI.

NOTE: Adjust the import below to match your actual generator class
and its __init__ signature (in_channels, features, etc.) in
src/pix2pix/generator_model.py.
"""

import io
import time

import torch
from PIL import Image
from torchvision import transforms

from src.pix2pix.generator_model import Generator  # adjust class name if different

# ---- Config (mirror what train.py used, or import from src/pix2pix/config.py) ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "src/pix2pix/gen.pth.tar"
IMAGE_SIZE = 256  # match whatever size the model was trained on

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class ModelService:
    """Loads the generator once and reuses it across requests."""

    def __init__(self):
        self.model = None
        self.load()

    def load(self):
        model = Generator().to(DEVICE)  # pass constructor args if your class needs them
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

        # Adjust this key if your checkpoint dict uses a different name
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

    def is_ready(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> tuple[bytes, float]:
        """Takes raw uploaded image bytes, returns (output_png_bytes, inference_ms)."""
        start = time.time()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = _transform(image).unsqueeze(0).to(DEVICE)

        output_tensor = self.model(input_tensor)

        # Undo normalization: [-1, 1] -> [0, 1]
        output_tensor = output_tensor.squeeze(0).cpu() * 0.5 + 0.5
        output_image = transforms.ToPILImage()(output_tensor.clamp(0, 1))

        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        elapsed_ms = (time.time() - start) * 1000

        return buffer.getvalue(), elapsed_ms


# Single shared instance, imported by main.py
model_service = ModelService()