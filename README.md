# Pix2Pix — Satellite Image Translation

A PyTorch implementation of **Pix2Pix** (conditional GAN, Isola et al. 2017) trained to translate satellite imagery to map-style images (or vice versa), depending on how `data/` is arranged.

## Project Structure

```
PixPix/
├── .venv/                          # Local virtual environment (not committed)
├── data/                           # Training/validation/test image pairs
├── evaluations/                    # Generated samples, metrics, evaluation outputs
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI app — /predict and /health endpoints
│   │   ├── inference.py            # Loads gen.pth.tar, runs inference
│   │   └── schemas.py              # Request/response models
│   └── pix2pix/
│       ├── __init__.py
│       ├── config.py               # Hyperparameters, paths, device settings
│       ├── dataset.py              # Dataset class / data loading & augmentation
│       ├── discriminator_model.py  # PatchGAN discriminator architecture
│       ├── disc.pth.tar            # Saved discriminator checkpoint (training only)
│       ├── generator_model.py      # U-Net generator architecture
│       ├── gen.pth.tar             # Saved generator checkpoint (used for inference)
│       ├── train.py                # Training loop entry point
│       └── utils.py                # Checkpointing, image saving, helper functions
├── frontend/                        # Web app (uploads image, shows result)
├── docker/
│   ├── api.Dockerfile              # Builds the backend API image
│   ├── web.Dockerfile              # Builds the frontend image
│   └── docker-compose.yml          # Runs the API (and frontend, once added) together
├── tests/                          # Unit tests
├── .github/
│   └── workflows/                  # CI/CD pipelines (test, build, deploy)
├── .dockerignore
├── .gitignore
├── .python-version                 # Python version pin (e.g. via pyenv)
├── pyproject.toml                  # Project metadata / build config
├── requirements.txt                # Python dependencies
└── README.md
```

### Module overview

| File | Purpose |
|---|---|
| `config.py` | Central place for hyperparameters (learning rate, batch size, image size, epochs), device selection, and file paths. |
| `dataset.py` | Loads paired satellite/map images, applies transforms/augmentations, and serves batches via `torch.utils.data.Dataset`/`DataLoader`. |
| `generator_model.py` | U-Net-based generator that maps an input image (e.g. satellite view) to the target domain (e.g. map view). |
| `discriminator_model.py` | PatchGAN discriminator that classifies overlapping patches of the generated/real image pairs as real or fake. |
| `train.py` | Orchestrates the adversarial + L1 training loop, logging, and checkpoint saving/loading. |
| `utils.py` | Shared helpers — saving/loading checkpoints (`gen.pth.tar`, `disc.pth.tar`), saving example predictions to `evaluations/`, etc. |
| `api/main.py` | FastAPI app exposing the trained model as a REST API (`/predict`, `/health`). |
| `api/inference.py` | Loads `gen.pth.tar` once at startup and runs inference on uploaded images. |

## From Training to Deployment

Training (`train.py`) produces two checkpoint files:

- **`gen.pth.tar`** — the generator's trained weights. This is the file that actually matters for deployment — it's what turns a satellite image into a map image.
- **`disc.pth.tar`** — the discriminator's weights. Only needed if you want to resume training later; it plays no role in inference/serving.

Once training is done, `gen.pth.tar` is loaded by `src/api/inference.py` and kept in memory by the FastAPI app (`src/api/main.py`). The model is **not** retrained or reloaded per request — it's loaded once at startup and reused, which is what makes the API responsive.

The flow end-to-end:

```
train.py  →  gen.pth.tar (+ disc.pth.tar)
                   │
                   ▼
        src/api/inference.py  (loads gen.pth.tar)
                   │
                   ▼
        src/api/main.py  (FastAPI, exposes /predict)
                   │
                   ▼
   Web frontend uploads a satellite image → gets back the generated map image
```

The API is containerized (see `docker/Dockerfile.api`) so it can run consistently in any environment, and `docker/docker-compose.yml` ties the API (and eventually the web frontend) together as one deployable unit.

## Setup

### 1. Prerequisites
- Python version as pinned in `.python-version`
- (Optional but recommended) an NVIDIA GPU + CUDA drivers for reasonable training speed

### 2. Clone and enter the project
```bash
git clone <your-repo-url> PixPix
cd PixPix
```

### 3. Create and activate a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you use `pyproject.toml` for dependency management instead (e.g. with `pip` build tooling or `poetry`/`uv`), install via that instead:
```bash
pip install -e .
```

### 5. Prepare the data
Place paired training images under `data/`, split into train/val (and optionally test) folders. Each image pair is typically stored as a single side-by-side image (input | target) or as two mirrored folders — check `dataset.py` for the exact expected layout, and update the paths in `config.py` accordingly.

### 6. Train the model
```bash
python -m src.pix2pix.train
```
Checkpoints are periodically saved to `src/pix2pix/gen.pth.tar` and `src/pix2pix/disc.pth.tar`. Adjust epochs, batch size, and learning rate in `config.py`.

### 7. Resume training / run inference
If `LOAD_MODEL` (or equivalent flag in `config.py`) is enabled, `train.py` will load the existing `gen.pth.tar` / `disc.pth.tar` checkpoints before continuing. Generated sample outputs are written to `evaluations/`.

### 8. Run tests
```bash
pytest tests/
```

### 9. Serve the trained model as an API
Once `gen.pth.tar` exists (from step 6), start the backend API:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
- `GET /health` — check the API and model are up
- `POST /predict` — upload a satellite image (`multipart/form-data`, field name `file`), get back the generated map image

Interactive API docs are available at `http://localhost:8000/docs`.

### 10. Run everything with Docker
Build and start the API in a container:
```bash
docker compose -f docker/docker-compose.yml up --build
```
The API will be available at `http://localhost:8000`, exactly as it is when run locally with `uvicorn`.

### 11. Frontend
The `frontend/` folder holds the web app that lets a user upload a satellite image and view the generated map image returned by `/predict`. Point its API base URL at wherever the backend is running (`http://localhost:8000` locally, or the deployed API URL in production). Its container is built from `docker/web.Dockerfile`.

## Notes
- This README assumes a standard Pix2Pix layout (U-Net generator + PatchGAN discriminator, trained with adversarial loss + L1 pixel loss). Adjust the "Module overview" section if your implementation differs.
- Update the `data/` layout and `config.py` description above with your project's actual dataset source (e.g. the SpaceNet or Maps satellite-to-map dataset) once finalized.