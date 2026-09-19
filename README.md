# Pix2Pix — Satellite Image Translation

A PyTorch implementation of **Pix2Pix** (conditional GAN, Isola et al. 2017) trained to translate satellite imagery to map-style images (or vice versa), depending on how `data/` is arranged.

## Project Structure

```
PixPix/
├── .venv/                          # Local virtual environment (not committed)
├── data/                           # Training/validation/test image pairs
├── evaluations/                    # Generated samples, metrics, evaluation outputs
├── src/
│   └── pix2pix/
│       ├── __init__.py
│       ├── config.py               # Hyperparameters, paths, device settings
│       ├── dataset.py              # Dataset class / data loading & augmentation
│       ├── discriminator_model.py  # PatchGAN discriminator architecture
│       ├── generator_model.py      # U-Net generator architecture
│       ├── train.py                # Training loop entry point
│       ├── utils.py                # Checkpointing, image saving, helper functions
│       ├── disc.pth.tar            # Saved discriminator checkpoint
│       └── gen.pth.tar             # Saved generator checkpoint
├── tests/                          # Unit tests
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

## Notes
- This README assumes a standard Pix2Pix layout (U-Net generator + PatchGAN discriminator, trained with adversarial loss + L1 pixel loss). Adjust the "Module overview" section if your implementation differs.
- Update the `data/` layout and `config.py` description above with your project's actual dataset source (e.g. the SpaceNet or Maps satellite-to-map dataset) once finalized.