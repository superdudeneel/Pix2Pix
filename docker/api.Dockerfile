# ---- Base image ----
# Use a slim Python image matching your .python-version
FROM python:3.11-slim

# ---- Working directory inside the container ----
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy application code ----
COPY src/ ./src/

# ---- Copy model checkpoints ----
# (Skip this if checkpoints are large/handled separately, e.g. mounted volume or downloaded at startup)
COPY src/pix2pix/gen.pth.tar ./src/pix2pix/gen.pth.tar

# ---- Expose the API port ----
EXPOSE 8000

# ---- Healthcheck (matches the /health endpoint in main.py) ----
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ---- Run the API ----
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]