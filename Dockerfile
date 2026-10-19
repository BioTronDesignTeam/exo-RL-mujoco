# Reproducible CPU environment for training, evaluation, and tests.
#
#   docker build -t exo-rl .
#   docker run --rm exo-rl                                    # run the test suite
#   docker run --rm -v "$PWD/runs:/app/runs" exo-rl \
#       python scripts/train.py --name mk1 --max-minutes 60   # train; results land in ./runs
#
# Headless only: no live viewer. Video recording works (MUJOCO_GL=osmesa).
# For GPU training, start from an nvidia/cuda base image and install the CUDA PyTorch build.
FROM python:3.11-slim

# OSMesa: software OpenGL so MuJoCo can render videos without a display.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libosmesa6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so code changes don't reinstall them (layer caching).
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "pytest", "-q"]
